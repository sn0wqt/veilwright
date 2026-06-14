"""End-to-end adversarial loop: Defender -> Attacker -> Utility Judge."""

from defender.attacker import Attacker
from defender.defender import Defender
from defender.types import (
    AdversarialIteration,
    AdversarialResult,
    AttackerOutput,
    DefenderInput,
    UtilityInput,
)
from defender.utility import UtilityJudge
from defender.utils import is_guess_correct


def run_adversarial_loop(
    text: str,
    target_attributes: list[str],
    max_iterations: int = 3,
    defender_model: str | None = None,
    attacker_model: str | None = None,
    utility_model: str | None = None,
    utility_threshold: float = 0.75,
    confidence_threshold: float = 0.7,
) -> AdversarialResult:
    """Run the adversarial loop until success or max iterations."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")
    if not text.strip():
        raise ValueError("text must be non-empty.")
    if not target_attributes:
        raise ValueError("target_attributes must contain at least one attribute.")
    if not 0.0 <= utility_threshold <= 1.0:
        raise ValueError("utility_threshold must be between 0.0 and 1.0.")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0.0 and 1.0.")

    defender = Defender(model=defender_model)
    attacker = Attacker(
        model=attacker_model,
        confidence_threshold=confidence_threshold,
    )
    utility = UtilityJudge(model=utility_model)

    iterations: list[AdversarialIteration] = []
    attacker_feedback: str | None = None
    ground_truth: dict[str, str] = {}

    for iteration in range(1, max_iterations + 1):
        defender_input = DefenderInput(
            text=text,
            target_attributes=target_attributes,
            iteration=iteration,
            attacker_feedback=attacker_feedback,
            ground_truth=ground_truth,
        )
        defender_output = defender.run(defender_input)

        if iteration == 1:
            ground_truth = defender_output.ground_truth

        attacker_output = attacker.run(defender_output.rewritten_text, target_attributes)

        guessed_attributes = _verified_successful_attributes(
            attacker_output=attacker_output,
            target_attributes=target_attributes,
            ground_truth=ground_truth,
            confidence_threshold=confidence_threshold,
        )
        attacker_output.successful_attributes = guessed_attributes
        attacker_success = bool(guessed_attributes)

        utility_output = utility.score(
            UtilityInput(
                original_text=text,
                rewritten_text=defender_output.rewritten_text,
                target_attributes=target_attributes,
            )
        )
        utility_pass = utility_output.score >= utility_threshold
        utility_output.passes = utility_pass

        iterations.append(
            AdversarialIteration(
                iteration=iteration,
                defender_output=defender_output,
                attacker_output=attacker_output,
                utility_output=utility_output,
                attacker_success=attacker_success,
                utility_pass=utility_pass,
            )
        )

        if (not attacker_success) and utility_pass:
            return AdversarialResult(
                final_rewritten_text=defender_output.rewritten_text,
                iterations=iterations,
                total_iterations=iteration,
                exit_reason="attacker_failed",
                ground_truth=ground_truth,
                final_utility_score=utility_output.score,
                success=True,
                confidence_threshold=confidence_threshold,
            )

        attacker_feedback = _build_iteration_feedback(
            attacker_output=attacker_output,
            utility_score=utility_output.score,
            utility_threshold=utility_threshold,
            utility_pass=utility_pass,
        )

    # If we reached here, the loop did not terminate successfully early
    last_iter = iterations[-1]
    if last_iter.attacker_success:
        exit_reason = "max_iterations_reached"
    else:
        exit_reason = "utility_too_low_at_max"

    return AdversarialResult(
        final_rewritten_text=last_iter.defender_output.rewritten_text,
        iterations=iterations,
        total_iterations=len(iterations),
        exit_reason=exit_reason,
        ground_truth=ground_truth,
        final_utility_score=last_iter.utility_output.score,
        success=False,
        confidence_threshold=confidence_threshold,
    )


def _verified_successful_attributes(
    attacker_output: AttackerOutput,
    target_attributes: list[str],
    ground_truth: dict[str, str],
    confidence_threshold: float,
) -> list[str]:
    """Return attributes the attacker successfully guessed after verification."""
    verified: list[str] = []
    for attr in target_attributes:
        truth = ground_truth.get(attr)
        if truth and truth.strip():
            guess = attacker_output.guesses.get(attr, "UNKNOWN")
            conf = attacker_output.confidence.get(attr, 0.0)
            if is_guess_correct(guess, truth, conf, attr, confidence_threshold):
                verified.append(attr)
        elif attr in attacker_output.successful_attributes:
            verified.append(attr)
    return verified


def _build_attacker_feedback(attacker_output: AttackerOutput) -> str:
    """Build a structured feedback string for the Defender from Attacker guesses."""
    lines = []
    for attr in attacker_output.successful_attributes:
        guess = attacker_output.guesses.get(attr, "UNKNOWN")
        if guess.upper() != "UNKNOWN":
            conf = attacker_output.confidence.get(attr, 0.0)
            reason = attacker_output.reasoning.get(attr, "No explanation provided.")
            lines.append(
                f"- {attr}: Guessed '{guess}' (confidence: {conf:.2f}). "
                f"Reasoning: {reason}"
            )
    if not lines:
        return "Attacker was unable to guess any target attributes."
    return "Attacker guesses and inference details from the previous round:\n" + "\n".join(lines)


def _build_iteration_feedback(
    attacker_output: AttackerOutput,
    utility_score: float,
    utility_threshold: float,
    utility_pass: bool,
) -> str:
    """Build privacy and utility feedback for the next Defender iteration."""
    attacker_feedback = _build_attacker_feedback(attacker_output)
    attacker_succeeded = bool(attacker_output.successful_attributes)

    if attacker_succeeded and utility_pass:
        return attacker_feedback

    if not attacker_succeeded and not utility_pass:
        return (
            "Privacy goal achieved — the attacker could not infer any target "
            "attributes.\n"
            f"However, the rewrite scored {utility_score:.0%} on utility "
            f"(threshold: {utility_threshold:.0%}).\n"
            "The rewrite removed or abstracted too much non-sensitive meaning.\n"
            "On the next iteration, focus on PRESERVING the narrative structure, "
            "relationships, and non-identifying details while keeping the target "
            "attributes hidden. Do NOT apply heavier anonymization — the privacy "
            "is already sufficient."
        )

    if attacker_succeeded and not utility_pass:
        guessed = ", ".join(attacker_output.successful_attributes)
        return (
            f"The attacker successfully guessed: {guessed}.\n"
            f"{attacker_feedback}\n"
            f"Additionally, the rewrite scored {utility_score:.0%} on utility "
            f"(threshold: {utility_threshold:.0%}) — some non-sensitive meaning "
            "was lost.\n"
            "On the next iteration, apply heavier anonymization for the guessed "
            "attributes while trying to preserve more non-sensitive narrative detail."
        )

    # Fallthrough: not attacker_succeeded and utility_pass.
    # Unreachable from run_adversarial_loop (which exits early in this case)
    # but kept for standalone callers of this function.
    return attacker_feedback
