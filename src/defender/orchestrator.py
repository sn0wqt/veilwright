"""End-to-end adversarial loop: Defender -> Attacker -> Utility Judge."""

from dataclasses import dataclass, field

from defender.attacker import Attacker, AttackerOutput
from defender.defender import Defender
from defender.types import DefenderInput, DefenderOutput, UtilityInput, UtilityOutput
from defender.utility import UtilityJudge


@dataclass
class AdversarialIteration:
    """Single iteration record for the loop."""

    iteration: int
    defender_output: DefenderOutput
    attacker_output: AttackerOutput
    utility_output: UtilityOutput
    attacker_success: bool
    utility_pass: bool

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "defender_output": self.defender_output.to_dict(),
            "attacker_output": self.attacker_output.to_dict(),
            "utility_output": self.utility_output.to_dict(),
            "attacker_success": self.attacker_success,
            "utility_pass": self.utility_pass,
        }


@dataclass
class AdversarialResult:
    """Final result from the adversarial loop."""

    iterations: list[AdversarialIteration] = field(default_factory=list)
    success: bool = False
    stop_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "stop_reason": self.stop_reason,
            "iterations": [it.to_dict() for it in self.iterations],
        }


def run_adversarial_loop(
    text: str,
    target_attributes: list[str],
    max_iterations: int = 3,
    defender_model: str | None = None,
    attacker_model: str | None = None,
    utility_model: str | None = None,
    utility_threshold: float = 0.75,
) -> AdversarialResult:
    """Run the adversarial loop until success or max iterations."""
    defender = Defender(model=defender_model)
    attacker = Attacker(model=attacker_model)
    utility = UtilityJudge(model=utility_model)

    iterations: list[AdversarialIteration] = []
    attacker_feedback: str | None = None

    for iteration in range(1, max_iterations + 1):
        defender_input = DefenderInput(
            text=text,
            target_attributes=target_attributes,
            iteration=iteration,
            attacker_feedback=attacker_feedback,
        )
        defender_output = defender.run(defender_input)

        attacker_output = attacker.run(defender_output.rewritten_text)
        attacker_success = attacker_output.has_any_entities()

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
                iterations=iterations,
                success=True,
                stop_reason="attacker_failed_and_utility_passed",
            )

        attacker_feedback = _build_attacker_feedback(attacker_output)

    return AdversarialResult(
        iterations=iterations,
        success=False,
        stop_reason="max_iterations_reached",
    )


def _build_attacker_feedback(attacker_output: AttackerOutput) -> str:
    """Build a compact feedback string for the Defender."""
    return (
        "Attacker guesses: "
        f"PERSON={attacker_output.person}; "
        f"LOC={attacker_output.loc}; "
        f"ORG={attacker_output.org}; "
        f"DATE={attacker_output.date}."
    )
