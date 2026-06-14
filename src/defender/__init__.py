"""Defender — semantic text anonymization agent."""

from defender.defender import Defender, DefenderError
from defender.attacker import Attacker, AttackerError
from defender.utility import UtilityJudge, UtilityError
from defender.orchestrator import run_adversarial_loop
from defender.types import (
    DefenderInput,
    DefenderOutput,
    StrategyRecord,
    UtilityInput,
    UtilityOutput,
    AttackerOutput,
    AdversarialIteration,
    AdversarialResult,
)


def run_defender(defender_input: DefenderInput) -> DefenderOutput:
    """Top-level convenience function for the orchestrator.

    Creates a Defender (reads GEMINI_API_KEY from .env) and runs the pipeline.
    """
    defender = Defender()
    return defender.run(defender_input)


__all__ = [
    "Defender",
    "DefenderError",
    "Attacker",
    "AttackerError",
    "AttackerOutput",
    "UtilityJudge",
    "UtilityError",
    "DefenderInput",
    "DefenderOutput",
    "StrategyRecord",
    "UtilityInput",
    "UtilityOutput",
    "AdversarialIteration",
    "AdversarialResult",
    "run_defender",
    "run_adversarial_loop",
]
