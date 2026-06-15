"""Defender — semantic anonymization toolkit and adversarial loop."""

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

    Creates a Defender using credentials from the environment or `.env` file
    and runs the anonymization pipeline.
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
