"""Defender — semantic text anonymization agent."""

from defender.defender import Defender, DefenderError
from defender.types import DefenderInput, DefenderOutput, StrategyRecord


def run_defender(defender_input: DefenderInput) -> DefenderOutput:
    """Top-level convenience function for the orchestrator.

    Creates a Defender (reads GEMINI_API_KEY from .env) and runs the pipeline.
    """
    defender = Defender()
    return defender.run(defender_input)


__all__ = [
    "Defender",
    "DefenderError",
    "DefenderInput",
    "DefenderOutput",
    "StrategyRecord",
    "run_defender",
]
