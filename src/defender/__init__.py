"""
Defender — the anonymization agent in a multi-agent semantic anonymization system.

Provides the top-level ``run_defender()`` function that the future orchestrator
will call, along with re-exports of the core types.

Example usage::

    from defender import run_defender, DefenderInput

    result = run_defender(DefenderInput(
        text="I remember watching the moon landing...",
        target_attributes=["Age", "Birth Year", "Exact Event"],
    ))
    print(result.rewritten_text)
"""

from defender.defender import Defender, DefenderError
from defender.types import DefenderInput, DefenderOutput, StrategyRecord


def run_defender(defender_input: DefenderInput) -> DefenderOutput:
    """
    Top-level convenience function for the orchestrator.

    Creates a :class:`Defender` instance (reading the API key from the
    ``GEMINI_API_KEY`` environment variable) and runs the full pipeline.

    Args:
        defender_input: Input payload with text and target attributes.

    Returns:
        The Defender's output with rewritten text and metadata.

    Raises:
        DefenderError: If the API key is missing or the LLM call fails.
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
