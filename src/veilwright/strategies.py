"""Rewrite strategy definitions for the Defender agent."""

from enum import Enum


class RewriteStrategy(str, Enum):
    """The three semantic rewrite strategies available to the Defender."""

    ABSTRACTION = "abstraction"
    SHIFTING = "shifting"
    OMISSION = "omission"


STRATEGY_DESCRIPTIONS: dict[RewriteStrategy, str] = {
    RewriteStrategy.ABSTRACTION: (
        "Replace a specific identifying clue with a vaguer, more general equivalent "
        "that preserves the narrative role but removes inferability. "
        "Example: 'watched the moon landing' -> 'watched a historic space event'."
    ),
    RewriteStrategy.SHIFTING: (
        "Replace the clue with a plausible but factually different reference, "
        "adjusting surrounding details for consistency. "
        "Example: 'moon landing in 1969' -> 'fall of the Berlin Wall in 1989', "
        "then shift the narrator's implied age accordingly."
    ),
    RewriteStrategy.OMISSION: (
        "Remove the clue entirely from the text. Use this only when abstraction "
        "and shifting would destroy too much of the original meaning or create "
        "obvious inconsistencies."
    ),
}

VALID_STRATEGY_NAMES: set[str] = {s.value for s in RewriteStrategy}
