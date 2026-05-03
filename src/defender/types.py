"""
Shared type definitions for the Defender agent.

These types define the interface contract between the Defender,
Attacker, and Utility Judge agents in the multi-agent semantic
anonymization system.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class StrategyRecord:
    """
    Record of which rewrite strategy was applied to a single target attribute.

    Attributes:
        attribute: The target attribute this record describes (e.g. "Age").
        strategy: One of "abstraction", "shifting", or "omission".
        reasoning: LLM's chain-of-thought reasoning for choosing this strategy.
    """

    attribute: str
    strategy: str  # "abstraction" | "shifting" | "omission"
    reasoning: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyRecord:
        return cls(
            attribute=data["attribute"],
            strategy=data["strategy"],
            reasoning=data["reasoning"],
        )


@dataclass
class DefenderInput:
    """
    Input payload for the Defender agent.

    Attributes:
        text: The original sensitive text to anonymize.
        target_attributes: List of attributes to hide (e.g. ["Age", "Birth Year"]).
        iteration: Which retry iteration this is (loop counter from orchestrator).
        attacker_feedback: Populated in later iterations with the Attacker's guess
            and reasoning, so the Defender can adapt its strategy.
    """

    text: str
    target_attributes: list[str]
    iteration: int = 1
    attacker_feedback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DefenderInput:
        return cls(
            text=data["text"],
            target_attributes=data["target_attributes"],
            iteration=data.get("iteration", 1),
            attacker_feedback=data.get("attacker_feedback"),
        )


@dataclass
class DefenderOutput:
    """
    Output payload from the Defender agent.

    Attributes:
        original_text: The unmodified input text.
        rewritten_text: Semantically rewritten text with target attributes obscured.
        target_attributes: The attributes that were targeted for hiding.
        strategies_used: Strategy records for each attribute.
        confidence: Self-assessed confidence (0.0–1.0) that the rewrite is sufficient.
        iteration: Which iteration produced this output.
        syntactic_pii_found: Explicit PII tokens found by the syntactic scanner pass.
    """

    original_text: str
    rewritten_text: str
    target_attributes: list[str]
    strategies_used: list[StrategyRecord]
    confidence: float
    iteration: int
    syntactic_pii_found: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_text": self.original_text,
            "rewritten_text": self.rewritten_text,
            "target_attributes": self.target_attributes,
            "strategies_used": [s.to_dict() for s in self.strategies_used],
            "confidence": self.confidence,
            "iteration": self.iteration,
            "syntactic_pii_found": self.syntactic_pii_found,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DefenderOutput:
        return cls(
            original_text=data["original_text"],
            rewritten_text=data["rewritten_text"],
            target_attributes=data["target_attributes"],
            strategies_used=[
                StrategyRecord.from_dict(s) for s in data["strategies_used"]
            ],
            confidence=data["confidence"],
            iteration=data["iteration"],
            syntactic_pii_found=data.get("syntactic_pii_found", []),
        )
