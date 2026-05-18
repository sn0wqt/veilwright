"""Shared type definitions for the Defender agent."""

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class StrategyRecord:
    """Record of which rewrite strategy was applied to a target attribute."""

    attribute: str
    strategy: str       # "abstraction" | "shifting" | "omission"
    reasoning: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyRecord":
        return cls(
            attribute=data["attribute"],
            strategy=data["strategy"],
            reasoning=data["reasoning"],
        )


@dataclass
class DefenderInput:
    """Input payload for the Defender agent."""

    text: str
    target_attributes: list[str]
    iteration: int = 1
    attacker_feedback: str | None = None
    ground_truth: dict[str, str] = field(default_factory=dict)  # optional: user-provided ground truth

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DefenderInput":
        return cls(
            text=data["text"],
            target_attributes=data["target_attributes"],
            iteration=data.get("iteration", 1),
            attacker_feedback=data.get("attacker_feedback"),
            ground_truth=data.get("ground_truth", {}),
        )


@dataclass
class DefenderOutput:
    """Output payload from the Defender agent."""

    original_text: str
    rewritten_text: str
    target_attributes: list[str]
    strategies_used: list[StrategyRecord]
    confidence: float
    iteration: int
    syntactic_pii_found: list[str] = field(default_factory=list)
    ground_truth: dict[str, str] = field(default_factory=dict)  # attribute -> actual value

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_text": self.original_text,
            "rewritten_text": self.rewritten_text,
            "target_attributes": self.target_attributes,
            "strategies_used": [s.to_dict() for s in self.strategies_used],
            "confidence": self.confidence,
            "iteration": self.iteration,
            "syntactic_pii_found": self.syntactic_pii_found,
            "ground_truth": self.ground_truth,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DefenderOutput":
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
            ground_truth=data.get("ground_truth", {}),
        )


@dataclass
class UtilityInput:
    """Input payload for the Utility Judge."""

    original_text: str
    rewritten_text: str
    target_attributes: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UtilityOutput:
    """Output payload from the Utility Judge."""

    original_text: str
    rewritten_text: str
    score: float
    rationale: str
    passes: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_text": self.original_text,
            "rewritten_text": self.rewritten_text,
            "score": self.score,
            "rationale": self.rationale,
            "passes": self.passes,
        }
