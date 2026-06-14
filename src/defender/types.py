"""Shared type definitions for the Defender agent."""

from dataclasses import dataclass, field


@dataclass
class StrategyRecord:
    """Record of which rewrite strategy was applied to a target attribute."""

    attribute: str
    strategy: str       # "abstraction" | "shifting" | "omission"
    reasoning: str

    def to_dict(self) -> dict[str, str]:
        return {
            "attribute": self.attribute,
            "strategy": self.strategy,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "StrategyRecord":
        return cls(
            attribute=str(data.get("attribute", "")),
            strategy=str(data.get("strategy", "omission")),
            reasoning=str(data.get("reasoning", "")),
        )


@dataclass
class DefenderInput:
    """Input payload for the Defender agent."""

    text: str
    target_attributes: list[str]
    iteration: int = 1
    attacker_feedback: str | None = None
    ground_truth: dict[str, str] = field(default_factory=dict)  # optional: user-provided ground truth

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "target_attributes": self.target_attributes,
            "iteration": self.iteration,
            "attacker_feedback": self.attacker_feedback,
            "ground_truth": self.ground_truth,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "DefenderInput":
        return cls(
            text=str(data.get("text", "")),
            target_attributes=_string_list(data.get("target_attributes", [])),
            iteration=_int_value(data.get("iteration"), 1),
            attacker_feedback=_optional_str(data.get("attacker_feedback")),
            ground_truth=_string_dict(data.get("ground_truth", {})),
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
    clue_map: dict[str, list[dict[str, str]]] = field(default_factory=dict)  # attribute -> list of clue dicts (from pre-pass)

    def to_dict(self) -> dict[str, object]:
        return {
            "original_text": self.original_text,
            "rewritten_text": self.rewritten_text,
            "target_attributes": self.target_attributes,
            "strategies_used": [s.to_dict() for s in self.strategies_used],
            "confidence": self.confidence,
            "iteration": self.iteration,
            "syntactic_pii_found": self.syntactic_pii_found,
            "ground_truth": self.ground_truth,
            "clue_map": self.clue_map,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "DefenderOutput":
        return cls(
            original_text=str(data.get("original_text", "")),
            rewritten_text=str(data.get("rewritten_text", "")),
            target_attributes=_string_list(data.get("target_attributes", [])),
            strategies_used=_strategy_records(data.get("strategies_used", [])),
            confidence=_float_value(data.get("confidence"), 0.0),
            iteration=_int_value(data.get("iteration"), 1),
            syntactic_pii_found=_string_list(data.get("syntactic_pii_found", [])),
            ground_truth=_string_dict(data.get("ground_truth", {})),
            clue_map=_clue_map(data.get("clue_map", {})),
        )


@dataclass
class UtilityInput:
    """Input payload for the Utility Judge."""

    original_text: str
    rewritten_text: str
    target_attributes: list[str] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "original_text": self.original_text,
            "rewritten_text": self.rewritten_text,
            "target_attributes": self.target_attributes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "UtilityInput":
        """Build UtilityInput from a possibly partial dictionary."""
        target_value = data.get("target_attributes")
        return cls(
            original_text=str(data.get("original_text", "")),
            rewritten_text=str(data.get("rewritten_text", "")),
            target_attributes=(
                _string_list(target_value) if target_value is not None else None
            ),
        )


@dataclass
class UtilityOutput:
    """Output payload from the Utility Judge."""

    original_text: str
    rewritten_text: str
    score: float
    rationale: str
    passes: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "original_text": self.original_text,
            "rewritten_text": self.rewritten_text,
            "score": self.score,
            "rationale": self.rationale,
            "passes": self.passes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "UtilityOutput":
        return cls(
            original_text=str(data.get("original_text", "")),
            rewritten_text=str(data.get("rewritten_text", "")),
            score=_float_value(data.get("score"), 0.0),
            rationale=str(data.get("rationale", "")),
            passes=bool(data.get("passes", False)),
        )


@dataclass
class AttackerOutput:
    """Parsed Attacker results."""

    guesses: dict[str, str]           # attribute -> best guess value (or "UNKNOWN")
    reasoning: dict[str, str]         # attribute -> explanation of inference chain
    confidence: dict[str, float]      # attribute -> confidence (0.0 to 1.0)
    successful_attributes: list[str]  # self-reported or orchestrator-verified successes
    raw_response: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "guesses": self.guesses,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "successful_attributes": self.successful_attributes,
            "raw_response": self.raw_response,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "AttackerOutput":
        return cls(
            guesses=_string_dict(data.get("guesses", {})),
            reasoning=_string_dict(data.get("reasoning", {})),
            confidence=_float_dict(data.get("confidence", {})),
            successful_attributes=_string_list(data.get("successful_attributes", [])),
            raw_response=str(data.get("raw_response", "")),
        )


@dataclass
class AdversarialIteration:
    """Single iteration record for the loop."""

    iteration: int
    defender_output: DefenderOutput
    attacker_output: AttackerOutput
    utility_output: UtilityOutput
    attacker_success: bool
    utility_pass: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "defender_output": self.defender_output.to_dict(),
            "attacker_output": self.attacker_output.to_dict(),
            "utility_output": self.utility_output.to_dict(),
            "attacker_success": self.attacker_success,
            "utility_pass": self.utility_pass,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "AdversarialIteration":
        defender_data = data.get("defender_output", {})
        attacker_data = data.get("attacker_output", {})
        utility_data = data.get("utility_output", {})
        return cls(
            iteration=_int_value(data.get("iteration"), 1),
            defender_output=DefenderOutput.from_dict(
                defender_data if isinstance(defender_data, dict) else {}
            ),
            attacker_output=AttackerOutput.from_dict(
                attacker_data if isinstance(attacker_data, dict) else {}
            ),
            utility_output=UtilityOutput.from_dict(
                utility_data if isinstance(utility_data, dict) else {}
            ),
            attacker_success=bool(data.get("attacker_success", False)),
            utility_pass=bool(data.get("utility_pass", False)),
        )


@dataclass
class AdversarialResult:
    """Final result from the adversarial loop."""

    final_rewritten_text: str
    iterations: list[AdversarialIteration]
    total_iterations: int
    exit_reason: str
    ground_truth: dict[str, str]
    final_utility_score: float
    success: bool
    confidence_threshold: float = 0.7

    def to_dict(self) -> dict[str, object]:
        return {
            "final_rewritten_text": self.final_rewritten_text,
            "iterations": [it.to_dict() for it in self.iterations],
            "total_iterations": self.total_iterations,
            "exit_reason": self.exit_reason,
            "ground_truth": self.ground_truth,
            "final_utility_score": self.final_utility_score,
            "success": self.success,
            "confidence_threshold": self.confidence_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "AdversarialResult":
        iterations_value = data.get("iterations", [])
        iterations = iterations_value if isinstance(iterations_value, list) else []
        return cls(
            final_rewritten_text=str(data.get("final_rewritten_text", "")),
            iterations=[
                AdversarialIteration.from_dict(it) for it in iterations if isinstance(it, dict)
            ],
            total_iterations=_int_value(data.get("total_iterations"), 0),
            exit_reason=str(data.get("exit_reason", "")),
            ground_truth=_string_dict(data.get("ground_truth", {})),
            final_utility_score=_float_value(data.get("final_utility_score"), 0.0),
            success=bool(data.get("success", False)),
            confidence_threshold=_float_value(data.get("confidence_threshold"), 0.7),
        )


def _optional_str(value: object) -> str | None:
    """Convert optional values to strings while preserving None."""
    return None if value is None else str(value)


def _string_list(value: object) -> list[str]:
    """Convert a list-like value into a list of strings."""
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _string_dict(value: object) -> dict[str, str]:
    """Convert a mapping into a string-to-string dictionary."""
    if not isinstance(value, dict):
        return {}
    return {str(key): str(val) for key, val in value.items()}


def _float_dict(value: object) -> dict[str, float]:
    """Convert a mapping into a string-to-float dictionary."""
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, val in value.items():
        result[str(key)] = _float_value(val, 0.0)
    return result


def _strategy_records(value: object) -> list[StrategyRecord]:
    """Convert raw strategy dictionaries into StrategyRecord instances."""
    if not isinstance(value, list):
        return []
    return [StrategyRecord.from_dict(item) for item in value if isinstance(item, dict)]


def _clue_map(value: object) -> dict[str, list[dict[str, str]]]:
    """Convert raw clue-map data into the expected nested string dictionaries."""
    if not isinstance(value, dict):
        return {}

    result: dict[str, list[dict[str, str]]] = {}
    for attr, clues in value.items():
        if not isinstance(clues, list):
            result[str(attr)] = []
            continue
        result[str(attr)] = [
            _string_dict(clue) for clue in clues if isinstance(clue, dict)
        ]
    return result


def _float_value(value: object, default: float) -> float:
    """Convert a value to float, or return a default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_value(value: object, default: int) -> int:
    """Convert a value to int, or return a default."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
