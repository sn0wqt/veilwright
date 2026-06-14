"""Utility helpers for JSON extraction and response validation."""

from collections.abc import Callable
import json
import re

from defender.strategies import VALID_STRATEGY_NAMES


def parse_llm_json(text: str) -> dict[str, object]:
    """Extract and parse a JSON object from an LLM response.

    Handles markdown code fences and surrounding commentary.
    """
    cleaned = text.strip()

    # strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # try direct parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # try to find the outermost { ... } block
    brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if brace_match:
        try:
            parsed = json.loads(brace_match.group())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("No JSON object found in LLM response", cleaned, 0)


def parse_validated_llm_json(
    initial_text: str,
    retry: Callable[[], str],
    validate: Callable[[dict[str, object]], list[str]],
    error_type: type[Exception],
    parse_error_message: str,
    validation_error_message: str,
) -> dict[str, object]:
    """Parse LLM JSON, retry once on parse/validation failure, then validate."""
    try:
        parsed = parse_llm_json(initial_text)
        errors = validate(parsed)
        if not errors:
            return parsed
    except json.JSONDecodeError:
        pass

    retry_text = retry()
    try:
        parsed = parse_llm_json(retry_text)
    except json.JSONDecodeError as exc:
        raise error_type(parse_error_message.format(error=exc)) from exc

    errors = validate(parsed)
    if errors:
        raise error_type(validation_error_message.format(errors=errors))

    return parsed


def validate_defender_response(
    data: dict[str, object], target_attributes: list[str] | None = None
) -> list[str]:
    """Check that an LLM response dict has the required fields. Returns error list.

    If target_attributes is provided, also validates that all attributes have strategies.
    """
    errors: list[str] = []

    if "rewritten_text" not in data:
        errors.append("Missing 'rewritten_text' field.")
    elif not isinstance(data["rewritten_text"], str):
        errors.append("'rewritten_text' must be a string.")
    elif not data["rewritten_text"].strip():
        errors.append("'rewritten_text' must be a non-empty string.")

    if "strategies_used" not in data:
        errors.append("Missing 'strategies_used' field.")
    elif not isinstance(data["strategies_used"], list):
        errors.append("'strategies_used' must be a list.")
    else:
        for i, record in enumerate(data["strategies_used"]):
            if not isinstance(record, dict):
                errors.append(f"strategies_used[{i}] must be a dict.")
                continue
            for key in ("attribute", "strategy", "reasoning"):
                if key not in record:
                    errors.append(f"strategies_used[{i}] missing '{key}'.")
                elif not isinstance(record[key], str):
                    errors.append(f"strategies_used[{i}].{key} must be a string.")
                elif not record[key].strip():
                    errors.append(
                        f"strategies_used[{i}].{key} must be a non-empty string."
                    )
            strategy = record.get("strategy")
            if isinstance(strategy, str):
                strategy_lower = strategy.lower()
                if strategy_lower not in VALID_STRATEGY_NAMES:
                    errors.append(
                        f"strategies_used[{i}] has invalid strategy "
                        f"'{strategy}'. Must be one of {VALID_STRATEGY_NAMES}."
                    )
                else:
                    # normalize to lowercase for downstream use
                    record["strategy"] = strategy_lower

        # check coverage: all target attributes must have a strategy (only if target_attributes provided)
        if target_attributes:
            covered = {s.get("attribute") for s in data["strategies_used"] if isinstance(s, dict)}
            missing = set(target_attributes) - covered
            if missing:
                errors.append(
                    f"Missing strategies for target attributes: {', '.join(sorted(missing))}"
                )

    if "confidence" not in data:
        errors.append("Missing 'confidence' field.")
    elif not isinstance(data["confidence"], (int, float)):
        errors.append("'confidence' must be a number.")
    elif not 0.0 <= data["confidence"] <= 1.0:
        errors.append("'confidence' must be between 0.0 and 1.0.")

    return errors


def validate_utility_response(data: dict[str, object]) -> list[str]:
    """Check that a Utility Judge response dict has the required fields."""
    errors: list[str] = []

    if "score" not in data:
        errors.append("Missing 'score' field.")
    elif not isinstance(data["score"], (int, float)):
        errors.append("'score' must be a number.")
    elif not 0.0 <= data["score"] <= 1.0:
        errors.append("'score' must be between 0.0 and 1.0.")

    if "rationale" not in data:
        errors.append("Missing 'rationale' field.")
    elif not isinstance(data["rationale"], str):
        errors.append("'rationale' must be a string.")
    elif not data["rationale"].strip():
        errors.append("'rationale' must be a non-empty string.")

    return errors


def _compare_guess_to_ground_truth(guess: str, truth: str, attr: str = "") -> bool:
    """Compare an attacker's guess against the ground truth value.

    Returns True if:
    - Exact match after lowercasing and stripping whitespace.
    - The guess is a substring of the ground truth (or vice versa).
    - For age-like attributes, numeric values are within 6.
    - For birth/year-like attributes, numeric values are within 5.
    Does NOT match if either value is empty, None, or "UNKNOWN".
    """
    if not guess or not truth:
        return False
    g = guess.strip().lower()
    t = truth.strip().lower()
    if g == "unknown" or t == "unknown" or not g or not t:
        return False

    # 1. Exact or substring match
    if (g == t) or (g in t) or (t in g):
        return True

    attr_lower = attr.lower()
    is_age_attr = "age" in attr_lower
    is_year_attr = any(word in attr_lower for word in ("birth", "year", "born"))
    if not (is_age_attr or is_year_attr):
        return False

    try:
        guess_nums = [int(n) for n in re.findall(r"\d+", g)]
        truth_nums = [int(n) for n in re.findall(r"\d+", t)]
    except ValueError:
        return False

    tolerance = 5 if is_year_attr else 6
    for guess_num in guess_nums:
        for truth_num in truth_nums:
            if abs(guess_num - truth_num) <= tolerance:
                return True

    return False


def is_guess_correct(
    guess: str,
    truth: str | None,
    confidence: float,
    attribute: str,
    threshold: float = 0.7,
) -> bool:
    """Return True when a confident attacker guess matches ground truth."""
    if truth is None:
        return False
    return confidence >= threshold and _compare_guess_to_ground_truth(
        guess,
        truth,
        attribute,
    )
