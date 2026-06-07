"""Utility helpers for JSON extraction and response validation."""

import json
import re

from defender.strategies import VALID_STRATEGY_NAMES


def parse_llm_json(text: str) -> dict:
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
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # try to find the outermost { ... } block
    brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group())

    raise json.JSONDecodeError("No JSON object found in LLM response", cleaned, 0)


def validate_defender_response(data: dict, target_attributes: list[str] | None = None) -> list[str]:
    """Check that an LLM response dict has the required fields. Returns error list.

    If target_attributes is provided, also validates that all attributes have strategies.
    """
    errors: list[str] = []

    if "rewritten_text" not in data:
        errors.append("Missing 'rewritten_text' field.")
    elif not isinstance(data["rewritten_text"], str):
        errors.append("'rewritten_text' must be a string.")

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
            if "strategy" in record:
                strategy_lower = record["strategy"].lower()
                if strategy_lower not in VALID_STRATEGY_NAMES:
                    errors.append(
                        f"strategies_used[{i}] has invalid strategy "
                        f"'{record['strategy']}'. Must be one of {VALID_STRATEGY_NAMES}."
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


def validate_utility_response(data: dict) -> list[str]:
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

    return errors
