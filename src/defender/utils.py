"""Utility helpers for JSON extraction and response validation."""

from collections.abc import Callable
import json
import re

from defender.strategies import VALID_STRATEGY_NAMES

_MONTHS: dict[str, int] = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_ROLE_EQUIVALENCE_GROUPS: tuple[tuple[frozenset[str], ...], ...] = (
    (frozenset({"president"}), frozenset({"head", "state"})),
    (frozenset({"senator"}), frozenset({"legislator"})),
    (frozenset({"politician"}), frozenset({"public", "servant"})),
)


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
        if isinstance(parsed, str) and parsed != cleaned:
            return parse_llm_json(parsed)
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


def _date_tuple(year: int, month: int, day: int) -> tuple[int, int, int] | None:
    """Return a comparable date tuple when components are in plausible ranges."""
    if 1 <= day <= 31 and 1 <= month <= 12 and year >= 1000:
        return (year, month, day)
    return None


def _parse_date_value(value: str) -> tuple[int, int, int] | None:
    """Parse common date formats into (year, month, day) without calendar validation."""
    normalized = value.strip().lower()

    numeric = re.search(r"\b(\d{1,4})[./-](\d{1,2})[./-](\d{1,4})\b", normalized)
    if numeric:
        first, second, third = (int(part) for part in numeric.groups())
        if first >= 1000:
            return _date_tuple(first, second, third)
        day_first = _date_tuple(third, second, first)
        if day_first is not None:
            return day_first
        return _date_tuple(third, first, second)

    month_names = "|".join(sorted(_MONTHS, key=len, reverse=True))
    month_first = re.search(
        rf"\b({month_names})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+(\d{{4}})\b",
        normalized,
    )
    if month_first:
        month_name, day, year = month_first.groups()
        return _date_tuple(int(year), _MONTHS[month_name], int(day))

    day_first = re.search(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_names})\.?[,]?\s+(\d{{4}})\b",
        normalized,
    )
    if day_first:
        day, month_name, year = day_first.groups()
        return _date_tuple(int(year), _MONTHS[month_name], int(day))

    return None


def _compare_guess_to_ground_truth(guess: str, truth: str, attr: str = "") -> bool:
    """Compare an attacker's guess against the ground truth value.

    Returns True if:
    - Exact match after lowercasing and stripping whitespace.
    - For compatible attribute types, one value is a meaningful substring or
      token subset of the other.
    - For date attributes, parsed dates are equal across common formats.
    - For birth/year-like attributes, numeric years are within 5.
    - For age-like attributes, numeric ages are within 6.
    - A standalone numeric/currency value appears inside a longer value.
    Does NOT match if either value is empty, None, or "UNKNOWN".
    """
    if not guess or not truth:
        return False
    g = guess.strip().lower()
    t = truth.strip().lower()
    if g == "unknown" or t == "unknown" or not g or not t:
        return False

    # 1. Exact match
    if g == t:
        return True

    attr_lower = attr.lower()

    if _has_standalone_number_match(g, t):
        return True

    # 2. Full Date matching (canonical date comparison across common formats)
    is_date_attr = "date" in attr_lower
    if is_date_attr:
        guess_date = _parse_date_value(g)
        truth_date = _parse_date_value(t)
        if guess_date is not None and truth_date is not None:
            return guess_date == truth_date

        g_words = set(re.findall(r"[a-z0-9]+", g))
        t_words = set(re.findall(r"[a-z0-9]+", t))
        return bool(g_words and t_words and g_words == t_words)

    # 3. Birth Year (or other Year attributes, but not date)
    is_year_attr = (
        any(word in attr_lower for word in ("birth", "year", "born"))
        and "date" not in attr_lower
    )
    if is_year_attr:
        guess_years = [int(y) for y in re.findall(r"\d+", g)]
        guess_years = [y for y in guess_years if y >= 1000]
        truth_years = [int(y) for y in re.findall(r"\d+", t)]
        truth_years = [y for y in truth_years if y >= 1000]
        if guess_years and truth_years:
            for gy in guess_years:
                for ty in truth_years:
                    if abs(gy - ty) <= 5:
                        return True

    # 4. Age matching (exclude 4-digit years)
    is_age_attr = "age" in attr_lower
    if is_age_attr:
        guess_ages = [int(n) for n in re.findall(r"\d+", g)]
        guess_ages = [n for n in guess_ages if n < 1000]
        truth_ages = [int(n) for n in re.findall(r"\d+", t)]
        truth_ages = [n for n in truth_ages if n < 1000]
        if guess_ages and truth_ages:
            for ga in guess_ages:
                for ta in truth_ages:
                    if abs(ga - ta) <= 6:
                        return True

    # 5. Attribute-aware partial matching. Avoid a global substring fallback:
    # "New York" is not a correct Employer guess for "New York City Police
    # Department", even though it is a literal substring.
    if _allows_token_subset(attr_lower, g, t):
        return True

    return False


def _allows_token_subset(attr_lower: str, guess: str, truth: str) -> bool:
    """Return True when partial token matching is appropriate for an attribute."""
    g_words = _significant_tokens(guess)
    t_words = _significant_tokens(truth)
    if not g_words or not t_words:
        return False

    if any(
        word in attr_lower
        for word in ("name", "identity", "person", "who", "leader", "figure")
    ):
        return g_words.issubset(t_words) or t_words.issubset(g_words)

    if "event" in attr_lower:
        return g_words.issubset(t_words) or t_words.issubset(g_words)

    if any(word in attr_lower for word in ("location", "place", "city", "country")):
        return g_words.issubset(t_words) or t_words.issubset(g_words)

    if any(word in attr_lower for word in ("income", "salary", "earning", "money")):
        return _has_matching_number(g_words, t_words)

    if any(
        word in attr_lower
        for word in ("profession", "occupation", "job", "office", "role")
    ):
        if _role_alias_match(g_words, t_words):
            return True

        smaller = g_words if len(g_words) <= len(t_words) else t_words
        larger = t_words if smaller is g_words else g_words
        return len(smaller) == 1 and smaller.issubset(larger)

    return False


def _significant_tokens(value: str) -> set[str]:
    """Return normalized non-noise tokens for fuzzy matching."""
    noise = {"mr", "mrs", "ms", "dr", "prof", "the", "a", "an", "of", "and"}
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if token not in noise and (len(token) >= 2 or token.isdigit())
    }


def _has_matching_number(left_tokens: set[str], right_tokens: set[str]) -> bool:
    """Return True when token sets share a numeric value."""
    left_numbers = {token for token in left_tokens if token.isdigit()}
    right_numbers = {token for token in right_tokens if token.isdigit()}
    return bool(left_numbers & right_numbers)


def _role_alias_match(left_tokens: set[str], right_tokens: set[str]) -> bool:
    """Return True when role terms describe the same broad public role."""
    for group in _ROLE_EQUIVALENCE_GROUPS:
        left_matches = [alias for alias in group if alias.issubset(left_tokens)]
        right_matches = [alias for alias in group if alias.issubset(right_tokens)]
        if left_matches and right_matches:
            return True
    return False


def _has_standalone_number_match(guess: str, truth: str) -> bool:
    """Match standalone numeric/currency values inside longer phrases."""
    guess_numbers = _number_tokens(guess)
    truth_numbers = _number_tokens(truth)
    if not guess_numbers or not truth_numbers:
        return False

    guess_is_number = _is_standalone_numeric_value(guess)
    truth_is_number = _is_standalone_numeric_value(truth)
    if not (guess_is_number or truth_is_number):
        return False

    return bool(set(guess_numbers) & set(truth_numbers))


def _number_tokens(value: str) -> list[str]:
    """Extract normalized integer-like number tokens from text."""
    return [token.replace(",", "") for token in re.findall(r"\d[\d,]*", value)]


def _is_standalone_numeric_value(value: str) -> bool:
    """Return True when text is essentially one numeric or currency value."""
    stripped = value.lower()
    stripped = re.sub(r"\b(years?|old|annually|per|year|month)\b", "", stripped)
    stripped = re.sub(r"[$€£,%+\s,.-]", "", stripped)
    return bool(stripped) and stripped.isdigit()


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
