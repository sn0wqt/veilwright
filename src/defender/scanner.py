"""
Syntactic PII scanner for the Defender agent.

Performs a first-pass scan over free text to detect and mask explicit PII
tokens (emails, phone numbers, dates, credit card numbers) before the text
is sent to the LLM for semantic rewriting. This reduces token leakage and
lets the LLM focus on semantic clues only.

Adapted from the DataShield project's SensitiveDataScanner — regex patterns
and phone validation logic are reused; all database/column-level logic has
been removed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import phonenumbers


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PIIMatch:
    """
    A single PII token found in the input text.

    Attributes:
        pii_type: Category label (e.g. "EMAIL", "PHONE", "DATE", "CREDIT_CARD", "PERSON").
        value: The original text that matched.
        start: Start index in the original text.
        end: End index in the original text.
        replacement: The placeholder token that replaced the value (e.g. "<EMAIL>").
    """

    pii_type: str
    value: str
    start: int
    end: int
    replacement: str


@dataclass
class ScanResult:
    """
    Output of the syntactic scanner pass.

    Attributes:
        pii_found: All PII matches detected in the text.
        masked_text: The input text with explicit PII replaced by placeholders.
    """

    pii_found: list[PIIMatch] = field(default_factory=list)
    masked_text: str = ""


# ---------------------------------------------------------------------------
# Compiled regex patterns (adapted from datashield/scanner.py)
# ---------------------------------------------------------------------------

# Email: standard addr-spec pattern
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

# Date patterns: ISO (YYYY-MM-DD), US (MM/DD/YYYY), EU (DD.MM.YYYY, DD/MM/YYYY)
_DATE_RE = re.compile(
    r"\b(?:"
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"   # ISO: 2024-01-15
    r"|"
    r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{4}"  # US/EU: 01/15/2024 or 15.01.2024
    r")\b"
)

# Credit card: 13–19 digits, optionally grouped with spaces or dashes
_CREDIT_CARD_RE = re.compile(
    r"\b(?:\d[ \-]?){13,19}\b"
)

# Phone: international format (+...) or common US/EU formats
# We lean on the phonenumbers library for validation, but need a regex to
# find candidates in free text first.
_PHONE_CANDIDATE_RE = re.compile(
    r"(?<!\d)"                         # not preceded by a digit
    r"(?:"
    r"\+\d[\d\s\-().]{7,18}\d"         # international: +1 (202) 555-1234
    r"|"
    r"\(?\d{2,4}\)?[\s\-.]?\d{3,4}[\s\-.]?\d{3,4}"  # local: (202) 555-1234
    r")"
    r"(?!\d)"                          # not followed by a digit
)

# Person placeholder pattern: <PERSON> tokens already in text (from upstream preprocessing)
_PERSON_TAG_RE = re.compile(r"<PERSON>")


# ---------------------------------------------------------------------------
# Luhn check (from datashield/scanner.py)
# ---------------------------------------------------------------------------

def _luhn_check(card_number: str) -> bool:
    """
    Validate a credit card number using the Luhn algorithm.

    Args:
        card_number: Digit string (spaces and dashes allowed).

    Returns:
        True if the number passes the Luhn checksum.
    """
    try:
        digits = [int(d) for d in card_number.replace(" ", "").replace("-", "")]
        digits.reverse()
        for i in range(1, len(digits), 2):
            digits[i] *= 2
            if digits[i] > 9:
                digits[i] -= 9
        return sum(digits) % 10 == 0
    except (ValueError, AttributeError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Phone validation (from datashield/scanner.py)
# ---------------------------------------------------------------------------

def _is_valid_phone(value: str) -> bool:
    """
    Validate a phone-number candidate using the phonenumbers library.

    Args:
        value: Candidate string.

    Returns:
        True if the value parses as a possible phone number.
    """
    try:
        parsed = phonenumbers.parse(value.strip(), None)
        return phonenumbers.is_possible_number(parsed)
    except phonenumbers.NumberParseException:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_text(text: str) -> ScanResult:
    """
    Scan free text for explicit PII and return a masked version.

    The scanner detects:
    - Email addresses
    - Phone numbers (validated via the phonenumbers library)
    - Dates in common formats
    - Credit card numbers (validated via the Luhn algorithm)
    - Existing ``<PERSON>`` placeholder tags

    Each detected token is replaced with a type-specific placeholder
    (e.g. ``<EMAIL>``, ``<PHONE>``) in the masked output text.

    Args:
        text: The raw input text to scan.

    Returns:
        A :class:`ScanResult` with all matches and the masked text.
    """
    matches: list[PIIMatch] = []

    # --- Emails ---
    for m in _EMAIL_RE.finditer(text):
        matches.append(PIIMatch(
            pii_type="EMAIL",
            value=m.group(),
            start=m.start(),
            end=m.end(),
            replacement="<EMAIL>",
        ))

    # --- Phone numbers (regex candidates → phonenumbers validation) ---
    for m in _PHONE_CANDIDATE_RE.finditer(text):
        candidate = m.group()
        if _is_valid_phone(candidate):
            # Avoid overlap with credit card matches
            matches.append(PIIMatch(
                pii_type="PHONE",
                value=candidate,
                start=m.start(),
                end=m.end(),
                replacement="<PHONE>",
            ))

    # --- Dates ---
    for m in _DATE_RE.finditer(text):
        matches.append(PIIMatch(
            pii_type="DATE",
            value=m.group(),
            start=m.start(),
            end=m.end(),
            replacement="<DATE>",
        ))

    # --- Credit cards ---
    for m in _CREDIT_CARD_RE.finditer(text):
        raw = m.group().replace(" ", "").replace("-", "")
        if len(raw) >= 13 and raw.isdigit() and _luhn_check(raw):
            matches.append(PIIMatch(
                pii_type="CREDIT_CARD",
                value=m.group(),
                start=m.start(),
                end=m.end(),
                replacement="<CREDIT_CARD>",
            ))

    # --- <PERSON> tags already in text ---
    for m in _PERSON_TAG_RE.finditer(text):
        matches.append(PIIMatch(
            pii_type="PERSON",
            value=m.group(),
            start=m.start(),
            end=m.end(),
            replacement="<PERSON>",
        ))

    # --- Deduplicate overlapping spans (keep longest) ---
    matches = _deduplicate_spans(matches)

    # --- Build masked text (replace from end to preserve indices) ---
    masked = text
    for match in sorted(matches, key=lambda m: m.start, reverse=True):
        masked = masked[:match.start] + match.replacement + masked[match.end:]

    return ScanResult(pii_found=matches, masked_text=masked)


def _deduplicate_spans(matches: list[PIIMatch]) -> list[PIIMatch]:
    """
    Remove overlapping PII matches, keeping the longest span.

    Args:
        matches: Unsorted list of PII matches.

    Returns:
        Deduplicated list sorted by start position.
    """
    if not matches:
        return []

    # Sort by start, then by descending length (prefer longer matches)
    sorted_matches = sorted(matches, key=lambda m: (m.start, -(m.end - m.start)))

    result: list[PIIMatch] = [sorted_matches[0]]
    for current in sorted_matches[1:]:
        prev = result[-1]
        if current.start >= prev.end:
            # No overlap
            result.append(current)
        # else: skip (overlapping, and previous is already longer or equal)

    return result
