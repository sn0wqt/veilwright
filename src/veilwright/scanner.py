"""Syntactic PII scanner for the Defender agent.

Two-pass detection:
  1. Optional spaCy NER — detects names, locations, orgs, money (reported but
     NOT masked, so the LLM can see them and reason about how to rewrite them)
  2. Regex — detects emails, phones, dates, credit cards. Emails, phones,
     and credit cards are masked before LLM calls; dates are detected but
     left visible for semantic rewriting.
"""

import re
from dataclasses import dataclass, field

try:
    import phonenumbers
except ImportError:
    phonenumbers = None

# Load the best installed spaCy model, with graceful regex-only fallback.
try:
    import spacy

    _nlp = None
    for _model_name in ("en_core_web_lg", "en_core_web_sm"):
        try:
            _nlp = spacy.load(_model_name)
            break
        except OSError:
            continue
except ImportError:
    _nlp = None


@dataclass
class PIIMatch:
    """A single PII token found in the input text."""

    pii_type: str       # e.g. "PERSON", "EMAIL", "PHONE", "LOCATION", etc.
    value: str
    start: int
    end: int
    replacement: str    # e.g. "<EMAIL>"
    mask: bool = True   # if False, detected but not masked in output


@dataclass
class ScanResult:
    """Output of the syntactic scanner pass."""

    pii_found: list[PIIMatch] = field(default_factory=list)
    masked_text: str = ""


# ---------------------------------------------------------------------------
# Regex patterns (for things spaCy can't catch)
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

_DATE_RE = re.compile(
    r"\b(?:"
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"           # ISO: 2024-01-15
    r"|"
    r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{4}"       # US/EU: 01/15/2024
    r")\b"
)

_CREDIT_CARD_RE = re.compile(
    r"\b(?:\d[ \-]?){13,19}\b"
)

_PHONE_CANDIDATE_RE = re.compile(
    r"(?<!\d)"
    r"(?:"
    r"\+\d[\d\s\-().]{7,18}\d"               # international: +1 (202) 555-1234
    r"|"
    r"\(?\d{2,4}\)?[\s\-.]?\d{3,4}[\s\-.]?\d{3,4}"  # local formats
    r")"
    r"(?!\d)"
)


# spaCy entity label -> our PII type + replacement tag
_NER_LABEL_MAP: dict[str, tuple[str, str]] = {
    "PERSON":  ("PERSON", "<PERSON>"),
    "GPE":     ("LOCATION", "<LOCATION>"),
    "LOC":     ("LOCATION", "<LOCATION>"),
    "ORG":     ("ORGANIZATION", "<ORGANIZATION>"),
    "MONEY":   ("MONEY", "<MONEY>"),
    "DATE":    ("DATE", "<DATE>"),
}

_CELESTIAL_PERSON_FALSE_POSITIVES = {"moon", "sun", "earth"}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _luhn_check(card_number: str) -> bool:
    """Validate a credit card number using the Luhn algorithm."""
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


def _is_valid_phone(value: str) -> bool:
    """Check if a string looks like a real phone number."""
    if phonenumbers is None:
        # Fallback if library missing: just accept regex candidates
        return True
    try:
        parsed = phonenumbers.parse(value.strip(), None)
        return phonenumbers.is_possible_number(parsed)
    except phonenumbers.NumberParseException:
        try:
            parsed = phonenumbers.parse(value.strip(), "US")
            return phonenumbers.is_possible_number(parsed)
        except phonenumbers.NumberParseException:
            return False


def _is_false_positive_person(value: str, text: str, start: int) -> bool:
    """Return True for common non-person terms spaCy can label as PERSON."""
    normalized = value.strip().lower()
    if normalized not in _CELESTIAL_PERSON_FALSE_POSITIVES:
        return False

    prefix = text[max(0, start - 5):start].lower()
    return prefix.endswith("the ")


def scan_text(text: str) -> ScanResult:
    """Scan free text for PII and return a masked version.

    PERSON entities and regex matches (emails, phones, credit cards) are
    MASKED before the LLM — direct identifiers we don't want to leak.
    Other NER entities (locations, orgs, money) are DETECTED but NOT masked —
    the LLM needs to see them to know what to rewrite semantically.
    """
    matches: list[PIIMatch] = []

    # --- Pass 1: spaCy NER ---
    if _nlp is not None:
        doc = _nlp(text)
        for ent in doc.ents:
            if ent.label_ in _NER_LABEL_MAP:
                if ent.label_ == "PERSON" and _is_false_positive_person(
                    ent.text,
                    text,
                    ent.start_char,
                ):
                    continue
                pii_type, replacement = _NER_LABEL_MAP[ent.label_]
                # PERSON gets masked (direct identifier), others are detect-only
                should_mask = ent.label_ == "PERSON"
                matches.append(PIIMatch(
                    pii_type=pii_type, value=ent.text,
                    start=ent.start_char, end=ent.end_char,
                    replacement=replacement, mask=should_mask,
                ))

    # --- Pass 2: regex ---

    # emails
    for m in _EMAIL_RE.finditer(text):
        matches.append(PIIMatch(
            pii_type="EMAIL", value=m.group(),
            start=m.start(), end=m.end(), replacement="<EMAIL>",
        ))

    # phone numbers (regex candidates -> phonenumbers validation)
    for m in _PHONE_CANDIDATE_RE.finditer(text):
        candidate = m.group()
        if _is_valid_phone(candidate):
            matches.append(PIIMatch(
                pii_type="PHONE", value=candidate,
                start=m.start(), end=m.end(), replacement="<PHONE>",
            ))

    # dates (regex catches structured dates spaCy might miss)
    for m in _DATE_RE.finditer(text):
        matches.append(PIIMatch(
            pii_type="DATE", value=m.group(),
            start=m.start(), end=m.end(), replacement="<DATE>",
            mask=False,
        ))

    # credit cards
    for m in _CREDIT_CARD_RE.finditer(text):
        raw = m.group().replace(" ", "").replace("-", "")
        if len(raw) >= 13 and raw.isdigit() and _luhn_check(raw):
            matches.append(PIIMatch(
                pii_type="CREDIT_CARD", value=m.group(),
                start=m.start(), end=m.end(), replacement="<CREDIT_CARD>",
            ))

    # deduplicate overlapping spans
    matches = _deduplicate_spans(matches)

    # build masked text — only replace entities with mask=True
    masked = text
    maskable = sorted(
        [m for m in matches if m.mask],
        key=lambda m: m.start, reverse=True,
    )
    for match in maskable:
        masked = masked[:match.start] + match.replacement + masked[match.end:]

    return ScanResult(pii_found=matches, masked_text=masked)


def _deduplicate_spans(matches: list[PIIMatch]) -> list[PIIMatch]:
    """Remove overlapping PII matches, keeping maskable direct PII first."""
    if not matches:
        return []

    sorted_matches = sorted(
        matches,
        key=lambda m: (not m.mask, -(m.end - m.start), m.start),
    )

    kept: list[PIIMatch] = []
    for current in sorted_matches:
        if all(current.end <= prev.start or current.start >= prev.end for prev in kept):
            kept.append(current)

    return sorted(kept, key=lambda m: (m.start, m.end))
