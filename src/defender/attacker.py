"""Attacker agent: tries to de-anonymize rewritten text."""

import json
import os
import re
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai

load_dotenv()


ATTACKER_PROMPT_TEMPLATE = """
You are the guessing agent in a multi-agent semantic anonymization system.

Your task is to apply chain of thoughts to de-anonymize the information from the text given and return both de-anonymized data and the text filled
with that de-anonymized data.
You should search for names, locations, organizations and dates relying on the context, given by text. For example,
if the text says that the person was born on 4th of August 1961 in Hawaii and during his career was a president of USA you should
suggest that the text is about Barack Obama. In addition, you should try to de-anonymize any other information with respect to already found names,
locations, organizations and dates.

Rules:
- Return ONLY a valid dictionary-like text which follows the given structure, where each of the keys contains a list of contents:
"PERSON": [Name1, Name2,...],
"LOC": [Location1, Location2,...],
"ORG": [Organization1, Organization2...],
"DATE": [Date1, Date2,...],
"TEXT_R": full text containing de-anonymized data
- If you are not able to de-anonymize hidden data, you should return json-like text containing empty lists for related keys
in the structure defined above and partially filled full text
- Do NOT invent any facts, dates or info
- Do NOT return text as json
- Do NOT miss the outer brackets of dictionary

Text:
{text}
"""


@dataclass
class AttackerOutput:
    """Parsed Attacker results."""

    person: list[str]
    loc: list[str]
    org: list[str]
    date: list[str]
    text_r: str
    raw_response: str

    def to_dict(self) -> dict:
        return {
            "PERSON": self.person,
            "LOC": self.loc,
            "ORG": self.org,
            "DATE": self.date,
            "TEXT_R": self.text_r,
            "raw_response": self.raw_response,
        }

    def has_any_entities(self) -> bool:
        return bool(self.person or self.loc or self.org or self.date)


class AttackerError(Exception):
    """Raised when the Attacker encounters an unrecoverable error."""


class Attacker:
    """Attacker wrapper around Gemini calls."""

    DEFAULT_MODEL = "gemini-3-flash-preview"
    MAX_TOKENS = 4096

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.model = model or self.DEFAULT_MODEL
        self._client: genai.Client | None = None
        self._fallback_client: genai.Client | None = None

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if resolved_key:
            self._client = genai.Client(api_key=resolved_key)

        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if credentials_path and project_id:
            location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
            self._fallback_client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
            )

        if self._client is None and self._fallback_client is not None:
            self._client = self._fallback_client
            self._fallback_client = None
        elif self._client is None:
            raise AttackerError(
                "No credentials found. Set GEMINI_API_KEY for AI Studio, "
                "or GOOGLE_APPLICATION_CREDENTIALS + GOOGLE_CLOUD_PROJECT "
                "for Vertex AI."
            )

    def run(self, text: str) -> AttackerOutput:
        """Run the attacker against one rewritten text."""
        prompt = ATTACKER_PROMPT_TEMPLATE.format(text=text)
        raw = self._call_llm([prompt])
        return _parse_attacker_output(raw)

    def _call_llm(self, contents: list[str]) -> str:
        try:
            return self._send(self._client, contents)
        except AttackerError as exc:
            err_str = str(exc)
            is_rate_limited = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            is_unavailable = "503" in err_str or "UNAVAILABLE" in err_str

            if (is_rate_limited or is_unavailable) and self._fallback_client:
                print(
                    "[attacker] Free tier hit limit, switching to Vertex AI (Cloud credits)...",
                    file=sys.stderr,
                )
                return self._send(self._fallback_client, contents)
            raise

    def _send(self, client: genai.Client, contents: list[str]) -> str:
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config={
                    "max_output_tokens": self.MAX_TOKENS,
                    "temperature": 0.7,
                },
            )
            text = response.text
            if not text:
                raise AttackerError(
                    "Gemini returned an empty response. The input may be too "
                    "long or the content may have been blocked."
                )
            return text
        except AttackerError:
            raise
        except Exception as exc:
            raise AttackerError(f"Gemini API error: {exc}") from exc


def _parse_attacker_output(raw: str) -> AttackerOutput:
    """Parse the attacker dictionary-like response into structured data."""
    text = raw.strip()

    # Try JSON first (in case the model ignores instructions).
    try:
        data = json.loads(text)
        return AttackerOutput(
            person=_normalize_list(data.get("PERSON")),
            loc=_normalize_list(data.get("LOC")),
            org=_normalize_list(data.get("ORG")),
            date=_normalize_list(data.get("DATE")),
            text_r=str(data.get("TEXT_R", "")),
            raw_response=raw,
        )
    except Exception:
        pass

    fields = {
        "PERSON": [],
        "LOC": [],
        "ORG": [],
        "DATE": [],
        "TEXT_R": "",
    }

    for key in ("PERSON", "LOC", "ORG", "DATE", "TEXT_R"):
        match = re.search(
            rf"['\"]?{key}['\"]?\s*:\s*(\[.*?\]|\".*?\"|'.*?'|[^,\n\r}}]+)",
            text,
            re.DOTALL,
        )
        if not match:
            continue
        value = match.group(1).strip()
        if key == "TEXT_R":
            fields[key] = _strip_quotes(value)
        else:
            fields[key] = _parse_list_value(value)

    return AttackerOutput(
        person=fields["PERSON"],
        loc=fields["LOC"],
        org=fields["ORG"],
        date=fields["DATE"],
        text_r=fields["TEXT_R"],
        raw_response=raw,
    )


def _normalize_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return _parse_list_value(value)
    return [str(value)]


def _parse_list_value(value: str) -> list[str]:
    cleaned = value.strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1].strip()
    if not cleaned:
        return []

    # Prefer quoted tokens if present.
    quoted = re.findall(r"\"([^\"]+)\"|'([^']+)'", cleaned)
    if quoted:
        tokens = [q[0] or q[1] for q in quoted]
        return [t.strip() for t in tokens if t.strip()]

    parts = [p.strip() for p in cleaned.split(",")]
    return [_strip_quotes(p) for p in parts if _strip_quotes(p)]


def _strip_quotes(value: str) -> str:
    cleaned = value.strip()
    if (cleaned.startswith("\"") and cleaned.endswith("\"")) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        cleaned = cleaned[1:-1]
    return cleaned.strip()
