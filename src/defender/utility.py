"""Utility Judge for scoring meaning preservation."""

import json
import os
import sys

from dotenv import load_dotenv
from google import genai

from defender.prompts import UTILITY_SYSTEM_PROMPT, UTILITY_RETRY_PROMPT, build_utility_prompt
from defender.types import UtilityInput, UtilityOutput
from defender.utils import parse_llm_json, validate_utility_response

load_dotenv()


class UtilityError(Exception):
    """Raised when the Utility Judge encounters an unrecoverable error."""


class UtilityJudge:
    """Scores meaning preservation between original and rewritten text."""

    DEFAULT_MODEL = "gemini-2.5-flash"
    MAX_TOKENS = 4096

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.model = model or self.DEFAULT_MODEL
        self._client: genai.Client | None = None
        self._fallback_client: genai.Client | None = None

        # Primary: AI Studio (free tier)
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if resolved_key:
            self._client = genai.Client(api_key=resolved_key)

        # Fallback: Vertex AI (Cloud credits)
        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if credentials_path and project_id:
            location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
            self._fallback_client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
            )

        # If no primary, promote fallback
        if self._client is None and self._fallback_client is not None:
            self._client = self._fallback_client
            self._fallback_client = None
        elif self._client is None:
            raise UtilityError(
                "No credentials found. Set GEMINI_API_KEY for AI Studio, "
                "or GOOGLE_APPLICATION_CREDENTIALS + GOOGLE_CLOUD_PROJECT "
                "for Vertex AI."
            )

    def score(self, utility_input: UtilityInput) -> UtilityOutput:
        """Score semantic preservation for a rewritten text."""
        prompt = build_utility_prompt(
            original_text=utility_input.original_text,
            rewritten_text=utility_input.rewritten_text,
            target_attributes=utility_input.target_attributes,
        )

        llm_text = self._call_llm([prompt])

        try:
            parsed = parse_llm_json(llm_text)
            errors = validate_utility_response(parsed)
            if errors:
                raise json.JSONDecodeError(
                    f"Validation errors: {'; '.join(errors)}",
                    llm_text,
                    0,
                )
        except json.JSONDecodeError:
            retry_text = self._call_llm([llm_text, UTILITY_RETRY_PROMPT])
            try:
                parsed = parse_llm_json(retry_text)
                errors = validate_utility_response(parsed)
                if errors:
                    raise UtilityError(
                        f"LLM response failed validation after retry: {errors}"
                    )
            except json.JSONDecodeError as exc:
                raise UtilityError(
                    f"Failed to parse LLM JSON after retry: {exc}"
                ) from exc

        return UtilityOutput(
            original_text=utility_input.original_text,
            rewritten_text=utility_input.rewritten_text,
            score=float(parsed["score"]),
            rationale=parsed["rationale"],
        )

    def _call_llm(self, contents: list[str]) -> str:
        """Send contents to the Gemini API, falling back to Vertex on rate limits."""
        try:
            return self._send(self._client, contents)
        except UtilityError as exc:
            err_str = str(exc)
            is_rate_limited = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            is_unavailable = "503" in err_str or "UNAVAILABLE" in err_str

            if (is_rate_limited or is_unavailable) and self._fallback_client:
                print(
                    "[utility] Free tier hit limit, switching to Vertex AI (Cloud credits)...",
                    file=sys.stderr,
                )
                return self._send(self._fallback_client, contents)
            raise

    def _send(self, client: genai.Client, contents: list[str]) -> str:
        """Send a request using a specific genai client."""
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config={
                    "system_instruction": UTILITY_SYSTEM_PROMPT,
                    "max_output_tokens": self.MAX_TOKENS,
                    "temperature": 0.2,
                },
            )
            text = response.text
            if not text:
                raise UtilityError(
                    "Gemini returned an empty response. The input may be too "
                    "long or the content may have been blocked."
                )
            return text
        except UtilityError:
            raise
        except Exception as exc:
            raise UtilityError(f"Gemini API error: {exc}") from exc
