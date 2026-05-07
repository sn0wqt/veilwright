"""Core Defender agent logic."""

import json
import os
import sys

from dotenv import load_dotenv
from google import genai

from defender.prompts import SYSTEM_PROMPT, RETRY_PROMPT, build_rewrite_prompt
from defender.scanner import scan_text
from defender.types import DefenderInput, DefenderOutput, StrategyRecord
from defender.utils import parse_llm_json, validate_defender_response

load_dotenv()


class DefenderError(Exception):
    """Raised when the Defender agent encounters an unrecoverable error."""


class Defender:
    """Rewrites free text to hide target attributes using a syntactic + LLM pipeline."""

    DEFAULT_MODEL = "gemini-2.5-flash"
    MAX_TOKENS = 16384

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
            raise DefenderError(
                "No credentials found. Set GEMINI_API_KEY for AI Studio, "
                "or GOOGLE_APPLICATION_CREDENTIALS + GOOGLE_CLOUD_PROJECT "
                "for Vertex AI."
            )

    def run(self, defender_input: DefenderInput) -> DefenderOutput:
        """Run the full Defender pipeline on the given input."""

        # Step 1: syntactic scanner pass
        scan_result = scan_text(defender_input.text)
        syntactic_pii = [
            f"{m.pii_type}: {m.value} [{'masked' if m.mask else 'detected'}]"
            for m in scan_result.pii_found
        ]

        # use masked text for the LLM so we don't leak explicit PII
        text_for_llm = scan_result.masked_text

        # Step 2: build prompt
        user_prompt = build_rewrite_prompt(
            text=text_for_llm,
            target_attributes=defender_input.target_attributes,
            iteration=defender_input.iteration,
            attacker_feedback=defender_input.attacker_feedback,
        )

        # Step 3: call the LLM
        contents = [user_prompt]
        llm_response_text = self._call_llm(contents)

        # Step 4: parse + validate
        try:
            parsed = parse_llm_json(llm_response_text)
            errors = validate_defender_response(parsed)
            if errors:
                raise json.JSONDecodeError(
                    f"Validation errors: {'; '.join(errors)}",
                    llm_response_text,
                    0,
                )
        except json.JSONDecodeError:
            # retry once with a stricter prompt
            contents.append(llm_response_text)
            contents.append(RETRY_PROMPT)
            retry_text = self._call_llm(contents)

            try:
                parsed = parse_llm_json(retry_text)
                errors = validate_defender_response(parsed)
                if errors:
                    raise DefenderError(
                        f"LLM response failed validation after retry: {errors}"
                    )
            except json.JSONDecodeError as exc:
                raise DefenderError(
                    f"Failed to parse LLM JSON after retry: {exc}"
                ) from exc

        # build output
        strategies = [StrategyRecord.from_dict(s) for s in parsed["strategies_used"]]

        return DefenderOutput(
            original_text=defender_input.text,
            rewritten_text=parsed["rewritten_text"],
            target_attributes=defender_input.target_attributes,
            strategies_used=strategies,
            confidence=float(parsed["confidence"]),
            iteration=defender_input.iteration,
            syntactic_pii_found=syntactic_pii,
        )

    def _call_llm(self, contents: list[str]) -> str:
        """Send contents to the Gemini API, falling back to Vertex on rate limits."""
        try:
            return self._send(self._client, contents)
        except DefenderError as exc:
            # Check for rate-limit (429) or overload (503)
            err_str = str(exc)
            is_rate_limited = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            is_unavailable = "503" in err_str or "UNAVAILABLE" in err_str

            if (is_rate_limited or is_unavailable) and self._fallback_client:
                print(
                    "[defender] Free tier hit limit, switching to Vertex AI (Cloud credits)...",
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
                    "system_instruction": SYSTEM_PROMPT,
                    "max_output_tokens": self.MAX_TOKENS,
                    "temperature": 0.7,
                },
            )
            text = response.text
            if not text:
                raise DefenderError(
                    "Gemini returned an empty response. The input may be too "
                    "long or the content may have been blocked."
                )
            return text
        except DefenderError:
            raise
        except Exception as exc:
            raise DefenderError(f"Gemini API error: {exc}") from exc
