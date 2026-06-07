"""Core Defender agent logic."""

import json
import os
import sys

from dotenv import load_dotenv
from google import genai

from defender.prompts import (
    REWRITE_SYSTEM_PROMPT,
    ANALYSIS_SYSTEM_PROMPT,
    RETRY_PROMPT,
    build_clue_enumeration_prompt,
    build_ground_truth_prompt,
    build_rewrite_prompt,
)
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

        if not defender_input.text.strip():
            raise DefenderError("Input text is empty.")

        estimated_tokens = len(defender_input.text) // 4
        if estimated_tokens > 6000:
            raise DefenderError(
                f"Input text too long (~{estimated_tokens} tokens). "
                f"Maximum supported: ~6000 tokens to leave room for prompt and output."
            )

        # --- Step 1: auto-extract ground truth if caller didn't provide it ---
        if defender_input.iteration == 1 and not defender_input.ground_truth and defender_input.target_attributes:
            defender_input.ground_truth = self.extract_ground_truth(
                defender_input.text, defender_input.target_attributes
            )

        # --- Step 2: scan for explicit PII ---
        scan_result = scan_text(defender_input.text)
        syntactic_pii = [
            f"{m.pii_type}: {m.value} [{'masked' if m.mask else 'detected'}]"
            for m in scan_result.pii_found
        ]

        # send masked text to LLM so we don't leak raw emails/phones/etc
        text_for_llm = scan_result.masked_text

        # --- Step 3: clue enumeration pre-pass (iteration 1 only) ---
        # On retries the attacker_feedback already tells us what clues slipped through,
        # so we don't need to re-enumerate from scratch.
        clue_map: dict = {}
        if defender_input.iteration == 1 and defender_input.target_attributes:
            clue_map = self._enumerate_clues(text_for_llm, defender_input.target_attributes)

        # --- Step 4: build the rewrite prompt with clue map baked in ---
        user_prompt = build_rewrite_prompt(
            text=text_for_llm,
            target_attributes=defender_input.target_attributes,
            iteration=defender_input.iteration,
            attacker_feedback=defender_input.attacker_feedback,
            clue_map=clue_map or None,
        )

        # --- Step 5: call the LLM ---
        contents = [user_prompt]
        llm_response_text = self._call_llm(contents)

        # --- Step 6: parse and validate ---
        needs_retry = False
        try:
            parsed = parse_llm_json(llm_response_text)
            errors = validate_defender_response(parsed, defender_input.target_attributes)
            if errors:
                needs_retry = True
        except json.JSONDecodeError:
            needs_retry = True

        if needs_retry:
            # retry once with stricter prompt
            retry_text = self._call_llm([user_prompt, RETRY_PROMPT])

            try:
                parsed = parse_llm_json(retry_text)
                errors = validate_defender_response(parsed, defender_input.target_attributes)
                if errors:
                    raise DefenderError(
                        f"LLM response failed validation after retry: {errors}"
                    )
            except json.JSONDecodeError as exc:
                raise DefenderError(
                    f"Failed to parse LLM JSON after retry: {exc}"
                ) from exc

        # --- Step 7: build output ---
        strategies = [StrategyRecord.from_dict(s) for s in parsed["strategies_used"]]

        return DefenderOutput(
            original_text=defender_input.text,
            rewritten_text=parsed["rewritten_text"],
            target_attributes=defender_input.target_attributes,
            strategies_used=strategies,
            confidence=float(parsed["confidence"]),
            iteration=defender_input.iteration,
            syntactic_pii_found=syntactic_pii,
            ground_truth=defender_input.ground_truth,
            clue_map=clue_map,
        )

    def extract_ground_truth(
        self, text: str, target_attributes: list[str]
    ) -> dict[str, str]:
        """Infer the actual values of target attributes from the original text.

        Uses an LLM call to perform any reasoning needed (e.g. event year +
        stated age → birth year). Fails silently and returns {} so the pipeline
        is never blocked by a bad response here.
        """
        if not target_attributes:
            return {}
        prompt = build_ground_truth_prompt(text, target_attributes)
        try:
            response = self._call_llm([prompt], system_prompt=ANALYSIS_SYSTEM_PROMPT)
            parsed = parse_llm_json(response)
            return {
                k: str(v)
                for k, v in parsed.get("ground_truth", {}).items()
                if v is not None
            }
        except Exception:
            # non-fatal — orchestrator can still run without ground truth
            return {}

    def _enumerate_clues(
        self, text: str, target_attributes: list[str]
    ) -> dict:
        """Pre-pass: map every inference chain that could reveal a target attribute.

        The result is injected into the rewrite prompt as an explicit checklist,
        so the rewriting LLM doesn't have to discover clues itself mid-task.
        Fails silently and returns {} so the pipeline degrades gracefully.
        """
        if not target_attributes:
            return {}
        prompt = build_clue_enumeration_prompt(text, target_attributes)
        try:
            response = self._call_llm([prompt], system_prompt=ANALYSIS_SYSTEM_PROMPT)
            parsed = parse_llm_json(response)
            return parsed.get("clue_map", {})
        except Exception:
            # non-fatal — rewrite will still run, just without the explicit checklist
            return {}

    def _call_llm(self, contents: list[str], system_prompt: str = REWRITE_SYSTEM_PROMPT) -> str:
        """Send contents to the Gemini API, falling back to Vertex on rate limits."""
        try:
            return self._send(self._client, contents, system_prompt)
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
                return self._send(self._fallback_client, contents, system_prompt)
            raise

    def _send(self, client: genai.Client, contents: list[str], system_prompt: str) -> str:
        """Send a request using a specific genai client."""
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config={
                    "system_instruction": system_prompt,
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
