"""
Core Defender agent logic.

Orchestrates the syntactic scanner pass and LLM-based semantic rewriting
to produce anonymized text where target attributes cannot be inferred.
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
load_dotenv()

from google import genai

from defender.prompts import SYSTEM_PROMPT, RETRY_PROMPT, build_rewrite_prompt
from defender.scanner import scan_text
from defender.types import DefenderInput, DefenderOutput, StrategyRecord
from defender.utils import parse_llm_json, validate_defender_response


class DefenderError(Exception):
    """Raised when the Defender agent encounters an unrecoverable error."""


class Defender:
    """
    The Defender agent — rewrites free text to hide target attributes.

    Pipeline:
        1. Run the syntactic scanner to detect and mask explicit PII.
        2. Build a chain-of-thought prompt for the LLM.
        3. Call the Gemini API and parse the structured JSON response.
        4. Retry once with a stricter prompt if JSON parsing fails.

    Args:
        api_key: Gemini API key. Falls back to ``GEMINI_API_KEY`` env var.
        model: Model identifier to use.

    Raises:
        DefenderError: If no API key is available.
    """

    DEFAULT_MODEL = "gemini-2.5-flash"
    MAX_TOKENS = 4096

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise DefenderError(
                "No Gemini API key provided. Set the GEMINI_API_KEY "
                "environment variable or pass api_key= to the Defender constructor."
            )

        self.model = model or self.DEFAULT_MODEL
        self._client = genai.Client(api_key=resolved_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, defender_input: DefenderInput) -> DefenderOutput:
        """
        Execute the full Defender pipeline.

        Args:
            defender_input: The input payload with text and target attributes.

        Returns:
            A :class:`DefenderOutput` with the rewritten text and metadata.

        Raises:
            DefenderError: On LLM call failure or unrecoverable parse errors.
        """
        # --- Step 1: Syntactic scanner pass ---
        scan_result = scan_text(defender_input.text)
        syntactic_pii = [
            f"{m.pii_type}: {m.value}" for m in scan_result.pii_found
        ]

        # Use masked text for the LLM prompt to reduce token leakage
        text_for_llm = scan_result.masked_text

        # --- Step 2: Build prompt ---
        user_prompt = build_rewrite_prompt(
            text=text_for_llm,
            target_attributes=defender_input.target_attributes,
            iteration=defender_input.iteration,
            attacker_feedback=defender_input.attacker_feedback,
        )

        # --- Step 3: Call the LLM ---
        contents = [user_prompt]
        llm_response_text = self._call_llm(contents)

        # --- Step 4: Parse and validate ---
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
            # --- Step 5: Retry once with stricter prompt ---
            contents.append(llm_response_text)  # model's previous reply
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
                    f"Failed to parse LLM JSON response after retry: {exc}"
                ) from exc

        # --- Build output ---
        strategies = [
            StrategyRecord.from_dict(s) for s in parsed["strategies_used"]
        ]

        return DefenderOutput(
            original_text=defender_input.text,
            rewritten_text=parsed["rewritten_text"],
            target_attributes=defender_input.target_attributes,
            strategies_used=strategies,
            confidence=float(parsed["confidence"]),
            iteration=defender_input.iteration,
            syntactic_pii_found=syntactic_pii,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_llm(self, contents: list[str]) -> str:
        """
        Send contents to the Gemini API and return the text response.

        Args:
            contents: Conversation contents as a list of strings.

        Returns:
            The model's text response.

        Raises:
            DefenderError: On API errors.
        """
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "max_output_tokens": self.MAX_TOKENS,
                    "temperature": 0.7,
                },
            )
            return response.text
        except Exception as exc:
            raise DefenderError(f"Gemini API error: {exc}") from exc
