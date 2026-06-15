"""Utility Judge for scoring meaning preservation."""

from defender.llm_client import GeminiClient
from defender.prompts import (
    UTILITY_RETRY_PROMPT,
    UTILITY_SYSTEM_PROMPT,
    build_utility_prompt,
)
from defender.types import UtilityInput, UtilityOutput
from defender.utils import parse_validated_llm_json, validate_utility_response


class UtilityError(Exception):
    """Raised when the Utility Judge encounters an unrecoverable error."""


class UtilityJudge:
    """Scores meaning preservation between original and rewritten text."""

    DEFAULT_MODEL = "gemini-2.5-flash"
    MAX_TOKENS = 4096

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.model = model or self.DEFAULT_MODEL
        self._llm = GeminiClient(
            model=self.model,
            label="utility",
            error_type=UtilityError,
            api_key=api_key,
        )

    def score(self, utility_input: UtilityInput) -> UtilityOutput:
        """Score semantic preservation for a rewritten text."""
        if not utility_input.original_text.strip():
            raise UtilityError("original_text must be non-empty.")
        if not utility_input.rewritten_text.strip():
            raise UtilityError("rewritten_text must be non-empty.")

        prompt = build_utility_prompt(
            original_text=utility_input.original_text,
            rewritten_text=utility_input.rewritten_text,
            target_attributes=utility_input.target_attributes,
        )

        llm_text = self._call_llm([prompt])

        parsed = parse_validated_llm_json(
            initial_text=llm_text,
            retry=lambda: self._call_llm([prompt, UTILITY_RETRY_PROMPT]),
            validate=validate_utility_response,
            error_type=UtilityError,
            parse_error_message="Failed to parse LLM JSON after retry: {error}",
            validation_error_message="LLM response failed validation after retry: {errors}",
        )

        rationale = parsed["rationale"]
        assert isinstance(rationale, str)

        return UtilityOutput(
            original_text=utility_input.original_text,
            rewritten_text=utility_input.rewritten_text,
            score=float(parsed["score"]),
            rationale=rationale,
        )

    def _call_llm(self, contents: list[str]) -> str:
        """Send contents to the Gemini API, falling back to Vertex on rate limits."""
        return self._llm.generate(
            contents=contents,
            system_prompt=UTILITY_SYSTEM_PROMPT,
            max_output_tokens=self.MAX_TOKENS,
            temperature=0.2,
            response_mime_type="application/json",
        )
