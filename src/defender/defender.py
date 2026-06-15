"""Core Defender agent logic."""

from defender.llm_client import GeminiClient
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
from defender.utils import (
    parse_llm_json,
    parse_validated_llm_json,
    validate_defender_response,
)


class DefenderError(Exception):
    """Raised when the Defender agent encounters an unrecoverable error."""


class Defender:
    """Rewrites free text to hide target attributes using a syntactic + LLM pipeline."""

    DEFAULT_MODEL = "gemini-2.5-flash"
    MAX_TOKENS = 16384

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.model = model or self.DEFAULT_MODEL
        self._llm = GeminiClient(
            model=self.model,
            label="defender",
            error_type=DefenderError,
            api_key=api_key,
        )

    def run(self, defender_input: DefenderInput) -> DefenderOutput:
        """Run the full Defender pipeline on the given input."""

        if not defender_input.text.strip():
            raise DefenderError("Input text is empty.")
        if not defender_input.target_attributes:
            raise DefenderError("At least one target attribute is required.")

        estimated_tokens = len(defender_input.text) // 4
        if estimated_tokens > 6000:
            raise DefenderError(
                f"Input text too long (~{estimated_tokens} tokens). "
                f"Maximum supported: ~6000 tokens to leave room for prompt and output."
            )

        # auto-extract ground truth if caller didn't provide it
        ground_truth = dict(defender_input.ground_truth)
        if defender_input.iteration == 1 and not ground_truth:
            ground_truth = self.extract_ground_truth(
                defender_input.text, defender_input.target_attributes
            )

        # scan for explicit PII
        scan_result = scan_text(defender_input.text)
        syntactic_pii = [
            f"{m.pii_type}: {m.value} [{'masked' if m.mask else 'detected'}]"
            for m in scan_result.pii_found
        ]

        # send masked text to LLM so we don't leak raw emails/phones/etc
        text_for_llm = scan_result.masked_text

        # clue enumeration pre-pass (iteration 1 only)
        # On retries the attacker_feedback already tells us what clues slipped through,
        # so we don't need to re-enumerate from scratch.
        clue_map: dict[str, list[dict[str, str]]] = {}
        if defender_input.iteration == 1 and defender_input.target_attributes:
            clue_map = self._enumerate_clues(text_for_llm, defender_input.target_attributes)

        # build the rewrite prompt with clue map baked in
        user_prompt = build_rewrite_prompt(
            text=text_for_llm,
            target_attributes=defender_input.target_attributes,
            iteration=defender_input.iteration,
            attacker_feedback=defender_input.attacker_feedback,
            clue_map=clue_map or None,
        )

        # call the LLM
        contents = [user_prompt]
        llm_response_text = self._call_llm(contents)

        # parse and validate
        parsed = parse_validated_llm_json(
            initial_text=llm_response_text,
            retry=lambda: self._call_llm([user_prompt, RETRY_PROMPT]),
            validate=lambda data: validate_defender_response(
                data, defender_input.target_attributes
            ),
            error_type=DefenderError,
            parse_error_message="Failed to parse LLM JSON after retry: {error}",
            validation_error_message="LLM response failed validation after retry: {errors}",
        )

        # build output
        strategies = [
            StrategyRecord.from_dict(s)
            for s in parsed["strategies_used"]
            if isinstance(s, dict)
        ]

        rewritten_text = parsed["rewritten_text"]
        assert isinstance(rewritten_text, str)

        return DefenderOutput(
            original_text=defender_input.text,
            rewritten_text=rewritten_text,
            target_attributes=defender_input.target_attributes,
            strategies_used=strategies,
            confidence=float(parsed["confidence"]),
            iteration=defender_input.iteration,
            syntactic_pii_found=syntactic_pii,
            ground_truth=ground_truth,
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
            ground_truth = parsed.get("ground_truth", {})
            if not isinstance(ground_truth, dict):
                return {}
            by_lower = {str(key).lower(): value for key, value in ground_truth.items()}
            result: dict[str, str] = {}
            for attr in target_attributes:
                value = ground_truth.get(attr, by_lower.get(attr.lower()))
                if value is not None:
                    result[attr] = _stringify_ground_truth_value(value)
            return result
        except Exception:
            # non-fatal — orchestrator can still run without ground truth
            return {}

    def _enumerate_clues(
        self, text: str, target_attributes: list[str]
    ) -> dict[str, list[dict[str, str]]]:
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
            clue_map = parsed.get("clue_map", {})
            return _normalize_clue_map(clue_map, target_attributes)
        except Exception:
            # non-fatal — rewrite will still run, just without the explicit checklist
            return {}

    def _call_llm(self, contents: list[str], system_prompt: str = REWRITE_SYSTEM_PROMPT) -> str:
        """Send contents to the Gemini API, falling back to Vertex on rate limits."""
        return self._llm.generate(
            contents=contents,
            system_prompt=system_prompt,
            max_output_tokens=self.MAX_TOKENS,
            temperature=0.7,
            response_mime_type="application/json",
        )


def _normalize_clue_map(
    clue_map: object,
    target_attributes: list[str],
) -> dict[str, list[dict[str, str]]]:
    """Keep clue-map data only for requested attributes and coerce values to strings."""
    if not isinstance(clue_map, dict):
        return {}

    by_lower = {str(key).lower(): value for key, value in clue_map.items()}
    normalized: dict[str, list[dict[str, str]]] = {}
    for attr in target_attributes:
        raw_clues = clue_map.get(attr, by_lower.get(attr.lower(), []))
        if not isinstance(raw_clues, list):
            normalized[attr] = []
            continue
        normalized[attr] = [
            {
                "clue": str(clue.get("clue", "")),
                "type": str(clue.get("type", "")),
                "inference": str(clue.get("inference", "")),
            }
            for clue in raw_clues
            if isinstance(clue, dict)
        ]
    return normalized


def _stringify_ground_truth_value(value: object) -> str:
    """Convert LLM-extracted ground-truth values into readable strings."""
    if isinstance(value, list):
        return "; ".join(str(item) for item in value if item is not None)
    if isinstance(value, dict):
        return "; ".join(
            f"{key}: {item}" for key, item in value.items() if item is not None
        )
    return str(value)
