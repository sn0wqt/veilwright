"""Attacker agent: tries to de-anonymize rewritten text targeting specific attributes."""

from json import JSONDecodeError

from defender.llm_client import GeminiClient
from defender.types import AttackerOutput
from defender.utils import parse_llm_json, parse_validated_llm_json


def _build_system_prompt(confidence_threshold: float) -> str:
    """Build the Attacker system prompt with the active confidence threshold."""
    return f"""You are an adversarial Attacker agent in a multi-agent semantic anonymization system.

Your goal is to red-team a defensive anonymization system by inferring the
requested target attributes from the rewritten text. This is a privacy
evaluation task for improving anonymization quality.

For each target attribute:
1. Analyze the clues, contextual hints, and temporal/relational markers.
2. Make the most specific best-effort guess supported by the text. Do not
   overuse "UNKNOWN": if the text supports a broad category or plausible value,
   return that guess with lower confidence rather than hiding uncertainty.
   Use "UNKNOWN" only when no meaningful clue remains for that attribute, and
   assign "UNKNOWN" a confidence of 0.0.
3. Assign a confidence score from 0.0 to 1.0 indicating how certain you are of your guess.

You MUST include exactly one entry for every requested target attribute in
`guesses`, `reasoning`, and `confidence`. Do not include attributes outside
the requested target list. In `reasoning`, provide a concise step-by-step
inference summary that explicitly names the clues you used. If you make a
low-confidence guess, explain why it is uncertain. If no clue remains, explain
why the value is UNKNOWN.
In `successful_attributes`, include only requested attributes where confidence
is >= {confidence_threshold:.2f} and the guess is not "UNKNOWN".

You must return your response as a valid JSON object with the following structure:
{{
  "guesses": {{
    "<attribute_name>": "<best guess or UNKNOWN>"
  }},
  "reasoning": {{
    "<attribute_name>": "<step-by-step inference details>"
  }},
  "confidence": {{
    "<attribute_name>": 0.85
  }},
  "successful_attributes": [
    "<attribute_name>"
  ]
}}

Repeat each object member for every requested target attribute. If no attribute
meets the success criteria, return an empty `successful_attributes` array.
No other text or markdown code fences outside the JSON.
"""


def _build_retry_prompt(confidence_threshold: float) -> str:
    """Build the Attacker retry prompt with the active confidence threshold."""
    return f"""Your previous response was not valid JSON. Please respond with ONLY a raw JSON object — no markdown code fences, no commentary, no text before or after the JSON.

The JSON must have exactly these keys:
- "guesses": object mapping attribute name to guess (string)
- "reasoning": object mapping attribute name to reasoning (string)
- "confidence": object mapping attribute name to float between 0.0 and 1.0
- "successful_attributes": list of requested attributes where confidence >= {confidence_threshold:.2f}

Respond now with the corrected JSON:
"""


def _build_repair_prompt(
    rewritten_text: str,
    target_attributes: list[str],
    confidence_threshold: float,
    previous_error: str,
) -> str:
    """Build a standalone final-repair prompt for malformed attacker output."""
    attributes_list = "\n".join(f"  - {attr}" for attr in target_attributes)
    return f"""Your previous attacker response could not be parsed as valid JSON.

Previous parser error:
{previous_error}

Return ONLY one raw JSON object. Do not include markdown, commentary, apologies,
or text outside the JSON object. Make best-effort guesses when clues support a
broad or uncertain value, using lower confidence for uncertainty. Use "UNKNOWN"
with confidence 0.0 only when no meaningful clue remains.

Target attributes:
{attributes_list}

Rewritten text:
\"\"\"
{rewritten_text}
\"\"\"

Required JSON object:
{{
  "guesses": {{
    "<attribute_name>": "<best guess or UNKNOWN>"
  }},
  "reasoning": {{
    "<attribute_name>": "<brief clue-based inference summary>"
  }},
  "confidence": {{
    "<attribute_name>": 0.0
  }},
  "successful_attributes": []
}}

Only include requested attributes. Include a name in `successful_attributes`
only when its confidence is >= {confidence_threshold:.2f} and its guess is not
"UNKNOWN".
"""


class AttackerError(Exception):
    """Raised when the Attacker encounters an unrecoverable error."""


class Attacker:
    """Attacker wrapper around Gemini calls."""

    DEFAULT_MODEL = "gemini-2.5-flash"
    MAX_TOKENS = 4096

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        confidence_threshold: float = 0.7,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise AttackerError("confidence_threshold must be between 0.0 and 1.0.")
        self.model = model or self.DEFAULT_MODEL
        self.confidence_threshold = confidence_threshold
        self._llm = GeminiClient(
            model=self.model,
            label="attacker",
            error_type=AttackerError,
            api_key=api_key,
        )

    def run(self, rewritten_text: str, target_attributes: list[str]) -> AttackerOutput:
        """Run the attacker against rewritten text to guess target attributes."""
        if not rewritten_text.strip():
            raise AttackerError("rewritten_text must be non-empty.")
        if not target_attributes:
            raise AttackerError("target_attributes must contain at least one attribute.")

        prompt = self._build_prompt(rewritten_text, target_attributes)
        raw_response = self._call_llm([prompt])

        try:
            parsed = parse_validated_llm_json(
                initial_text=raw_response,
                retry=lambda: self._call_llm(
                    [prompt, _build_retry_prompt(self.confidence_threshold)]
                ),
                validate=lambda data: self._validate_response(data, target_attributes),
                error_type=AttackerError,
                parse_error_message="Failed to parse Attacker JSON after retry: {error}",
                validation_error_message=(
                    "Attacker response failed validation after retry: {errors}"
                ),
            )
        except AttackerError as exc:
            repair_prompt = _build_repair_prompt(
                rewritten_text=rewritten_text,
                target_attributes=target_attributes,
                confidence_threshold=self.confidence_threshold,
                previous_error=str(exc),
            )
            repair_response = self._call_llm([repair_prompt], temperature=0.0)
            try:
                parsed = parse_llm_json(repair_response)
            except JSONDecodeError as repair_exc:
                snippet = repair_response[:500].replace("\n", "\\n")
                raise AttackerError(
                    "Failed to parse Attacker JSON after final repair attempt: "
                    f"{repair_exc}. Raw repair response starts with: {snippet!r}"
                ) from repair_exc

            errors = self._validate_response(parsed, target_attributes)
            if errors:
                raise AttackerError(
                    "Attacker response failed validation after final repair "
                    f"attempt: {errors}"
                ) from exc
            raw_response = repair_response

        # Ensure all target attributes exist in parsed output and recompute successes.
        guesses_raw = parsed.get("guesses", {})
        reasoning_raw = parsed.get("reasoning", {})
        confidence_raw = parsed.get("confidence", {})
        guesses = (
            {str(k): str(v) for k, v in guesses_raw.items()}
            if isinstance(guesses_raw, dict)
            else {}
        )
        reasoning = (
            {str(k): str(v) for k, v in reasoning_raw.items()}
            if isinstance(reasoning_raw, dict)
            else {}
        )
        confidence_values = confidence_raw if isinstance(confidence_raw, dict) else {}

        confidence: dict[str, float] = {}
        successful_attributes: list[str] = []
        for attr in target_attributes:
            if attr not in guesses:
                guesses[attr] = "UNKNOWN"
            if attr not in reasoning:
                reasoning[attr] = "No reasoning provided."

            try:
                confidence[attr] = float(confidence_values.get(attr, 0.0))
            except (ValueError, TypeError):
                confidence[attr] = 0.0

            if guesses[attr].strip().upper() == "UNKNOWN":
                confidence[attr] = 0.0

            if (
                confidence[attr] >= self.confidence_threshold
                and guesses[attr].strip().upper() != "UNKNOWN"
            ):
                successful_attributes.append(attr)

        return AttackerOutput(
            guesses=guesses,
            reasoning=reasoning,
            confidence=confidence,
            successful_attributes=successful_attributes,
            raw_response=raw_response,
        )

    def _build_prompt(self, text: str, target_attributes: list[str]) -> str:
        """Build the Attacker user prompt for a rewritten text."""
        attributes_list = "\n".join(f"  - {attr}" for attr in target_attributes)
        return f"""Analyze the following rewritten text and infer the target attributes.

Target attributes to infer:
{attributes_list}

Rewritten Text:
\"\"\"
{text}
\"\"\"

Return the result as a valid JSON object matching the requested schema.
"""

    def _validate_response(
        self, data: dict[str, object], target_attributes: list[str]
    ) -> list[str]:
        """Validate the attacker's structured JSON response."""
        errors: list[str] = []
        for key in ("guesses", "reasoning", "confidence"):
            if key not in data:
                errors.append(f"Missing '{key}' field.")
            elif not isinstance(data[key], dict):
                errors.append(f"'{key}' must be a dictionary.")
            else:
                field = data[key]
                missing = set(target_attributes) - {str(attr) for attr in field}
                if missing:
                    errors.append(
                        f"'{key}' missing target attributes: {', '.join(sorted(missing))}"
                    )
                for attr in target_attributes:
                    value = field.get(attr)
                    if key in ("guesses", "reasoning") and not isinstance(value, str):
                        errors.append(f"'{key}.{attr}' must be a string.")
                    if key == "confidence":
                        if not isinstance(value, (int, float)):
                            errors.append(f"'confidence.{attr}' must be a number.")
                        elif not 0.0 <= float(value) <= 1.0:
                            errors.append(
                                f"'confidence.{attr}' must be between 0.0 and 1.0."
                            )
        successful = data.get("successful_attributes")
        if successful is not None and not isinstance(successful, list):
            errors.append("'successful_attributes' must be a list when provided.")
        elif isinstance(successful, list):
            requested = set(target_attributes)
            for attr in successful:
                if not isinstance(attr, str):
                    errors.append("'successful_attributes' entries must be strings.")
                elif attr not in requested:
                    errors.append(
                        f"'successful_attributes' contains unknown attribute '{attr}'."
                    )
        return errors

    def _call_llm(self, contents: list[str], temperature: float = 0.7) -> str:
        """Send contents to Gemini through the shared client."""
        return self._llm.generate(
            contents=contents,
            system_prompt=_build_system_prompt(self.confidence_threshold),
            max_output_tokens=self.MAX_TOKENS,
            temperature=temperature,
            response_mime_type="application/json",
        )
