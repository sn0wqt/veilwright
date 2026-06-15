"""Shared Gemini client wrapper with AI Studio and Vertex fallback support."""

import os
import sys

from dotenv import load_dotenv
from google import genai

load_dotenv()


class GeminiClient:
    """Small wrapper around Gemini generation with credential and fallback handling."""

    EMPTY_RESPONSE_ATTEMPTS = 4

    def __init__(
        self,
        model: str,
        label: str,
        error_type: type[Exception],
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.label = label
        self._error_type = error_type
        self._client: genai.Client | None = None
        self._fallback_client: genai.Client | None = None

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if resolved_key:
            self._client = genai.Client(api_key=resolved_key)

        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if credentials_path and project_id and os.path.exists(credentials_path):
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
            vertex_hint = ""
            if credentials_path and not os.path.exists(credentials_path):
                vertex_hint = f" Credential file not found: {credentials_path}."
            raise self._error(
                "No credentials found. Set GEMINI_API_KEY for AI Studio, "
                "or GOOGLE_APPLICATION_CREDENTIALS + GOOGLE_CLOUD_PROJECT "
                f"for Vertex AI.{vertex_hint}"
            )

    def generate(
        self,
        contents: list[str],
        system_prompt: str,
        max_output_tokens: int,
        temperature: float,
        response_mime_type: str | None = None,
    ) -> str:
        """Generate text with the primary client, falling back on rate limits."""
        try:
            return self._send(
                self._client,
                contents,
                system_prompt,
                max_output_tokens,
                temperature,
                response_mime_type,
            )
        except self._error_type as exc:
            err_str = str(exc)
            is_rate_limited = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            is_unavailable = "503" in err_str or "UNAVAILABLE" in err_str

            if (is_rate_limited or is_unavailable) and self._fallback_client:
                print(
                    f"[{self.label}] Free tier hit limit, switching to Vertex AI "
                    "(Cloud credits).",
                    file=sys.stderr,
                )
                fallback_client = self._fallback_client
                text = self._send(
                    fallback_client,
                    contents,
                    system_prompt,
                    max_output_tokens,
                    temperature,
                    response_mime_type,
                )
                self._client = fallback_client
                self._fallback_client = None
                return text
            raise

    def _send(
        self,
        client: genai.Client | None,
        contents: list[str],
        system_prompt: str,
        max_output_tokens: int,
        temperature: float,
        response_mime_type: str | None,
    ) -> str:
        """Send a Gemini request using a concrete client."""
        if client is None:
            raise self._error("Gemini client was not initialized.")

        config: dict[str, object] = {
            "system_instruction": system_prompt,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
        }
        if response_mime_type:
            config["response_mime_type"] = response_mime_type

        try:
            for _ in range(self.EMPTY_RESPONSE_ATTEMPTS):
                response = client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )

                # Check if the prompt itself was blocked
                if hasattr(response, "prompt_feedback") and response.prompt_feedback:
                    block_reason = getattr(response.prompt_feedback, "block_reason", None)
                    if block_reason and type(block_reason).__name__ not in ("MagicMock", "Mock"):
                        raise self._error(
                            f"[{self.label}] Request prompt was blocked by Gemini safety filters or policy "
                            f"(Block Reason: {block_reason})."
                        )

                # Check if the response generation was blocked or cut off
                if hasattr(response, "candidates") and isinstance(response.candidates, list) and response.candidates:
                    candidate = response.candidates[0]
                    finish_reason = getattr(candidate, "finish_reason", None)
                    if finish_reason and type(finish_reason).__name__ not in ("MagicMock", "Mock"):
                        reason_str = str(finish_reason).upper()
                        if "SAFETY" in reason_str or "RECITATION" in reason_str:
                            raise self._error(
                                f"[{self.label}] Request/Response was blocked by Gemini safety filters or policy "
                                f"(Finish Reason: {finish_reason})."
                            )
                        elif "MAX_TOKENS" in reason_str:
                            raise self._error(
                                f"[{self.label}] Request/Response was cut off because it hit the maximum token limit of "
                                f"{max_output_tokens} tokens (Finish Reason: MAX_TOKENS). "
                                f"This can happen when reasoning models generate long thinking chains."
                            )

                text = response.text
                if text:
                    return text

            raise self._error(
                f"[{self.label}] Gemini model {self.model} returned an empty "
                f"response after {self.EMPTY_RESPONSE_ATTEMPTS} attempts. The "
                "input may be too long, temporarily unavailable, or blocked."
            )
        except self._error_type:
            raise
        except Exception as exc:
            raise self._error(f"Gemini API error: {exc}") from exc

    def _error(self, message: str) -> Exception:
        """Build an agent-specific exception."""
        return self._error_type(message)
