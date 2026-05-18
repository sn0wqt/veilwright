"""LLM prompt templates for the Defender agent."""

from defender.strategies import STRATEGY_DESCRIPTIONS, RewriteStrategy


SYSTEM_PROMPT = """\
You are the Defender agent in a multi-agent semantic anonymization system.

Your task is to rewrite text so that specific target attributes CANNOT be \
inferred by an adversarial Attacker agent, while preserving as much of the \
original meaning and narrative as possible.

You have three rewrite strategies available:

1. **Abstraction** — {abstraction}
2. **Shifting** — {shifting}
3. **Omission** — {omission}

RULES:
- You MUST reason step-by-step about each target attribute before rewriting.
- For each attribute, identify ALL clues in the text that could reveal it.
- Choose the lightest-touch strategy that removes inferability.
- The rewritten text must read naturally — no placeholders, no brackets, \
no meta-commentary.
- Maintain internal consistency: if you shift one detail, adjust all \
related details to match.
- Return your response as valid JSON (and nothing else).
""".format(
    abstraction=STRATEGY_DESCRIPTIONS[RewriteStrategy.ABSTRACTION],
    shifting=STRATEGY_DESCRIPTIONS[RewriteStrategy.SHIFTING],
    omission=STRATEGY_DESCRIPTIONS[RewriteStrategy.OMISSION],
)


UTILITY_SYSTEM_PROMPT = """\
You are the Utility Judge in a multi-agent semantic anonymization system.

Your task is to score how much of the original, non-sensitive meaning was
preserved in the rewritten text. The rewritten text may intentionally remove
or alter clues that reveal target attributes.

RULES:
- Focus on semantic meaning preservation: events, relationships, tone, and intent.
- Do NOT penalize removal or abstraction of target attributes.
- Penalize factual contradictions, missing core meaning, or added facts.
- Return your response as valid JSON (and nothing else).
"""


def build_rewrite_prompt(
    text: str,
    target_attributes: list[str],
    iteration: int = 1,
    attacker_feedback: str | None = None,
) -> str:
    """Build the user message for the Defender's rewrite request."""
    attributes_list = "\n".join(f"  - {attr}" for attr in target_attributes)

    prompt = f"""\
Rewrite the following text to hide these target attributes:

{attributes_list}

TEXT:
\"\"\"
{text}
\"\"\"
"""

    if iteration > 1:
        prompt += f"""
This is iteration {iteration}. Your previous rewrite was NOT sufficient — \
the Attacker was able to guess one or more attributes. You MUST apply a \
heavier rewrite this time.
"""

    if attacker_feedback:
        prompt += f"""
ATTACKER FEEDBACK FROM PREVIOUS ROUND:
\"\"\"
{attacker_feedback}
\"\"\"

Use this feedback to understand what clues the Attacker exploited, and \
make sure to eliminate them in this rewrite.
"""

    prompt += """
INSTRUCTIONS:
1. For each target attribute, identify every clue in the text that could \
reveal it (direct mentions, contextual hints, temporal markers, etc.).
2. Choose a strategy (abstraction, shifting, or omission) for each attribute. \
Prefer abstraction > shifting > omission in terms of meaning preservation.
3. Apply all strategies simultaneously to produce a single rewritten text.
4. Self-assess your confidence (0.0–1.0) that an adversarial LLM could NOT \
guess any of the target attributes from the rewritten text.

Return ONLY a JSON object with this exact structure (no markdown fences, no \
extra text):

{
  "rewritten_text": "...",
  "strategies_used": [
    {
      "attribute": "...",
      "strategy": "abstraction | shifting | omission",
      "reasoning": "..."
    }
  ],
  "confidence": 0.85
}
"""

    return prompt


def build_utility_prompt(
    original_text: str,
    rewritten_text: str,
    target_attributes: list[str] | None = None,
) -> str:
    """Build the user message for the Utility Judge scoring request."""
    targets = "\n".join(f"  - {attr}" for attr in (target_attributes or []))

    prompt = f"""\
Evaluate how well the rewritten text preserves the original meaning.

TARGET ATTRIBUTES (allowed to be hidden or altered):
{targets if targets else "  - (none provided)"}

ORIGINAL TEXT:
\"\"\"
{original_text}
\"\"\"

REWRITTEN TEXT:
\"\"\"
{rewritten_text}
\"\"\"

Return ONLY a JSON object with this exact structure (no markdown fences, no extra text):

{{
    "score": 0.0,
    "rationale": "..."
}}
"""

    return prompt


RETRY_PROMPT = """\
Your previous response was not valid JSON. Please respond with ONLY a raw JSON \
object — no markdown code fences, no commentary, no text before or after the JSON.

The JSON must have exactly these keys:
- "rewritten_text": string
- "strategies_used": array of objects, each with "attribute", "strategy", "reasoning"
- "confidence": number between 0.0 and 1.0

Respond now with the corrected JSON:
"""


UTILITY_RETRY_PROMPT = """\
Your previous response was not valid JSON. Please respond with ONLY a raw JSON \
object — no markdown code fences, no commentary, no text before or after the JSON.

The JSON must have exactly these keys:
- "score": number between 0.0 and 1.0
- "rationale": string

Respond now with the corrected JSON:
"""
