"""LLM prompt templates for the Defender agent.

Exports:
    REWRITE_SYSTEM_PROMPT       — system instruction for rewrite calls
    ANALYSIS_SYSTEM_PROMPT      — system instruction for GT/clue enumeration calls
    RETRY_PROMPT                — appended when LLM returns invalid JSON
    build_clue_enumeration_prompt — pre-pass: map all inference chains per attribute
    build_ground_truth_prompt   — extract actual attribute values from original text
    build_rewrite_prompt        — main rewrite instruction (accepts optional clue_map)
"""

from defender.strategies import STRATEGY_DESCRIPTIONS, RewriteStrategy


REWRITE_SYSTEM_PROMPT = """\
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


ANALYSIS_SYSTEM_PROMPT = """\
You are a privacy analysis assistant. Your task is to carefully analyze text \
and extract information or identify inference patterns as requested.

Be thorough, precise, and return your response as valid JSON (and nothing else).
"""


def build_clue_enumeration_prompt(text: str, target_attributes: list[str]) -> str:
    """Build the prompt for the clue enumeration pre-pass.

    This runs before the rewrite so the LLM knows exactly what inference
    chains exist for each target attribute before deciding how to neutralize them.
    """
    attributes_list = "\n".join(f"  - {attr}" for attr in target_attributes)

    return f"""\
You are a privacy analyst preparing a rewrite plan. Your job is to identify \
EVERY clue in the text that could help an adversary infer each target attribute.

Think carefully about all clue types:
- **Direct**: explicit statements ("I was six years old")
- **Contextual**: events or dates that anchor other attributes ("moon landing" \
implies 1969, which combined with an age gives a birth year)
- **Multi-hop**: chains of inference across multiple clues
- **Relational**: references that imply demographic facts ("my retirement party" \
implies older age)

Target attributes to protect:
{attributes_list}

TEXT:
\"\"\"
{text}
\"\"\"

Return ONLY a JSON object with this exact structure (no markdown, no extra text):

{{
  "clue_map": {{
    "<attribute_name>": [
      {{
        "clue": "<text fragment>",
        "type": "direct | contextual | multi-hop | relational",
        "inference": "<explanation of how this clue reveals the attribute>"
      }}
    ]
  }}
}}

Only include attributes from the target list. If no clues exist for an attribute, \
use an empty list. Be exhaustive — a missed clue is a privacy failure.
"""


def build_ground_truth_prompt(text: str, target_attributes: list[str]) -> str:
    """Build the prompt for auto-extracting ground truth values from text.

    Performs any inferences needed (e.g. event year + stated age → birth year)
    so the Judge can compare attacker guesses against actual values.
    """
    attributes_list = "\n".join(f"  - {attr}" for attr in target_attributes)

    return f"""\
Read the following text and determine the actual value of each target attribute \
for the person described. Perform any necessary inferences — do not just copy \
surface text, reason to the underlying fact.

For example: "I was six years old when I watched the moon landing" → \
Birth Year = 1963 (inferred: 1969 − 6 = 1963).

Target attributes:
{attributes_list}

TEXT:
\"\"\"
{text}
\"\"\"

Return ONLY a JSON object with this exact structure (no markdown, no extra text):

{{
  "ground_truth": {{
    "<attribute_name>": "<actual value>",
    ...
  }}
}}

If a value genuinely cannot be determined from the text, use null. \
Be precise — give specific values, not ranges, where the text supports it.
"""


def build_rewrite_prompt(
    text: str,
    target_attributes: list[str],
    iteration: int = 1,
    attacker_feedback: str | None = None,
    clue_map: dict | None = None,
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

    if clue_map:
        prompt += """
KNOWN INFERENCE CHAINS (you MUST neutralize ALL of these — missing even one \
is a privacy failure):
"""
        for attr, clues in clue_map.items():
            if clues:
                prompt += f"\n{attr}:\n"
                for clue in clues:
                    prompt += (
                        f"  - \"{clue['clue']}\" "
                        f"({clue['type']}): {clue['inference']}\n"
                    )
        prompt += "\n"

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

    if clue_map:
        prompt += """
INSTRUCTIONS:
1. Use the KNOWN INFERENCE CHAINS above as your checklist — every listed clue \
must be addressed. Do not rely on your own clue identification at this stage.
2. Choose a strategy (abstraction, shifting, or omission) for each attribute. \
Prefer abstraction > shifting > omission in terms of meaning preservation.
3. Apply all strategies simultaneously to produce a single rewritten text.
4. Self-assess your confidence (0.0–1.0) that an adversarial LLM could NOT \
guess any of the target attributes from the rewritten text.
"""
    else:
        prompt += """
INSTRUCTIONS:
1. For each target attribute, identify every clue in the text that could \
reveal it (direct mentions, contextual hints, temporal markers, etc.).
2. Choose a strategy (abstraction, shifting, or omission) for each attribute. \
Prefer abstraction > shifting > omission in terms of meaning preservation.
3. Apply all strategies simultaneously to produce a single rewritten text.
4. Self-assess your confidence (0.0–1.0) that an adversarial LLM could NOT \
guess any of the target attributes from the rewritten text.
"""

    prompt += """
IMPORTANT: You MUST include exactly one entry in the `strategies_used` array \
for EVERY target attribute requested. If an attribute is not present in the \
text, use the "omission" strategy and explicitly state that it was absent.

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


RETRY_PROMPT = """\
Your previous response was not valid JSON. Please respond with ONLY a raw JSON \
object — no markdown code fences, no commentary, no text before or after the JSON.

The JSON must have exactly these keys:
- "rewritten_text": string
- "strategies_used": array of objects, each with "attribute", "strategy", "reasoning"
- "confidence": number between 0.0 and 1.0

Respond now with the corrected JSON:
"""
