# Defender Output Format - Integration Guide

## For the Attacker & Judge Teams

The Defender outputs a structured JSON object that contains everything needed for the adversarial loop.

## Output Structure

```python
{
    "original_text": str,           # The original sensitive text
    "rewritten_text": str,          # The anonymized version
    "target_attributes": [str],     # List of attributes to hide (e.g., ["Age", "Birth Year"])
    "strategies_used": [            # What the Defender did for each attribute
        {
            "attribute": str,       # e.g., "Age"
            "strategy": str,        # "abstraction" | "shifting" | "omission"
            "reasoning": str        # Why this strategy was chosen
        }
    ],
    "confidence": float,            # 0.0-1.0, Defender's self-assessment
    "iteration": int,               # Which iteration this is (starts at 1)
    "syntactic_pii_found": [str],   # PII detected by scanner (e.g., ["EMAIL: test@example.com [masked]"])
    "ground_truth": {               # IMPORTANT: The actual sensitive values
        str: str                    # attribute -> actual value
    },
    "clue_map": {                   # NOTE: Only populated on iteration == 1
        str: [                      # attribute -> list of clue dicts
            {
                "clue": str,
                "type": str,        # "direct" | "contextual" | "multi-hop" | "relational"
                "inference": str
            }
        ]
    }
}
```

## The `ground_truth` Field

The `ground_truth` field is a dictionary mapping each target attribute to its actual value:

```python
{
    "Age": "6 years old",
    "Birth Year": "1963",
    "Location": "Paris",
    "Profession": "Software Engineer"
}
```

### How It Works

The `ground_truth` can be **user-provided** when calling the Defender, or **auto-extracted** if omitted.

**Auto-extraction (default):** If you don't provide `ground_truth`, the Defender makes an LLM call on iteration 1 to infer the actual values from the original text. The orchestrator then reuses those extracted values for all subsequent iterations by passing them back in `DefenderInput.ground_truth`.

**User-provided:** If you already know the true values, pass them in `DefenderInput.ground_truth`. The Defender will use those values directly and skip auto-extraction:

```python
result = run_defender(DefenderInput(
    text=text,
    target_attributes=target_attributes,
    ground_truth={
        "Age": "46",
        "Profession": "Cardiologist",
    },
))
```

In the built-in adversarial loop, `run_adversarial_loop()` passes an empty `ground_truth` dictionary on iteration 1, lets the Defender auto-extract the values once, then carries `defender_output.ground_truth` forward into every later `DefenderInput`. Ground truth is not re-extracted on iteration 2+.

### Usage Example

```python
from defender import run_defender, DefenderInput

result = run_defender(DefenderInput(
    text="I was born in 1990 in Paris and work as a software engineer.",
    target_attributes=["Birth Year", "Location", "Profession"],
    ground_truth={
        "Birth Year": "1990",
        "Location": "Paris",
        "Profession": "Software Engineer"
    }
))

# Now the orchestrator can compare:
# - result.ground_truth["Birth Year"] = "1990" (actual)
# - attacker_guess["Birth Year"] = "1990" (guessed)
# - Match? Yes → privacy failed, retry with stronger anonymization for that clue
```

## The `clue_map` Field

The `clue_map` field is populated only on `iteration == 1`. It contains the Defender's pre-pass map of direct, contextual, multi-hop, and relational clues that could reveal each target attribute. On later iterations the Defender relies on `attacker_feedback` from the previous round instead of re-running clue enumeration.

## For the Attacker

Your output should match this structure so the Judge can easily compare:

```python
{
    "guesses": {
        "Age": "6 years old",
        "Birth Year": "1963"
    },
    "reasoning": {
        "Age": "The rewritten text still says the narrator was a young child at the event.",
        "Birth Year": "The event clue plus age suggests the early 1960s."
    },
    "confidence": {
        "Age": 0.8,
        "Birth Year": 0.75
    },
    "successful_attributes": ["Age", "Birth Year"],
    "raw_response": "<optional raw LLM response when serialized from AttackerOutput>"
}
```

The Attacker must key `guesses`, `reasoning`, and `confidence` by the exact requested target attributes. Use `"UNKNOWN"` for any value that cannot be inferred. The orchestrator verifies `attacker_output.guesses` against `defender_output.ground_truth`; when ground truth is available, it overwrites `successful_attributes` in the stored iteration with the attributes that actually matched.

## For the Utility Judge

You'll receive:
1. `defender_output.original_text` - The original
2. `defender_output.rewritten_text` - The anonymized version
3. `defender_output.target_attributes` - Attributes that were allowed to be hidden or altered

Your job is only to score utility: how much non-sensitive meaning was preserved when comparing `original_text` and `rewritten_text`. Do not penalize removal, abstraction, or shifting of the requested target attributes.

## For the Orchestrator

The orchestrator compares `defender_output.ground_truth` against `attacker_output.guesses`. A target attribute counts as successfully guessed only when the attacker's confidence is at least the configured confidence threshold and the guess fuzzy-matches ground truth. The default threshold is `0.7`; pass `confidence_threshold=0.8` to `run_adversarial_loop()` or use the CLI `--confidence-threshold` flag to override it. Fuzzy matching includes exact/substring matches and controlled numeric proximity for age/year attributes. If ground truth is unavailable for an attribute, the orchestrator falls back to `attacker_output.successful_attributes`.

The loop continues while either privacy fails or utility is below threshold. Feedback sent back to the Defender includes both signals: attacker reasoning for successfully guessed attributes, and utility-recovery guidance when privacy passes but too much non-sensitive meaning was lost.

## CLI Usage

When using the CLI, you can't provide ground truth (it's for programmatic use). The CLI is just for manual testing:

```bash
defender anonymize --text "I was born in 1990." --attributes "Birth Year"
```

For the full adversarial loop, use the CLI or Python API:

```bash
defender adversarial --text "I moved to Paris after finishing medical school." --attributes "Location,Profession" --iterations 3 --no-json
```

```python
from defender import run_adversarial_loop

result = run_adversarial_loop(
    text="I moved to Paris after finishing medical school.",
    target_attributes=["Location", "Profession"],
    max_iterations=3,
    confidence_threshold=0.7,
)

print(result.success)
print(result.exit_reason)
print(result.final_rewritten_text)
print(result.confidence_threshold)
```
