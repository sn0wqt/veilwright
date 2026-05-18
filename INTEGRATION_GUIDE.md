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

**Auto-extraction (default):** If you don't provide `ground_truth`, the Defender makes an LLM call to infer the actual values from the original text. This happens once on the first call, then the extracted values are reused for all subsequent iterations.

**User-provided (recommended for orchestrator):** For better performance and consistency, the orchestrator should extract ground truth once and pass it to all Defender iterations:

```python
# Orchestrator extracts once
ground_truth = extract_ground_truth_once(text, target_attributes)

# Pass to all iterations
for iteration in range(1, max_iterations + 1):
    result = run_defender(DefenderInput(
        text=text,
        target_attributes=target_attributes,
        ground_truth=ground_truth,  # ← Reuse across iterations
        iteration=iteration,
    ))
```

**Why user-provided is better:** While the Defender *can* auto-extract ground truth, doing it at the Orchestrator level is recommended for production. It saves the Defender an extra LLM call (reducing latency and token cost), and ensures the Judge and Attacker receive a stable, consistent set of ground truth values across all adversarial iterations.

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

# Now the Judge can compare:
# - result.ground_truth["Birth Year"] = "1990" (actual)
# - attacker_guess["Birth Year"] = "1990" (guessed)
# - Match? Yes → Defender failed, retry with heavier rewrite
```

## For the Attacker

Your output should match this structure so the Judge can easily compare:

```python
{
    "guesses": {
        "Age": "6 years old",
        "Birth Year": "1963",
        # ... one entry per target_attributes
    },
    "reasoning": str,  # Your chain-of-thought
    "confidence": float  # 0.0-1.0
}
```

The Judge will then compare `defender_output.ground_truth` vs `attacker_output.guesses`.

## For the Judge

You'll receive:
1. `defender_output.original_text` - The original
2. `defender_output.rewritten_text` - The anonymized version
3. `defender_output.ground_truth` - The actual sensitive values
4. `attacker_output.guesses` - What the Attacker guessed

Your job:
1. **Privacy score**: Did the Attacker guess correctly? Compare `ground_truth` vs `guesses`
2. **Utility score**: How much meaning was preserved? Compare `original_text` vs `rewritten_text`

## CLI Usage

When using the CLI, you can't provide ground truth (it's for programmatic use). The CLI is just for manual testing:

```bash
defender anonymize --text "I was born in 1990." --attributes "Birth Year"
```

For the full adversarial loop, use the Python API:

```python
from defender import run_defender, DefenderInput

defender_output = run_defender(DefenderInput(
    text="...",
    target_attributes=["Age", "Location"],
    ground_truth={"Age": "30", "Location": "New York"}
))

# Pass defender_output to Attacker
# Pass both to Judge
```


