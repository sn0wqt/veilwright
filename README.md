# Adversarial Anonymization System (Defender)

An end-to-end multi-agent semantic text anonymization system featuring three collaborative agents: a **Defender** that anonymizes text, an **Attacker** that attempts to de-anonymize target attributes, and a **Utility Judge** that scores meaning preservation.

## System Overview

The `defender` CLI is the entry point for the complete proof-of-concept:

1. **Defender** — rewrites text to hide target attributes
2. **Attacker** — receives the rewritten text and tries to guess the hidden attributes
3. **Utility Judge** — scores how much of the original meaning was preserved

The system runs as an adversarial loop until privacy and utility pass in the same iteration. If the Attacker guesses correctly, the Defender retries with stronger anonymization for the leaked attributes. If privacy passes but utility is too low, the Defender retries with lighter-touch preservation of non-sensitive meaning.

## How It Works

The anonymization pipeline combines syntactic scanning, semantic analysis, and
LLM rewriting:

1. **Ground-truth extraction** — on the first iteration, the Defender infers the actual target values so attacker guesses can be verified later.
2. **Clue enumeration** — on the first iteration, the Defender maps direct, contextual, relational, and multi-hop clues for each target attribute.
3. **Syntactic scanner** — spaCy NER plus regex detection, with regex-only fallback if spaCy is unavailable. Direct identifiers such as names, emails, phone numbers, and credit card numbers are masked before reaching the LLM. Contextual clues such as dates, locations, organizations, and money are detected but usually left visible so the LLM can rewrite them semantically.
4. **Semantic rewriting** — a Gemini LLM applies one of three strategies per target attribute:
   - **Abstraction** — replace specific clues with vaguer equivalents (e.g. "moon landing" → "historic space event")
   - **Shifting** — replace clues with plausible but different references
   - **Omission** — remove clues entirely (last resort)

The full adversarial loop also includes:

5. **Attacker** — attempts to de-anonymize rewritten text using contextual inference.
6. **Utility Judge** — scores how much of the original meaning is preserved (0.0-1.0).

## Evaluation Logic

The adversarial loop reports `success=True` only when both conditions hold in the
same iteration:

- the Attacker has no verified successful guesses for any target attribute;
- the Utility Judge score is greater than or equal to the utility threshold
  (default: `0.75`).

An attacker guess is a verified privacy failure only when it fuzzy-matches the
ground truth and the attacker's confidence is at least the configured confidence
threshold (default: `0.7`). Fuzzy matching handles exact matches, identity/event/
location token matches, common role aliases such as `head of state` and
`president`, date normalization, and controlled age/year numeric tolerance.

The Utility Judge is separate from privacy verification. It uses Gemini to score
how much non-sensitive meaning was preserved, and it is explicitly instructed not
to penalize removal or abstraction of the target attributes.

Ground truth is auto-extracted by the Defender on the first iteration if it is
not supplied by the caller. The orchestrator reuses that same ground truth for
later iterations so it is not re-extracted. The Defender also builds a `clue_map`
on the first iteration only; later iterations use attacker and utility feedback
instead of repeating the clue pre-pass.

## Installation

Requires Python 3.12 or newer.

```bash
# 1. Install uv if you do not already have it.
pip install uv

# 2. Install project dependencies, including the small spaCy model.
uv sync --extra dev
```

The default install uses the small spaCy English model. To upgrade the syntactic scanner to the large model, install the optional `large` extra:

```bash
uv sync --extra dev --extra large
```

If spaCy or a model is unavailable, the scanner falls back to regex-only detection and the rest of the application still runs.

## Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

For Google AI Studio, set:

```text
GEMINI_API_KEY=your-api-key-here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/) → Get API key.

For Vertex AI fallback with Google Cloud credits, add:

```text
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global
```

If both are configured, the toolkit uses AI Studio first and automatically switches to Vertex AI when the primary client is rate-limited or unavailable.

Do not submit real `.env` files or service-account JSON keys. They are intentionally ignored by Git.

## Usage

```bash
# View all available commands
uv run defender --help

# View help for a specific command
uv run defender anonymize --help
```

### Anonymize (Single Pass)

Inline text:

```bash
uv run defender anonymize --text "I remember watching the moon landing with my father. It was a huge event to see Neil Armstrong become the first man on the Moon. Funnily enough, this is the only specific memory I have from when I was six years old." --attributes "Age,Birth Year,Exact Event"
```

From a file:

```bash
uv run defender anonymize --file input.txt --attributes "Age,Birth Year,Exact Event"
```

With a specific model:

```bash
uv run defender anonymize --text "I was six years old during the moon landing." --attributes "Age" --model gemini-2.5-pro
```

JSON output (for machine-to-machine communication):

```bash
uv run defender anonymize --text "I was six years old during the moon landing." --attributes "Age" --json
```

### Adversarial Loop (Defender + Attacker + Utility Judge)

```bash
uv run defender adversarial --text "I remember watching the moon landing with my father when I was six years old." --attributes "Age,Birth Year,Exact Event" --iterations 3 --json
```

Readable terminal summary:

```bash
uv run defender adversarial --text "I remember watching the moon landing with my father when I was six years old." --attributes "Age,Birth Year,Exact Event" --iterations 3 --no-json
```

Save a JSON report file:

```bash
uv run defender adversarial --file input.txt --attributes "Age,Birth Year,Exact Event" --iterations 3 --json --out report.json
```

You can customize models and utility threshold:

```bash
uv run defender adversarial --text "I started residency after medical school and now lead a hospital clinic." --attributes "Profession" --defender-model gemini-2.5-flash --attacker-model gemini-2.5-flash --utility-model gemini-2.5-flash --utility-threshold 0.75 --confidence-threshold 0.7 --json
```

An attacker guess counts as a privacy failure only when it fuzzy-matches the ground truth and the attacker's confidence is at least the configured confidence threshold. The default is `0.7`, chosen to be privacy-sensitive for plausible semantic leaks.

### List Available Models

```bash
uv run defender models
```

## Python API

Single-pass anonymization:

```python
from defender import run_defender, DefenderInput

result = run_defender(DefenderInput(
    text="I remember watching the moon landing with my father.",
    target_attributes=["Age", "Birth Year", "Exact Event"],
))

print(result.rewritten_text)
print(result.strategies_used)
print(result.confidence)
```

Full adversarial loop:

```python
from defender import run_adversarial_loop

result = run_adversarial_loop(
    text="I remember watching the moon landing with my father.",
    target_attributes=["Age", "Birth Year"],
    max_iterations=5,
)

print(result.success)
print(result.exit_reason)
print(result.final_rewritten_text)
print(result.final_utility_score)
```

Important result fields:

- `DefenderOutput.ground_truth`: inferred or user-provided target values.
- `DefenderOutput.clue_map`: first-iteration semantic inference clues.
- `AttackerOutput.guesses`: target attribute -> best-effort attacker guess.
- `AttackerOutput.reasoning`: clues used for each attacker guess.
- `AttackerOutput.confidence`: attacker confidence per attribute.
- `AdversarialResult.iterations`: full per-iteration Defender, Attacker, and
  Utility Judge outputs.

## Project Structure

```text
src/defender/
├── __init__.py      # Top-level API: run_defender()
├── __main__.py      # CLI entry point
├── attacker.py      # Attacker prompt + parsing logic
├── defender.py      # Core pipeline: scanner → LLM → parse
├── llm_client.py    # Shared Gemini client + Vertex fallback wrapper
├── orchestrator.py  # Adversarial loop runner
├── prompts.py       # System prompts, user prompt builders, retry prompts
├── scanner.py       # spaCy + regex PII detection with regex-only fallback
├── strategies.py    # RewriteStrategy enum + descriptions
├── types.py         # Shared dataclasses for all agents and loop results
├── utility.py       # Utility Judge scorer
└── utils.py         # JSON extraction + response validation
```

## License

MIT — see [LICENSE.md](LICENSE.md).
