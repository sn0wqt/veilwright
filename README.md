# Defender

The Defender agent — the anonymization component of a multi-agent semantic anonymization system. It rewrites free text to hide specific target attributes by neutralizing semantic clues that an LLM could exploit, while preserving the underlying meaning of the text.

## System Overview

This project is part of a three-agent adversarial system:

1. **Defender** (this project) — rewrites text to hide target attributes
2. **Attacker** — receives the rewritten text and tries to guess the hidden attributes
3. **Utility Judge** — scores how much of the original meaning was preserved

The system runs as an adversarial loop: if the Attacker guesses correctly, the Defender retries with a heavier rewrite, until the Attacker fails but the Utility Judge still gives a passing score.

## How It Works

The Defender uses a two-pass pipeline:

1. **Syntactic scanner** — regex-based detection of explicit PII (emails, phone numbers, dates, credit card numbers, `<PERSON>` tags). These are masked before reaching the LLM.
2. **Semantic rewriting** — a Gemini LLM applies one of three strategies per target attribute:
   - **Abstraction** — replace specific clues with vaguer equivalents (e.g. "moon landing" → "historic space event")
   - **Shifting** — replace clues with plausible but different references
   - **Omission** — remove clues entirely (last resort)

## Installation

```bash
# Install uv if you haven't already
pip install uv

# Sync dependencies
uv sync --extra dev

# Download the spaCy NER model
uv pip install en_core_web_sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

## Configuration

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your-api-key-here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/) → Get API key.

## Usage

```bash
# View all available commands
defender --help

# View help for a specific command
defender anonymize --help
```

### Defend (Anonymize Text)

Inline text:

```bash
defender anonymize --text "I remember watching the moon landing with my father. It was a huge event to see Neil Armstrong become the first man on the Moon. Funnily enough, this is the only specific memory I have from when I was six years old." --attributes "Age,Birth Year,Exact Event"
```

From a file:

```bash
defender anonymize --file input.txt --attributes "Age,Birth Year,Exact Event"
```

With a specific model:

```bash
defender anonymize --text "..." --attributes "Age" --model gemini-2.5-pro
```

JSON output (for machine-to-machine communication):

```bash
defender anonymize --text "..." --attributes "Age" --json
```

### List Available Models

```bash
defender models
```

## Python API

The Defender exposes a clean interface for integration with the Attacker and Utility Judge:

```python
from defender import run_defender, DefenderInput

result = run_defender(DefenderInput(
    text="I remember watching the moon landing...",
    target_attributes=["Age", "Birth Year", "Exact Event"],
))

print(result.rewritten_text)
print(result.strategies_used)
print(result.confidence)
```

For the adversarial loop (with Attacker feedback):

```python
result = run_defender(DefenderInput(
    text="I remember watching the moon landing...",
    target_attributes=["Age", "Birth Year"],
    iteration=2,
    attacker_feedback="The narrator watched the moon landing at age 6, so born ~1963.",
))
```

## Project Structure

```text
src/defender/
├── __init__.py      # Top-level API: run_defender()
├── __main__.py      # CLI entry point
├── defender.py      # Core pipeline: scanner → LLM → parse
├── prompts.py       # System prompt, user prompt builder, retry prompt
├── scanner.py       # Regex-based PII detection + masking
├── strategies.py    # RewriteStrategy enum + descriptions
├── types.py         # DefenderInput, DefenderOutput, StrategyRecord
└── utils.py         # JSON extraction + response validation
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT — see [LICENSE.md](LICENSE.md).
