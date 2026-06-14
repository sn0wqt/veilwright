# Defender

The Defender agent — the anonymization component of a multi-agent semantic anonymization system. It rewrites free text to hide specific target attributes by neutralizing semantic clues that an LLM could exploit, while preserving the underlying meaning of the text.

## System Overview

This project is part of a three-agent adversarial system:

1. **Defender** (this project) — rewrites text to hide target attributes
2. **Attacker** — receives the rewritten text and tries to guess the hidden attributes
3. **Utility Judge** — scores how much of the original meaning was preserved

The system runs as an adversarial loop until privacy and utility pass in the same iteration. If the Attacker guesses correctly, the Defender retries with stronger anonymization for the leaked attributes. If privacy passes but utility is too low, the Defender retries with lighter-touch preservation of non-sensitive meaning.

## How It Works

The Defender uses a two-pass pipeline:

1. **Syntactic scanner** — spaCy NER plus regex detection. Direct identifiers such as names, emails, phone numbers, and credit card numbers are masked before reaching the LLM. Contextual clues such as dates, locations, organizations, and money are detected but usually left visible so the LLM can rewrite them semantically.
2. **Semantic rewriting** — a Gemini LLM applies one of three strategies per target attribute:
   - **Abstraction** — replace specific clues with vaguer equivalents (e.g. "moon landing" → "historic space event")
   - **Shifting** — replace clues with plausible but different references
   - **Omission** — remove clues entirely (last resort)

The full adversarial loop also includes:

3. **Attacker** — attempts to de-anonymize rewritten text using contextual inference.
4. **Utility Judge** — scores how much of the original meaning is preserved (0.0-1.0).

## Installation

```bash
# Install uv if you haven't already
pip install uv

# Sync dependencies, including dev tools and the spaCy model
uv sync --extra dev
```

## Configuration

Create a `.env` file in the project root. For Google AI Studio:

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

If both are configured, Defender uses AI Studio first and automatically switches to Vertex AI when the primary client is rate-limited or unavailable.

## Usage

```bash
# View all available commands
defender --help

# View help for a specific command
defender anonymize --help
```

### Anonymize (Single Pass)

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
defender anonymize --text "I was six years old during the moon landing." --attributes "Age" --model gemini-2.5-pro
```

JSON output (for machine-to-machine communication):

```bash
defender anonymize --text "I was six years old during the moon landing." --attributes "Age" --json
```

### Adversarial Loop (Defender + Attacker + Utility Judge)

```bash
defender adversarial --text "I remember watching the moon landing with my father when I was six years old." --attributes "Age,Birth Year,Exact Event" --iterations 3 --json
```

Readable terminal summary:

```bash
defender adversarial --text "I remember watching the moon landing with my father when I was six years old." --attributes "Age,Birth Year,Exact Event" --iterations 3 --no-json
```

Save a JSON report file:

```bash
defender adversarial --file input.txt --attributes "Age,Birth Year,Exact Event" --iterations 3 --json --out report.json
```

You can customize models and utility threshold:

```bash
defender adversarial --text "I started residency after medical school and now lead a hospital clinic." --attributes "Profession" --defender-model gemini-2.5-flash --attacker-model gemini-3-flash-preview --utility-model gemini-2.5-flash --utility-threshold 0.75 --confidence-threshold 0.7 --json
```

An attacker guess counts as a privacy failure only when it fuzzy-matches the ground truth and the attacker's confidence is at least the configured confidence threshold. The default is `0.7`, chosen to be privacy-sensitive for plausible semantic leaks.

### List Available Models

```bash
defender models
```

## Python API

The Defender exposes a clean interface for integration with the Attacker and Utility Judge:

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

For the adversarial loop (with Attacker feedback):

```python
result = run_defender(DefenderInput(
    text="I remember watching the moon landing with my father.",
    target_attributes=["Age", "Birth Year"],
    iteration=2,
    attacker_feedback="The narrator watched the moon landing at age 6, so born ~1963.",
    ground_truth=previous_result.ground_truth,  # Important: carry over from iter 1
))
```

*See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for full details on the `ground_truth` and `clue_map` fields.*

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
├── scanner.py       # spaCy + regex PII detection and masking
├── strategies.py    # RewriteStrategy enum + descriptions
├── types.py         # Shared dataclasses for all agents and loop results
├── utility.py       # Utility Judge scorer
└── utils.py         # JSON extraction + response validation
```

## License

MIT — see [LICENSE.md](LICENSE.md).
