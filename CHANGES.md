# CHANGES.md

This document summarizes the feedback received during the poster session and how we responded in the final proof-of-concept.

## 1. Make the system easier to use from other systems

**Feedback:** Add a wrapper or interface so the anonymizer can be used by other systems.

**Status:** Partially incorporated.

We did not build a separate service wrapper because this project is a research proof-of-concept, and the feedback itself noted that a wrapper was probably not necessary for evaluation. Instead, we exposed the core functionality through simple interfaces:

- `run_anonymizer()` / `run_defender()` for a single anonymization pass.
- `run_adversarial_loop()` for the full Defender/Attacker/Utility Judge loop.
- `veilwright anonymize` and `veilwright adversarial` CLI commands for manual testing.
- `README.md` documenting setup, configuration, CLI usage, API usage, and the
  main output fields.

This keeps the project easy to run for evaluation while still making integration possible.

## 2. Define the utility score more clearly

**Feedback:** The poster mentioned a Utility Judge, but the scoring criteria were unclear. Reviewers asked whether the score measured factual consistency, readability, meaning preservation, task usefulness, or something else.

**Status:** Incorporated.

The Utility Judge is now defined as an LLM-based semantic meaning-preservation score from `0.0` to `1.0`. It compares the original and rewritten texts while treating the requested target attributes as intentionally hideable. The judge is instructed to:

- preserve non-sensitive meaning, narrative structure, relationships, tone, and intent;
- not penalize abstraction, shifting, or removal of the target attributes;
- penalize lost non-sensitive meaning, factual contradictions, hallucinated details, and overly destructive rewrites.

These criteria are implemented in the Utility Judge prompt and documented in
`README.md`.

## 3. Finish the adversarial loop for rapid testing

**Feedback:** Finish the end-to-end loop so the complete system can be tested quickly instead of only showing separate components.

**Status:** Fully incorporated.

The proof-of-concept now includes a complete automated adversarial loop:

- The Defender rewrites the text.
- The Attacker guesses each requested target attribute from the rewritten text.
- The Utility Judge scores meaning preservation on every iteration.
- The orchestrator exits successfully only when the Attacker fails on all target attributes and the utility score meets the configured threshold in the same iteration.
- Ground truth is extracted once on the first iteration and reused afterward.
- Attacker feedback, including the reasoning behind successful guesses, is passed back to the Defender.
- Utility feedback is passed back when privacy succeeds but the rewrite loses too much non-sensitive meaning.

The loop can be run with:

```bash
veilwright adversarial --text "..." --attributes "Age,Birth Year,Exact Event" --iterations 5 --no-json
```

## 4. Add concrete per-iteration results and more diverse examples

**Feedback:** Add experimental results showing how identity leakage is reduced after each anonymization iteration. Include more examples beyond political figures, such as medical or social-media-style text. Expand limitations around adversarial attacks, bias, and failure cases.

**Status:** Partially incorporated.

The adversarial CLI now prints readable per-iteration results, including:

- rewritten text for each iteration;
- the Attacker's guess for every target attribute;
- confidence values;
- reasoning for each inference;
- whether each guess matched ground truth;
- the Utility Judge score.

We tested the system on several input types, including the moon-landing example from the project description, a medical-professional style biography, a high-income professional profile, and a public political biography. These cover different leakage patterns: inferred age and birth year, profession and income, and globally unique public identity.

We did not build a broad benchmark suite or social-media dataset before the deadline. That remains a limitation of this proof-of-concept.

## 5. Clarify the evaluation methodology and system flow

**Feedback:** The evaluation methodology could be clearer and more objective. Reviewers also suggested making the Defender/Attacker/Judge flow more visible in the poster layout.

**Status:** Partially incorporated.

The software now exposes the full evaluation state for each `AdversarialIteration`: Defender output, Attacker output, verified successful attributes, Utility Judge score, and whether privacy failed. The CLI also shows the Defender -> Attacker -> Utility Judge progression in a readable form, making the evaluation process easier to inspect and reproduce.

We did not replace the LLM-based Utility Judge with a separate quantitative metric such as embedding similarity. That would be useful future work, but it was outside the scope of the final proof-of-concept. The current score should therefore be interpreted as an LLM-based utility judgment, not as a fully objective metric.

## Remaining limitations

- Outputs can vary between runs because all three agents rely on LLM behavior.
- Public figures and uniquely identifying career histories are difficult to anonymize without losing substantial utility.
- The Utility Judge is useful for rapid iteration, but it is still subjective.
- The loop has a cost and latency tradeoff because each iteration requires multiple LLM calls.
- Stronger attackers, biased model assumptions, or external world knowledge may still recover hidden attributes from some texts.
