# Contributing to Defender

Thank you for contributing to the Defender agent! To ensure a high-quality codebase and meet our course requirements, please follow the guidelines below.

## Main Branch Protection

- The `main` branch is **protected**.
- **Direct commits to `main` are strictly prohibited**.
- All changes must be submitted via a Pull Request (PR).

## Branching Strategy

We use the following naming convention for branches:

- `feat/name-of-feature` — for new features (e.g., `feat/add-shifting-strategy`)
- `fix/name-of-bug` — for bug fixes (e.g., `fix/json-parsing-edge-case`)
- `test/name-of-test` — for adding or updating tests (e.g., `test/scanner-phone-validation`)
- `chore/name-of-task` — for maintenance or refactoring (e.g., `chore/update-dependencies`)
- `docs/name-of-doc` — for documentation-only changes (e.g., `docs/update-readme`)

## Pull Request (PR) Process

1. **Create a branch** — always branch off from `main`.
2. **Commit changes** — keep commits small with descriptive messages.
3. **Open a PR** — submit your PR against the `main` branch.
4. **Code review**:
   - Every PR must be reviewed by at least one other group member.
   - Every student must submit at least one PR and review at least one PR.
5. **Merge** — only merge after receiving 1 approval and resolving all conversations.

## Quality Standards

- **Documentation** — all new public classes and functions must have docstrings.
- **Tests** — new features should include unit tests. Use mocks instead of real API calls.
- **Dependencies** — if you add a new package, update `pyproject.toml` and run `uv lock`.

## Development Setup

1. Clone the repository.
2. Install uv: `pip install uv`
3. Sync dependencies: `uv sync --extra dev`
4. Create a `.env` file with your `GEMINI_API_KEY` (see README).
5. Run tests: `uv run pytest`
