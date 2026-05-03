# Contributing to DataShield

Thank you for contributing to DataShield! To ensure a high-quality codebase and meet our course requirements, please follow the guidelines below.

## 🚫 Main Branch Protection
* The `main` branch is **protected**.
* **Direct commits to `main` are strictly prohibited**.
* All changes must be submitted via a Pull Request (PR).

## 🔀 Branching Strategy
We use the following naming convention for branches:
* `feat/name-of-feature`: For new features (e.g., `feat/data-anonymizer`).
* `fix/name-of-bug`: For bug fixes (e.g., `fix/crash-on-invalid-input`).
* `test/name-of-test`: For adding or updating tests (e.g., `test/add-unit-tests`).
* `chore/name-of-task`: For maintenance, refactoring, or documentation (e.g., `chore/update-dependencies`).
* `docs/name-of-doc`: For documentation-only changes (e.g., `docs/update-readme`).

## 📝 Pull Request (PR) Process
1.  **Create a Branch**: Always branch off from `main`.
2.  **Commit Changes**: Keep commits small and use descriptive messages.
3.  **Open a PR**: Submit your PR against the `main` branch.
4.  **Code Review**:
    * **Requirement**: Every PR must be reviewed by at least one other group member.
    * **Requirement**: Every student must submit at least one PR and review at least one PR during the project lifecycle.
5.  **Merge**: You may only merge after receiving 1 approval and resolving all conversation threads.

## ✅ Quality Standards
* **Documentation**: All new public classes and functions must have docstrings (one-line summaries minimum).
* **Tests**: New features should include unit tests. Avoid calling real external APIs in tests; use mocks instead.
* **Dependencies**: If you add a new package, update `pyproject.toml` and run `uv lock` to update `uv.lock`.

## 🚀 Development Setup
1.  Clone the repository.
2.  Install uv: `pip install uv` (if not already installed).
3.  Sync dependencies: `uv sync --extra dev` (this will create a virtual environment and install all dependencies).
4.  Run tests: `uv run pytest`.
