"""
CLI entry point for the Defender agent.

Usage::

    python -m defender \\
      --text "I remember watching the moon landing..." \\
      --attributes "Age" "Birth Year" "Exact Event"

    python -m defender \\
      --file input.txt \\
      --attributes "Age" "Birth Year" "Exact Event"
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from defender.defender import Defender, DefenderError
from defender.types import DefenderInput

app = typer.Typer(
    name="defender",
    help="Defender agent — semantic text anonymization for adversarial PII protection.",
    add_completion=False,
)
console = Console()


@app.command("models")
def list_models() -> None:
    """List available Gemini models that can be used with --model."""
    import os
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        console.print("[bold red]Error:[/bold red] GEMINI_API_KEY not set in .env")
        raise typer.Exit(code=1)

    client = genai.Client(api_key=api_key)

    console.print("\n[bold cyan]Available Gemini Models:[/bold cyan]\n")

    # Filter out non-text models (image, embedding, tts, robotics, etc.)
    skip_keywords = {"embedding", "image", "tts", "robotics", "audio", "live", "banana"}

    try:
        for model in client.models.list():
            model_id = model.name or ""
            if "gemini" not in model_id.lower():
                continue
            if any(kw in model_id.lower() for kw in skip_keywords):
                continue
            # Strip the "models/" prefix
            clean_id = model_id.removeprefix("models/")
            console.print(f"  [green]{clean_id}[/green]")
    except Exception as exc:
        console.print(f"[bold red]API Error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    console.print()


@app.command("run")
def defend(
    text: str | None = typer.Option(
        None,
        "--text",
        "-t",
        help="The sensitive text to anonymize (inline). Use --file for longer texts.",
    ),
    file: Path | None = typer.Option(
        None,
        "--file",
        "-f",
        help="Path to a text file containing the sensitive text.",
        exists=True,
        readable=True,
    ),
    attributes: str = typer.Option(
        ...,
        "--attributes",
        "-a",
        help='Comma-separated target attributes to hide (e.g. "Age,Birth Year,Exact Event").',
    ),
    iterations: int = typer.Option(
        1,
        "--iterations",
        "-n",
        help="Number of defender iterations to run.",
        min=1,
    ),
    model: str = typer.Option(
        "gemini-2.5-flash",
        "--model",
        "-m",
        help="Gemini model to use (e.g. gemini-2.5-flash, gemini-2.5-pro).",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        help="Output raw JSON instead of pretty-printed results.",
    ),
) -> None:
    """
    Run the Defender agent on the given text and target attributes.

    Provide input via --text (inline) or --file (from file). Exactly one is required.
    """
    # --- Resolve input text ---
    if text and file:
        console.print(
            "[bold red]Error:[/bold red] Provide either --text or --file, not both."
        )
        raise typer.Exit(code=1)
    if not text and not file:
        console.print("[bold red]Error:[/bold red] Provide either --text or --file.")
        raise typer.Exit(code=1)

    if file:
        try:
            text = file.read_text(encoding="utf-8")
        except Exception as exc:
            console.print(f"[bold red]Error reading file:[/bold red] {exc}")
            raise typer.Exit(code=1)

    # --- Parse attributes ---
    attr_list = [a.strip() for a in attributes.split(",") if a.strip()]
    if not attr_list:
        console.print("[bold red]Error:[/bold red] No attributes provided.")
        raise typer.Exit(code=1)

    try:
        defender = Defender(model=model)
    except DefenderError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    attacker_feedback: str | None = None

    for iteration in range(1, iterations + 1):
        defender_input = DefenderInput(
            text=text,
            target_attributes=attr_list,
            iteration=iteration,
            attacker_feedback=attacker_feedback,
        )

        try:
            result = defender.run(defender_input)
        except DefenderError as exc:
            console.print(f"[bold red]Defender error:[/bold red] {exc}")
            raise typer.Exit(code=1)

        if output_json:
            console.print_json(json.dumps(result.to_dict(), indent=2))
        else:
            _pretty_print(result, iteration, iterations)

        # In a full system, the Attacker would evaluate here and provide
        # feedback; for now, use the previous rewrite as input for the
        # next iteration.
        text = result.rewritten_text


def _pretty_print(result, iteration: int, total_iterations: int) -> None:
    """Render a DefenderOutput with rich formatting."""
    console.print()
    console.rule(
        f"[bold cyan]Defender Output — Iteration {iteration}/{total_iterations}[/bold cyan]"
    )

    # --- Syntactic PII ---
    if result.syntactic_pii_found:
        pii_table = Table(
            title="🔍 Syntactic PII Detected (pre-masked before LLM)",
            show_header=True,
            header_style="bold magenta",
        )
        pii_table.add_column("#", style="dim", width=4)
        pii_table.add_column("Type", style="cyan")
        pii_table.add_column("Value", style="red")
        for i, pii in enumerate(result.syntactic_pii_found, 1):
            parts = pii.split(": ", 1)
            pii_type = parts[0] if len(parts) == 2 else "PII"
            pii_value = parts[1] if len(parts) == 2 else pii
            pii_table.add_row(str(i), pii_type, pii_value)
        console.print(pii_table)
        console.print()

    # --- Strategies ---
    strat_table = Table(
        title="🛡️  Rewrite Strategies",
        show_header=True,
        header_style="bold magenta",
    )
    strat_table.add_column("Attribute", style="cyan", min_width=15)
    strat_table.add_column("Strategy", style="green", min_width=12)
    strat_table.add_column("Reasoning", style="white", max_width=80)
    for s in result.strategies_used:
        strat_table.add_row(s.attribute, s.strategy, s.reasoning)
    console.print(strat_table)
    console.print()

    # --- Rewritten text ---
    console.print(
        Panel(
            result.rewritten_text,
            title="[bold green]Rewritten Text[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )

    # --- Confidence ---
    confidence_pct = result.confidence * 100
    color = (
        "green" if confidence_pct >= 80 else "yellow" if confidence_pct >= 50 else "red"
    )
    console.print(
        f"\n[bold]Confidence:[/bold] [{color}]{confidence_pct:.0f}%[/{color}]"
    )
    console.print()


def main() -> None:
    """Entry point for the ``defender`` console script."""
    app()


if __name__ == "__main__":
    main()
