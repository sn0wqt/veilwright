"""
CLI entry point for the Defender agent.

Usage::

    python -m defender \\
      --text "I remember watching the moon landing..." \\
      --attributes "Age" "Birth Year" "Exact Event" \\
      --iterations 1
"""

from __future__ import annotations

import json

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


@app.command()
def defend(
    text: str = typer.Option(
        ...,
        "--text", "-t",
        help="The original sensitive text to anonymize.",
    ),
    attributes: list[str] = typer.Option(
        ...,
        "--attributes", "-a",
        help="Target attributes to hide (repeat for multiple).",
    ),
    iterations: int = typer.Option(
        1,
        "--iterations", "-n",
        help="Number of defender iterations to run.",
        min=1,
    ),
    model: str = typer.Option(
        "gemini-2.5-flash",
        "--model", "-m",
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
    """
    try:
        defender = Defender(model=model)
    except DefenderError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    attacker_feedback: str | None = None

    for iteration in range(1, iterations + 1):
        defender_input = DefenderInput(
            text=text,
            target_attributes=attributes,
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
    console.print(Panel(
        result.rewritten_text,
        title="[bold green]Rewritten Text[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    # --- Confidence ---
    confidence_pct = result.confidence * 100
    color = "green" if confidence_pct >= 80 else "yellow" if confidence_pct >= 50 else "red"
    console.print(
        f"\n[bold]Confidence:[/bold] [{color}]{confidence_pct:.0f}%[/{color}]"
    )
    console.print()


def main() -> None:
    """Entry point for the ``defender`` console script."""
    app()


if __name__ == "__main__":
    main()
