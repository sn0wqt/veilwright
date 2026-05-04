"""CLI entry point for the Defender agent."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from defender.defender import Defender, DefenderError
from defender.types import DefenderInput

load_dotenv()

app = typer.Typer(
    name="defender",
    help="Defender agent — semantic text anonymization for adversarial PII protection.",
    add_completion=False,
)
console = Console()


@app.command("models")
def list_models() -> None:
    """List available Gemini models that can be used with --model."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        console.print("[bold red]Error:[/bold red] GEMINI_API_KEY not set in .env")
        raise typer.Exit(code=1)

    client = genai.Client(api_key=api_key)

    console.print("\n[bold cyan]Available Gemini Models:[/bold cyan]\n")

    # skip non-text models
    skip_keywords = {"embedding", "image", "tts", "robotics", "audio", "live", "banana"}

    try:
        for model in client.models.list():
            model_id = model.name or ""
            if "gemini" not in model_id.lower():
                continue
            if any(kw in model_id.lower() for kw in skip_keywords):
                continue
            clean_id = model_id.removeprefix("models/")
            console.print(f"  [green]{clean_id}[/green]")
    except Exception as exc:
        console.print(f"[bold red]API Error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    console.print()


@app.command("anonymize")
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
    """Run the Defender agent on the given text and target attributes."""

    # resolve input text
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

    # parse attributes
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

        # in a full system the Attacker would evaluate here;
        # for now just feed the previous rewrite into the next iteration
        text = result.rewritten_text


def _pretty_print(result, iteration: int, total: int) -> None:
    """Render a DefenderOutput with rich formatting."""
    console.print()
    console.rule(f"[bold cyan]Defender Output - Iteration {iteration}/{total}[/bold cyan]")

    # syntactic PII
    if result.syntactic_pii_found:
        pii_table = Table(
            title="PII Detected by Scanner (NER + Regex)",
            show_header=True,
            header_style="bold magenta",
        )
        pii_table.add_column("#", style="dim", width=4)
        pii_table.add_column("Type", style="cyan")
        pii_table.add_column("Value", style="white")
        pii_table.add_column("Action")
        for i, pii in enumerate(result.syntactic_pii_found, 1):
            # format: "TYPE: value [masked]" or "TYPE: value [detected]"
            bracket_idx = pii.rfind(" [")
            if bracket_idx != -1:
                main_part = pii[:bracket_idx]
                action = pii[bracket_idx + 2:-1]  # "masked" or "detected"
            else:
                main_part = pii
                action = "?"
            parts = main_part.split(": ", 1)
            pii_type = parts[0] if len(parts) == 2 else "PII"
            pii_value = parts[1] if len(parts) == 2 else main_part
            # color-code: masked = bold red, detected = dim yellow
            if action == "masked":
                styled_action = "[bold red]masked[/bold red]"
                styled_value = f"[red]{pii_value}[/red]"
            else:
                styled_action = "[dim yellow]detected[/dim yellow]"
                styled_value = f"[yellow]{pii_value}[/yellow]"
            pii_table.add_row(str(i), pii_type, styled_value, styled_action)
        console.print(pii_table)
        console.print()

    # strategies
    strategy_table = Table(
        title="Rewrite Strategies",
        show_header=True,
        header_style="bold magenta",
    )
    strategy_table.add_column("Attribute", style="cyan", min_width=15)
    strategy_table.add_column("Strategy", style="green", min_width=12)
    strategy_table.add_column("Reasoning", style="white", max_width=80)
    for s in result.strategies_used:
        strategy_table.add_row(s.attribute, s.strategy, s.reasoning)
    console.print(strategy_table)
    console.print()

    # rewritten text
    console.print(
        Panel(
            result.rewritten_text,
            title="[bold green]Rewritten Text[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )

    # confidence
    pct = result.confidence * 100
    color = "green" if pct >= 80 else "yellow" if pct >= 50 else "red"
    console.print(f"\n[bold]Confidence:[/bold] [{color}]{pct:.0f}%[/{color}]")
    console.print()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
