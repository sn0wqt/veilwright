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
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")

    if not api_key and not (credentials_path and project_id):
        console.print(
            "[bold red]Error:[/bold red] Set GEMINI_API_KEY for AI Studio, "
            "or GOOGLE_APPLICATION_CREDENTIALS and GOOGLE_CLOUD_PROJECT for Vertex AI."
        )
        raise typer.Exit(code=1)

    if credentials_path and project_id:
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        client = genai.Client(vertexai=True, project=project_id, location=location)
    else:
        client = genai.Client(api_key=api_key)

    console.print("\n[bold cyan]Available Gemini Models:[/bold cyan]\n")

    # skip non-text models and version-pinned ones that don't work on both
    skip_keywords = {"embedding", "image", "tts", "robotics", "audio", "live", "banana"}
    # version-pinned suffixes only available on Vertex, not AI Studio free tier
    skip_suffixes = ("-002", "-003", "1.5-pro")
    # dated preview/exp names only on Vertex
    import re
    _dated_re = re.compile(r"-(preview|exp)-\d{2}-\d{2}$")

    try:
        for model in client.models.list():
            model_id = model.name or ""
            if "gemini" not in model_id.lower():
                continue
            if any(kw in model_id.lower() for kw in skip_keywords):
                continue
            # strip prefixes: "publishers/google/models/" (Vertex) or "models/" (AI Studio)
            clean_id = model_id.removeprefix("publishers/google/models/").removeprefix("models/")
            
            if any(clean_id.endswith(s) for s in skip_suffixes):
                continue
            if _dated_re.search(clean_id):
                continue
                
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
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show all detected PII entities, not just masked ones.",
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
    original_text = text  # preserve original for iterations

    for iteration in range(1, iterations + 1):
        defender_input = DefenderInput(
            text=original_text,  # always rewrite from original
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
            _pretty_print(result, iteration, iterations, verbose)

        # in a full system the Attacker would evaluate here and provide feedback
        # for now, just simulate iteration without actual feedback


def _pretty_print(result, iteration: int, total: int, verbose: bool = False) -> None:
    """Render a DefenderOutput with rich formatting."""
    console.print()
    console.rule(f"[bold cyan]Defender Output - Iteration {iteration}/{total}[/bold cyan]")

    # syntactic PII
    if result.syntactic_pii_found:
        pii_to_show = result.syntactic_pii_found
        if not verbose:
            pii_to_show = [pii for pii in pii_to_show if "[masked]" in pii]

        if pii_to_show:
            pii_table = Table(
                title="PII Detected by Scanner (NER + Regex)",
                show_header=True,
                header_style="bold magenta",
            )
            pii_table.add_column("#", style="dim", width=4)
            pii_table.add_column("Type", style="cyan")
            pii_table.add_column("Value", style="white")
            pii_table.add_column("Action")
            for i, pii in enumerate(pii_to_show, 1):
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
                    styled_action = "[yellow]detected[/yellow]"
                    styled_value = f"[dim]{pii_value}[/dim]"
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
