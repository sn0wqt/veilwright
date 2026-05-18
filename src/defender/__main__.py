"""CLI entry point for the Defender agent."""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from defender.defender import Defender, DefenderError
from defender.types import DefenderInput, DefenderOutput

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

    skip_keywords = {"embedding", "image", "tts", "robotics", "audio", "live", "banana"}
    skip_suffixes = ("-002", "-003", "1.5-pro")
    _dated_re = re.compile(r"-(preview|exp)-\d{2}-\d{2}$")

    try:
        for model in client.models.list():
            model_id = model.name or ""
            if "gemini" not in model_id.lower():
                continue
            if any(kw in model_id.lower() for kw in skip_keywords):
                continue
            clean_id = (
                model_id
                .removeprefix("publishers/google/models/")
                .removeprefix("models/")
            )
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
        Defender.DEFAULT_MODEL,
        "--model",
        "-m",
        help="Gemini model to use (e.g. gemini-2.5-flash, gemini-2.5-pro).",
    ),
    output_json: bool = typer.Option(
        False,
        "--json",
        help="Output raw JSON instead of pretty-printed results.",
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write results to a .txt file. Each iteration is separated by two blank lines.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show all detected PII entities, not just masked ones.",
    ),
) -> None:
    """Run the Defender agent on the given text and target attributes."""

    # --- resolve input ---
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

    attr_list = [a.strip() for a in attributes.split(",") if a.strip()]
    if not attr_list:
        console.print("[bold red]Error:[/bold red] No attributes provided.")
        raise typer.Exit(code=1)

    try:
        defender = Defender(model=model)
    except DefenderError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    # ground truth is extracted once on iteration 1 then carried forward so
    # every iteration in the output shows the same consistent values
    ground_truth: dict[str, str] = {}
    attacker_feedback: str | None = None
    output_blocks: list[str] = []

    for iteration in range(1, iterations + 1):
        defender_input = DefenderInput(
            text=text,
            target_attributes=attr_list,
            iteration=iteration,
            attacker_feedback=attacker_feedback,
            ground_truth=ground_truth,
        )

        try:
            result = defender.run(defender_input)
        except DefenderError as exc:
            console.print(f"[bold red]Defender error:[/bold red] {exc}")
            raise typer.Exit(code=1)

        # lock in ground truth after first attempt (even if it failed and returned {})
        if iteration == 1:
            ground_truth = result.ground_truth

        if output_json:
            console.print_json(json.dumps(result.to_dict(), indent=2))
        else:
            _pretty_print(result, iteration, iterations, verbose)

        if output_file:
            output_blocks.append(_format_txt_block(result, iteration, iterations))

        # in the full system the Attacker feeds back here via attacker_feedback;
        # for solo CLI testing iterations run without external feedback

    if output_file and output_blocks:
        try:
            output_file.write_text("\n\n".join(output_blocks), encoding="utf-8")
            console.print(f"[dim]Results saved to [bold]{output_file}[/bold][/dim]")
        except Exception as exc:
            console.print(f"[bold red]Error writing output file:[/bold red] {exc}")
            raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _format_txt_block(result: DefenderOutput, iteration: int, total: int) -> str:
    """Format one iteration as a plain-text block for the output file."""
    lines: list[str] = []

    lines.append(f"=== Iteration {iteration}/{total} ===")
    lines.append("")

    # ground truth — present on every iteration since it's carried forward
    if result.ground_truth:
        lines.append("Ground Truth:")
        for attr, val in result.ground_truth.items():
            lines.append(f"  {attr}: {val}")
        lines.append("")

    # clue map — only populated on iteration 1 (pre-pass doesn't re-run on retries)
    if result.clue_map:
        lines.append("Identified Clues:")
        for attr, clues in result.clue_map.items():
            if clues:
                lines.append(f"  {attr}:")
                for clue in clues:
                    lines.append(
                        f"    - \"{clue.get('clue', '?')}\" ({clue.get('type', '?')}): {clue.get('inference', '?')}"
                    )
        lines.append("")

    lines.append("Strategies Used:")
    for s in result.strategies_used:
        lines.append(f"  [{s.strategy.upper()}] {s.attribute}: {s.reasoning}")
    lines.append("")

    lines.append("Rewritten Text:")
    lines.append(result.rewritten_text)
    lines.append("")

    pct = result.confidence * 100
    lines.append(f"Confidence: {pct:.0f}%")

    masked = [p for p in result.syntactic_pii_found if "[masked]" in p]
    if masked:
        lines.append("")
        lines.append("Masked PII:")
        for p in masked:
            lines.append(f"  {p}")

    return "\n".join(lines)


def _pretty_print(
    result: DefenderOutput, iteration: int, total: int, verbose: bool = False
) -> None:
    """Render a DefenderOutput with rich formatting."""
    console.print()
    console.rule(f"[bold cyan]Defender Output — Iteration {iteration}/{total}[/bold cyan]")

    # --- ground truth ---
    if result.ground_truth:
        gt_table = Table(
            title="Ground Truth (LLM-inferred from original text)",
            show_header=True,
            header_style="bold blue",
        )
        gt_table.add_column("Attribute", style="cyan", min_width=20)
        gt_table.add_column("Value", style="white")
        for attr, val in result.ground_truth.items():
            gt_table.add_row(attr, str(val))
        console.print(gt_table)
        console.print()

    # --- clue map (iteration 1 only) ---
    if result.clue_map:
        clue_table = Table(
            title="Identified Inference Clues (pre-pass)",
            show_header=True,
            header_style="bold yellow",
        )
        clue_table.add_column("Attribute", style="cyan", min_width=15)
        clue_table.add_column("Type", style="yellow", min_width=10)
        clue_table.add_column("Clue", style="white", min_width=25)
        clue_table.add_column("Inference", style="dim", max_width=60)
        for attr, clues in result.clue_map.items():
            for clue in clues:
                clue_table.add_row(
                    attr,
                    clue.get("type", ""),
                    clue.get("clue", ""),
                    clue.get("inference", ""),
                )
        console.print(clue_table)
        console.print()

    # --- syntactic PII ---
    if result.syntactic_pii_found:
        pii_to_show = result.syntactic_pii_found
        if not verbose:
            pii_to_show = [p for p in pii_to_show if "[masked]" in p]

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
                bracket_idx = pii.rfind(" [")
                if bracket_idx != -1:
                    main_part = pii[:bracket_idx]
                    action = pii[bracket_idx + 2:-1]
                else:
                    main_part = pii
                    action = "?"
                parts = main_part.split(": ", 1)
                pii_type = parts[0] if len(parts) == 2 else "PII"
                pii_value = parts[1] if len(parts) == 2 else main_part
                if action == "masked":
                    styled_action = "[bold red]masked[/bold red]"
                    styled_value = f"[red]{pii_value}[/red]"
                else:
                    styled_action = "[yellow]detected[/yellow]"
                    styled_value = f"[dim]{pii_value}[/dim]"
                pii_table.add_row(str(i), pii_type, styled_value, styled_action)
            console.print(pii_table)
            console.print()

    # --- strategies ---
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

    # --- rewritten text ---
    console.print(
        Panel(
            result.rewritten_text,
            title="[bold green]Rewritten Text[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )

    # --- confidence ---
    pct = result.confidence * 100
    color = "green" if pct >= 80 else "yellow" if pct >= 50 else "red"
    console.print(f"\n[bold]Confidence:[/bold] [{color}]{pct:.0f}%[/{color}]")
    console.print()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
