from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from jev_skill_router.models import Skill
from jev_skill_router.router import SkillRouter

console = Console()

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_DEFAULT_SKILLS_FILE = _DATA_DIR / "skills.json"


def _load_skills(skills_file: Path) -> list[Skill]:
    if not skills_file.exists():
        console.print(f"[red]Skills file not found:[/red] {skills_file}")
        sys.exit(1)
    with open(skills_file) as f:
        raw = json.load(f)
    return [Skill(**s) for s in raw]


def _decision_color(decision: str) -> str:
    return {"route": "green", "review": "yellow", "none": "red"}.get(decision, "white")


def _decision_icon(decision: str) -> str:
    return {"route": "[bold green]ROUTE[/]", "review": "[bold yellow]REVIEW[/]", "none": "[bold red]NONE[/]"}.get(decision, decision)


@click.command()
@click.argument("query")
@click.option(
    "--skills-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to skills JSON file (default: data/skills.json)",
)
@click.option("--show-probs", is_flag=True, default=False, help="Show full probability table")
@click.option("--json-output", is_flag=True, default=False, help="Output raw JSON result")
def main(query: str, skills_file: Path | None, show_probs: bool, json_output: bool) -> None:
    """Route QUERY to the best-matching skill using TypeSafe's Jev.

    Example:

        jev-skill-router "Extract tables from this PDF"
    """
    resolved_file = skills_file or _DEFAULT_SKILLS_FILE
    skills = _load_skills(resolved_file)

    try:
        with SkillRouter(skills) as router:
            result = router.route(query)
    except EnvironmentError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Routing failed:[/red] {e}")
        sys.exit(1)

    if json_output:
        click.echo(result.model_dump_json(indent=2))
        return

    # ── Summary panel ────────────────────────────────────────────────────────
    decision_display = _decision_icon(result.decision)
    skill_display = (
        f"[bold cyan]{result.selected_skill}[/]"
        if result.selected_skill
        else "[dim]— no match —[/dim]"
    )

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold dim", justify="right")
    summary.add_column()
    summary.add_row("Query", f"[italic]{query}[/italic]")
    summary.add_row("Selected Skill", skill_display)
    summary.add_row("Decision", decision_display)
    summary.add_row("Confidence", f"{result.confidence:.1%}")
    summary.add_row("Latency", f"{result.latency_ms:.1f} ms")

    color = _decision_color(result.decision)
    console.print()
    console.print(
        Panel(summary, title="[bold]Jev Skill Router[/bold]", border_style=color, expand=False)
    )

    # ── Optional probability table ────────────────────────────────────────────
    if show_probs and result.probabilities:
        sorted_probs = sorted(result.probabilities.items(), key=lambda x: x[1], reverse=True)[:10]
        prob_table = Table(
            "Skill", "Probability",
            title="Top probabilities",
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold dim",
        )
        for name, prob in sorted_probs:
            bar = "█" * int(prob * 20)
            prob_table.add_row(name, f"{prob:.1%}  {bar}")
        console.print(prob_table)

    console.print()
