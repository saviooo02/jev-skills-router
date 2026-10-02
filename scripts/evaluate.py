"""Evaluation script: Jev SkillRouter vs LLM baseline router.

Usage:
    python scripts/evaluate.py                          # OpenRouter free model (default)
    python scripts/evaluate.py --provider anthropic
    python scripts/evaluate.py --provider openai
    python scripts/evaluate.py --limit 10               # quick smoke test
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich import box

load_dotenv()

# Allow running as `python scripts/evaluate.py` from the project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jev_skill_router.models import Skill
from jev_skill_router.router import SkillRouter

console = Console()

_ROOT = Path(__file__).parent.parent
_SKILLS_FILE = _ROOT / "data" / "skills.json"
_QUERIES_FILE = _ROOT / "data" / "eval_queries.json"
_RESULTS_DIR = _ROOT / "results"

# Cost estimates (USD per 1k tokens, rough industry figures)
_LLM_COST_PER_1K_INPUT = {"anthropic": 0.00025, "openai": 0.00015, "openrouter": 0.0}
_JEV_COST_PER_CALL = 0.000010
_LLM_AVG_TOKENS_PER_CALL = 400

# OpenRouter model for the LLM baseline
_OPENROUTER_MODEL = "~anthropic/claude-haiku-latest"
# Seconds to wait between LLM calls — avoids upstream 429s on shared-pool models
_LLM_CALL_DELAY_S = 0.5
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class QueryResult:
    id: str
    query: str
    gold_skill: str | None
    query_type: str

    jev_skill: str | None
    jev_decision: str
    jev_confidence: float
    jev_latency_ms: float
    jev_correct: bool

    llm_skill: str | None
    llm_latency_ms: float
    llm_correct: bool


@dataclass
class RouterMetrics:
    name: str
    accuracy: float
    wrong_skill_rate: float
    none_rate: float
    median_latency_ms: float
    estimated_cost_usd: float
    n: int


# ─────────────────────────────────────────────────────────────────────────────
# LLM baseline router
# ─────────────────────────────────────────────────────────────────────────────


def _build_skill_list(skills: list[Skill]) -> str:
    return "\n".join(f"- {s.name}: {s.description}" for s in skills)


def _parse_llm_skill(text: str, valid_names: set[str]) -> str | None:
    """Extract skill name from LLM response; return None if none found."""
    text = text.strip().lower()
    if "none" in text or "no match" in text or "no skill" in text:
        return None
    for name in valid_names:
        if name.lower() in text:
            return name
    # Last-resort: first word that matches
    for word in text.split():
        clean = word.strip(".,\"'`")
        if clean in valid_names:
            return clean
    return None


def llm_route_anthropic(query: str, skills: list[Skill]) -> tuple[str | None, float]:
    import anthropic  # local import to avoid hard requirement when using openai

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic(api_key=api_key)
    skill_list = _build_skill_list(skills)

    t0 = time.perf_counter()
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        system=(
            "You are a skill router. Given a user query and a list of skills, "
            "respond with ONLY the skill name that best matches. "
            "If no skill matches well, respond with 'none'.\n\n"
            f"Available skills:\n{skill_list}"
        ),
        messages=[{"role": "user", "content": query}],
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    raw = message.content[0].text
    valid_names = {s.name for s in skills}
    return _parse_llm_skill(raw, valid_names), latency_ms


def llm_route_openai(query: str, skills: list[Skill]) -> tuple[str | None, float]:
    import openai  # local import

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = openai.OpenAI(api_key=api_key)
    skill_list = _build_skill_list(skills)

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=64,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a skill router. Given a user query and a list of skills, "
                    "respond with ONLY the skill name that best matches. "
                    "If no skill matches well, respond with 'none'.\n\n"
                    f"Available skills:\n{skill_list}"
                ),
            },
            {"role": "user", "content": query},
        ],
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    raw = response.choices[0].message.content or ""
    valid_names = {s.name for s in skills}
    return _parse_llm_skill(raw, valid_names), latency_ms


def llm_route_openrouter(query: str, skills: list[Skill]) -> tuple[str | None, float]:
    """Route via a free OpenRouter model (OpenAI-compatible API)."""
    import openai  # local import — openai library works with any OpenAI-compatible endpoint

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")

    client = openai.OpenAI(
        api_key=api_key,
        base_url=_OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/savioshajioff/jev-skill-router",
            "X-Title": "jev-skill-router-eval",
        },
    )
    skill_list = _build_skill_list(skills)

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=_OPENROUTER_MODEL,
        max_tokens=32,  # skill names are short; cap tokens to save quota
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a skill router. Given a user query and a list of skills, "
                    "respond with ONLY the exact skill name that best matches. "
                    "If no skill matches, respond with 'none'. No explanation.\n\n"
                    f"Available skills:\n{skill_list}"
                ),
            },
            {"role": "user", "content": query},
        ],
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    raw = response.choices[0].message.content or ""
    valid_names = {s.name for s in skills}
    return _parse_llm_skill(raw, valid_names), latency_ms


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def _is_correct(predicted: str | None, gold: str | None) -> bool:
    """A prediction is correct when it exactly matches the gold label (None == None)."""
    return predicted == gold


def _compute_metrics(
    results: list[QueryResult],
    name: Literal["jev", "llm"],
    provider: str,
    n_queries: int,
) -> RouterMetrics:
    skills_field = f"{name}_skill"
    correct_field = f"{name}_correct"
    latency_field = f"{name}_latency_ms"

    corrects = [getattr(r, correct_field) for r in results]
    latencies = [getattr(r, latency_field) for r in results]
    predicted_skills = [getattr(r, skills_field) for r in results]
    golds = [r.gold_skill for r in results]

    accuracy = sum(corrects) / len(corrects)

    # Wrong skill = predicted a skill but it was the wrong one (not "no skill when there should be none")
    wrong_skill = sum(
        1 for p, g in zip(predicted_skills, golds) if p is not None and p != g
    )
    wrong_skill_rate = wrong_skill / len(results)

    none_rate = sum(1 for p in predicted_skills if p is None) / len(results)

    median_latency = statistics.median(latencies)

    if name == "jev":
        cost = n_queries * _JEV_COST_PER_CALL
    else:
        cost_per_k = _LLM_COST_PER_1K_INPUT.get(provider, 0.0002)
        cost = (
            n_queries
            * (_LLM_AVG_TOKENS_PER_1K := _LLM_AVG_TOKENS_PER_CALL / 1000)
            * cost_per_k
        )

    return RouterMetrics(
        name=name.upper(),
        accuracy=accuracy,
        wrong_skill_rate=wrong_skill_rate,
        none_rate=none_rate,
        median_latency_ms=median_latency,
        estimated_cost_usd=cost,
        n=len(results),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────


def _print_summary(jev: RouterMetrics, llm: RouterMetrics, provider: str) -> None:
    llm_label = (
        f"LLM (OpenRouter · claude-haiku)"
        if provider == "openrouter"
        else f"LLM ({provider})"
    )
    table = Table(
        title="Jev vs LLM Baseline — Routing Accuracy",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
    )
    table.add_column("Metric", style="bold dim")
    table.add_column("Jev (TypeSafe)", justify="right")
    table.add_column(llm_label, justify="right")

    def pct(v: float) -> str:
        return f"{v:.1%}"

    def ms(v: float) -> str:
        return f"{v:.0f} ms"

    def cost(v: float) -> str:
        return "FREE" if v == 0.0 else f"${v:.5f}"

    def diff(a: float, b: float, higher_is_better: bool = True) -> str:
        delta = a - b
        if abs(delta) < 0.001:
            return ""
        sign = "+" if delta > 0 else ""
        color = "green" if (delta > 0) == higher_is_better else "red"
        return f" [dim {color}]({sign}{delta:.1%})[/]"

    acc_diff = diff(jev.accuracy, llm.accuracy)
    wrong_diff = diff(
        jev.wrong_skill_rate, llm.wrong_skill_rate, higher_is_better=False
    )
    lat_diff_raw = jev.median_latency_ms - llm.median_latency_ms
    lat_color = "green" if lat_diff_raw < 0 else "red"
    lat_sign = "+" if lat_diff_raw > 0 else ""
    lat_diff = (
        f" [dim {lat_color}]({lat_sign}{lat_diff_raw:.0f} ms)[/]"
        if abs(lat_diff_raw) > 1
        else ""
    )

    table.add_row("Accuracy", f"{pct(jev.accuracy)}{acc_diff}", pct(llm.accuracy))
    table.add_row(
        "Wrong Skill Rate",
        f"{pct(jev.wrong_skill_rate)}{wrong_diff}",
        pct(llm.wrong_skill_rate),
    )
    table.add_row("None Rate", pct(jev.none_rate), pct(llm.none_rate))
    table.add_row(
        "Median Latency",
        f"{ms(jev.median_latency_ms)}{lat_diff}",
        ms(llm.median_latency_ms),
    )
    table.add_row(
        "Est. Cost (n queries)",
        cost(jev.estimated_cost_usd),
        cost(llm.estimated_cost_usd),
    )
    table.add_row("Queries Evaluated", str(jev.n), str(llm.n))

    console.print()
    console.print(table)
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


@click.command()
@click.option(
    "--provider",
    type=click.Choice(["openrouter", "anthropic", "openai"]),
    default="openrouter",
    show_default=True,
    help="LLM provider for the baseline router",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Limit evaluation to the first N queries (useful for quick checks)",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Path for the detailed JSON results file (default: results/evaluation.json)",
)
def main(provider: str, limit: int | None, output: Path | None) -> None:
    """Compare Jev SkillRouter against an LLM baseline on the eval dataset."""

    # Load data
    with open(_SKILLS_FILE) as f:
        skills = [Skill(**s) for s in json.load(f)]
    with open(_QUERIES_FILE) as f:
        queries = json.load(f)

    if limit:
        queries = queries[:limit]

    n = len(queries)
    console.print(
        f"\n[bold]Evaluating {n} queries[/bold] | Jev vs [cyan]{provider}[/cyan] baseline\n"
    )

    llm_fn = {
        "openrouter": llm_route_openrouter,
        "anthropic": llm_route_anthropic,
        "openai": llm_route_openai,
    }[provider]

    # Preflight: check Jev key (OpenRouter preferred, TypeSafe direct as fallback)
    jev_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
    if not jev_key:
        console.print(
            "[red]No Jev key found. Set OPENROUTER_API_KEY (recommended) or TYPESAFE_API_KEY — aborting.[/red]"
        )
        sys.exit(1)
    jev_via = (
        "OpenRouter" if os.environ.get("OPENROUTER_API_KEY") else "TypeSafe direct"
    )
    console.print(f"  [dim]Jev via:[/dim] [cyan]{jev_via}[/cyan]")
    llm_key_name = {
        "openrouter": "OPENROUTER_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }[provider]
    if not os.environ.get(llm_key_name):
        console.print(f"[red]{llm_key_name} not set — aborting.[/red]")
        sys.exit(1)

    if provider == "openrouter":
        console.print(f"  [dim]LLM model:[/dim] [cyan]{_OPENROUTER_MODEL}[/cyan]")

    results: list[QueryResult] = []

    with SkillRouter(skills) as router:
        for i, q in enumerate(queries, 1):
            qid = q["id"]
            query_text = q["query"]
            gold = q["gold_skill"]
            qtype = q["type"]

            console.print(
                (
                    f"  [{i:02d}/{n}] [dim]{qid}[/dim] {query_text[:60]}…"
                    if len(query_text) > 60
                    else f"  [{i:02d}/{n}] [dim]{qid}[/dim] {query_text}"
                ),
                end=" ",
            )

            # Jev
            try:
                jev_result = router.route(query_text)
                jev_skill = jev_result.selected_skill
                jev_decision = jev_result.decision
                jev_conf = jev_result.confidence
                jev_lat = jev_result.latency_ms
            except Exception as e:
                console.print(f"[red]Jev error: {e}[/red]")
                jev_skill, jev_decision, jev_conf, jev_lat = None, "error", 0.0, 0.0

            # LLM
            try:
                llm_skill, llm_lat = llm_fn(query_text, skills)
            except Exception as e:
                console.print(f"[red]LLM error: {e}[/red]")
                llm_skill, llm_lat = None, 0.0
            time.sleep(_LLM_CALL_DELAY_S)

            jev_correct = _is_correct(jev_skill, gold)
            llm_correct = _is_correct(llm_skill, gold)

            jev_icon = "[green]✓[/]" if jev_correct else "[red]✗[/]"
            llm_icon = "[green]✓[/]" if llm_correct else "[red]✗[/]"
            console.print(
                f"  Jev:{jev_icon}[dim]{jev_lat:.0f}ms[/dim]  LLM:{llm_icon}[dim]{llm_lat:.0f}ms[/dim]"
            )

            results.append(
                QueryResult(
                    id=qid,
                    query=query_text,
                    gold_skill=gold,
                    query_type=qtype,
                    jev_skill=jev_skill,
                    jev_decision=jev_decision,
                    jev_confidence=jev_conf,
                    jev_latency_ms=jev_lat,
                    jev_correct=jev_correct,
                    llm_skill=llm_skill,
                    llm_latency_ms=llm_lat,
                    llm_correct=llm_correct,
                )
            )

    # Metrics
    jev_metrics = _compute_metrics(results, "jev", provider, n)
    llm_metrics = _compute_metrics(results, "llm", provider, n)

    _print_summary(jev_metrics, llm_metrics, provider)

    # Save results
    _RESULTS_DIR.mkdir(exist_ok=True)
    out_path = output or _RESULTS_DIR / "evaluation.json"
    payload = {
        "metadata": {
            "n_queries": n,
            "llm_provider": provider,
            "skills_file": str(_SKILLS_FILE),
            "queries_file": str(_QUERIES_FILE),
        },
        "summary": {
            "jev": asdict(jev_metrics),
            "llm": asdict(llm_metrics),
        },
        "results": [asdict(r) for r in results],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    console.print(f"[dim]Full results saved to[/dim] [bold]{out_path}[/bold]\n")


if __name__ == "__main__":
    main()
