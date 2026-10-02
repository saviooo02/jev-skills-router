# jev-skill-router

```
       _                   __   _ ____                    __
      (_)__ _   __   _____/ /__(_) / /  _________  __  __/ /____  _____
     / / _ \ | / /  / ___/ //_/ / / /  / ___/ __ \/ / / / __/ _ \/ ___/
    / /  __/ |/ /  (__  ) ,< / / / /  / /  / /_/ / /_/ / /_/  __/ /
 __/ /\___/|___/  /____/_/|_/_/_/_/  /_/   \____/\__,_/\__/\___/_/
/___/
              ~ route agent skills in milliseconds, not seconds ~
```

Fast, cheap AI agent skill routing powered by [TypeSafe's Jev](https://typesafe.ai) — a System One decision model that replaces an LLM call for routing with single-digit-millisecond latency at a fraction of the cost.

---

## The Problem

Every AI agent framework eventually faces the same bottleneck: **deciding which skill or tool to invoke** for a given user query. The naive approach is to ask an LLM — but that adds 200–1500 ms of latency and burns tokens on a task that is really just classification.

Jev solves this. It is a purpose-built decision model that answers structured questions (choice, yes/no, score) about text in under 10 ms without the overhead of a general-purpose language model. This package wraps Jev into a clean Python `SkillRouter` class so you can drop it into any agent pipeline.

---

## Installation

```bash
# clone or download the repo, then:
pip install -e ".[dev]"
```

Copy and populate the environment file:

```bash
cp .env.example .env
# edit .env — add TYPESAFE_API_KEY and optionally ANTHROPIC_API_KEY / OPENAI_API_KEY
```

---

## Quick Start

### Python API

```python
from jev_skill_router import Skill, SkillRouter

skills = [
    Skill(name="pdf_extractor",  description="Extract text, tables, and metadata from PDF documents"),
    Skill(name="web_scraper",    description="Scrape and parse content from websites and web pages"),
    Skill(name="code_generator", description="Write, complete, or scaffold code in any programming language"),
    Skill(name="email_composer", description="Draft professional emails, replies, and follow-ups"),
]

with SkillRouter(skills) as router:
    result = router.route("Extract all tables from this quarterly earnings report PDF")

print(result.selected_skill)   # "pdf_extractor"
print(result.confidence)       # e.g. 0.94
print(result.decision)         # "route" | "review" | "none"
print(result.latency_ms)       # e.g. 8.2
```

### CLI

```bash
# Route a single query using the bundled skill catalog
jev-skill-router "Extract tables from this PDF"

# With probability breakdown
jev-skill-router "Write a Python function to parse dates" --show-probs

# Use a custom skills file
jev-skill-router "Schedule a team standup" --skills-file my_skills.json

# Raw JSON output
jev-skill-router "Translate this document to French" --json-output
```

**Example output:**

```
╭─────────────────── Jev Skill Router ───────────────────╮
│  Query          Extract tables from this PDF            │
│  Selected Skill pdf_extractor                           │
│  Decision       ROUTE                                   │
│  Confidence     94.0%                                   │
│  Latency        8.1 ms                                  │
╰────────────────────────────────────────────────────────╯
```

### Confidence thresholds

| Confidence  | Decision | Meaning                                   |
| ----------- | -------- | ----------------------------------------- |
| ≥ 0.75      | `route`  | High confidence — proceed automatically   |
| 0.45 – 0.74 | `review` | Medium confidence — consider human review |
| < 0.45      | `none`   | Low confidence — no skill matched         |

Thresholds are configurable:

```python
router = SkillRouter(skills, high_confidence_threshold=0.80, medium_confidence_threshold=0.50)
```

---

## Running the Evaluation

Compare Jev routing against an LLM baseline across 38 labeled queries:

```bash
# vs OpenRouter free model — Llama 3.1 8B (default, no cost)
python scripts/evaluate.py

# vs Anthropic Claude Haiku
python scripts/evaluate.py --provider anthropic

# vs OpenAI GPT-4o-mini
python scripts/evaluate.py --provider openai

# Quick smoke test with first 10 queries
python scripts/evaluate.py --limit 10
```

The default baseline uses `meta-llama/llama-3.1-8b-instruct:free` via OpenRouter, so the comparison costs nothing beyond your Jev API calls.

Results are saved to `results/evaluation.json`.

### Evaluation Results

> Run `python scripts/evaluate.py` to populate this table with real numbers.

| Metric               | Jev (TypeSafe) | LLM (Claude Haiku) |
| -------------------- | :------------: | :----------------: |
| Accuracy             |       —        |         —          |
| Wrong Skill Rate     |       —        |         —          |
| None Rate            |       —        |         —          |
| Median Latency       |       —        |         —          |
| Est. Cost (38 calls) |       —        |         —          |

---

## Project Structure

```
jev-skill-router/
├── src/jev_skill_router/
│   ├── __init__.py        # Public API
│   ├── models.py          # Skill, RoutingResult (Pydantic)
│   ├── router.py          # SkillRouter — core Jev integration
│   └── cli.py             # rich terminal CLI
├── data/
│   ├── skills.json        # 22 sample skills
│   └── eval_queries.json  # 38 labeled evaluation queries
├── scripts/
│   └── evaluate.py        # Jev vs LLM benchmark runner
├── results/               # Created at runtime by evaluate.py
├── tests/
│   └── test_router.py     # Unit tests (no API calls)
├── .env.example
└── pyproject.toml
```

---

## Running Tests

```bash
pytest
```

Tests cover model construction, confidence-to-decision mapping, and router validation — no API key required.

---

## How It Works

```
                     ┌───────────────────────────────┐
  "Extract tables    │          SkillRouter          │
   from this PDF" ──▶│  skills ──▶ Choice question   │
                     │  + __none__ fallback option   │
                     └───────────────┬───────────────┘
                                     │  system_one(state=query, ...)
                                     ▼
                     ┌───────────────────────────────┐
                     │         TypeSafe Jev          │
                     │  structured prediction, <10ms │
                     └───────────────┬───────────────┘
                                     │  skill + confidence + probs
                                     ▼
                     ┌───────────────────────────────┐
                     │      Threshold policy         │
                     └───┬───────────┬───────────┬───┘
                         │           │           │
                    ≥ 0.75      0.45–0.74     < 0.45
                         ▼           ▼           ▼
                     ┌───────┐  ┌────────┐   ┌───────┐
                     │ ROUTE │  │ REVIEW │   │ NONE  │
                     └───────┘  └────────┘   └───────┘
```

1. **SkillRouter** builds a `Choice` question with one criterion per skill (name → description) plus a `__none__` fallback option.
2. It calls `TypeSafeClient.system_one(state=query, questions={"skill": Choice(...)})`.
3. Jev returns the selected skill name, a confidence score (0–1), and the full probability distribution.
4. A simple threshold policy converts confidence into a routing decision (`route` / `review` / `none`).

The key insight is that Jev treats routing as **structured prediction**, not generation — so it skips sampling, KV-cache management, and token-by-token decoding. The result is ~10× faster and ~100× cheaper than asking an LLM the same question.

---
