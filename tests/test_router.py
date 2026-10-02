"""Basic unit tests for SkillRouter (no API calls required)."""

from __future__ import annotations

import pytest

from jev_skill_router.models import RoutingResult, Skill
from jev_skill_router.router import SkillRouter, _NONE_LABEL


# ─────────────────────────────────────────────────────────────────────────────
# Model tests
# ─────────────────────────────────────────────────────────────────────────────

def test_skill_creation():
    skill = Skill(name="pdf_extractor", description="Extract text from PDFs")
    assert skill.name == "pdf_extractor"
    assert skill.description == "Extract text from PDFs"


def test_routing_result_is_routed():
    result = RoutingResult(
        selected_skill="pdf_extractor",
        confidence=0.92,
        decision="route",
        latency_ms=12.3,
    )
    assert result.is_routed is True


def test_routing_result_not_routed_when_review():
    result = RoutingResult(
        selected_skill="pdf_extractor",
        confidence=0.60,
        decision="review",
        latency_ms=8.0,
    )
    assert result.is_routed is False


def test_routing_result_not_routed_when_none():
    result = RoutingResult(
        selected_skill=None,
        confidence=0.30,
        decision="none",
        latency_ms=9.1,
    )
    assert result.is_routed is False
    assert result.selected_skill is None


def test_routing_result_probabilities_optional():
    result = RoutingResult(
        selected_skill="code_generator",
        confidence=0.85,
        decision="route",
        latency_ms=11.0,
    )
    assert result.probabilities is None


# ─────────────────────────────────────────────────────────────────────────────
# SkillRouter construction tests
# ─────────────────────────────────────────────────────────────────────────────

def test_router_rejects_empty_skills():
    with pytest.raises(ValueError, match="At least one skill"):
        SkillRouter([], api_key="fake-key")


def test_router_rejects_missing_api_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    skills = [Skill(name="pdf_extractor", description="Extract text from PDFs")]
    with pytest.raises(EnvironmentError, match="OPENROUTER_API_KEY"):
        SkillRouter(skills)


def test_router_skills_property():
    skills = [
        Skill(name="pdf_extractor", description="Extract text from PDFs"),
        Skill(name="web_scraper", description="Scrape web pages"),
    ]
    router = SkillRouter(skills, api_key="fake-key-for-test")
    assert len(router.skills) == 2
    assert router.skills[0].name == "pdf_extractor"
    router.close()


def test_router_criteria_includes_none_label():
    skills = [Skill(name="pdf_extractor", description="Extract text from PDFs")]
    router = SkillRouter(skills, api_key="fake-key-for-test")
    assert _NONE_LABEL in router._criteria
    router.close()


def test_router_context_manager():
    skills = [Skill(name="pdf_extractor", description="Extract text from PDFs")]
    with SkillRouter(skills, api_key="fake-key-for-test") as router:
        assert len(router.skills) == 1
