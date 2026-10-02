from __future__ import annotations

import os
import time
from typing import Sequence

from dotenv import load_dotenv
from typesafe_sdk import Choice, TypeSafeClient

from jev_skill_router.models import RoutingResult, Skill

load_dotenv()

# Confidence thresholds for routing decisions
_HIGH_CONFIDENCE = 0.75
_MEDIUM_CONFIDENCE = 0.45

_NONE_LABEL = "__none__"
_NONE_DESCRIPTION = "None of the available skills match this query"

_OPENROUTER_BASE_URL = "https://openrouter.ai/api"
_OPENROUTER_JEV_MODEL = "~typesafe/jev-latest"


def _resolve_client(api_key: str | None, model: str | None) -> TypeSafeClient:
    """Resolve API key and base URL, preferring OpenRouter over direct TypeSafe."""
    # Explicit key passed in → use it as-is (caller sets base_url via env if needed)
    if api_key:
        return TypeSafeClient(api_key=api_key, model=model)

    # OPENROUTER_API_KEY → route Jev through openrouter.ai
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        return TypeSafeClient(
            api_key=openrouter_key,
            base_url=_OPENROUTER_BASE_URL,
            model=model or _OPENROUTER_JEV_MODEL,
        )

    # TYPESAFE_API_KEY → direct TypeSafe endpoint
    typesafe_key = os.environ.get("TYPESAFE_API_KEY")
    if typesafe_key:
        return TypeSafeClient(api_key=typesafe_key, model=model)

    raise EnvironmentError(
        "No API key found. Set OPENROUTER_API_KEY (recommended) or TYPESAFE_API_KEY, "
        "or pass api_key= to SkillRouter()."
    )


class SkillRouter:
    """Routes a natural-language query to the best-matching skill using Jev.

    Jev is TypeSafe's System One decision model — a fast structured-decision
    engine available via OpenRouter (OPENROUTER_API_KEY) or TypeSafe directly
    (TYPESAFE_API_KEY). It returns a typed choice with confidence scores rather
    than generating text, making it ideal as a cheap pre-filter for AI agents.
    """

    def __init__(
        self,
        skills: Sequence[Skill],
        *,
        api_key: str | None = None,
        model: str | None = None,
        high_confidence_threshold: float = _HIGH_CONFIDENCE,
        medium_confidence_threshold: float = _MEDIUM_CONFIDENCE,
    ) -> None:
        if not skills:
            raise ValueError("At least one skill is required.")

        self._skills = list(skills)
        self._high = high_confidence_threshold
        self._medium = medium_confidence_threshold

        self._client = _resolve_client(api_key, model)

        # Build Choice criteria once: {skill_name: description, "__none__": ...}
        self._criteria: dict[str, str] = {
            skill.name: skill.description for skill in self._skills
        }
        self._criteria[_NONE_LABEL] = _NONE_DESCRIPTION

    def route(self, query: str) -> RoutingResult:
        """Route a query to the best-matching skill.

        Args:
            query: The user's natural-language request.

        Returns:
            A RoutingResult with the selected skill, confidence, decision, and
            the full probability distribution over all skills.
        """
        t0 = time.perf_counter()

        response = self._client.system_one(
            state=query,
            questions={
                "skill": Choice(
                    instructions=(
                        "Which skill best handles this request? "
                        "Choose '__none__' if none are a good match."
                    ),
                    criteria=self._criteria,
                )
            },
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        answer = response.choices["skill"]
        chosen = answer.choice
        confidence = answer.confidence
        probabilities = dict(answer.probabilities)

        # Treat a "none" selection the same as low confidence
        if chosen == _NONE_LABEL:
            selected_skill = None
            decision = "none"
        else:
            selected_skill = chosen
            if confidence >= self._high:
                decision = "route"
            elif confidence >= self._medium:
                decision = "review"
            else:
                decision = "none"
                selected_skill = None

        # Clean up the internal none label from the returned probabilities
        probabilities.pop(_NONE_LABEL, None)

        return RoutingResult(
            selected_skill=selected_skill,
            confidence=confidence,
            decision=decision,
            latency_ms=round(latency_ms, 2),
            probabilities=probabilities,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SkillRouter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def skills(self) -> list[Skill]:
        return list(self._skills)
