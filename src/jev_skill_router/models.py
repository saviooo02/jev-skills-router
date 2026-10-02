from __future__ import annotations

from pydantic import BaseModel, Field


class Skill(BaseModel):
    name: str = Field(description="Unique identifier for the skill")
    description: str = Field(description="What this skill does, used as routing criteria")

    def __hash__(self) -> int:
        return hash(self.name)


class RoutingResult(BaseModel):
    selected_skill: str | None = Field(
        description="Name of the matched skill, or None when no match found"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Jev confidence score (0-1)"
    )
    decision: str = Field(
        description="'route' (high confidence), 'review' (medium), or 'none' (low)"
    )
    latency_ms: float = Field(description="Round-trip time to Jev API in milliseconds")
    probabilities: dict[str, float] | None = Field(
        default=None,
        description="Per-skill probability distribution from Jev"
    )

    @property
    def is_routed(self) -> bool:
        return self.decision == "route"
