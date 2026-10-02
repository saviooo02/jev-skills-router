"""jev-skill-router: Fast AI agent skill routing via TypeSafe's Jev decision model."""

from jev_skill_router.models import RoutingResult, Skill
from jev_skill_router.router import SkillRouter

__version__ = "0.1.0"
__all__ = ["Skill", "RoutingResult", "SkillRouter"]
