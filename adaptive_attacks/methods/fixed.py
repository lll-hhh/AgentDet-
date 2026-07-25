"""Static payload baseline exposed through the adaptive attack interface."""

from __future__ import annotations

from typing import Any

from ..base import (
    AdaptiveAttackMethod,
    AttackCandidate,
    AttackFeedback,
    InjectionSite,
)
from ..registry import register_attack_method

from core.message_types import Case


@register_attack_method
class FixedPayloadAttack(AdaptiveAttackMethod):
    name = "fixed"

    def generate(
        self,
        *,
        case: Case,
        injection_sites: list[InjectionSite],
        defense_profile: dict[str, Any],
        feedback: AttackFeedback | None,
        iteration: int,
    ) -> AttackCandidate:
        del case, defense_profile, feedback
        return AttackCandidate(
            replacements={site.site_id: site.unsafe_text for site in injection_sites},
            strategy="reuse original static injection payload",
            metadata={"iteration": iteration},
        )


__all__ = ["FixedPayloadAttack"]
