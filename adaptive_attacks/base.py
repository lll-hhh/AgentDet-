"""Core contracts for benchmark-native adaptive attacks."""

from __future__ import annotations

from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from core.message_types import Case


JsonPathPart = str | int


@dataclass(frozen=True)
class InjectionSite:
    """One mutable untrusted text field associated with an injection point."""

    site_id: str
    location: str
    mechanism: str
    target: str
    field_path: tuple[JsonPathPart, ...]
    safe_text: str
    unsafe_text: str
    baseline_unsafe_text: str = ""

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "site_id": self.site_id,
            "location": self.location,
            "mechanism": self.mechanism,
            "target": self.target,
            "field_path": list(self.field_path),
        }
        if include_text:
            payload["safe_text"] = self.safe_text
            payload["unsafe_text"] = self.unsafe_text
        return payload


@dataclass(frozen=True)
class AttackFeedback:
    """Coarse execution feedback exposed to a bounded gray-box attacker."""

    attack_successful: bool = False
    attack_success_total: bool = False
    task_completed: bool = False
    defense_triggered: bool = False
    blocked: bool = False
    blocked_by: str = ""
    confirmation_count: int = 0
    skipped_leaf_count: int = 0
    target_tool_called: bool = False
    runtime_error: bool = False
    notes: tuple[str, ...] = ()
    visible_trace: dict[str, Any] = field(default_factory=dict)

    def score(self) -> float:
        """Rank candidates without exposing internal defense reasoning."""

        if self.runtime_error:
            return -1000.0
        score = 100.0 if self.attack_successful else 0.0
        if self.target_tool_called:
            score += 20.0
        if self.task_completed:
            score += 2.0
        score -= float(self.confirmation_count) * 3.0
        score -= float(self.skipped_leaf_count) * 2.0
        if self.blocked:
            score -= 10.0
        return score

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack_successful": self.attack_successful,
            "attack_success_total": self.attack_success_total,
            "task_completed": self.task_completed,
            "defense_triggered": self.defense_triggered,
            "blocked": self.blocked,
            "blocked_by": self.blocked_by,
            "confirmation_count": self.confirmation_count,
            "skipped_leaf_count": self.skipped_leaf_count,
            "target_tool_called": self.target_tool_called,
            "runtime_error": self.runtime_error,
            "notes": list(self.notes),
            "candidate_score": self.score(),
            "visible_trace": deepcopy(self.visible_trace),
        }


@dataclass(frozen=True)
class AttackCandidate:
    """A method-produced replacement set for declared injection sites."""

    replacements: dict[str, str]
    strategy: str = ""
    raw_response: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "replacements": dict(self.replacements),
            "strategy": self.strategy,
            "raw_response": self.raw_response,
            "usage": dict(self.usage),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class InjectionTask:
    """Attacker objective extracted from one unsafe AgentDET case."""

    goal: str
    goal_type: str = ""
    injection_points: tuple[dict[str, Any], ...] = ()
    attacker_states: tuple[dict[str, Any], ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_case(cls, case: Case) -> "InjectionTask":
        attack_info = deepcopy(dict(case.attack_info or {}))
        points = attack_info.get("injection_points", [])
        states = attack_info.get("attacker_states", [])
        return cls(
            goal=str(attack_info.get("attacker_intent", "") or ""),
            goal_type=str(attack_info.get("goal_type", "") or ""),
            injection_points=tuple(
                deepcopy(point) for point in points if isinstance(point, dict)
            ),
            attacker_states=tuple(
                deepcopy(state) for state in states if isinstance(state, dict)
            ),
            raw=attack_info,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "goal_type": self.goal_type,
            "injection_points": [deepcopy(point) for point in self.injection_points],
            "attacker_states": [deepcopy(state) for state in self.attacker_states],
        }


@dataclass(frozen=True)
class AdaptiveAttackContext:
    """AgentDojo-style context supplied to one adaptive attack method.

    The target pipeline is intentionally exposed as the live executor object. Attack
    implementations may inspect it for white-box experiments, but ordinary attacks
    should only use its public attributes and the coarse feedback provided here.
    """

    user_task: Case
    injection_task: InjectionTask
    target_pipeline: Any
    tool_definitions: dict[str, dict[str, Any]]
    injection_candidates: tuple[InjectionSite, ...]
    ground_truth: tuple[dict[str, Any], ...]
    defense_profile: dict[str, Any] = field(default_factory=dict)
    feedback: AttackFeedback | None = None
    iteration: int = 0

    @property
    def case(self) -> Case:
        """Compatibility alias for existing AgentDET attack methods."""

        return self.user_task

    @property
    def injection_sites(self) -> list[InjectionSite]:
        """Return mutable injection fields visible on the declared task path."""

        return list(self.injection_candidates)

    @property
    def available_tool_names(self) -> tuple[str, ...]:
        return tuple(self.tool_definitions)

    def get_injection_candidates(self) -> list[InjectionSite]:
        """Mirror AgentDojo's injection-candidate helper."""

        return list(self.injection_candidates)

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        """Serialize public attack context without serializing the live pipeline."""

        return {
            "user_task": {
                "id": self.user_task.id,
                "domain": self.user_task.domain,
                "scenario": self.user_task.scenario,
                "user_instruction": self.user_task.user_instruction,
                "tools": list(self.user_task.tools),
            },
            "injection_task": self.injection_task.to_dict(),
            "target_pipeline": {
                "type": type(self.target_pipeline).__name__,
                "name": str(
                    getattr(self.target_pipeline, "pipeline_name", "")
                    or getattr(self.target_pipeline, "victim_model", "")
                    or getattr(self.target_pipeline, "name", "")
                ),
            },
            "tool_definitions": deepcopy(self.tool_definitions),
            "injection_candidates": [
                site.to_dict(include_text=include_text)
                for site in self.injection_candidates
            ],
            "ground_truth": [deepcopy(step) for step in self.ground_truth],
            "defense_profile": deepcopy(self.defense_profile),
            "feedback": self.feedback.to_dict() if self.feedback is not None else None,
            "iteration": self.iteration,
        }


def build_attack_context(
    *,
    case: Case,
    target_pipeline: Any,
    tool_definitions: dict[str, dict[str, Any]],
    injection_sites: list[InjectionSite],
    defense_profile: dict[str, Any] | None = None,
    feedback: AttackFeedback | None = None,
    iteration: int = 0,
) -> AdaptiveAttackContext:
    """Build the stable public context passed to registered attack methods."""

    return AdaptiveAttackContext(
        user_task=case,
        injection_task=InjectionTask.from_case(case),
        target_pipeline=target_pipeline,
        tool_definitions=deepcopy(tool_definitions),
        injection_candidates=tuple(injection_sites),
        ground_truth=tuple(deepcopy(step) for step in (case.ground_truth or [])),
        defense_profile=deepcopy(dict(defense_profile or {})),
        feedback=feedback,
        iteration=int(iteration),
    )


class AdaptiveAttackMethod(ABC):
    """Extension point for adaptive payload generation methods.

    New plugins should override :meth:`attack`. The legacy :meth:`generate`
    signature remains available so existing AgentDET experiments keep working.
    """

    name: str

    def attack(self, context: AdaptiveAttackContext) -> AttackCandidate:
        """Generate one candidate from the complete benchmark context."""

        return self.generate(
            case=context.user_task,
            injection_sites=context.injection_sites,
            defense_profile=context.defense_profile,
            feedback=context.feedback,
            iteration=context.iteration,
        )

    def generate(
        self,
        *,
        case: Case,
        injection_sites: list[InjectionSite],
        defense_profile: dict[str, Any],
        feedback: AttackFeedback | None,
        iteration: int,
    ) -> AttackCandidate:
        """Legacy generation hook retained for existing attack implementations."""

        raise NotImplementedError(
            f"{type(self).__name__} must override attack(context) or generate(...)."
        )


__all__ = [
    "AdaptiveAttackMethod",
    "AdaptiveAttackContext",
    "AttackCandidate",
    "AttackFeedback",
    "InjectionTask",
    "InjectionSite",
    "JsonPathPart",
    "build_attack_context",
]
