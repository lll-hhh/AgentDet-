"""Extensible adaptive attack support for AgentDET."""

from .base import (
    AdaptiveAttackContext,
    AdaptiveAttackMethod,
    AttackCandidate,
    AttackFeedback,
    InjectionTask,
    InjectionSite,
    build_attack_context,
)
from .registry import (
    create_attack_method,
    list_attack_methods,
    load_attack_module,
    load_attack_modules,
    register_attack_method,
)
from . import methods as _methods  # noqa: F401

__all__ = [
    "AdaptiveAttackContext",
    "AdaptiveAttackMethod",
    "AttackCandidate",
    "AttackFeedback",
    "InjectionTask",
    "InjectionSite",
    "build_attack_context",
    "create_attack_method",
    "list_attack_methods",
    "load_attack_module",
    "load_attack_modules",
    "register_attack_method",
]
