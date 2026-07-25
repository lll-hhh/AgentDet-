"""Registration and construction of adaptive attack methods."""

from __future__ import annotations

import importlib
import importlib.util
from hashlib import sha256
from pathlib import Path
import sys
from typing import Any, TypeVar

from .base import AdaptiveAttackMethod


ATTACK_METHODS: dict[str, type[AdaptiveAttackMethod]] = {}
AttackType = TypeVar("AttackType", bound=type[AdaptiveAttackMethod])


def register_attack_method(
    attack_class: AttackType | None = None,
    *,
    name: str | None = None,
):
    """Register an attack class as a decorator or direct function call."""

    def _register(cls: AttackType) -> AttackType:
        attack_name = str(name or getattr(cls, "name", "") or "").strip().lower()
        if not attack_name:
            raise ValueError("Adaptive attack classes must define a non-empty name.")
        ATTACK_METHODS[attack_name] = cls
        return cls

    if attack_class is None:
        return _register
    return _register(attack_class)


def create_attack_method(name: str, **kwargs: Any) -> AdaptiveAttackMethod:
    normalized = str(name or "").strip().lower()
    if normalized not in ATTACK_METHODS:
        raise ValueError(
            f"Unknown adaptive attack method '{name}'. "
            f"Available methods: {sorted(ATTACK_METHODS)}"
        )
    return ATTACK_METHODS[normalized](**kwargs)


def list_attack_methods() -> list[str]:
    return sorted(ATTACK_METHODS)


def load_attack_module(module_reference: str) -> None:
    """Import an attack plugin by dotted module name or Python file path."""

    reference = str(module_reference or "").strip()
    if not reference:
        raise ValueError("Attack module reference cannot be empty.")

    candidate = Path(reference).expanduser()
    if candidate.suffix.lower() == ".py" or candidate.exists():
        path = candidate.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Adaptive attack module does not exist: {path}")
        digest = sha256(str(path).encode("utf-8")).hexdigest()[:12]
        module_name = f"agentdet_adaptive_plugin_{path.stem}_{digest}"
        if module_name in sys.modules:
            return
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load adaptive attack module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return

    importlib.import_module(reference)


def load_attack_modules(module_references: list[str]) -> None:
    for module_reference in module_references:
        load_attack_module(module_reference)


__all__ = [
    "ATTACK_METHODS",
    "create_attack_method",
    "load_attack_module",
    "load_attack_modules",
    "list_attack_methods",
    "register_attack_method",
]
