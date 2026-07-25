
from typing import Any

from .base import DefenseHook
from .registry import HookRegistry


from .input.simple import SandwichingHook, InstructionalHook, SpotlightingHook

from .reasoning.reasoning import (
    CaMeLReasoningHook,
)




HOOKS: dict[str, type[DefenseHook]] = {

    "sandwiching": SandwichingHook,
    "instructional": InstructionalHook,
    "spotlighting": SpotlightingHook,


    "camel_reasoning": CaMeLReasoningHook,
}

PLANNER_ALIASES: dict[str, str] = {
    "ipiguard": "ipiguard",
    "gardener": "gardener",
    "camel": "camel",
    "drift": "drift",
    "rtbas": "rtbas",
}

RUNTIME_ALIASES: dict[str, str] = {
    "fides": "fides",
    "drift": "drift",
    "tool_filter": "tool_filter",
    "piguard": "piguard",
    "pi_guard": "piguard",
    "pi-guard": "piguard",
    "protectai": "protectai",
    "camel": "camel",
    "rtbas": "rtbas",
    "autopermissions": "autopermissions_hitl",
    "autopermissions_hitl": "autopermissions_hitl",
    "auto_permissions": "autopermissions_hitl",
}


def create_hook(name: str, **config) -> DefenseHook:
    name_lower = name.lower().strip()

    if name_lower not in HOOKS:
        raise ValueError(
            f"Unknown hook: {name}. "
            f"Available hooks: {list(HOOKS.keys())}"
        )

    hook_class = HOOKS[name_lower]
    return hook_class(**config)


def create_hook_registry(config: dict[str, Any], defense_client: dict[str, Any] | None = None) -> HookRegistry:
    registry = HookRegistry()
    dc = defense_client or {}

    for layer in ["input", "plan", "reasoning", "action", "finish"]:
        hooks_config = config.get(layer, [])

        for hook_config in hooks_config:
            if isinstance(hook_config, str):

                try:
                    hook_class = HOOKS.get(hook_config.lower().strip())
                    extra = {}
                    if hook_class is not None and getattr(hook_class, "requires_model", False):
                        extra = {k: v for k, v in dc.items() if v is not None}
                    hook = create_hook(hook_config, **extra)
                    registry.register(hook)
                except (ValueError, TypeError) as e:
                    print(f"Warning: skipping invalid hook '{hook_config}': {e}")
            elif isinstance(hook_config, dict):

                raw_config = dict(hook_config)
                name = raw_config.pop("name", "")
                if not name:
                    print(f"Warning: skipping hook config without a name: {hook_config}")
                    continue
                try:
                    hook_class = HOOKS.get(name.lower().strip())
                    if hook_class is not None and getattr(hook_class, "requires_model", False):
                        for key in ("api_key", "base_url", "model"):
                            if key not in raw_config and dc.get(key) is not None:
                                raw_config[key] = dc[key]
                    hook = create_hook(name, **raw_config)
                    registry.register(hook)
                except (ValueError, TypeError) as e:
                    print(f"Warning: skipping invalid hook '{name}': {e}")

    return registry


def parse_defense_string(defense_str: str) -> dict[str, Any]:
    if not defense_str:
        return {
            "planner": "",
            "runtime": "",
            "input": [],
            "plan": [],
            "reasoning": [],
            "action": [],
            "finish": [],
        }

    defense_names = [d.strip() for d in defense_str.split("+") if d.strip()]

    config: dict[str, Any] = {
        "planner": "",
        "runtime": "",
        "input": [],
        "plan": [],
        "reasoning": [],
        "action": [],
        "finish": [],
    }

    for raw_name in defense_names:
        name = raw_name.lower()
        if name in {"camel", "drift", "rtbas"}:
            if name == "camel":
                planner_name = "camel"
                runtime_name = "camel"
                config["reasoning"].append("camel_reasoning")
            elif name == "drift":
                planner_name = "drift"
                runtime_name = "drift"
            else:
                planner_name = "rtbas"
                runtime_name = "rtbas"
            if config["planner"] and config["planner"] != planner_name:
                raise ValueError(
                    f"Multiple planner defenses requested: '{config['planner']}' and "
                    f"'{planner_name}'. Specify at most one planner."
                )
            if config["runtime"] and config["runtime"] != runtime_name:
                raise ValueError(
                    f"Multiple runtime defenses requested: '{config['runtime']}' and "
                    f"'{runtime_name}'. Specify at most one runtime defense."
                )
            config["planner"] = planner_name
            config["runtime"] = runtime_name
            continue

        planner_name = PLANNER_ALIASES.get(name, "")
        if planner_name:
            if config["planner"] and config["planner"] != planner_name:
                raise ValueError(
                    f"Multiple planner defenses requested: '{config['planner']}' and "
                    f"'{planner_name}'. Specify at most one planner."
                )
            config["planner"] = planner_name
            continue

        runtime_name = RUNTIME_ALIASES.get(name, "")
        if runtime_name:
            if config["runtime"] and config["runtime"] != runtime_name:
                raise ValueError(
                    f"Multiple runtime defenses requested: '{config['runtime']}' and "
                    f"'{runtime_name}'. Specify at most one runtime defense."
                )
            config["runtime"] = runtime_name
            continue

        if name not in HOOKS:
            planner_names = sorted(set(PLANNER_ALIASES.values()))
            runtime_names = sorted(set(RUNTIME_ALIASES.values()))
            hook_names = sorted(HOOKS.keys())
            raise ValueError(
                f"Unknown defense: '{raw_name}'. "
                f"Available planners: {planner_names}; "
                f"available runtimes: {runtime_names}; "
                f"available hooks: {hook_names}."
            )

        hook_class = HOOKS[name]
        layer = hook_class.layer
        config[layer].append(name)

    return config


def list_available_hooks() -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {
        "input": [],
        "plan": [],
        "reasoning": [],
        "action": [],
        "finish": [],
    }

    for name, hook_class in HOOKS.items():
        info = {
            "name": name,
            "layer": hook_class.layer,
            "is_blocking": hook_class.is_blocking,
            "requires_model": hook_class.requires_model,
        }
        result[hook_class.layer].append(info)

    return result
