IPIGUARD_PLANNER = "ipiguard"
DRIFT_PLANNER = "drift"
DRIFT_FORMAL_PLANNER = "drift_formal"
FIDES_RUNTIME = "fides"
DRIFT_RUNTIME = "drift"
DRIFT_FORMAL_RUNTIME = "drift_formal"
TOOL_FILTER_RUNTIME = "tool_filter"
PIGUARD_RUNTIME = "piguard"
PROTECTAI_RUNTIME = "protectai"
CAMEL_PLANNER = "camel"
CAMEL_RUNTIME = "camel"
RTBAS_PLANNER = "rtbas"
RTBAS_RUNTIME = "rtbas"
AUTOPERMISSIONS_RUNTIME = "autopermissions_hitl"

PLANNER_NAMES = {
    IPIGUARD_PLANNER,
    CAMEL_PLANNER,
    DRIFT_PLANNER,
    DRIFT_FORMAL_PLANNER,
    RTBAS_PLANNER,
}

RUNTIME_NAMES = {
    FIDES_RUNTIME,
    DRIFT_RUNTIME,
    DRIFT_FORMAL_RUNTIME,
    TOOL_FILTER_RUNTIME,
    PIGUARD_RUNTIME,
    PROTECTAI_RUNTIME,
    CAMEL_RUNTIME,
    RTBAS_RUNTIME,
    AUTOPERMISSIONS_RUNTIME,
}

HOOK_LAYERS = ("input", "plan", "reasoning", "action", "finish")


def _normalize_name(name: str) -> str:
    return str(name or "").strip().lower()


def _hook_entry_name(entry: object) -> str:
    if isinstance(entry, str):
        return _normalize_name(entry)
    if isinstance(entry, dict):
        return _normalize_name(entry.get("name", ""))
    return ""


def _contains_hook_name(entries: list[object], name: str) -> bool:
    target = _normalize_name(name)
    if not target:
        return False
    return any(_hook_entry_name(entry) == target for entry in entries)


def _filter_reserved_entries(entries: list[object], reserved_names: set[str]) -> list[object]:
    return [entry for entry in entries if _hook_entry_name(entry) not in reserved_names]


def _planner_name(defense_config: dict) -> str:
    planner = _normalize_name(defense_config.get("planner", ""))
    if planner:
        return planner

    for layer in ("plan", "action", "reasoning", "input", "finish"):
        hooks = list(defense_config.get(layer, []) or [])
        for planner_name in (
            IPIGUARD_PLANNER,
            CAMEL_PLANNER,
            DRIFT_PLANNER,
            DRIFT_FORMAL_PLANNER,
            RTBAS_PLANNER,
        ):
            if _contains_hook_name(hooks, planner_name):
                return planner_name
    return ""


def _runtime_name(defense_config: dict) -> str:
    runtime = _normalize_name(defense_config.get("runtime", ""))
    if runtime:
        return runtime

    for layer in ("reasoning", "action", "plan", "input", "finish"):
        hooks = list(defense_config.get(layer, []) or [])
        for runtime_name in (
            FIDES_RUNTIME,
            DRIFT_RUNTIME,
            DRIFT_FORMAL_RUNTIME,
            TOOL_FILTER_RUNTIME,
            PIGUARD_RUNTIME,
            PROTECTAI_RUNTIME,
            CAMEL_RUNTIME,
            RTBAS_RUNTIME,
            AUTOPERMISSIONS_RUNTIME,
        ):
            if _contains_hook_name(hooks, runtime_name):
                return runtime_name
    return ""


def get_planner_name(defense_config: dict) -> str:
    return _planner_name(defense_config)


def get_runtime_name(defense_config: dict) -> str:
    return _runtime_name(defense_config)


def extract_hook_config(defense_config: dict) -> dict:
    """
    Return a pure hook config (planner defenses removed).

    IPIGuard is treated as a planner, not a hook. This helper keeps only real
    hook defenses while preserving composability with non-IPIGuard hooks.
    """
    runtime = _runtime_name(defense_config)
    reserved_names = set(PLANNER_NAMES) | set(RUNTIME_NAMES)

    input_hooks = _filter_reserved_entries(list(defense_config.get("input", []) or []), reserved_names)
    plan_hooks = _filter_reserved_entries(list(defense_config.get("plan", []) or []), reserved_names)
    reasoning_hooks = _filter_reserved_entries(list(defense_config.get("reasoning", []) or []), reserved_names)
    action_hooks = _filter_reserved_entries(list(defense_config.get("action", []) or []), reserved_names)
    finish_hooks = _filter_reserved_entries(list(defense_config.get("finish", []) or []), reserved_names)

    planner = _planner_name(defense_config)
    if planner:
        plan_hooks = [entry for entry in plan_hooks if _hook_entry_name(entry) != planner]

    if runtime:
        reasoning_hooks = [entry for entry in reasoning_hooks if _hook_entry_name(entry) != runtime]

    return {
        "input": input_hooks,
        "plan": plan_hooks,
        "reasoning": reasoning_hooks,
        "action": action_hooks,
        "finish": finish_hooks,
    }


def collect_defense_config_warnings(defense_config: dict) -> list[str]:
    """
    Return human-readable warnings for likely malformed defense configurations.

    This helps avoid silent no-op behavior when planner/runtime defenses are
    mistakenly placed in hook lists.
    """
    warnings: list[str] = []

    for layer in HOOK_LAYERS:
        hooks = list(defense_config.get(layer, []) or [])
        names = {_hook_entry_name(entry) for entry in hooks}
        names = {name for name in names if name}

        misplaced_planners = sorted(name for name in names if name in PLANNER_NAMES and layer != "plan")
        if misplaced_planners:
            warnings.append(
                f"planner defenses {misplaced_planners} found under '{layer}' hooks; "
                "they will be treated as planner plugins, not hooks."
            )

        misplaced_runtimes = sorted(name for name in names if name in RUNTIME_NAMES and layer != "reasoning")
        if misplaced_runtimes:
            warnings.append(
                f"runtime defenses {misplaced_runtimes} found under '{layer}' hooks; "
                "they will be treated as runtime plugins, not hooks."
            )

    return warnings
