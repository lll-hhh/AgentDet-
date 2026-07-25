from __future__ import annotations

import uuid
from typing import Any

from core.world.environment import WorldState


def world_of(tool: Any) -> WorldState:
    return getattr(tool, "world", WorldState.get_instance())


def get_root(tool: Any, name: str, default: Any = None) -> Any:
    world = world_of(tool)
    if isinstance(getattr(world, "extra_state", None), dict) and name in world.extra_state:
        return world.extra_state[name]
    snapshot = world.to_dict()
    if name in snapshot:
        return snapshot[name]
    return default


def set_root(tool: Any, name: str, value: Any) -> Any:
    world = world_of(tool)
    if not isinstance(getattr(world, "extra_state", None), dict):
        world.extra_state = {}
    world.extra_state[name] = value
    return value


def ensure_list_root(tool: Any, name: str) -> list[Any]:
    value = get_root(tool, name)
    if isinstance(value, list):
        if name not in getattr(world_of(tool), "extra_state", {}):
            set_root(tool, name, value)
        return value
    return set_root(tool, name, [])


def ensure_object_root(tool: Any, name: str) -> dict[str, Any]:
    value = get_root(tool, name)
    if isinstance(value, dict):
        if name not in getattr(world_of(tool), "extra_state", {}):
            set_root(tool, name, value)
        return value
    return set_root(tool, name, {})


def find_one(records: list[dict[str, Any]], **filters: Any) -> dict[str, Any] | None:
    for item in records:
        if not isinstance(item, dict):
            continue
        matched = True
        for key, value in filters.items():
            if value is None:
                continue
            if item.get(key) != value:
                matched = False
                break
        if matched:
            return item
    return None


def find_many(records: list[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        matched = True
        for key, value in filters.items():
            if value is None:
                continue
            if item.get(key) != value:
                matched = False
                break
        if matched:
            out.append(item)
    return out


def text_result(payload: dict[str, Any] | None, *, label: str) -> dict[str, Any]:
    if not payload:
        return {"status": "error", "message": f"{label} not found."}
    return {"status": "success", **payload}


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def match_query(item: dict[str, Any], query: str, fields: list[str]) -> bool:
    needle = str(query or "").strip().lower()
    if not needle:
        return False
    for field in fields:
        value = item.get(field)
        if value is None:
            continue
        if needle in str(value).lower():
            return True
    return False


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def lookup_by_arg(
    tool: Any,
    *,
    root: str,
    arg_value: Any,
    id_field: str,
    label: str,
) -> dict[str, Any]:
    if arg_value is None or arg_value == "":
        return {"status": "error", "message": f"Missing required argument: {id_field}"}
    item = find_one(ensure_list_root(tool, root), **{id_field: arg_value})
    if not item:
        return {"status": "error", "message": f"{label} not found."}
    return {"status": "success", **item}


def list_by_arg(
    tool: Any,
    *,
    root: str,
    arg_value: Any,
    id_field: str,
    container_key: str,
    label: str,
) -> dict[str, Any]:
    if arg_value is None or arg_value == "":
        return {"status": "error", "message": f"Missing required argument: {id_field}"}
    items = find_many(ensure_list_root(tool, root), **{id_field: arg_value})
    if not items:
        return {"status": "error", "message": f"{label} not found."}
    return {"status": "success", id_field: arg_value, container_key: items, "count": len(items)}
