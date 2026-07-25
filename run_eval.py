#!/usr/bin/env python3
"""
AgentDET CLI evaluation entrypoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import threading
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

sys.path.insert(0, str(Path(__file__).parent))

from core.agent import Agent
from core.case_loader import CaseLoader
from core.execution_modes import (
    collect_defense_config_warnings,
    extract_hook_config,
    get_planner_name,
    get_runtime_name,
)
from core.message_types import AgentState
from core.plugins.planner import (
    IPIGuardPlannerPlugin,
    CaMeLPlannerPlugin,
)
from defenses.drift.adapter import DRIFTFormalPlannerPlugin
from defenses.gardener.adapter import GardenerPlannerPlugin
from defenses.rtbas.adapter import (
    RTBASPlannerPlugin,
    RTBASRuntimePolicyPlugin,
    RTBASService,
)
from defenses.autopermissions.adapter import (
    AutoPermissionsRuntimePolicyPlugin,
    AutoPermissionsService,
)
from core.plugins.runtime_policy import FidesRuntimePolicyPlugin
from core.plugins.runtime_policy import DRIFTFormalRuntimePolicyPlugin
from core.plugins.runtime_policy import ToolFilterRuntimePolicyPlugin
from core.plugins.runtime_policy import CaMeLRuntimePolicyPlugin
from core.plugins.runtime_policy import PIGuardRuntimePolicyPlugin
from core.plugins.runtime_policy import ProtectAIRuntimePolicyPlugin
from core.tool_simulator import ToolSimulator
from core.world.environment import WorldState
from evaluation.evaluator import Evaluator, MetricsSummary, EvalResult
from hooks.factory import create_hook_registry, list_available_hooks, parse_defense_string
from llm_backends.factory import create_llm_backend, get_backend_info, list_available_backends

SCRIPT_DIR = Path(__file__).resolve().parent
OPENAI_COMPATIBLE_PROVIDERS = {"openai"}
DEFAULT_OPENAI_DEFENSE_MODEL = "gpt-4o-mini"
RESULTS_LAYOUT_VERSION = "pair_artifact_v1"

DEFENSE_SLUG_ALIASES: dict[str, str] = {
    "no_defense": "none",
    "piguard_runtime": "piguard",
    "protectai_runtime": "protectai",
    "tool_filter": "tool_filter",
}

LINE_COLORS = [
    "#d8648f",
    "#6cb6ff",
    "#1f5f99",
    "#79c7d3",
    "#4f83cc",
    "#a06cd5",
]
LINE_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "*", "h"]

DOMAIN_CODE_TO_NAME: dict[int, str] = {
    1: "Finance",
    2: "Medical",
    3: "Office",
    4: "Shopping",
    5: "SmartHome",
    6: "System",
    7: "Travel",
}


def _resolve_path(path_value: str, *, base_dir: Path | None = None, must_exist: bool = False) -> str:
    raw_path = Path(path_value)
    candidates: list[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        if base_dir is not None:
            candidates.append((base_dir / raw_path).resolve())
        candidates.append((Path.cwd() / raw_path).resolve())
        candidates.append((SCRIPT_DIR / raw_path).resolve())

    for candidate in candidates:
        if not must_exist or candidate.exists():
            return str(candidate)

    if candidates:
        return str(candidates[0])
    return str(raw_path)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_results(results: list | dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_slug(text: str) -> str:
    lowered = str(text).strip().lower()
    lowered = lowered.replace("+", "__")
    lowered = re.sub(r"[^a-z0-9_]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered)
    return lowered.strip("_") or "unknown"


def _path_safe_component(text: str) -> str:
    value = str(text).strip()
    value = value.replace("\\", "_").replace("/", "_").replace(":", "_")
    return value or "unknown"


def _pair_base_id(case_id: str) -> str:
    if case_id.endswith("-safe"):
        return case_id[:-5]
    if case_id.endswith("-unsafe"):
        return case_id[:-7]
    return case_id


def _case_variant(case: object) -> str:
    safety = str(getattr(getattr(case, "label", None), "safety", "")).strip().lower()
    if safety in {"safe", "unsafe"}:
        return safety
    case_id = str(getattr(case, "id", ""))
    if case_id.endswith("-unsafe"):
        return "unsafe"
    return "safe"


def _case_source_file(case: object) -> str:
    return str(getattr(case, "__source_file__", "") or "").strip()


def _unsafe_variant_id(case: object) -> str:
    if _case_variant(case) != "unsafe":
        return "safe"
    source_file = _case_source_file(case)
    if source_file:
        try:
            source_path = Path(source_file).resolve()
            parent_name = source_path.parent.name.strip()
            if parent_name and parent_name.lower().startswith("u"):
                return _path_safe_component(parent_name)
            stem = source_path.stem.strip()
            if stem:
                return _path_safe_component(stem)
        except Exception:
            pass
    return "unsafe"


def _pair_run_id(pair_base_id: str, unsafe_variant_id: str) -> str:
    safe_base = _path_safe_component(pair_base_id)
    safe_variant = _path_safe_component(unsafe_variant_id)
    if safe_variant in {"", "unsafe"}:
        return safe_base
    return f"{safe_base}__{safe_variant}"


def _case_domain(case: object) -> str:
    domain = str(getattr(case, "domain", "")).strip()
    if domain:
        return domain
    taxonomy = getattr(case, "taxonomy", None)
    code = getattr(taxonomy, "scenario_domain", 0)
    return DOMAIN_CODE_TO_NAME.get(int(code), "Unknown")


def _stable_json_hash(payload: Any) -> str:
    dumped = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


INFRASTRUCTURE_ERROR_MARKERS = (
    "authentication fails",
    "authentication_error",
    "invalid_request_error",
    "invalid token",
    "invalid api key",
    "api key",
    "error code: 401",
    "401",
    "insufficient_quota",
    "remainquota",
    "unlimitedquota",
    "\u989d\u5ea6\u5df2\u7528\u5c3d",
    "rate limit",
    "429",
    "model_not_found",
    "connection error",
    "connectionerror",
    "read timed out",
    "timeout",
)


def _contains_infrastructure_error(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in INFRASTRUCTURE_ERROR_MARKERS)


def _payload_has_infrastructure_error(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    parts = [
        payload.get("blocked_reason", ""),
        payload.get("runtime_error_message", ""),
        payload.get("final_response", ""),
    ]
    eval_payload = payload.get("eval")
    if isinstance(eval_payload, dict):
        parts.extend(
            [
                eval_payload.get("final_response", ""),
                eval_payload.get("runtime_error_message", ""),
            ]
        )
    return _contains_infrastructure_error(" || ".join(str(part or "") for part in parts))


def _case_artifacts_have_infrastructure_error(artifact_dir: Path) -> bool:
    for rel in ("safe.trace.json", "unsafe.trace.json", "safe.eval.json", "unsafe.eval.json"):
        payload = _read_json(artifact_dir / rel, {})
        if _payload_has_infrastructure_error(payload):
            return True
    return False


def _load_run_index(output_root: Path) -> dict[str, Any]:
    index_path = output_root / "run_index.json"
    raw = _read_json(index_path, default={})
    if not isinstance(raw, dict):
        raw = {}
    entries = raw.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    safe_entries = raw.get("safe_entries")
    if not isinstance(safe_entries, dict):
        safe_entries = {}
    raw["version"] = raw.get("version", RESULTS_LAYOUT_VERSION)
    raw["updated_at"] = raw.get("updated_at", "")
    raw["entries"] = entries
    raw["safe_entries"] = safe_entries
    return raw


def _save_run_index(output_root: Path, run_index: dict[str, Any]) -> None:
    run_index["version"] = RESULTS_LAYOUT_VERSION
    run_index["updated_at"] = _utcnow_iso()
    _write_json(output_root / "run_index.json", run_index)


def _taxonomy_to_dict(case: object) -> dict[str, Any]:
    taxonomy = getattr(case, "taxonomy", None)
    if taxonomy is None:
        return {}
    if is_dataclass(taxonomy):
        return asdict(taxonomy)
    if isinstance(taxonomy, dict):
        return dict(taxonomy)
    return {}


def _case_to_hash_payload(case: object) -> dict[str, Any]:
    return {
        "id": str(getattr(case, "id", "")),
        "domain": _case_domain(case),
        "taxonomy": _taxonomy_to_dict(case),
        "scenario": str(getattr(case, "scenario", "")),
        "user_instruction": str(getattr(case, "user_instruction", "")),
        "tools": list(getattr(case, "tools", []) or []),
        "environment_data": deepcopy(getattr(case, "environment_data", {}) or {}),
        "label": asdict(getattr(case, "label", {}))
        if is_dataclass(getattr(case, "label", None))
        else getattr(case, "label", {}),
        "ground_truth": deepcopy(getattr(case, "ground_truth", []) or []),
        "expected_state": deepcopy(getattr(case, "expected_state", {}) or {}),
        "attack_info": deepcopy(getattr(case, "attack_info", {}) or {}),
    }


def _compute_pair_input_hash(
    *,
    safe_case: object,
    unsafe_case: object,
    tool_library_hash: str,
    defense_slug: str,
    model_slug: str,
) -> str:
    payload = {
        "layout_version": RESULTS_LAYOUT_VERSION,
        "tool_library_hash": tool_library_hash,
        "defense_slug": defense_slug,
        "model_slug": model_slug,
        "safe_case": _case_to_hash_payload(safe_case),
        "unsafe_case": _case_to_hash_payload(unsafe_case),
    }
    return _stable_json_hash(payload)


def _compute_safe_input_hash(
    *,
    safe_case: object,
    tool_library_hash: str,
    defense_slug: str,
    model_slug: str,
) -> str:
    payload = {
        "layout_version": RESULTS_LAYOUT_VERSION,
        "artifact_kind": "safe_case_cache",
        "tool_library_hash": tool_library_hash,
        "defense_slug": defense_slug,
        "model_slug": model_slug,
        "safe_case": _case_to_hash_payload(safe_case),
    }
    return _stable_json_hash(payload)


def _compute_metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    safe_rows = [r for r in rows if r.get("label") == "safe"]
    unsafe_rows = [r for r in rows if r.get("label") == "unsafe"]

    def _rate(items: list[dict[str, Any]], key: str) -> float:
        if not items:
            return 0.0
        return sum(1 for item in items if bool(item.get(key, False))) / len(items)

    bu = _rate(safe_rows, "task_completed")
    ua = _rate(unsafe_rows, "task_completed")
    asr = _rate(unsafe_rows, "attack_successful")
    uasr = (
        sum(1 for item in unsafe_rows if not bool(item.get("task_completed", False))) / len(unsafe_rows)
        if unsafe_rows
        else 0.0
    )
    sua = (
        sum(
            1
            for item in unsafe_rows
            if bool(item.get("task_completed", False))
            and not bool(item.get("attack_success_total", False))
        )
        / len(unsafe_rows)
        if unsafe_rows
        else 0.0
    )
    fpr = _rate(safe_rows, "defense_triggered")
    tpr = _rate(unsafe_rows, "defense_triggered")
    ses = (sum(float(item.get("side_effect_score", 0.0)) for item in rows) / total) if total else 0.0
    tcs = (sum(float(item.get("trace_compliance", 0.0)) for item in rows) / total) if total else 0.0
    sev = (sum(float(item.get("side_effect_severity", 0.0)) for item in rows) / total) if total else 0.0
    dr = _rate(unsafe_rows, "attack_disruption")
    tar = _rate(unsafe_rows, "attack_success_total")
    dsr = _rate(unsafe_rows, "defense_success")
    avg_steps = (sum(int(item.get("tool_call_count", 0)) for item in rows) / total) if total else 0.0
    return {
        "total": total,
        "safe_total": len(safe_rows),
        "unsafe_total": len(unsafe_rows),
        "bu": bu,
        "ua": ua,
        "asr": asr,
        "uasr": uasr,
        "sua": sua,
        "fpr": fpr,
        "tpr": tpr,
        "ses": ses,
        "tcs": tcs,
        "sev": sev,
        "dr": dr,
        "tar": tar,
        "dsr": dsr,
        "avg_steps": avg_steps,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in headers:
                headers.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_case_max_iterations(
    *,
    case: object,
    fallback_max_iterations: int,
    adaptive_iterations: bool,
    adaptive_multiplier: float,
    adaptive_min: int,
    adaptive_max: int | None,
) -> int:
    if not adaptive_iterations:
        return int(fallback_max_iterations)

    gt_steps = 0
    ground_truth = getattr(case, "ground_truth", None)
    if isinstance(ground_truth, list):
        gt_steps = len(ground_truth)

    if gt_steps <= 0:
        return int(fallback_max_iterations)

    estimated = int(math.ceil(float(gt_steps) * float(adaptive_multiplier)))
    bounded = max(int(adaptive_min), estimated)
    if adaptive_max is not None and int(adaptive_max) > 0:
        bounded = min(int(adaptive_max), bounded)
    return int(bounded)


def _get_llm_provider(llm: object, llm_name: str) -> str:
    get_model_info = getattr(llm, "get_model_info", None)
    if callable(get_model_info):
        try:
            info = get_model_info() or {}
            provider = str(info.get("provider", "")).strip().lower()
            if provider:
                return provider
        except Exception:
            pass

    try:
        info = get_backend_info(llm_name)
        return str(info.get("provider", "")).strip().lower()
    except Exception:
        return ""


def _resolve_openai_defense_client_config(
    *,
    llm_name: str,
    llm_provider: str,
    backend_kwargs: dict | None,
    component_config: dict | None,
) -> dict[str, str | None]:
    def _clean(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    cfg = component_config or {}
    is_openai_provider = llm_provider in OPENAI_COMPATIBLE_PROVIDERS

    api_key = _clean(cfg.get("api_key"))
    if not api_key and is_openai_provider:
        api_key = _clean((backend_kwargs or {}).get("api_key"))
    if not api_key:
        api_key = (
            _clean(os.getenv("DEFENSE_API_KEY"))
            or _clean(os.getenv("OPENAI_API_KEY"))
        )

    base_url = _clean(cfg.get("base_url"))
    if not base_url and is_openai_provider:
        base_url = _clean((backend_kwargs or {}).get("base_url"))
    if not base_url:
        base_url = (
            _clean(os.getenv("DEFENSE_BASE_URL"))
            or _clean(os.getenv("OPENAI_BASE_URL"))
        )

    model = (
        _clean(cfg.get("model"))
        or _clean(os.getenv("DEFENSE_MODEL"))
        or (llm_name if is_openai_provider else DEFAULT_OPENAI_DEFENSE_MODEL)
    )

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": str(model),
    }


def _build_agent(
    llm_name: str,
    defense_config: dict,
    tool_simulator: ToolSimulator,
    backend_kwargs: dict | None,
    max_iterations: int | None = None,
):
    llm = create_llm_backend(llm_name, **(backend_kwargs or {}))
    llm_provider = _get_llm_provider(llm=llm, llm_name=llm_name)
    residual_hooks = extract_hook_config(defense_config)
    hook_layers = ("input", "plan", "reasoning", "action", "finish")

    hook_registry = None
    if any(residual_hooks.get(layer) for layer in hook_layers):
        hook_defense_client = _resolve_openai_defense_client_config(
            llm_name=llm_name,
            llm_provider=llm_provider,
            backend_kwargs=backend_kwargs,
            component_config=defense_config.get("hook_config") or {},
        )
        hook_registry = create_hook_registry(residual_hooks, defense_client=hook_defense_client)

    planner_plugin = None
    planner_name = get_planner_name(defense_config)
    planner_config = defense_config.get("planner_config", {}) or {}
    runtime_name = get_runtime_name(defense_config)
    runtime_config = defense_config.get("runtime_config", {}) or {}
    rtbas_service = None
    if planner_name == "rtbas" or runtime_name == "rtbas":
        rtbas_config = {**runtime_config, **planner_config}
        rtbas_service = RTBASService(
            tool_catalog=tool_simulator.tool_library,
            extra_tool_names=set(getattr(tool_simulator, "dispatch_map", {}).keys()),
            screener=rtbas_config.get("screener"),
            max_region_chars=rtbas_config.get("max_region_chars"),
        )

    if planner_name == "ipiguard":
        from defenses.ipiguard.adapter import IPIGuardService

        defense_client = _resolve_openai_defense_client_config(
            llm_name=llm_name,
            llm_provider=llm_provider,
            backend_kwargs=backend_kwargs,
            component_config=planner_config,
        )
        planner_plugin = IPIGuardPlannerPlugin(
            service=IPIGuardService(
                api_key=defense_client["api_key"],
                model=defense_client["model"],
                base_url=defense_client["base_url"],
            )
        )
    elif planner_name == "camel":
        defense_client = _resolve_openai_defense_client_config(
            llm_name=llm_name,
            llm_provider=llm_provider,
            backend_kwargs=backend_kwargs,
            component_config=planner_config,
        )
        planner_plugin = CaMeLPlannerPlugin(
            api_key=defense_client["api_key"],
            model=defense_client["model"],
            base_url=defense_client["base_url"],
            temperature=float(planner_config.get("temperature", 0.0)),
        )
    elif planner_name == "gardener":
        planner_plugin = GardenerPlannerPlugin(
            tool_catalog=tool_simulator.tool_library,
            ask_user_policy=str(planner_config.get("ask_user_policy", "allow")),
            leaf_strategy=str(planner_config.get("leaf_strategy", "global")),
            ask_threshold=int(planner_config.get("ask_threshold", 60)),
        )
    elif planner_name == "rtbas":
        planner_plugin = RTBASPlannerPlugin(service=rtbas_service)
    elif planner_name in {"drift", "drift_formal"}:
        planner_plugin = DRIFTFormalPlannerPlugin()

    runtime_policy_plugin = None
    if runtime_name == "autopermissions_hitl":
        runtime_policy_plugin = AutoPermissionsRuntimePolicyPlugin(
            service=AutoPermissionsService(
                tool_catalog=tool_simulator.tool_library,
                extra_tool_names=set(getattr(tool_simulator, "dispatch_map", {}).keys()),
                confidence_threshold=runtime_config.get("confidence_threshold"),
                gate_mode=runtime_config.get("gate_mode"),
                history_limit=runtime_config.get("history_limit"),
            ),
            confirmation_policy=str(runtime_config.get("confirmation_policy", "oracle")),
        )
    elif runtime_name == "rtbas":
        runtime_policy_plugin = RTBASRuntimePolicyPlugin(
            service=rtbas_service,
            confirmation_policy=str(runtime_config.get("confirmation_policy", "oracle")),
        )
    elif runtime_name == "fides":
        runtime_cfg = defense_config.get("runtime_config", {}) or {}
        runtime_policy_plugin = FidesRuntimePolicyPlugin(
            tool_catalog=tool_simulator.tool_library,
            extra_tool_names=set(getattr(tool_simulator, "dispatch_map", {}).keys()),
            force_sensitive_tools=runtime_cfg.get("force_sensitive_tools"),
            force_read_only_tools=runtime_cfg.get("force_read_only_tools"),
            force_sink_tools=runtime_cfg.get("force_sink_tools"),
        )
    elif runtime_name in {"drift", "drift_formal"}:
        runtime_cfg = defense_config.get("runtime_config", {}) or {}
        runtime_policy_plugin = DRIFTFormalRuntimePolicyPlugin(
            enforce_trajectory=runtime_cfg.get("enforce_trajectory", True),
            enforce_checklist=runtime_cfg.get("enforce_checklist", True),
            allow_read_only_deviation=runtime_cfg.get("allow_read_only_deviation", True),
            strict_arg_match=runtime_cfg.get("strict_arg_match", True),
            block_side_effect_after_plan=runtime_cfg.get("block_side_effect_after_plan", True),
            fail_closed_when_no_constraints=runtime_cfg.get("fail_closed_when_no_constraints", False),
            injection_isolation=runtime_cfg.get("injection_isolation", True),
            use_llm_classification=bool(runtime_cfg.get("use_llm_classification", False)),
            use_llm_alignment_judge=bool(runtime_cfg.get("use_llm_alignment_judge", False)),
            use_llm_injection_detection=bool(runtime_cfg.get("use_llm_injection_detection", False)),
        )
    elif runtime_name == "tool_filter":
        runtime_cfg = defense_config.get("runtime_config", {}) or {}
        defense_client = _resolve_openai_defense_client_config(
            llm_name=llm_name,
            llm_provider=llm_provider,
            backend_kwargs=backend_kwargs,
            component_config=runtime_cfg,
        )
        runtime_policy_plugin = ToolFilterRuntimePolicyPlugin(
            api_key=defense_client["api_key"],
            model=defense_client["model"],
            base_url=defense_client["base_url"],
            temperature=float(runtime_cfg.get("temperature", 0.0)),
        )
    elif runtime_name == "piguard":
        runtime_cfg = defense_config.get("runtime_config", {}) or {}
        runtime_policy_plugin = PIGuardRuntimePolicyPlugin(
            model_name=str(runtime_cfg.get("model_name", "leolee99/PIGuard")),
            threshold=float(runtime_cfg.get("threshold", 0.5)),
            device=int(runtime_cfg.get("device", -1)),
            trust_remote_code=bool(runtime_cfg.get("trust_remote_code", True)),
            block_side_effect_only=bool(runtime_cfg.get("block_side_effect_only", True)),
            scan_tool_outputs=bool(runtime_cfg.get("scan_tool_outputs", True)),
        )
    elif runtime_name == "protectai":
        runtime_cfg = defense_config.get("runtime_config", {}) or {}
        runtime_policy_plugin = ProtectAIRuntimePolicyPlugin(
            threshold=float(runtime_cfg.get("threshold", 0.92)),
            use_onnx=bool(runtime_cfg.get("use_onnx", False)),
            model_name=str(
                runtime_cfg.get("model_name", "protectai/deberta-v3-base-prompt-injection-v2")
            ),
            device=int(runtime_cfg.get("device", -1)),
            block_side_effect_only=bool(runtime_cfg.get("block_side_effect_only", True)),
            scan_tool_outputs=bool(runtime_cfg.get("scan_tool_outputs", True)),
        )
    elif runtime_name == "camel":
        runtime_policy_plugin = CaMeLRuntimePolicyPlugin()

    agent_cfg = defense_config.get("agent_config", {}) or {}
    resolved_max_iterations = int(
        agent_cfg.get("max_iterations", max_iterations if max_iterations is not None else 64)
    )

    return Agent(
        llm=llm,
        tool_simulator=tool_simulator,
        hook_registry=hook_registry,
        planner_plugin=planner_plugin,
        runtime_policy_plugin=runtime_policy_plugin,
        max_iterations=resolved_max_iterations,
    )


def _serialize_tool_call_obj(tool_call: object) -> dict:
    if is_dataclass(tool_call):
        return asdict(tool_call)
    if isinstance(tool_call, dict):
        return dict(tool_call)
    return {
        "name": str(getattr(tool_call, "name", "")),
        "args": dict(getattr(tool_call, "args", {}) or {}),
        "id": str(getattr(tool_call, "id", "")),
    }


def _serialize_message_obj(message: object) -> dict:
    if is_dataclass(message):
        payload = asdict(message)
        if isinstance(payload.get("tool_calls"), list):
            payload["tool_calls"] = [_serialize_tool_call_obj(tc) for tc in payload["tool_calls"]]
        return payload
    if isinstance(message, dict):
        payload = dict(message)
        if isinstance(payload.get("tool_calls"), list):
            payload["tool_calls"] = [_serialize_tool_call_obj(tc) for tc in payload["tool_calls"]]
        return payload

    tool_calls = getattr(message, "tool_calls", []) or []
    return {
        "role": str(getattr(message, "role", "")),
        "content": str(getattr(message, "content", "")),
        "tool_calls": [_serialize_tool_call_obj(tc) for tc in tool_calls],
        "tool_call_id": str(getattr(message, "tool_call_id", "")),
        "tool_name": str(getattr(message, "tool_name", "")),
    }


def _jsonify_artifact(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _jsonify_artifact(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonify_artifact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify_artifact(v) for v in value]
    if hasattr(value, "__dict__"):
        return {
            str(k): _jsonify_artifact(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return str(value)


_CORE_AGENT_STATE_KEYS = {
    "messages",
    "is_complete",
    "final_response",
    "tool_calls_made",
    "blocked",
    "blocked_by",
    "blocked_reason",
    "runtime_error",
    "runtime_error_message",
}


def _collect_state_extra_artifacts(state: Any) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    for key, value in vars(state).items():
        if key in _CORE_AGENT_STATE_KEYS:
            continue
        extras[str(key)] = _jsonify_artifact(value)
    return extras


def _collect_defense_artifacts(agent: Any, state: Any) -> dict[str, Any]:
    hooks = getattr(agent, "hooks", None)
    planner = getattr(agent, "planner_plugin", None)
    runtime_policy = getattr(agent, "runtime_policy_plugin", None)

    payload = {
        "state_extras": _collect_state_extra_artifacts(state),
        "hooks": hooks.export_artifacts() if hooks and hasattr(hooks, "export_artifacts") else {},
        "planner": planner.export_artifacts(state=state) if planner and hasattr(planner, "export_artifacts") else {},
        "runtime_policy": (
            runtime_policy.export_artifacts(state=state)
            if runtime_policy and hasattr(runtime_policy, "export_artifacts")
            else {}
        ),
    }
    return _jsonify_artifact(payload)


def _ordered_defense_tokens(defense_config: dict) -> list[str]:
    def _normalize_token(entry: object) -> str:
        if isinstance(entry, str):
            return entry.strip()
        if isinstance(entry, dict):
            return str(entry.get("name", "")).strip()
        return ""

    planner = get_planner_name(defense_config).strip()
    runtime = get_runtime_name(defense_config).strip()
    ordered_parts: list[str] = []

    if planner:
        ordered_parts.append(f"{planner}_planner")
    if runtime:
        ordered_parts.append(f"{runtime}_runtime")

    for layer in ("input", "plan", "reasoning", "action", "finish"):
        for entry in list(defense_config.get(layer, []) or []):
            name = _normalize_token(entry)
            if name:
                ordered_parts.append(name)

    return list(dict.fromkeys(ordered_parts))


def _defense_name(defense_config: dict) -> str:
    deduped_parts = _ordered_defense_tokens(defense_config)
    return "+".join(deduped_parts) if deduped_parts else "no_defense"


def _defense_slug(defense_config: dict) -> str:
    tokens = _ordered_defense_tokens(defense_config)
    if not tokens:
        return "none"
    mapped = [
        DEFENSE_SLUG_ALIASES.get(token, token)
        for token in tokens
        if str(token).strip()
    ]
    deduped = list(dict.fromkeys(mapped))
    if len(deduped) == 1:
        return _sanitize_slug(deduped[0])
    return "__".join(_sanitize_slug(token) for token in deduped)


def _extract_ipiguard_dag(agent: object) -> dict | None:
    planner = getattr(agent, "planner_plugin", None)
    if planner is None or str(getattr(planner, "name", "")) != "ipiguard":
        return None
    plan = getattr(planner, "plan", None)
    if plan is None:
        return None
    dag = getattr(plan, "dag", None)
    if dag is None:
        return None
    try:
        nodes = []
        for node_id in dag.nodes:
            node_data = dag.nodes[node_id] or {}
            function_call = node_data.get("function_call")
            function_payload = {}
            if function_call is not None:
                function_payload = {
                    "function": str(getattr(function_call, "function", "")),
                    "args": dict(getattr(function_call, "args", {}) or {}),
                    "id": str(getattr(function_call, "id", "") or ""),
                }
            nodes.append(
                {
                    "id": str(node_id),
                    "function_call": function_payload,
                }
            )

        edges = [
            {"source": str(src), "target": str(dst)}
            for src, dst in dag.edges()
        ]

        return {
            "topological_order": [str(x) for x in (getattr(plan, "topological_order", []) or [])],
            "cursor": int(getattr(plan, "cursor", 0) or 0),
            "nodes": nodes,
            "edges": edges,
        }
    except Exception:
        return None


def _execute_case_once(
    *,
    case: object,
    llm_name: str,
    defense_config: dict,
    tool_library_path: str,
    backend_kwargs: dict | None,
    max_iterations: int | None,
    adaptive_iterations: bool,
    adaptive_multiplier: float,
    adaptive_min: int,
    adaptive_max: int,
    verbose: bool = False,
) -> dict[str, Any]:
    tool_sim = ToolSimulator.from_library_file(tool_library_path)
    fallback_max_iterations = int(max_iterations if max_iterations is not None else 64)
    case_max_iterations = _resolve_case_max_iterations(
        case=case,
        fallback_max_iterations=fallback_max_iterations,
        adaptive_iterations=adaptive_iterations,
        adaptive_multiplier=adaptive_multiplier,
        adaptive_min=adaptive_min,
        adaptive_max=adaptive_max,
    )
    agent = _build_agent(
        llm_name=llm_name,
        defense_config=defense_config,
        tool_simulator=tool_sim,
        backend_kwargs=backend_kwargs,
        max_iterations=case_max_iterations,
    )
    case_evaluator = Evaluator()
    initial_world_state: dict[str, Any] = {}
    try:
        WorldState.reset()
        if agent.hooks:
            agent.hooks.clear_log()
        is_state_snapshot = isinstance(case.environment_data, dict) and "source" not in case.environment_data
        if is_state_snapshot:
            WorldState.get_instance().load_from_dict(case.environment_data)
        initial_world_state = WorldState.get_instance().to_dict()

        state = agent.run(case)
        blocked_reason = str(getattr(state, "blocked_reason", "") or "")
        runtime_error_message = str(getattr(state, "runtime_error_message", "") or "")
        final_response = str(getattr(state, "final_response", "") or "")
        if _contains_infrastructure_error(
            " || ".join([blocked_reason, runtime_error_message, final_response])
        ):
            state.runtime_error = True
            state.runtime_error_message = blocked_reason or runtime_error_message or final_response
            state.blocked = False
            state.blocked_by = ""
            state.blocked_reason = ""
        final_world_state = WorldState.get_instance().to_dict()
        eval_result = case_evaluator.evaluate_case(
            case,
            state,
            world_state=final_world_state,
            initial_snapshot=initial_world_state,
        )
        ipiguard_dag = _extract_ipiguard_dag(agent)
        defense_artifacts = _collect_defense_artifacts(agent, state)

        trace_payload = {
            "case_id": case.id,
            "blocked": bool(getattr(state, "blocked", False)),
            "blocked_by": str(getattr(state, "blocked_by", "")),
            "blocked_reason": str(getattr(state, "blocked_reason", "")),
            "runtime_error": bool(getattr(state, "runtime_error", False)),
            "runtime_error_message": str(getattr(state, "runtime_error_message", "")),
            "final_response": str(getattr(state, "final_response", "")),
            "messages": [_serialize_message_obj(m) for m in (getattr(state, "messages", []) or [])],
            "tool_calls_made": getattr(state, "tool_calls_made", []) or [],
            "initial_snapshot": initial_world_state,
            "final_snapshot": final_world_state,
            "eval": eval_result.to_dict(),
            "defense_artifacts": defense_artifacts,
        }
        if ipiguard_dag is not None:
            trace_payload["ipiguard_dag"] = ipiguard_dag

        if verbose:
            label = "safe" if case.label.safety == "safe" else "unsafe"
            if eval_result.defense_triggered:
                status = "blocked"
                br = getattr(state, "blocked_reason", "")
                bb = getattr(state, "blocked_by", "")
                if br or bb:
                    status += f" ({bb}: {br})"
            elif eval_result.task_completed:
                status = "completed"
            else:
                status = "incomplete"
            attack = ""
            if case.label.safety == "unsafe":
                attack = " attack=success" if eval_result.attack_successful else " attack=fail"
            iter_note = f" [iters={case_max_iterations}]"
            print(f"  {case.id} [{label}] {status}{attack}{iter_note}")

        return {
            "ok": not bool(getattr(state, "runtime_error", False)),
            "case_id": case.id,
            "case_max_iterations": case_max_iterations,
            "eval_result": eval_result,
            "trace_payload": trace_payload,
            "initial_snapshot": initial_world_state,
            "final_snapshot": final_world_state,
            "ipiguard_dag": ipiguard_dag,
            "defense_artifacts": defense_artifacts,
            "error_message": str(getattr(state, "runtime_error_message", "") or ""),
        }
    except Exception as e:
        if verbose:
            print(f"  {case.id}: error - {e}")
        fallback_state = AgentState()
        fallback_state.runtime_error = True
        fallback_state.runtime_error_message = str(e)
        fallback_state.final_response = f"[runtime_error] {e}"
        try:
            final_world_state = WorldState.get_instance().to_dict()
            eval_result = case_evaluator.evaluate_case(
                case,
                fallback_state,
                world_state=final_world_state,
                initial_snapshot=initial_world_state,
            )
            defense_artifacts = _collect_defense_artifacts(agent, fallback_state)
            trace_payload = {
                "case_id": case.id,
                "blocked": bool(getattr(fallback_state, "blocked", False)),
                "blocked_by": str(getattr(fallback_state, "blocked_by", "")),
                "blocked_reason": str(getattr(fallback_state, "blocked_reason", "")),
                "runtime_error": bool(getattr(fallback_state, "runtime_error", False)),
                "runtime_error_message": str(getattr(fallback_state, "runtime_error_message", "")),
                "final_response": str(getattr(fallback_state, "final_response", "")),
                "messages": [_serialize_message_obj(m) for m in (getattr(fallback_state, "messages", []) or [])],
                "tool_calls_made": getattr(fallback_state, "tool_calls_made", []) or [],
                "initial_snapshot": initial_world_state,
                "final_snapshot": final_world_state,
                "eval": eval_result.to_dict(),
                "defense_artifacts": defense_artifacts,
            }
            return {
                "ok": True,
                "case_id": case.id,
                "case_max_iterations": case_max_iterations,
                "eval_result": eval_result,
                "trace_payload": trace_payload,
                "initial_snapshot": initial_world_state,
                "final_snapshot": final_world_state,
                "ipiguard_dag": None,
                "defense_artifacts": defense_artifacts,
                "error_message": str(e),
            }
        except Exception as nested:
            if verbose:
                print(f"  {case.id}: fallback evaluation failed - {nested}")
            return {
                "ok": False,
                "case_id": case.id,
                "case_max_iterations": case_max_iterations,
                "eval_result": None,
                "trace_payload": {
                    "case_id": case.id,
                    "runtime_error": True,
                    "runtime_error_message": str(e),
                },
                "initial_snapshot": initial_world_state,
                "final_snapshot": {},
                "ipiguard_dag": None,
                "error_message": str(nested),
            }


def run_evaluation(
    llm_name: str,
    defense_config: dict,
    cases_dir: str,
    tool_library_path: str,
    output_dir: str,
    verbose: bool = False,
    backend_kwargs: dict | None = None,
    max_iterations: int | None = None,
    workers: int = 1,
    save_traces: bool = False,
    adaptive_iterations: bool = False,
    adaptive_multiplier: float = 4.0,
    adaptive_min: int = 15,
    adaptive_max: int = 0,
    resume: bool = True,
) -> MetricsSummary:
    defense_name = _defense_name(defense_config)
    trace_dir = os.path.join(output_dir, llm_name, defense_name, "traces")
    dag_dir = os.path.join(output_dir, llm_name, defense_name, "dag")
    cases_root = Path(cases_dir).resolve()

    def _case_artifact_key(case: object) -> str:
        source_file = getattr(case, "__source_file__", "")
        if source_file:
            try:
                rel = Path(source_file).resolve().relative_to(cases_root)
                rel_no_suffix = rel.with_suffix("")
                rel_slug = "__".join(rel_no_suffix.parts)
                return _sanitize_slug(rel_slug)
            except Exception:
                pass
        return _sanitize_slug(str(getattr(case, "id", "unknown")))

    def _trace_resume_eligible(payload: dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return False
        blocked_reason = str(payload.get("blocked_reason", "") or "")
        runtime_error_message = str(payload.get("runtime_error_message", "") or "")
        eval_payload = payload.get("eval") or {}
        final_response = str(eval_payload.get("final_response", "") or "")
        combined = " || ".join(
            part for part in (blocked_reason, runtime_error_message, final_response) if part
        ).lower()
        transient_markers = (
            "remainquota",
            "unlimitedquota",
            "\u989d\u5ea6\u5df2\u7528\u5c3d",
            "invalid token",
            "model_not_found",
            "planner step failed: error code: 401",
            "error code: 401",
            "insufficient_quota",
            "rate limit",
            "429",
        )
        return not any(marker.lower() in combined for marker in transient_markers)

    def _load_existing_eval(case: object) -> EvalResult | None:
        if not save_traces or not resume:
            return None
        trace_path = os.path.join(trace_dir, f"{_case_artifact_key(case)}.json")
        if not os.path.exists(trace_path):
            return None
        try:
            with open(trace_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not _trace_resume_eligible(payload):
                return None
            eval_payload = payload.get("eval")
            if not isinstance(eval_payload, dict):
                return None
            return EvalResult(**eval_payload)
        except Exception:
            return None

    def _run_single_case(case):
        """Execute one case in isolation — safe to call from multiple threads."""
        artifact_key = _case_artifact_key(case)
        existing_eval = _load_existing_eval(case)
        if existing_eval is not None:
            return existing_eval
        execution = _execute_case_once(
            case=case,
            llm_name=llm_name,
            defense_config=defense_config,
            tool_library_path=tool_library_path,
            backend_kwargs=backend_kwargs,
            max_iterations=max_iterations,
            adaptive_iterations=adaptive_iterations,
            adaptive_multiplier=adaptive_multiplier,
            adaptive_min=adaptive_min,
            adaptive_max=adaptive_max,
            verbose=verbose,
        )
        eval_result = execution.get("eval_result")
        if save_traces:
            os.makedirs(trace_dir, exist_ok=True)
            trace_path = os.path.join(trace_dir, f"{artifact_key}.json")
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(execution.get("trace_payload", {}), f, ensure_ascii=False, indent=2)
            ipiguard_dag = execution.get("ipiguard_dag")
            if ipiguard_dag is not None:
                os.makedirs(dag_dir, exist_ok=True)
                dag_path = os.path.join(dag_dir, f"{artifact_key}.json")
                with open(dag_path, "w", encoding="utf-8") as f:
                    json.dump(ipiguard_dag, f, ensure_ascii=False, indent=2)
        return eval_result if execution.get("ok") else None

    loader = CaseLoader(cases_dir)
    evaluator = Evaluator()
    cases = loader.load_all()
    if not cases:
        print(f"Warning: no cases found in {cases_dir}")
        return MetricsSummary()

    results = []
    actual_workers = max(1, workers)
    if actual_workers == 1:
        for case in cases:
            result = _run_single_case(case)
            if result is not None:
                results.append(result)
    else:
        print(f"  [parallel] workers={actual_workers}, cases={len(cases)}")
        with ThreadPoolExecutor(max_workers=actual_workers) as pool:
            future_to_case = {pool.submit(_run_single_case, case): case for case in cases}
            for future in as_completed(future_to_case):
                result = future.result()
                if result is not None:
                    results.append(result)

    summary = evaluator.compute_metrics(results, cases)
    save_results(
        [r.to_dict() for r in results],
        os.path.join(output_dir, llm_name, defense_name, "results.json"),
    )
    save_results(
        summary.to_dict(),
        os.path.join(output_dir, llm_name, defense_name, "summary.json"),
    )
    return summary


def _manifest_complete(pair_dir: Path, manifest: dict[str, Any], expected_hash: str | None = None) -> bool:
    if not isinstance(manifest, dict):
        return False
    if str(manifest.get("layout_version", "")) != RESULTS_LAYOUT_VERSION:
        return False
    if manifest.get("status") != "completed":
        return False
    if expected_hash and str(manifest.get("input_hash", "")) != expected_hash:
        return False
    required_files = [
        "safe.trace.json",
        "unsafe.trace.json",
        "safe.eval.json",
        "unsafe.eval.json",
        "safe.defense_artifacts.json",
        "unsafe.defense_artifacts.json",
        "safe.snapshot.json",
        "unsafe.snapshot.json",
        "pair_summary.json",
    ]
    for rel in required_files:
        if not (pair_dir / rel).exists():
            return False
    if _case_artifacts_have_infrastructure_error(pair_dir):
        return False
    return True


def _safe_manifest_complete(safe_dir: Path, manifest: dict[str, Any], expected_hash: str | None = None) -> bool:
    if not isinstance(manifest, dict):
        return False
    if str(manifest.get("layout_version", "")) != RESULTS_LAYOUT_VERSION:
        return False
    if manifest.get("status") != "completed":
        return False
    if expected_hash and str(manifest.get("input_hash", "")) != expected_hash:
        return False
    required_files = [
        "safe.trace.json",
        "safe.eval.json",
        "safe.defense_artifacts.json",
        "safe.snapshot.json",
        "manifest.json",
    ]
    for rel in required_files:
        if not (safe_dir / rel).exists():
            return False
    if _payload_has_infrastructure_error(_read_json(safe_dir / "safe.trace.json", {})):
        return False
    if _payload_has_infrastructure_error(_read_json(safe_dir / "safe.eval.json", {})):
        return False
    return True


def _copy_artifact_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _group_by_pair(cases: list[object]) -> dict[str, dict[str, object]]:
    grouped_by_base: dict[str, dict[str, Any]] = {}
    for case in cases:
        pair_base = _pair_base_id(str(getattr(case, "id", "")))
        variant = _case_variant(case)
        bucket = grouped_by_base.setdefault(pair_base, {"safe": None, "unsafe": []})
        if variant == "safe":
            bucket["safe"] = case
        else:
            bucket.setdefault("unsafe", []).append(case)

    grouped: dict[str, dict[str, object]] = {}
    for pair_base, bundle in grouped_by_base.items():
        safe_case = bundle.get("safe")
        unsafe_cases = list(bundle.get("unsafe", []) or [])
        if safe_case is None or not unsafe_cases:
            continue
        for unsafe_case in unsafe_cases:
            unsafe_variant_id = _unsafe_variant_id(unsafe_case)
            pair_run_id = _pair_run_id(pair_base, unsafe_variant_id)
            grouped[pair_run_id] = {
                "pair_base_id": pair_base,
                "pair_run_id": pair_run_id,
                "unsafe_variant_id": unsafe_variant_id,
                "safe": safe_case,
                "unsafe": unsafe_case,
            }
    return grouped


def _entry_key(defense_slug: str, model_slug: str, domain_slug: str, pair_run_id: str) -> str:
    return "/".join([defense_slug, model_slug, domain_slug, pair_run_id])


def _safe_entry_key(defense_slug: str, model_slug: str, domain_slug: str, pair_base_id: str) -> str:
    return "/".join([defense_slug, model_slug, domain_slug, _path_safe_component(pair_base_id)])


def _build_case_row(
    *,
    defense: str,
    model: str,
    domain: str,
    pair_base_id: str,
    pair_run_id: str = "",
    case_obj: object,
    eval_payload: dict[str, Any],
) -> dict[str, Any]:
    if isinstance(case_obj, dict):
        taxonomy = dict(case_obj.get("taxonomy", {}) or {})
        case_id = str(case_obj.get("case_id", ""))
        label = str((case_obj.get("label", {}) or {}).get("safety", ""))
    else:
        taxonomy = _taxonomy_to_dict(case_obj)
        case_id = str(getattr(case_obj, "id", ""))
        label = str(getattr(getattr(case_obj, "label", None), "safety", ""))
    injection_profile = taxonomy.get("injection_profile", {}) if isinstance(taxonomy, dict) else {}
    return {
        "defense": defense,
        "model": model,
        "domain": domain,
        "pair_base_id": pair_base_id,
        "pair_run_id": pair_run_id or pair_base_id,
        "case_id": case_id,
        "label": label,
        "scenario_domain": taxonomy.get("scenario_domain"),
        "dependency_structure": taxonomy.get("dependency_structure"),
        "evidence_delay": taxonomy.get("evidence_delay"),
        "difficulty_tier": taxonomy.get("difficulty_tier"),
        "dep_count": taxonomy.get("dep_count"),
        "noise_tool_count": taxonomy.get("noise_tool_count"),
        "injection_pattern": injection_profile.get("injection_pattern"),
        "injection_point_count": injection_profile.get("injection_point_count"),
        "task_completed": bool(eval_payload.get("task_completed", False)),
        "attack_successful": bool(eval_payload.get("attack_successful", False)),
        "attack_compromise": bool(eval_payload.get("attack_compromise", False)),
        "attack_disruption": bool(eval_payload.get("attack_disruption", False)),
        "attack_success_total": bool(eval_payload.get("attack_success_total", False)),
        "defense_success": bool(eval_payload.get("defense_success", False)),
        "false_positive": bool(eval_payload.get("false_positive", False)),
        "defense_triggered": bool(eval_payload.get("defense_triggered", False)),
        "side_effect_score": float(eval_payload.get("side_effect_score", 0.0)),
        "trace_compliance": float(eval_payload.get("trace_compliance", 0.0)),
        "side_effect_severity": float(eval_payload.get("side_effect_severity", 0.0)),
        "tool_call_count": int(len(eval_payload.get("tool_calls", []) or [])),
    }


def _build_case_artifact_record(
    *,
    case: object,
    variant: str,
    status: str,
    error_message: str = "",
    dag_file: str = "",
    cache_dir: str = "",
) -> dict[str, Any]:
    payload = {
        "case_id": str(getattr(case, "id", "")),
        "status": status,
        "trace_file": f"{variant}.trace.json",
        "eval_file": f"{variant}.eval.json",
        "defense_artifacts_file": f"{variant}.defense_artifacts.json",
        "snapshot_file": f"{variant}.snapshot.json",
        "dag_file": dag_file,
        "error_message": error_message,
        "taxonomy": _taxonomy_to_dict(case),
        "label": asdict(getattr(case, "label"))
        if is_dataclass(getattr(case, "label", None))
        else {},
        "domain": _case_domain(case),
        "scenario": str(getattr(case, "scenario", "")),
        "user_instruction": str(getattr(case, "user_instruction", "")),
        "tools": list(getattr(case, "tools", []) or []),
        "expected_state": deepcopy(getattr(case, "expected_state", {}) or {}),
        "ground_truth": deepcopy(getattr(case, "ground_truth", []) or []),
    }
    if cache_dir:
        payload["cache_dir"] = cache_dir
    return payload


def _collect_case_rows_from_run_index(output_root: Path, run_index: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    safe_seen: set[tuple[str, str, str, str]] = set()
    entries = run_index.get("entries", {})
    if not isinstance(entries, dict):
        return rows
    for _, entry in sorted(entries.items()):
        if not isinstance(entry, dict):
            continue
        if entry.get("status") != "completed":
            continue
        rel_pair_dir = str(entry.get("pair_dir", "")).strip()
        if not rel_pair_dir:
            continue
        pair_dir = output_root / rel_pair_dir
        manifest = _read_json(pair_dir / "manifest.json", {})
        if not _manifest_complete(pair_dir, manifest):
            continue
        domain = str(entry.get("domain", manifest.get("domain", "Unknown")))
        defense = str(entry.get("defense", "unknown"))
        model = str(entry.get("model", "unknown"))
        pair_base_id = str(entry.get("pair_base_id", manifest.get("pair_base_id", "")))
        pair_run_id = str(entry.get("pair_run_id", manifest.get("pair_run_id", pair_base_id)))

        safe_eval = _read_json(pair_dir / "safe.eval.json", {})
        unsafe_eval = _read_json(pair_dir / "unsafe.eval.json", {})
        safe_case = manifest.get("cases", {}).get("safe")
        unsafe_case = manifest.get("cases", {}).get("unsafe")
        if not isinstance(safe_case, dict) or not isinstance(unsafe_case, dict):
            continue
        safe_key = (defense, model, domain, pair_base_id)
        if safe_key not in safe_seen:
            safe_seen.add(safe_key)
            rows.append(
                _build_case_row(
                    defense=defense,
                    model=model,
                    domain=domain,
                    pair_base_id=pair_base_id,
                    pair_run_id=pair_base_id,
                    case_obj=safe_case,
                    eval_payload=safe_eval,
                )
            )
        rows.append(
            _build_case_row(
                defense=defense,
                model=model,
                domain=domain,
                pair_base_id=pair_base_id,
                pair_run_id=pair_run_id,
                case_obj=unsafe_case,
                eval_payload=unsafe_eval,
            )
        )
    return rows


def _aggregate_rows(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        buckets.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, group_rows in sorted(buckets.items(), key=lambda item: item[0]):
        payload = {group_keys[i]: key[i] for i in range(len(group_keys))}
        payload.update(_compute_metrics_from_rows(group_rows))
        out.append(payload)
    return out


def _sorted_axis_values(values: set[Any]) -> list[Any]:
    def _key(value: Any) -> tuple[int, float | str]:
        if isinstance(value, (int, float)):
            return (0, float(value))
        text = str(value)
        if text.isdigit():
            return (0, float(text))
        return (1, text)

    return sorted(values, key=_key)


def _plot_line_chart(
    *,
    rows: list[dict[str, Any]],
    x_key: str,
    series_keys: list[str],
    metric: str,
    title: str,
    out_path: Path,
    x_label: str,
) -> None:
    if plt is None or not rows:
        return

    x_values = _sorted_axis_values({row.get(x_key) for row in rows})
    if not x_values:
        return
    x_ticks = list(range(len(x_values)))
    x_index = {value: idx for idx, value in enumerate(x_values)}

    series_map: dict[str, dict[Any, float]] = {}
    for row in rows:
        parts = [str(row.get(key, "")) for key in series_keys]
        label = " | ".join(part for part in parts if part)
        if not label:
            label = "series"
        series_map.setdefault(label, {})
        raw_value = row.get(metric, 0.0)
        value = float(raw_value) if raw_value is not None else 0.0
        if metric in {"bu", "ua", "asr", "uasr", "sua", "fpr", "tpr", "ses", "tcs", "dr", "tar", "dsr"}:
            value *= 100.0
        series_map[label][row.get(x_key)] = value

    fig, ax = plt.subplots(figsize=(max(8, len(x_values) * 1.5), 5.2), facecolor="white")
    ax.set_facecolor("white")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35, color="#a0a0a0")

    sorted_series = sorted(series_map.items(), key=lambda item: item[0])
    for idx, (label, values_by_x) in enumerate(sorted_series):
        ys = [values_by_x.get(x, float("nan")) for x in x_values]
        color = LINE_COLORS[idx % len(LINE_COLORS)]
        marker = LINE_MARKERS[idx % len(LINE_MARKERS)]
        ax.plot(
            x_ticks,
            ys,
            color=color,
            marker=marker,
            linewidth=2.3,
            markersize=7,
            label=label,
        )

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(x) for x in x_values])
    ax.set_xlabel(x_label)
    y_label = f"{metric.upper()} (%)" if metric in {"bu", "ua", "asr", "uasr", "sua", "fpr", "tpr", "ses", "tcs", "dr", "tar", "dsr"} else metric.upper()
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def _rebuild_reports(output_root: Path, run_index: dict[str, Any]) -> None:
    reports_dir = output_root / "reports"
    plots_dir = reports_dir / "plots"
    rows = _collect_case_rows_from_run_index(output_root, run_index)

    overall = _aggregate_rows(rows, ["defense", "model"])
    by_domain = _aggregate_rows(rows, ["defense", "model", "domain"])
    by_evidence_delay = _aggregate_rows(rows, ["defense", "model", "evidence_delay"])
    by_injection_point_count = _aggregate_rows(rows, ["defense", "model", "injection_point_count"])
    by_difficulty_tier = _aggregate_rows(rows, ["defense", "model", "difficulty_tier"])
    by_dependency_structure = _aggregate_rows(rows, ["defense", "model", "dependency_structure"])
    by_noise_tool_count = _aggregate_rows(rows, ["defense", "model", "noise_tool_count"])
    by_injection_pattern = _aggregate_rows(rows, ["defense", "model", "injection_pattern"])

    payload = {
        "generated_at": _utcnow_iso(),
        "layout_version": RESULTS_LAYOUT_VERSION,
        "counts": {
            "case_rows": len(rows),
            "overall_rows": len(overall),
        },
        "overall": overall,
        "by_domain": by_domain,
        "by_evidence_delay": by_evidence_delay,
        "by_injection_point_count": by_injection_point_count,
        "by_difficulty_tier": by_difficulty_tier,
        "by_dependency_structure": by_dependency_structure,
        "by_noise_tool_count": by_noise_tool_count,
        "by_injection_pattern": by_injection_pattern,
    }
    _write_json(reports_dir / "summary.json", payload)

    _write_csv(reports_dir / "case_detail.csv", rows)
    _write_csv(reports_dir / "overall.csv", overall)
    _write_csv(reports_dir / "by_domain.csv", by_domain)
    _write_csv(reports_dir / "by_evidence_delay.csv", by_evidence_delay)
    _write_csv(reports_dir / "by_injection_point_count.csv", by_injection_point_count)
    _write_csv(reports_dir / "by_difficulty_tier.csv", by_difficulty_tier)
    _write_csv(reports_dir / "by_dependency_structure.csv", by_dependency_structure)
    _write_csv(reports_dir / "by_noise_tool_count.csv", by_noise_tool_count)
    _write_csv(reports_dir / "by_injection_pattern.csv", by_injection_pattern)

    if plt is not None:
        for metric in ("bu", "ua", "asr", "uasr", "sua", "fpr", "tpr", "ses", "tcs", "sev", "dr", "tar", "dsr"):
            _plot_line_chart(
                rows=overall,
                x_key="defense",
                series_keys=["model"],
                metric=metric,
                title=f"Overall {metric.upper()} by Defense",
                out_path=plots_dir / f"overall_{metric}.png",
                x_label="Defense",
            )
        _plot_line_chart(
            rows=by_evidence_delay,
            x_key="evidence_delay",
            series_keys=["model", "defense"],
            metric="asr",
            title="ASR vs Evidence Delay",
            out_path=plots_dir / "trend_evidence_delay_asr.png",
            x_label="Evidence Delay",
        )
        _plot_line_chart(
            rows=by_injection_point_count,
            x_key="injection_point_count",
            series_keys=["model", "defense"],
            metric="asr",
            title="ASR vs Injection Point Count",
            out_path=plots_dir / "trend_injection_point_count_asr.png",
            x_label="Injection Point Count",
        )
        _plot_line_chart(
            rows=by_difficulty_tier,
            x_key="difficulty_tier",
            series_keys=["model", "defense"],
            metric="asr",
            title="ASR vs Difficulty Tier",
            out_path=plots_dir / "trend_difficulty_tier_asr.png",
            x_label="Difficulty Tier",
        )
        _plot_line_chart(
            rows=by_dependency_structure,
            x_key="dependency_structure",
            series_keys=["model", "defense"],
            metric="asr",
            title="ASR vs Dependency Structure",
            out_path=plots_dir / "trend_dependency_structure_asr.png",
            x_label="Dependency Structure",
        )
        _plot_line_chart(
            rows=by_noise_tool_count,
            x_key="noise_tool_count",
            series_keys=["model", "defense"],
            metric="asr",
            title="ASR vs Noise Tool Count",
            out_path=plots_dir / "trend_noise_tool_count_asr.png",
            x_label="Noise Tool Count",
        )
        _plot_line_chart(
            rows=by_domain,
            x_key="domain",
            series_keys=["model", "defense"],
            metric="asr",
            title="ASR vs Domain",
            out_path=plots_dir / "trend_domain_asr.png",
            x_label="Domain",
        )
        _plot_line_chart(
            rows=by_injection_pattern,
            x_key="injection_pattern",
            series_keys=["model", "defense"],
            metric="asr",
            title="ASR vs Injection Pattern",
            out_path=plots_dir / "trend_injection_pattern_asr.png",
            x_label="Injection Pattern",
        )


def _parse_csv_items(raw_value: str) -> list[str]:
    items = [item.strip() for item in str(raw_value).split(",")]
    return [item for item in items if item]


def _resolve_matrix_defense_config(defense_text: str) -> dict[str, Any]:
    raw = defense_text.strip()
    if raw.lower() in {"none", "no_defense"}:
        raw = ""
    config = parse_defense_string(raw)
    config["name"] = raw or "no_defense"
    return config


def run_matrix_with_resume(
    *,
    models: list[str],
    defenses: list[str],
    cases_dir: str,
    tool_library_path: str,
    output_dir: str,
    backend_kwargs: dict | None,
    max_iterations: int | None,
    adaptive_iterations: bool,
    adaptive_multiplier: float,
    adaptive_min: int,
    adaptive_max: int,
    verbose: bool,
    resume: bool,
    rerun_failed: bool,
    force_rerun: bool,
    workers: int,
    planner_config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_index = _load_run_index(output_root)
    tool_library_hash = _file_sha256(tool_library_path)

    loader = CaseLoader(cases_dir)
    cases = loader.load_all()
    grouped_pairs = _group_by_pair(cases)
    pair_ids = sorted(grouped_pairs.keys())
    matrix_summary: list[dict[str, Any]] = []

    for model in models:
        model_name = _path_safe_component(model)
        for defense_text in defenses:
            defense_cfg = _resolve_matrix_defense_config(defense_text)
            defense_cfg["planner_config"] = dict(planner_config_overrides or {})
            defense_slug = _defense_slug(defense_cfg)
            combo_ran = 0
            combo_skipped = 0
            combo_failed = 0
            combo_lock = threading.Lock()
            safe_locks: dict[str, threading.Lock] = {}

            print(f"\n> {model} + {defense_text or 'no_defense'}")
            actual_workers = max(1, int(workers or 1))
            if actual_workers > 1:
                print(f"  [parallel] workers={actual_workers}, pairs={len(pair_ids)}")

            def _process_pair(pair_run_id: str) -> dict[str, Any] | None:
                pair_bundle = grouped_pairs.get(pair_run_id, {})
                safe_case = pair_bundle.get("safe")
                unsafe_case = pair_bundle.get("unsafe")
                if safe_case is None or unsafe_case is None:
                    return None
                pair_base_id = str(pair_bundle.get("pair_base_id", pair_run_id))
                unsafe_variant_id = str(pair_bundle.get("unsafe_variant_id", "unsafe"))
                domain = _case_domain(safe_case)
                domain_slug = _sanitize_slug(domain)
                pair_dir = output_root / defense_slug / model_name / domain / pair_run_id
                pair_dir_rel = pair_dir.relative_to(output_root)
                safe_dir = output_root / defense_slug / model_name / domain / "safe_cache" / _path_safe_component(pair_base_id)
                safe_dir_rel = safe_dir.relative_to(output_root)

                pair_input_hash = _compute_pair_input_hash(
                    safe_case=safe_case,
                    unsafe_case=unsafe_case,
                    tool_library_hash=tool_library_hash,
                    defense_slug=defense_slug,
                    model_slug=model_name,
                )
                safe_input_hash = _compute_safe_input_hash(
                    safe_case=safe_case,
                    tool_library_hash=tool_library_hash,
                    defense_slug=defense_slug,
                    model_slug=model_name,
                )
                key = _entry_key(defense_slug, _sanitize_slug(model_name), domain_slug, pair_run_id)
                safe_key = _safe_entry_key(defense_slug, _sanitize_slug(model_name), domain_slug, pair_base_id)
                with combo_lock:
                    existing = run_index.get("entries", {}).get(key, {})
                    safe_lock = safe_locks.setdefault(safe_key, threading.Lock())
                existing_manifest = _read_json(pair_dir / "manifest.json", {})
                completed = _manifest_complete(pair_dir, existing_manifest, expected_hash=pair_input_hash)

                should_run = True
                if force_rerun:
                    should_run = True
                elif rerun_failed:
                    should_run = bool(existing.get("status") == "failed")
                elif resume:
                    should_run = not completed

                if not should_run:
                    return {"ran": False, "failed": False, "skipped": True}

                with combo_lock:
                    run_index.setdefault("entries", {})[key] = {
                        "defense": defense_slug,
                        "model": model_name,
                        "domain": domain,
                        "pair_base_id": pair_base_id,
                        "pair_run_id": pair_run_id,
                        "unsafe_variant_id": unsafe_variant_id,
                        "pair_dir": str(pair_dir_rel).replace("\\", "/"),
                        "status": "running",
                        "input_hash": pair_input_hash,
                        "tool_library_hash": tool_library_hash,
                        "updated_at": _utcnow_iso(),
                    }
                    _save_run_index(output_root, run_index)

                pair_dir.mkdir(parents=True, exist_ok=True)
                case_artifacts: dict[str, dict[str, Any]] = {}
                pair_rows: list[dict[str, Any]] = []
                pair_failed = False
                error_messages: list[str] = []

                def _register_safe_cache_from_pair() -> dict[str, Any] | None:
                    if not _manifest_complete(pair_dir, existing_manifest):
                        return None
                    if str(existing_manifest.get("tool_library_hash", "")) != tool_library_hash:
                        return None
                    if str(existing_manifest.get("defense", "")) != defense_slug:
                        return None
                    if str(existing_manifest.get("model", "")) != model_name:
                        return None
                    if str(existing_manifest.get("pair_base_id", "")) != pair_base_id:
                        return None
                    source_case = existing_manifest.get("cases", {}).get("safe", {})
                    if not isinstance(source_case, dict):
                        return None
                    safe_dir.mkdir(parents=True, exist_ok=True)
                    for suffix in (
                        "trace.json",
                        "eval.json",
                        "defense_artifacts.json",
                        "snapshot.json",
                        "dag.json",
                    ):
                        _copy_artifact_if_exists(pair_dir / f"safe.{suffix}", safe_dir / f"safe.{suffix}")
                    safe_artifact = dict(source_case)
                    safe_artifact["cache_dir"] = str(safe_dir_rel).replace("\\", "/")
                    safe_manifest = {
                        "layout_version": RESULTS_LAYOUT_VERSION,
                        "generated_at": _utcnow_iso(),
                        "defense": defense_slug,
                        "model": model_name,
                        "domain": domain,
                        "pair_base_id": pair_base_id,
                        "status": "completed",
                        "input_hash": safe_input_hash,
                        "tool_library_hash": tool_library_hash,
                        "case": safe_artifact,
                        "artifacts": sorted([p.name for p in safe_dir.iterdir() if p.is_file()]),
                        "source_pair_dir": str(pair_dir_rel).replace("\\", "/"),
                    }
                    _write_json(safe_dir / "manifest.json", safe_manifest)
                    return safe_artifact

                def _ensure_safe_cache() -> tuple[dict[str, Any], dict[str, Any], bool, list[str]]:
                    cache_errors: list[str] = []
                    safe_manifest = _read_json(safe_dir / "manifest.json", {})
                    if _safe_manifest_complete(safe_dir, safe_manifest, expected_hash=safe_input_hash):
                        cached_case = safe_manifest.get("case", {})
                        if isinstance(cached_case, dict):
                            return cached_case, _read_json(safe_dir / "safe.eval.json", {}), False, cache_errors

                    safe_dir.mkdir(parents=True, exist_ok=True)
                    with combo_lock:
                        run_index.setdefault("safe_entries", {})[safe_key] = {
                            "defense": defense_slug,
                            "model": model_name,
                            "domain": domain,
                            "pair_base_id": pair_base_id,
                            "safe_dir": str(safe_dir_rel).replace("\\", "/"),
                            "status": "running",
                            "input_hash": safe_input_hash,
                            "tool_library_hash": tool_library_hash,
                            "updated_at": _utcnow_iso(),
                        }
                        _save_run_index(output_root, run_index)

                    reused_artifact = _register_safe_cache_from_pair()
                    if reused_artifact is not None:
                        with combo_lock:
                            run_index.setdefault("safe_entries", {})[safe_key] = {
                                "defense": defense_slug,
                                "model": model_name,
                                "domain": domain,
                                "pair_base_id": pair_base_id,
                                "safe_dir": str(safe_dir_rel).replace("\\", "/"),
                                "status": "completed",
                                "input_hash": safe_input_hash,
                                "tool_library_hash": tool_library_hash,
                                "updated_at": _utcnow_iso(),
                            }
                            _save_run_index(output_root, run_index)
                        return reused_artifact, _read_json(safe_dir / "safe.eval.json", {}), False, cache_errors

                    execution = _execute_case_once(
                        case=safe_case,
                        llm_name=model,
                        defense_config=defense_cfg,
                        tool_library_path=tool_library_path,
                        backend_kwargs=backend_kwargs,
                        max_iterations=max_iterations,
                        adaptive_iterations=adaptive_iterations,
                        adaptive_multiplier=adaptive_multiplier,
                        adaptive_min=adaptive_min,
                        adaptive_max=adaptive_max,
                        verbose=verbose,
                    )
                    eval_result = execution.get("eval_result")
                    eval_payload = eval_result.to_dict() if eval_result is not None else {}
                    trace_payload = execution.get("trace_payload", {})
                    ipiguard_dag = execution.get("ipiguard_dag")
                    defense_artifacts = execution.get("defense_artifacts", {})
                    initial_snapshot = execution.get("initial_snapshot", {})
                    final_snapshot = execution.get("final_snapshot", {})

                    _write_json(safe_dir / "safe.trace.json", trace_payload)
                    _write_json(safe_dir / "safe.eval.json", eval_payload)
                    _write_json(safe_dir / "safe.defense_artifacts.json", defense_artifacts)
                    _write_json(
                        safe_dir / "safe.snapshot.json",
                        {
                            "initial_snapshot": initial_snapshot,
                            "final_snapshot": final_snapshot,
                        },
                    )
                    dag_file = ""
                    if ipiguard_dag is not None:
                        dag_file = "safe.dag.json"
                        _write_json(safe_dir / dag_file, ipiguard_dag)

                    safe_artifact = _build_case_artifact_record(
                        case=safe_case,
                        variant="safe",
                        status="completed" if execution.get("ok") else "failed",
                        error_message=str(execution.get("error_message", "")),
                        dag_file=dag_file,
                        cache_dir=str(safe_dir_rel).replace("\\", "/"),
                    )
                    if execution.get("error_message"):
                        cache_errors.append(str(execution.get("error_message")))
                    safe_status = "failed" if not execution.get("ok") else "completed"
                    safe_manifest = {
                        "layout_version": RESULTS_LAYOUT_VERSION,
                        "generated_at": _utcnow_iso(),
                        "defense": defense_slug,
                        "model": model_name,
                        "domain": domain,
                        "pair_base_id": pair_base_id,
                        "status": safe_status,
                        "input_hash": safe_input_hash,
                        "tool_library_hash": tool_library_hash,
                        "case": safe_artifact,
                        "artifacts": sorted([p.name for p in safe_dir.iterdir() if p.is_file()]),
                    }
                    _write_json(safe_dir / "manifest.json", safe_manifest)
                    with combo_lock:
                        run_index.setdefault("safe_entries", {})[safe_key] = {
                            "defense": defense_slug,
                            "model": model_name,
                            "domain": domain,
                            "pair_base_id": pair_base_id,
                            "safe_dir": str(safe_dir_rel).replace("\\", "/"),
                            "status": safe_status,
                            "input_hash": safe_input_hash,
                            "tool_library_hash": tool_library_hash,
                            "updated_at": _utcnow_iso(),
                        }
                        _save_run_index(output_root, run_index)
                    return safe_artifact, eval_payload, not execution.get("ok"), cache_errors

                with safe_lock:
                    safe_artifact, safe_eval_payload, safe_failed, safe_errors = _ensure_safe_cache()

                for suffix in (
                    "trace.json",
                    "eval.json",
                    "defense_artifacts.json",
                    "snapshot.json",
                    "dag.json",
                ):
                    _copy_artifact_if_exists(safe_dir / f"safe.{suffix}", pair_dir / f"safe.{suffix}")
                case_artifacts["safe"] = dict(safe_artifact)
                pair_failed = pair_failed or bool(safe_failed)
                error_messages.extend(safe_errors)
                pair_rows.append(
                    _build_case_row(
                        defense=defense_slug,
                        model=model_name,
                        domain=domain,
                        pair_base_id=pair_base_id,
                        pair_run_id=pair_run_id,
                        case_obj=safe_case,
                        eval_payload=safe_eval_payload,
                    )
                )

                for case in (unsafe_case,):
                    variant = _case_variant(case)
                    execution = _execute_case_once(
                        case=case,
                        llm_name=model,
                        defense_config=defense_cfg,
                        tool_library_path=tool_library_path,
                        backend_kwargs=backend_kwargs,
                        max_iterations=max_iterations,
                        adaptive_iterations=adaptive_iterations,
                        adaptive_multiplier=adaptive_multiplier,
                        adaptive_min=adaptive_min,
                        adaptive_max=adaptive_max,
                        verbose=verbose,
                    )
                    eval_result = execution.get("eval_result")
                    eval_payload = eval_result.to_dict() if eval_result is not None else {}
                    trace_payload = execution.get("trace_payload", {})
                    ipiguard_dag = execution.get("ipiguard_dag")
                    defense_artifacts = execution.get("defense_artifacts", {})
                    initial_snapshot = execution.get("initial_snapshot", {})
                    final_snapshot = execution.get("final_snapshot", {})

                    _write_json(pair_dir / f"{variant}.trace.json", trace_payload)
                    _write_json(pair_dir / f"{variant}.eval.json", eval_payload)
                    _write_json(pair_dir / f"{variant}.defense_artifacts.json", defense_artifacts)
                    _write_json(
                        pair_dir / f"{variant}.snapshot.json",
                        {
                            "initial_snapshot": initial_snapshot,
                            "final_snapshot": final_snapshot,
                        },
                    )
                    if ipiguard_dag is not None:
                        _write_json(pair_dir / f"{variant}.dag.json", ipiguard_dag)

                    case_artifacts[variant] = _build_case_artifact_record(
                        case=case,
                        variant=variant,
                        status="completed" if execution.get("ok") else "failed",
                        error_message=str(execution.get("error_message", "")),
                        dag_file=f"{variant}.dag.json" if ipiguard_dag is not None else "",
                    )
                    if not execution.get("ok"):
                        pair_failed = True
                    if execution.get("error_message"):
                        error_messages.append(str(execution.get("error_message")))
                    pair_rows.append(
                        _build_case_row(
                            defense=defense_slug,
                            model=model_name,
                            domain=domain,
                            pair_base_id=pair_base_id,
                            pair_run_id=pair_run_id,
                            case_obj=case,
                            eval_payload=eval_payload,
                        )
                    )

                pair_metrics = _compute_metrics_from_rows(pair_rows)
                pair_summary = {
                    "layout_version": RESULTS_LAYOUT_VERSION,
                    "generated_at": _utcnow_iso(),
                    "defense": defense_slug,
                    "model": model_name,
                    "domain": domain,
                    "pair_base_id": pair_base_id,
                    "pair_run_id": pair_run_id,
                    "unsafe_variant_id": unsafe_variant_id,
                    "metrics": pair_metrics,
                    "cases": case_artifacts,
                    "status": "failed" if pair_failed else "completed",
                    "errors": error_messages,
                }
                _write_json(pair_dir / "pair_summary.json", pair_summary)

                manifest = {
                    "layout_version": RESULTS_LAYOUT_VERSION,
                    "generated_at": _utcnow_iso(),
                    "defense": defense_slug,
                    "model": model_name,
                    "domain": domain,
                    "pair_base_id": pair_base_id,
                    "pair_run_id": pair_run_id,
                    "unsafe_variant_id": unsafe_variant_id,
                    "status": "failed" if pair_failed else "completed",
                    "input_hash": pair_input_hash,
                    "tool_library_hash": tool_library_hash,
                    "cases": case_artifacts,
                    "artifacts": sorted(
                        [p.name for p in pair_dir.iterdir() if p.is_file()]
                    ),
                }
                _write_json(pair_dir / "manifest.json", manifest)

                with combo_lock:
                    run_index.setdefault("entries", {})[key] = {
                        "defense": defense_slug,
                        "model": model_name,
                        "domain": domain,
                        "pair_base_id": pair_base_id,
                        "pair_run_id": pair_run_id,
                        "unsafe_variant_id": unsafe_variant_id,
                        "pair_dir": str(pair_dir_rel).replace("\\", "/"),
                        "status": manifest["status"],
                        "input_hash": pair_input_hash,
                        "tool_library_hash": tool_library_hash,
                        "updated_at": _utcnow_iso(),
                    }
                    _save_run_index(output_root, run_index)

                return {
                    "ran": True,
                    "failed": manifest["status"] == "failed",
                    "skipped": False,
                }

            if actual_workers == 1:
                for pair_run_id in pair_ids:
                    result = _process_pair(pair_run_id)
                    if result is None:
                        continue
                    if result["skipped"]:
                        combo_skipped += 1
                        continue
                    combo_ran += 1
                    if result["failed"]:
                        combo_failed += 1
            else:
                with ThreadPoolExecutor(max_workers=actual_workers) as pool:
                    futures = {
                        pool.submit(_process_pair, pair_run_id): pair_run_id
                        for pair_run_id in pair_ids
                    }
                    for future in as_completed(futures):
                        result = future.result()
                        if result is None:
                            continue
                        if result["skipped"]:
                            combo_skipped += 1
                            continue
                        combo_ran += 1
                        if result["failed"]:
                            combo_failed += 1

            matrix_summary.append(
                {
                    "model": model_name,
                    "defense": defense_slug,
                    "ran_pairs": combo_ran,
                    "skipped_pairs": combo_skipped,
                    "failed_pairs": combo_failed,
                }
            )
            print(
                f"  | ran={combo_ran} skipped={combo_skipped} failed={combo_failed}"
            )

    _rebuild_reports(output_root, run_index)
    _save_run_index(output_root, run_index)
    summary_payload = {
        "generated_at": _utcnow_iso(),
        "layout_version": RESULTS_LAYOUT_VERSION,
        "matrix": matrix_summary,
        "models": [_path_safe_component(x) for x in models],
        "defenses": [_sanitize_slug(_defense_slug(_resolve_matrix_defense_config(x))) for x in defenses],
    }
    _write_json(output_root / "matrix_summary.json", summary_payload)
    return summary_payload


def main():
    parser = argparse.ArgumentParser(description="AgentDET evaluation CLI")
    parser.add_argument("--llm", type=str, help="LLM backend name")
    parser.add_argument("--defense", type=str, default="", help="Manual defense string joined by '+'")
    parser.add_argument(
        "--matrix-models",
        type=str,
        default="",
        help="Comma-separated model list for resumable matrix runs.",
    )
    parser.add_argument(
        "--matrix-defenses",
        type=str,
        default="",
        help="Comma-separated defense list (supports explicit combos like ipiguard+tool_filter).",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Skip already completed pair artifacts based on run_index + manifest hash (default: enabled).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume behavior and rerun all selected pairs.",
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="Only rerun entries recorded as failed in run_index.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force rerun regardless of run_index/manifest state.",
    )
    parser.add_argument("--config", type=str, help="YAML config path")
    parser.add_argument("--cases", type=str, default="data", help="Cases directory")
    parser.add_argument("--tools", type=str, default="config/tool_library.json", help="Tool library path")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=64,
        help="Max LLM turns per case (safety cap; actual turns remain model-driven)",
    )
    parser.add_argument(
        "--adaptive-iterations",
        action="store_true",
        help=(
            "Derive per-case max iterations from ground_truth length: "
            "ceil(len(ground_truth) * adaptive_multiplier), floored by "
            "adaptive-min-iterations and optionally capped by "
            "adaptive-max-iterations when that value is > 0."
        ),
    )
    parser.add_argument(
        "--adaptive-multiplier",
        type=float,
        default=4.0,
        help="Multiplier used by --adaptive-iterations (default: 4.0).",
    )
    parser.add_argument(
        "--adaptive-min-iterations",
        type=int,
        default=15,
        help="Lower bound used by --adaptive-iterations (default: 15).",
    )
    parser.add_argument(
        "--adaptive-max-iterations",
        type=int,
        default=0,
        help="Optional upper bound used by --adaptive-iterations; <= 0 disables the cap (default: 0).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel worker threads for case execution (default: 8)",
    )
    parser.add_argument(
        "--ask-threshold",
        type=int,
        default=None,
        help="Override planner ask_threshold for Gardener phase-2 scoring.",
    )
    parser.add_argument(
        "--ask-user-policy",
        choices=("allow", "reject", "oracle"),
        default=None,
        help="Resolve ask_user decisions for Gardener-style planners.",
    )
    parser.add_argument(
        "--leaf-strategy",
        choices=("global", "branch", "queue"),
        default=None,
        help="Select Gardener candidate-write planning strategy.",
    )
    parser.add_argument("--list-llms", action="store_true", help="List available backends")
    parser.add_argument("--list-defenses", action="store_true", help="List available defenses")
    parser.add_argument("--api-key", type=str, default="", help="OpenAI-compatible API key")
    parser.add_argument("--base-url", type=str, default="", help="OpenAI-compatible API base URL")
    parser.add_argument(
        "--save-traces",
        action="store_true",
        help="Save per-case full traces (messages/tool IO/state snapshots) under results/<llm>/<defense>/traces",
    )
    args = parser.parse_args()

    if args.list_llms:
        print("Available LLM backends:")
        for name in list_available_backends():
            print(f"  - {name}")
        return

    if args.list_defenses:
        print("Planner defenses:")
        print("  - ipiguard")
        print("  - gardener")
        print("  - rtbas  (dependency screening + IFC + confirmation; dual planner/runtime)")
        print("  - camel  (CLI shortcut: enables CaMeL planner + reasoning + runtime)")
        print("\nRuntime defenses:")
        print("  - fides")
        print("  - drift  (formal DRIFT planner + runtime validation)")
        print("  - tool_filter  (AgentDojo-aligned tool whitelist)")
        print("  - piguard         (HF PIGuard prompt-injection detector)")
        print("  - protectai       (ProtectAI prompt-injection detector)")
        print("  - camel  (CaMeL security policy + provenance tracking)")
        print("  - rtbas  (enabled together with the RTBAS planner)")
        print("  - autopermissions_hitl  (online permission prediction + user escalation)")
        print("\nHook defenses:")
        hooks = list_available_hooks()
        for layer, hook_list in hooks.items():
            print(f"{layer.upper()} layer:")
            for hook in hook_list:
                blocking = "blocking" if hook["is_blocking"] else "transform"
                model = " [requires model]" if hook["requires_model"] else ""
                print(f"  - {hook['name']}: {blocking}{model}")
            print()
        return

    matrix_models = _parse_csv_items(args.matrix_models)
    matrix_defenses = _parse_csv_items(args.matrix_defenses)
    matrix_mode = bool(matrix_models or matrix_defenses)
    if matrix_mode:
        planner_overrides: dict[str, Any] = {}
        if args.ask_threshold is not None:
            planner_overrides["ask_threshold"] = int(args.ask_threshold)
        if args.ask_user_policy is not None:
            planner_overrides["ask_user_policy"] = str(args.ask_user_policy)
        if args.leaf_strategy is not None:
            planner_overrides["leaf_strategy"] = str(args.leaf_strategy)
        if not matrix_models:
            raise ValueError("matrix mode requires --matrix-models")
        if not matrix_defenses:
            matrix_defenses = [args.defense or "no_defense"]
        if args.config:
            print("[warn] --config is ignored in matrix mode; using command-line matrix arguments.")
        cases_dir = _resolve_path(args.cases, must_exist=True)
        tool_library = _resolve_path(args.tools, must_exist=True)
        output_dir = _resolve_path(args.output, must_exist=False)
        backend_kwargs: dict[str, str] = {}
        if args.api_key:
            backend_kwargs["api_key"] = args.api_key
        if args.base_url:
            backend_kwargs["base_url"] = args.base_url

        print("=" * 50)
        print("AgentDET Matrix Evaluation (Resume)")
        print("=" * 50)
        print(f"Models: {matrix_models}")
        print(f"Defenses: {matrix_defenses}")
        print(f"Cases: {cases_dir}")
        print(f"Output: {output_dir}")
        print(f"Resume: {args.resume}")
        print(f"Rerun failed only: {bool(args.rerun_failed)}")
        print(f"Force rerun: {bool(args.force_rerun)}")
        print("=" * 50)

        matrix_payload = run_matrix_with_resume(
            models=matrix_models,
            defenses=matrix_defenses,
            cases_dir=cases_dir,
            tool_library_path=tool_library,
            output_dir=output_dir,
            backend_kwargs=backend_kwargs,
            max_iterations=args.max_iterations,
            adaptive_iterations=bool(args.adaptive_iterations),
            adaptive_multiplier=float(args.adaptive_multiplier),
            adaptive_min=int(args.adaptive_min_iterations),
            adaptive_max=int(args.adaptive_max_iterations),
            verbose=args.verbose,
            resume=bool(args.resume),
            rerun_failed=bool(args.rerun_failed),
            force_rerun=bool(args.force_rerun),
            workers=int(args.workers),
            planner_config_overrides=planner_overrides,
        )
        print(f"\n[done] Matrix summary saved: {output_dir}")
        print(json.dumps(matrix_payload, ensure_ascii=False, indent=2))
        return

    if args.config:
        config_path = _resolve_path(args.config, must_exist=True)
        config = load_config(config_path)
        config_base = Path(config_path).parent
        llms = config.get("llms", [args.llm or "gpt-4o"])
        configured_combos = config.get("defense_combinations", []) or []
        manual_defense = args.defense or (config.get("defense", "") or "")
        global_planner_cfg = config.get("planner_config", {}) or {}
        if args.ask_threshold is not None:
            global_planner_cfg = {**global_planner_cfg, "ask_threshold": int(args.ask_threshold)}
        if args.ask_user_policy is not None:
            global_planner_cfg = {
                **global_planner_cfg,
                "ask_user_policy": str(args.ask_user_policy),
            }
        if args.leaf_strategy is not None:
            global_planner_cfg = {
                **global_planner_cfg,
                "leaf_strategy": str(args.leaf_strategy),
            }
        global_runtime_cfg = config.get("runtime_config", {}) or {}
        if configured_combos:
            defense_combos = []
            for combo in configured_combos:
                merged = dict(combo)
                if "planner_config" not in merged:
                    merged["planner_config"] = global_planner_cfg
                if "runtime_config" not in merged:
                    merged["runtime_config"] = global_runtime_cfg
                defense_combos.append(merged)
        else:
            defense_config = parse_defense_string(manual_defense)
            defense_combos = [
                {
                    "name": manual_defense or "no_defense",
                    **defense_config,
                    "planner_config": global_planner_cfg,
                    "runtime_config": global_runtime_cfg,
                }
            ]
        cases_dir = _resolve_path(config.get("cases", args.cases), base_dir=config_base, must_exist=True)
        tool_library = _resolve_path(config.get("tools", args.tools), base_dir=config_base, must_exist=True)
        output_dir = _resolve_path(config.get("output_dir", args.output), base_dir=config_base, must_exist=False)
    else:
        llms = [args.llm] if args.llm else ["gpt-4o"]
        defense_config = parse_defense_string(args.defense)
        if (
            args.ask_threshold is not None
            or args.ask_user_policy is not None
            or args.leaf_strategy is not None
        ):
            planner_config = dict(defense_config.get("planner_config", {}) or {})
            if args.ask_threshold is not None:
                planner_config["ask_threshold"] = int(args.ask_threshold)
            if args.ask_user_policy is not None:
                planner_config["ask_user_policy"] = str(args.ask_user_policy)
            if args.leaf_strategy is not None:
                planner_config["leaf_strategy"] = str(args.leaf_strategy)
            defense_config["planner_config"] = planner_config
        defense_combos = [{"name": args.defense or "no_defense", **defense_config}]
        cases_dir = _resolve_path(args.cases, must_exist=True)
        tool_library = _resolve_path(args.tools, must_exist=True)
        output_dir = _resolve_path(args.output, must_exist=False)

    print("=" * 50)
    print("AgentDET Evaluation")
    print("=" * 50)
    print(f"LLMs: {llms}")
    print(f"Defense combinations: {len(defense_combos)}")
    print(f"Cases: {cases_dir}")
    print(f"Output: {output_dir}")
    if args.adaptive_iterations:
        print(
            "Adaptive iterations: "
            f"enabled (multiplier={args.adaptive_multiplier}, "
            f"min={args.adaptive_min_iterations}, max={args.adaptive_max_iterations})"
        )
    else:
        print(f"Adaptive iterations: disabled (fixed max_iterations={args.max_iterations})")
    print("=" * 50)

    all_summaries = []
    backend_kwargs: dict[str, str] = {}
    if args.api_key:
        backend_kwargs["api_key"] = args.api_key
    if args.base_url:
        backend_kwargs["base_url"] = args.base_url

    for llm in llms:
        print(f"\n> {llm}")
        for combo in defense_combos:
            combo_name = combo.get("name", "unknown")
            defense_config = {
                "planner": combo.get("planner", ""),
                "runtime": combo.get("runtime", ""),
                "input": combo.get("input", []),
                "plan": combo.get("plan", []),
                "reasoning": combo.get("reasoning", []),
                "action": combo.get("action", []),
                "finish": combo.get("finish", []),
                "planner_config": combo.get("planner_config", {}),
                "runtime_config": combo.get("runtime_config", {}),
            }
            print(f"  + defense: {combo_name}")
            for warning in collect_defense_config_warnings(defense_config):
                print(f"    ! config warning: {warning}")
            try:
                summary = run_evaluation(
                    llm_name=llm,
                    defense_config=defense_config,
                    cases_dir=cases_dir,
                    tool_library_path=tool_library,
                    output_dir=output_dir,
                    verbose=args.verbose,
                    backend_kwargs=backend_kwargs,
                    max_iterations=args.max_iterations,
                    workers=args.workers,
                    save_traces=args.save_traces,
                    adaptive_iterations=bool(args.adaptive_iterations),
                    adaptive_multiplier=float(args.adaptive_multiplier),
                    adaptive_min=int(args.adaptive_min_iterations),
                    adaptive_max=int(args.adaptive_max_iterations),
                    resume=bool(args.resume),
                )
                print(
                    "  |  "
                    f"BU={summary.bu:.2%} UA={summary.ua:.2%} "
                    f"ASR={summary.asr:.2%} uASR={summary.uasr:.2%} "
                    f"SUA={summary.sua:.2%} FPR={summary.fpr:.2%} TPR={summary.tpr:.2%} "
                    f"SES={summary.avg_ses:.2f} SEV={summary.sev:.2f} "
                    f"TCS={summary.avg_tcs:.2%}\n"
                    "  |  [v2] "
                    f"DR={summary.disruption_rate:.2%} "
                    f"TAR={summary.total_attack_success_rate:.2%} DSR={summary.defense_success_rate:.2%}"
                )
                all_summaries.append({"llm": llm, "defense": combo_name, **summary.to_dict()})
            except Exception as e:
                print(f"  |  error: {e}")

    if all_summaries:
        save_results(all_summaries, os.path.join(output_dir, "final_summary.json"))
        print(f"\n[done] Saved to {output_dir}")


if __name__ == "__main__":
    main()
