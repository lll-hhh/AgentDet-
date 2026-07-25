"""
Runtime policy plugins for the unified agent loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

from defenses.fides.adapter import (
    FidesRuntimeService,
    FidesTraceEntry,
    PolicyViolation,
    build_fides_runtime_service,
)

from defenses.drift.adapter import DRIFTRuntimeService, DRIFTFormalRuntimeService

from defenses.tool_filter.adapter import ToolFilterService

from defenses.camel.adapter import CaMeLSecurityPolicyEngine

from defenses.camel.tool_policy_map import build_tool_policy_map

from defenses.piguard.adapter import PIGuardService

from defenses.protectai.adapter import ProtectAIService


_PIGUARD_SERVICE_CACHE: dict[tuple[str, float, int, bool], PIGuardService] = {}
_PROTECTAI_SERVICE_CACHE: dict[tuple[str, float, bool, int], ProtectAIService] = {}
_PIGUARD_SERVICE_CACHE_LOCK = threading.Lock()
_PROTECTAI_SERVICE_CACHE_LOCK = threading.Lock()


def _get_cached_piguard_service(
    *,
    model_name: str,
    threshold: float,
    device: int,
    trust_remote_code: bool,
) -> PIGuardService:
    key = (str(model_name), float(threshold), int(device), bool(trust_remote_code))
    with _PIGUARD_SERVICE_CACHE_LOCK:
        service = _PIGUARD_SERVICE_CACHE.get(key)
        if service is None:
            service = PIGuardService(
                model_name=model_name,
                threshold=threshold,
                device=device,
                trust_remote_code=trust_remote_code,
            )
            _PIGUARD_SERVICE_CACHE[key] = service
        return service


def _get_cached_protectai_service(
    *,
    model_name: str,
    threshold: float,
    use_onnx: bool,
    device: int,
) -> ProtectAIService:
    key = (str(model_name), float(threshold), bool(use_onnx), int(device))
    with _PROTECTAI_SERVICE_CACHE_LOCK:
        service = _PROTECTAI_SERVICE_CACHE.get(key)
        if service is None:
            service = ProtectAIService(
                threshold=threshold,
                use_onnx=use_onnx,
                model_name=model_name,
                device=device,
            )
            _PROTECTAI_SERVICE_CACHE[key] = service
        return service


def _artifact_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _artifact_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_artifact_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return {
            str(k): _artifact_jsonable(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return str(value)


@dataclass
class RuntimePolicyDecision:
    allowed: bool = True
    skip: bool = False
    blocked_by: str = ""
    reason: str = ""


class RuntimePolicyPlugin:
    name = "none"

    def start(
        self,
        *,
        case: Any,
        state: Any,
        tools: list[dict[str, Any]],
        llm: Any,
    ) -> None:
        return None

    def before_tool_call(
        self,
        *,
        case: Any,
        state: Any,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> RuntimePolicyDecision:
        return RuntimePolicyDecision(allowed=True)

    def on_tool_result(
        self,
        *,
        case: Any,
        state: Any,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_output: Any,
        tool_content: str,
    ) -> None:
        return None

    def finish(self, *, case: Any, state: Any) -> None:
        return None

    def export_artifacts(self, *, state: Any) -> dict[str, Any]:
        return {"name": self.name}


class NoRuntimePolicyPlugin(RuntimePolicyPlugin):
    name = "none"


class FidesRuntimePolicyPlugin(RuntimePolicyPlugin):
    name = "fides"

    def __init__(
        self,
        *,
        service: FidesRuntimeService | None = None,
        tool_catalog: dict[str, Any] | None = None,
        extra_tool_names: Iterable[str] | None = None,
        force_sensitive_tools: Iterable[str] | None = None,
        force_read_only_tools: Iterable[str] | None = None,
        force_sink_tools: Iterable[str] | None = None,
    ):
        if service is not None:
            self.service = service
        else:
            self.service, _ = build_fides_runtime_service(
                tool_catalog=tool_catalog or {},
                extra_tool_names=extra_tool_names,
                force_sensitive_tools=force_sensitive_tools,
                force_read_only_tools=force_read_only_tools,
                force_sink_tools=force_sink_tools,
            )
        self.trace: list[FidesTraceEntry] = []
        self.current_label = self.service.initial_label()

    def start(
        self,
        *,
        case: Any,
        state: Any,
        tools: list[dict[str, Any]],
        llm: Any,
    ) -> None:
        self.trace = []
        self.current_label = self.service.initial_label()

        for msg in getattr(state, "messages", []):
            role = getattr(msg, "role", "")
            if role == "system":
                self.current_label = self.current_label.join(
                    self.service.label_from_source("system", getattr(msg, "content", ""))
                )
            elif role == "user":
                self.current_label = self.current_label.join(
                    self.service.label_from_source("user", getattr(msg, "content", ""))
                )

    def before_tool_call(
        self,
        *,
        case: Any,
        state: Any,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> RuntimePolicyDecision:
        try:
            self.service.check_tool_call(
                trace=self.trace,
                tool_name=tool_name,
                tool_args=tool_args,
                label=self.current_label,
            )
        except PolicyViolation as e:
            return RuntimePolicyDecision(
                allowed=False,
                blocked_by="fides_runtime",
                reason=f"Fides policy violation: {e.reason}",
            )
        return RuntimePolicyDecision(allowed=True)

    def on_tool_result(
        self,
        *,
        case: Any,
        state: Any,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_output: Any,
        tool_content: str,
    ) -> None:
        self.trace.append(
            FidesTraceEntry(
                tool_name=tool_name,
                tool_args=dict(tool_args),
                label=self.current_label,
            )
        )
        self.current_label = self.service.update_label_with_tool_output(
            current_label=self.current_label,
            tool_name=tool_name,
            tool_output=tool_output,
        )

    def export_artifacts(self, *, state: Any) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace": [_artifact_jsonable(entry) for entry in self.trace],
            "current_label": str(self.current_label),
        }


class DRIFTRuntimePolicyPlugin(RuntimePolicyPlugin):
    name = "drift"

    def __init__(
        self,
        *,
        service: DRIFTRuntimeService | None = None,
        enforce_trajectory: bool = True,
        enforce_checklist: bool = True,
        allow_read_only_deviation: bool = True,
        strict_arg_match: bool = True,
        block_side_effect_after_plan: bool = True,
        fail_closed_when_no_constraints: bool = False,
        injection_isolation: bool = True,
        use_llm_classification: bool = False,
        use_llm_alignment_judge: bool = False,
        use_llm_injection_detection: bool = False,
    ):
        self.service = service or DRIFTRuntimeService(
            enforce_trajectory=enforce_trajectory,
            enforce_checklist=enforce_checklist,
            allow_read_only_deviation=allow_read_only_deviation,
            strict_arg_match=strict_arg_match,
            block_side_effect_after_plan=block_side_effect_after_plan,
            fail_closed_when_no_constraints=fail_closed_when_no_constraints,
            injection_isolation=injection_isolation,
            use_llm_classification=use_llm_classification,
            use_llm_alignment_judge=use_llm_alignment_judge,
            use_llm_injection_detection=use_llm_injection_detection,
        )

    def start(
        self,
        *,
        case: Any,
        state: Any,
        tools: list[dict[str, Any]],
        llm: Any,
    ) -> None:
        self.service.initialize(case=case, tools=tools, llm=llm)
        setattr(
            state,
            "drift_initial_trajectory",
            list(self.service.state.planned_trajectory),
        )
        setattr(state, "drift_constraint_source", self.service.state.source)

    def before_tool_call(
        self,
        *,
        case: Any,
        state: Any,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> RuntimePolicyDecision:
        violation = self.service.validate_tool_call(
            tool_name=tool_name,
            tool_args=tool_args,
        )
        if violation is not None:
            return RuntimePolicyDecision(
                allowed=False,
                blocked_by="drift_runtime",
                reason=f"DRIFT validation failed: {violation.reason}",
            )
        return RuntimePolicyDecision(allowed=True)

    def on_tool_result(
        self,
        *,
        case: Any,
        state: Any,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_output: Any,
        tool_content: str,
    ) -> None:
        self.service.on_tool_executed(tool_name=tool_name)

        sanitized_content, removed_lines = self.service.sanitize_tool_output(tool_content)
        if removed_lines:
            if getattr(state, "messages", None):
                last_msg = state.messages[-1]
                if getattr(last_msg, "role", "") == "tool":
                    last_msg.content = sanitized_content
            removed_log = list(getattr(state, "drift_removed_injection_lines", []))
            removed_log.append(
                {
                    "tool_name": tool_name,
                    "removed_lines": removed_lines,
                }
            )
            setattr(state, "drift_removed_injection_lines", removed_log)

    def export_artifacts(self, *, state: Any) -> dict[str, Any]:
        return {
            "name": self.name,
            "constraint_state": _artifact_jsonable(self.service.state),
            "tool_permissions": dict(self.service.tool_permissions or {}),
            "removed_injection_lines": list(getattr(state, "drift_removed_injection_lines", []) or []),
            "initial_trajectory": list(getattr(state, "drift_initial_trajectory", []) or []),
            "constraint_source": str(getattr(state, "drift_constraint_source", "")),
        }


class DRIFTFormalRuntimePolicyPlugin(RuntimePolicyPlugin):
    name = "drift"

    def __init__(
        self,
        *,
        service: DRIFTFormalRuntimeService | None = None,
        enforce_trajectory: bool = True,
        enforce_checklist: bool = True,
        allow_read_only_deviation: bool = True,
        strict_arg_match: bool = True,
        block_side_effect_after_plan: bool = True,
        fail_closed_when_no_constraints: bool = False,
        injection_isolation: bool = True,
        use_llm_classification: bool = False,
        use_llm_alignment_judge: bool = False,
        use_llm_injection_detection: bool = False,
    ):
        self.service = service or DRIFTFormalRuntimeService(
            enforce_trajectory=enforce_trajectory,
            enforce_checklist=enforce_checklist,
            allow_read_only_deviation=allow_read_only_deviation,
            strict_arg_match=strict_arg_match,
            block_side_effect_after_plan=block_side_effect_after_plan,
            fail_closed_when_no_constraints=fail_closed_when_no_constraints,
            injection_isolation=injection_isolation,
            use_llm_classification=use_llm_classification,
            use_llm_alignment_judge=use_llm_alignment_judge,
            use_llm_injection_detection=use_llm_injection_detection,
        )

    def start(
        self,
        *,
        case: Any,
        state: Any,
        tools: list[dict[str, Any]],
        llm: Any,
    ) -> None:
        self.service.initialize(case=case, tools=tools, llm=llm, state=state)

    def before_tool_call(
        self,
        *,
        case: Any,
        state: Any,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> RuntimePolicyDecision:
        violation = self.service.validate_tool_call(
            tool_name=tool_name,
            tool_args=tool_args,
        )
        if violation is not None:
            return RuntimePolicyDecision(
                allowed=False,
                blocked_by="drift_runtime",
                reason=f"DRIFT validation failed: {violation.reason}",
            )
        return RuntimePolicyDecision(allowed=True)

    def on_tool_result(
        self,
        *,
        case: Any,
        state: Any,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_output: Any,
        tool_content: str,
    ) -> None:
        self.service.on_tool_executed(tool_name=tool_name)

    def export_artifacts(self, *, state: Any) -> dict[str, Any]:
        return {
            "name": self.name,
            "constraint_state": _artifact_jsonable(self.service.state),
            "tool_permissions": dict(self.service.tool_permissions or {}),
            "constraint_source": str(getattr(state, "drift_constraint_source", "")),
        }


class ToolFilterRuntimePolicyPlugin(RuntimePolicyPlugin):
    """
    AgentDojo-aligned tool filter as a runtime policy plugin.

    Upstream reference: ``OpenAILLMToolFilter`` in agentdojo (commit 462c88dd).

    Lifecycle:
    - ``start()``: call LLM once to build a task-relevant tool whitelist.
    - ``before_tool_call()``: block any tool not in the whitelist.
    """

    name = "tool_filter"

    def __init__(
        self,
        *,
        service: ToolFilterService | None = None,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ):
        if service is not None:
            self.service = service
        else:
            self.service = ToolFilterService(
                api_key=api_key,
                model=model,
                base_url=base_url,
                temperature=temperature,
            )
        self.allowed_tools: set[str] = set()
        self.raw_response: str = ""
        self.all_tool_names: list[str] = []

    def start(
        self,
        *,
        case: Any,
        state: Any,
        tools: list[dict[str, Any]],
        llm: Any,
    ) -> None:
        user_instruction = getattr(case, "user_instruction", "")
        system_message = getattr(case, "profile", "")

        result = self.service.generate_whitelist(
            user_instruction=user_instruction,
            tools=tools,
            system_message=system_message,
        )
        self.allowed_tools = result.allowed_tools
        self.raw_response = result.raw_response
        self.all_tool_names = result.all_tool_names

        setattr(state, "tool_filter_whitelist", sorted(self.allowed_tools))
        setattr(state, "tool_filter_all_tools", self.all_tool_names)
        setattr(state, "tool_filter_raw_response", self.raw_response)

    def before_tool_call(
        self,
        *,
        case: Any,
        state: Any,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> RuntimePolicyDecision:
        if not self.allowed_tools:
            return RuntimePolicyDecision(
                allowed=False,
                blocked_by="tool_filter",
                reason=(
                    "ToolFilter whitelist is empty (LLM generation failed "
                    "or returned no tools). Blocking all calls (fail-closed)."
                ),
            )

        if tool_name in self.allowed_tools:
            return RuntimePolicyDecision(allowed=True)

        return RuntimePolicyDecision(
            allowed=False,
            blocked_by="tool_filter",
            reason=(
                f"Tool '{tool_name}' not in AgentDojo-style whitelist. "
                f"Allowed: {sorted(self.allowed_tools)}"
            ),
        )

    def export_artifacts(self, *, state: Any) -> dict[str, Any]:
        return {
            "name": self.name,
            "allowed_tools": sorted(self.allowed_tools),
            "all_tool_names": list(self.all_tool_names or []),
            "raw_response": self.raw_response,
            "prompt": str(getattr(self.service, "prompt", "")),
            "model": str(getattr(self.service, "model", "")),
            "temperature": float(getattr(self.service, "temperature", 0.0) or 0.0),
        }


class _ClassifierBasedPIRuntimePolicyPlugin(RuntimePolicyPlugin):
    """
    Shared runtime logic for classifier-based prompt-injection defenses.

    Behavior:
    - Detect injection in user instruction at `start()`.
    - Optionally continue scanning tool outputs.
    - Once detected, block subsequent side-effect calls (or all calls).
    """

    name = "pi_detector"

    def __init__(
        self,
        *,
        service: Any,
        blocked_by_name: str,
        block_side_effect_only: bool = True,
        scan_tool_outputs: bool = True,
    ):
        self.service = service
        self.blocked_by_name = str(blocked_by_name)
        self.block_side_effect_only = bool(block_side_effect_only)
        self.scan_tool_outputs = bool(scan_tool_outputs)
        self.side_effect_tools: set[str] = set()
        self.detected: bool = False
        self.detections: list[dict[str, Any]] = []

    def _record_detection(self, *, source: str, text: str) -> None:
        result = self.service.classify(text)
        if not getattr(result, "is_injection", False):
            return
        self.detected = True
        self.detections.append(
            {
                "source": source,
                "score": float(getattr(result, "score", 0.0) or 0.0),
                "label": str(getattr(result, "label", "") or ""),
            }
        )

    def start(
        self,
        *,
        case: Any,
        state: Any,
        tools: list[dict[str, Any]],
        llm: Any,
    ) -> None:
        tool_catalog = {
            "tools": {
                t.get("name"): t
                for t in (tools or [])
                if isinstance(t, dict) and isinstance(t.get("name"), str) and t.get("name")
            }
        }
        policy_map = build_tool_policy_map(tool_catalog=tool_catalog)
        self.side_effect_tools = {
            name for name, spec in policy_map.items() if bool(spec.get("side_effect", True))
        }
        self.detected = False
        self.detections = []
        self._record_detection(
            source="user_instruction",
            text=str(getattr(case, "user_instruction", "") or ""),
        )
        setattr(state, f"{self.blocked_by_name}_detected", self.detected)
        setattr(state, f"{self.blocked_by_name}_detections", list(self.detections))

    def before_tool_call(
        self,
        *,
        case: Any,
        state: Any,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> RuntimePolicyDecision:
        if not self.detected:
            return RuntimePolicyDecision(allowed=True)
        if self.block_side_effect_only and tool_name not in self.side_effect_tools:
            return RuntimePolicyDecision(allowed=True)
        return RuntimePolicyDecision(
            allowed=False,
            blocked_by=self.blocked_by_name,
            reason=(
                f"{self.blocked_by_name}: prompt-injection signal detected; "
                f"blocking tool '{tool_name}'."
            ),
        )

    def on_tool_result(
        self,
        *,
        case: Any,
        state: Any,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_output: Any,
        tool_content: str,
    ) -> None:
        if not self.scan_tool_outputs:
            return
        if self.detected and not self.block_side_effect_only:
            return
        self._record_detection(source=f"tool:{tool_name}", text=tool_content or "")
        setattr(state, f"{self.blocked_by_name}_detected", self.detected)
        setattr(state, f"{self.blocked_by_name}_detections", list(self.detections))

    def export_artifacts(self, *, state: Any) -> dict[str, Any]:
        return {
            "name": self.name,
            "blocked_by_name": self.blocked_by_name,
            "detected": self.detected,
            "detections": list(self.detections or []),
            "side_effect_tools": sorted(self.side_effect_tools),
            "block_side_effect_only": self.block_side_effect_only,
            "scan_tool_outputs": self.scan_tool_outputs,
            "service": {
                "class": type(self.service).__name__,
                "model_name": str(getattr(self.service, "model_name", "")),
                "threshold": getattr(self.service, "threshold", None),
                "device": getattr(self.service, "device", None),
                "use_onnx": getattr(self.service, "use_onnx", None),
            },
        }


class PIGuardRuntimePolicyPlugin(_ClassifierBasedPIRuntimePolicyPlugin):
    """Runtime policy backed by `leolee99/PIGuard`."""

    name = "piguard"

    def __init__(
        self,
        *,
        service: PIGuardService | None = None,
        model_name: str = "leolee99/PIGuard",
        threshold: float = 0.5,
        device: int = -1,
        trust_remote_code: bool = True,
        block_side_effect_only: bool = True,
        scan_tool_outputs: bool = True,
    ):
        super().__init__(
            service=service
            or _get_cached_piguard_service(
                model_name=model_name,
                threshold=threshold,
                device=device,
                trust_remote_code=trust_remote_code,
            ),
            blocked_by_name="piguard",
            block_side_effect_only=block_side_effect_only,
            scan_tool_outputs=scan_tool_outputs,
        )


class ProtectAIRuntimePolicyPlugin(_ClassifierBasedPIRuntimePolicyPlugin):
    """Runtime policy backed by ProtectAI prompt-injection detectors."""

    name = "protectai"

    def __init__(
        self,
        *,
        service: ProtectAIService | None = None,
        threshold: float = 0.92,
        use_onnx: bool = False,
        model_name: str = "protectai/deberta-v3-base-prompt-injection-v2",
        device: int = -1,
        block_side_effect_only: bool = True,
        scan_tool_outputs: bool = True,
    ):
        super().__init__(
            service=service
            or _get_cached_protectai_service(
                threshold=threshold,
                use_onnx=use_onnx,
                model_name=model_name,
                device=device,
            ),
            blocked_by_name="protectai",
            block_side_effect_only=block_side_effect_only,
            scan_tool_outputs=scan_tool_outputs,
        )


class CaMeLRuntimePolicyPlugin(RuntimePolicyPlugin):
    """
    CaMeL security policy as a runtime policy plugin.

    Upstream reference: ``SecurityPolicyEngine`` + ``CaMeLValue`` provenance.

    Lifecycle:
    - ``start()``: register user instruction as trusted data source.
    - ``before_tool_call()``: check provenance of arguments for side-effect tools.
    - ``after_tool_call()``: register tool output as untrusted data source.
    """

    name = "camel"

    def __init__(self, *, engine: CaMeLSecurityPolicyEngine | None = None):
        self.engine = engine or CaMeLSecurityPolicyEngine()

    def start(
        self,
        *,
        case: Any,
        state: Any,
        tools: list[dict[str, Any]],
        llm: Any,
    ) -> None:
        user_instruction = getattr(case, "user_instruction", "")
        tool_catalog = {
            "tools": {
                t.get("name"): t
                for t in (tools or [])
                if isinstance(t, dict) and isinstance(t.get("name"), str) and t.get("name")
            }
        }
        self.engine.merge_tool_catalog(tool_catalog)
        self.engine.register_user_instruction(user_instruction)

        side_effect_tools = [
            name for name, p in self.engine.policy_map.items() if p["side_effect"]
        ]
        read_only_tools = [
            name for name, p in self.engine.policy_map.items() if not p["side_effect"]
        ]
        setattr(state, "camel_side_effect_tools", sorted(side_effect_tools))
        setattr(state, "camel_read_only_tools", sorted(read_only_tools))
        setattr(state, "camel_policy_checks", [])

    def before_tool_call(
        self,
        *,
        case: Any,
        state: Any,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> RuntimePolicyDecision:
        result = self.engine.check_policy(tool_name, tool_args)

        checks = getattr(state, "camel_policy_checks", [])
        checks.append(result.to_log_dict())
        setattr(state, "camel_policy_checks", checks)

        if result.allowed:
            return RuntimePolicyDecision(allowed=True)

        return RuntimePolicyDecision(
            allowed=False,
            blocked_by="camel",
            reason=result.reason,
        )

    def on_tool_result(
        self,
        *,
        case: Any,
        state: Any,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_output: Any,
        tool_content: str,
    ) -> None:
        """Register tool output as untrusted data for provenance tracking."""
        self.engine.register_tool_output(tool_name, tool_output)

    def export_artifacts(self, *, state: Any) -> dict[str, Any]:
        return {
            "name": self.name,
            "side_effect_tools": list(getattr(state, "camel_side_effect_tools", []) or []),
            "read_only_tools": list(getattr(state, "camel_read_only_tools", []) or []),
            "policy_checks": list(getattr(state, "camel_policy_checks", []) or []),
            "engine_check_log": _artifact_jsonable(getattr(self.engine, "check_log", [])),
            "policy_map": _artifact_jsonable(getattr(self.engine, "policy_map", {})),
        }
