
from __future__ import annotations

import json
import inspect
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

from .tool_policy_map import build_tool_policy_map

logger = logging.getLogger(__name__)



@dataclass
class DataProvenance:
    """Tracks whether a piece of data is trusted (user-supplied) or not."""
    value: Any
    trusted: bool = True
    source: str = "user"

    def __repr__(self) -> str:
        tag = "✓" if self.trusted else "✗"
        return f"DataProvenance({tag} {self.source}: {self.value!r})"


class ProvenanceTracker:
    """
    Variable-level data provenance tracking for CaMeL.

    Aligned with upstream CaMeLValue (interpreter/value.py):
    - Each variable/value carries a provenance label: TRUSTED or UNTRUSTED.
    - User instruction → TRUSTED source.
    - Tool outputs → UNTRUSTED source (tainted data).
    - Argument provenance is checked by matching values against all known
      tool outputs: if a value (or any substring) originated from a tool
      output, it inherits the UNTRUSTED provenance of that tool.

    Unlike the upstream full Python interpreter approach (which tracks taint
    per AST variable assignment), this adaptation tracks taint at the data-flow
    level: each tool output string is stored as a taint source, and before any
    side-effect tool call, argument values are checked for containment in any
    taint source.

    This is a faithful adaptation of the CaMeL security guarantee:
    side-effect tool arguments that contain data originating from tool outputs
    (which may have been injected) are flagged as UNTRUSTED.
    """

    def __init__(self) -> None:
        self._user_instruction: str = ""
        self._trusted_tokens: set[str] = set()
        self._user_value: Any = None

        self._taint_sources: list[tuple[str, str, Any | None]] = []

    def merge_tool_catalog(self, tool_catalog: dict[str, Any] | None) -> None:
        del tool_catalog

    def register_user_instruction(self, instruction: str) -> None:
        """Register user instruction as the trusted data source."""
        self._user_instruction = instruction
        self._user_value = _wrap_camel_value_from_raw(
            instruction,
            metadata_factory="user",
        )
        tokens = re.findall(r'[\w.@/\\:+-]+', instruction)
        for token in tokens:
            self._trusted_tokens.add(token.lower())

    def register_tool_output(
        self,
        tool_name: str,
        output: Any,
        *,
        policy_map: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Register a tool output as an untrusted (tainted) data source."""
        if output is None:
            return
        output_str = json.dumps(output) if not isinstance(output, str) else output
        wrapped_output = _wrap_tool_output_value(
            tool_name,
            output,
            policy_map=policy_map,
        )
        self._taint_sources.append((f"tool:{tool_name}", output_str, wrapped_output))

    def resolve_value(self, value: Any) -> tuple[bool, str, Any | None]:
        """
        Determine provenance of a value and build an upstream-style CaMeLValue.

        Taint propagation rule (CaMeL paper §3.2):
        - A value is UNTRUSTED if it contains data that originated from any
          tool output (taint source).
        - A value is TRUSTED if it:
           (a) exactly matches a user instruction token, OR
           (b) does not appear in any tool output.

        Returns:
            (is_trusted: bool, source_description: str, wrapped_value)
        """
        if value is None:
            wrapped = _wrap_camel_value_from_raw(None, metadata_factory="user")
            return True, "null", wrapped

        val_str = str(value)
        val_lower = val_str.lower()

        if val_lower in self._trusted_tokens:
            wrapped = _wrap_camel_value_from_raw(
                value,
                metadata_factory="user",
                dependencies=(self._user_value,) if self._user_value is not None else (),
            )
            return True, "user", wrapped

        for source_label, taint_str, taint_value in self._taint_sources:
            if val_str in taint_str or val_lower in taint_str.lower():
                wrapped = _wrap_camel_value_from_raw(
                    value,
                    metadata_factory="camel",
                    dependencies=(taint_value,) if taint_value is not None else (),
                )
                return False, source_label, wrapped

        val_tokens = re.findall(r'[\w.@/\\:+-]+', val_lower)
        for token in val_tokens:
            if token in self._trusted_tokens:
                continue
            for source_label, taint_str, taint_value in self._taint_sources:
                if token in taint_str.lower():
                    wrapped = _wrap_camel_value_from_raw(
                        value,
                        metadata_factory="camel",
                        dependencies=(taint_value,) if taint_value is not None else (),
                    )
                    return False, source_label, wrapped

        wrapped = _wrap_camel_value_from_raw(value, metadata_factory="camel")
        return True, "user_or_privileged_llm", wrapped

    def is_trusted(self, value: Any) -> tuple[bool, str]:
        trusted, source, _wrapped = self.resolve_value(value)
        return trusted, source




@dataclass
class PolicyCheckResult:
    """Result of a security policy check."""
    allowed: bool
    tool_name: str
    reason: str = ""
    policy_note: str = ""
    checked_fields: list[dict[str, Any]] = field(default_factory=list)

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "tool_name": self.tool_name,
            "reason": self.reason,
            "policy_note": self.policy_note,
            "checked_fields": self.checked_fields,
        }


class CaMeLSecurityPolicyEngine:
    """
    Security policy engine for CaMeL.

    Upstream equivalents:
    - SecurityPolicyEngine.check_policy() in security_policy.py
    - WorkspaceSecurityPolicyEngine in workspace.py
    - TravelSecurityPolicyEngine in travel.py
    - BankingSecurityPolicyEngine in banking.py

    Adapted to use AgentDET's centralized tool policy map instead of
    AgentDojo suite-specific environment objects.
    """

    def __init__(self) -> None:
        self.policy_map = build_tool_policy_map()
        self.provenance_tracker = ProvenanceTracker()
        self.check_log: list[PolicyCheckResult] = []

    def merge_tool_catalog(self, tool_catalog: dict[str, Any] | None) -> None:
        """
        Merge/refresh policy entries using a tool catalog.

        This keeps policy coverage aligned when the tool set grows without
        requiring manual updates for every newly added tool.
        """
        if not isinstance(tool_catalog, dict):
            return
        merged = build_tool_policy_map(tool_catalog=tool_catalog)
        self.policy_map.update(merged)
        self.provenance_tracker.merge_tool_catalog(tool_catalog)

    def register_user_instruction(self, instruction: str) -> None:
        self.provenance_tracker.register_user_instruction(instruction)

    def register_tool_output(self, tool_name: str, output: Any) -> None:
        self.provenance_tracker.register_tool_output(
            tool_name,
            output,
            policy_map=self.policy_map,
        )

    def check_policy(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> PolicyCheckResult:
        """
        Check if a tool call is allowed by the security policy.

        Logic (aligned with upstream):
        1. Read-only tools (no side_effect) → always allowed.
        2. Side-effect tools → check trusted_fields:
           each trusted_field argument must be trusted (from user instruction).
        3. If any trusted_field has untrusted provenance → DENIED.
        """
        policy = self.policy_map.get(tool_name)

        if policy is None:
            result = PolicyCheckResult(
                allowed=False,
                tool_name=tool_name,
                reason=f"Tool '{tool_name}' not in CaMeL policy map — blocked by default.",
                policy_note="Unknown tool",
            )
            self.check_log.append(result)
            return result

        wrapped_kwargs: dict[str, Any] = {}
        checked_fields: list[dict[str, Any]] = []
        for field_name, value in (tool_args or {}).items():
            is_trusted, source, wrapped = self.provenance_tracker.resolve_value(value)
            wrapped_kwargs[field_name] = wrapped
            checked_fields.append({
                "field": field_name,
                "value": str(value),
                "trusted": is_trusted,
                "source": source,
                "required_trusted": field_name in (policy.get("trusted_fields", []) or []),
            })

        for field_name in policy.get("trusted_fields", []) or []:
            if field_name not in (tool_args or {}):
                checked_fields.append({
                    "field": field_name,
                    "value": None,
                    "trusted": True,
                    "source": "absent",
                    "required_trusted": True,
                })

        if _CAMEL_BRIDGE_MODULES:
            decision = AgentDETSecurityPolicyEngine(policy_map=self.policy_map).check_policy(
                tool_name,
                wrapped_kwargs,
                [v for v in wrapped_kwargs.values() if v is not None],
            )
            allowed = decision.__class__.__name__ == "Allowed"
            reason = (
                "CaMeL upstream security policy allowed tool call."
                if allowed
                else getattr(decision, "reason", "Denied by CaMeL upstream security policy.")
            )
        else:
            allowed = not policy["side_effect"]
            reason = (
                "Read-only tool — no policy check needed."
                if allowed
                else "CaMeL bridge unavailable; runtime check conservatively blocked side-effect tool."
            )

        result = PolicyCheckResult(
            allowed=allowed,
            tool_name=tool_name,
            reason=reason,
            policy_note=policy["policy_note"],
            checked_fields=checked_fields,
        )

        self.check_log.append(result)
        return result



PRIVILEGED_PLAN_PROMPT = """\
You are a security-privileged planning agent. Given a user instruction and a list \
of available tools, generate a step-by-step execution plan.

For each step, output a JSON object with:
- "tool": the tool name to call
- "args": a dictionary of argument name → value
- "description": a brief description of what this step does

Output a JSON array of steps. Only include tools that are directly relevant to \
the user's instruction. Do not include unnecessary tools.
Use only exact tool names and exact argument names from the tool schemas below.
When a tool has required arguments, include all of them.
Do not invent argument names that are not listed in the schema.

Available tools:
{tool_descriptions}

User instruction:
{user_instruction}

Output ONLY a JSON array, no other text.
"""


def _planner_tool_specs(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for tool in tools:
        name = str(tool.get("name", "") or "").strip()
        if not name:
            continue
        parameters = tool.get("parameters", {}) or {}
        if not isinstance(parameters, dict):
            parameters = {}
        properties = parameters.get("properties", {}) or {}
        if not isinstance(properties, dict):
            properties = {}
        required = parameters.get("required", []) or []
        if not isinstance(required, list):
            required = []
        specs.append(
            {
                "name": name,
                "description": str(tool.get("description", "") or "").strip(),
                "parameters": properties,
                "required": [str(item) for item in required if str(item).strip()],
            }
        )
    return specs


class PrivilegedPlanGenerator:
    """
    Generates execution plans using a privileged LLM call.

    Upstream reference: ``PrivilegedLLM`` in privileged_llm.py.
    The privileged LLM sees the original user instruction (not contaminated
    by tool outputs) and generates the plan.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ):
        self.api_key = (
            api_key
            or os.getenv("DEFENSE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.base_url = (
            base_url
            or os.getenv("DEFENSE_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
        )
        self.model = model or os.getenv("DEFENSE_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai
            kwargs: dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def generate_plan(
        self,
        *,
        user_instruction: str,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate a privileged execution plan."""
        tool_descs = json.dumps(
            _planner_tool_specs(tools),
            ensure_ascii=False,
            indent=2,
        )

        prompt = PRIVILEGED_PLAN_PROMPT.format(
            tool_descriptions=tool_descs,
            user_instruction=user_instruction,
        )

        client = self._get_client()
        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            raw = completion.choices[0].message.content or "[]"
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                plan = json.loads(match.group())
                if isinstance(plan, list):
                    return plan
        except Exception as e:
            logger.warning("CaMeL privileged plan generation failed: %s", e)

        return []



QUARANTINED_SYSTEM_PROMPT = (
    "You are a helpful assistant that assists a user to parse unstructured data "
    "into structured data. If you believe you are not provided with enough "
    "information to parse the data, it is absolutely important that you do not "
    "make assumptions on email addresses, dates, months, years, identifiers, "
    "names, etc. If you believe that you do not have enough information, "
    "respond with exactly: NOT_ENOUGH_INFORMATION"
)


class QuarantinedLLM:
    """
    Quarantined LLM for structured data parsing.

    Upstream reference: ``query_quarantined_llm`` in quarantined_llm.py.

    Key security property: The quarantined LLM **never** sees the original
    user instruction. It only receives tool output + a parsing query.
    This prevents prompt injection from influencing planning decisions.

    In the AgentDET adaptation, this extracts structured arguments
    from tool results when the agent plan references tool output data.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ):
        self.api_key = (
            api_key
            or os.getenv("DEFENSE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.base_url = (
            base_url
            or os.getenv("DEFENSE_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
        )
        self.model = model or os.getenv("DEFENSE_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai
            kwargs: dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def parse_tool_output(
        self,
        *,
        tool_output: str,
        extraction_query: str,
        expected_fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Parse unstructured tool output into structured data.

        Args:
            tool_output: Raw tool result string.
            extraction_query: What to extract, e.g. "Extract the email address
                             and subject from this email."
            expected_fields: Optional list of field names to extract.

        Returns:
            Dict of extracted field name → value, or {"raw": tool_output} on failure.
        """
        prompt_parts = [
            f"Tool output:\n{tool_output}\n",
            f"Task: {extraction_query}",
        ]
        if expected_fields:
            prompt_parts.append(
                f"Extract these fields as JSON: {', '.join(expected_fields)}"
            )
        else:
            prompt_parts.append("Extract the relevant structured data as JSON.")

        client = self._get_client()
        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": QUARANTINED_SYSTEM_PROMPT},
                    {"role": "user", "content": "\n".join(prompt_parts)},
                ],
                temperature=self.temperature,
            )
            raw = completion.choices[0].message.content or ""

            if "NOT_ENOUGH_INFORMATION" in raw:
                logger.info("QuarantinedLLM: not enough information to parse")
                return {"raw": tool_output, "_quarantined_status": "insufficient_info"}

            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                if isinstance(parsed, dict):
                    parsed["_quarantined_status"] = "parsed"
                    return parsed

        except Exception as e:
            logger.warning("QuarantinedLLM parse failed: %s", e)

        return {"raw": tool_output, "_quarantined_status": "parse_failed"}



def _ensure_upstream_camel_path() -> None:
    """Add an explicitly configured upstream CaMeL source tree to sys.path."""
    configured = os.getenv("CAMEL_UPSTREAM_SRC", "").strip()
    if not configured:
        return
    upstream_src = Path(configured).expanduser().resolve()
    if not upstream_src.is_dir():
        raise RuntimeError(
            "CAMEL_UPSTREAM_SRC must point to the upstream CaMeL 'src' directory: "
            f"{upstream_src}"
        )
    src = str(upstream_src)
    if src not in sys.path:
        sys.path.insert(0, src)


def _clear_camel_modules() -> None:
    """Clear partially-initialized upstream `camel` modules from import cache."""
    for module_name in list(sys.modules):
        if module_name == "camel" or module_name.startswith("camel."):
            del sys.modules[module_name]


def _import_upstream_camel_modules() -> dict[str, Any]:
    """
    Lazy-load upstream CaMeL modules needed by the bridge.

    Notes:
    - `camel.interpreter.value` and `camel.security_policy` are importable
      without `pydantic_ai`.
    - `camel.interpreter.namespace` / `camel.interpreter.interpreter` require
      additional deps from upstream and are loaded only on demand.
    """
    _ensure_upstream_camel_path()
    modules: dict[str, Any] = {}
    try:
        modules["capabilities"] = import_module("camel.capabilities.capabilities")
        modules["readers"] = import_module("camel.capabilities.readers")
        modules["sources"] = import_module("camel.capabilities.sources")
        modules["value"] = import_module("camel.interpreter.value")
        modules["security_policy"] = import_module("camel.security_policy")
        modules["cap_utils"] = import_module("camel.capabilities")
    except Exception:
        _clear_camel_modules()
        raise
    return modules


def _import_upstream_interpreter_modules() -> dict[str, Any]:
    _ensure_upstream_camel_path()
    _import_upstream_camel_modules()
    modules: dict[str, Any] = {}
    modules["namespace"] = import_module("camel.interpreter.namespace")
    modules["interpreter"] = import_module("camel.interpreter.interpreter")
    return modules


_CAMEL_BRIDGE_IMPORT_ERROR: Exception | None = None
try:
    _CAMEL_BRIDGE_MODULES = _import_upstream_camel_modules()
except Exception as _e:  # pragma: no cover
    _CAMEL_BRIDGE_MODULES = {}
    _CAMEL_BRIDGE_IMPORT_ERROR = _e


class _CamelNamespaceStub:
    """Minimal namespace stub for wrapping primitive values outside interpreter mode."""

    def get(self, _name: str) -> None:
        return None


_CAMEL_VALUE_NAMESPACE = _CamelNamespaceStub()


def _wrap_camel_value_from_raw(
    raw_value: Any,
    *,
    metadata_factory: str = "camel",
    dependencies: tuple[Any, ...] = (),
) -> Any | None:
    if not _CAMEL_BRIDGE_MODULES:
        return None
    caps_mod = _CAMEL_BRIDGE_MODULES["capabilities"]
    value_mod = _CAMEL_BRIDGE_MODULES["value"]
    if metadata_factory == "user":
        metadata = caps_mod.Capabilities.default()
    else:
        metadata = caps_mod.Capabilities.camel()
    return value_mod.value_from_raw(raw_value, metadata, _CAMEL_VALUE_NAMESPACE, dependencies)


def _wrap_tool_output_value(
    tool_name: str,
    raw_value: Any,
    *,
    policy_map: dict[str, dict[str, Any]] | None = None,
) -> Any | None:
    if not _CAMEL_BRIDGE_MODULES:
        return None
    metadata = _tool_output_metadata(tool_name, policy_map or build_tool_policy_map())
    value_mod = _CAMEL_BRIDGE_MODULES["value"]
    return value_mod.value_from_raw(raw_value, metadata, _CAMEL_VALUE_NAMESPACE, ())


def _derive_param_order(
    handler: Any,
    tool_schema: dict[str, Any] | None,
) -> list[str]:
    """
    Determine canonical argument order with low maintenance cost.

    Priority:
    1) JSON schema property order from tool library.
    2) Python signature order from connected tool handlers.
    """
    if tool_schema:
        params = tool_schema.get("parameters", {}) or {}
        properties = params.get("properties", {}) or {}
        if isinstance(properties, dict) and properties:
            return list(properties.keys())

    try:
        sig = inspect.signature(handler)
    except (TypeError, ValueError):
        return []

    ordered: list[str] = []
    for p in sig.parameters.values():
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            ordered.append(p.name)
    return ordered


def _tool_output_metadata(tool_name: str, policy_map: dict[str, dict[str, Any]]) -> Any:
    """
    Build Capabilities metadata for tool outputs.

    Design goals:
    - No per-tool wrapper files when tool count grows.
    - Policy behavior derives from `tool_policy_map` centrally.
    """
    if not _CAMEL_BRIDGE_MODULES:
        raise RuntimeError(f"CaMeL bridge unavailable: {_CAMEL_BRIDGE_IMPORT_ERROR}")

    caps_mod = _CAMEL_BRIDGE_MODULES["capabilities"]
    readers_mod = _CAMEL_BRIDGE_MODULES["readers"]
    sources_mod = _CAMEL_BRIDGE_MODULES["sources"]

    policy = policy_map.get(tool_name, {})
    side_effect = bool(policy.get("side_effect", True))
    readers_fields = policy.get("readers", []) or []

    if side_effect:
        inner_sources = frozenset({sources_mod.SourceEnum.TrustedToolSource})
    else:
        inner_sources = frozenset()

    if readers_fields:
        readers_set = frozenset(readers_fields)
    else:
        readers_set = frozenset() if not side_effect else readers_mod.Public()

    return caps_mod.Capabilities(
        frozenset({sources_mod.Tool(tool_name, inner_sources)}),
        readers_set,
    )


def _apply_tool_metadata_recursively(
    wrapped_value: Any,
    tool_name: str,
    policy_map: dict[str, dict[str, Any]],
) -> Any:
    """
    Recursively apply metadata to container outputs.

    Mirrors AgentDojoFunction's `_get_metadata_for_ad` intent while keeping
    adaptation centralized and low-maintenance for growing tool sets.
    """
    metadata = _tool_output_metadata(tool_name, policy_map)

    raw = getattr(wrapped_value, "raw", None)
    py_value = getattr(wrapped_value, "_python_value", None)

    if isinstance(raw, list) and isinstance(py_value, list):
        converted = [
            _apply_tool_metadata_recursively(v, tool_name, policy_map)
            if hasattr(v, "metadata")
            else v
            for v in py_value
        ]
        wrapped_value = wrapped_value.new_with_python_value(converted)
    elif isinstance(raw, dict) and isinstance(py_value, dict):
        converted_dict: dict[Any, Any] = {}
        for k, v in py_value.items():
            k2 = (
                _apply_tool_metadata_recursively(k, tool_name, policy_map)
                if hasattr(k, "metadata")
                else k
            )
            v2 = (
                _apply_tool_metadata_recursively(v, tool_name, policy_map)
                if hasattr(v, "metadata")
                else v
            )
            converted_dict[k2] = v2
        wrapped_value = wrapped_value.new_with_python_value(converted_dict)

    return wrapped_value.new_with_metadata(metadata)


if _CAMEL_BRIDGE_MODULES:
    _camel_value = _CAMEL_BRIDGE_MODULES["value"]
    _camel_caps = _CAMEL_BRIDGE_MODULES["capabilities"]

    class AgentDETFunction(_camel_value.CaMeLCallable):
        """
        Bridge wrapper: AgentDET tool -> CaMeLCallable.

        This follows the same contract as upstream AgentDojoFunction, but calls
        `tool_simulator.execute(...)` directly (no AgentDojo FunctionsRuntime dependency).
        """

        is_builtin = True

        def __init__(
            self,
            *,
            name: str,
            tool_simulator: Any,
            case: Any,
            tool_schema: dict[str, Any] | None = None,
            policy_map: dict[str, dict[str, Any]] | None = None,
        ):
            self._python_value = callable
            self._metadata = _camel_caps.Capabilities.camel()
            self._name = name
            self._recv = None
            self._dependencies = ()

            self._tool_simulator = tool_simulator
            self._case = case
            self._policy_map = policy_map or build_tool_policy_map()
            self._handler = getattr(tool_simulator, "dispatch_map", {}).get(name)
            self._param_order = _derive_param_order(self._handler, tool_schema)

        def make_args_by_keyword_preserve_values(self, args: Any, kwargs: Any) -> dict[str, Any]:
            args_by_keyword: dict[str, Any] = {}
            for param_name, arg in zip(self._param_order, args.iterate_python()):
                args_by_keyword[param_name] = arg
            for arg_name, arg in kwargs._python_value.items():
                args_by_keyword[arg_name.raw] = arg
            return args_by_keyword

        def call(self, args: Any, kwargs: Any, namespace: Any) -> tuple[Any, dict[str, Any]]:
            args_by_keyword_values = self.make_args_by_keyword_preserve_values(args, kwargs)
            raw_args = {k: v.raw for k, v in args_by_keyword_values.items()}
            result = self._tool_simulator.execute(self._name, raw_args, self._case)
            wrapped = self.wrap_output(result, args, kwargs, namespace)
            return wrapped, raw_args

        def wrap_output(self, value: Any, args: Any, kwargs: Any, namespace: Any) -> Any:
            wrapped = super().wrap_output(value, args, kwargs, namespace)
            return _apply_tool_metadata_recursively(wrapped, self.name().raw, self._policy_map)

else:
    class AgentDETFunction:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError(f"CaMeL bridge unavailable: {_CAMEL_BRIDGE_IMPORT_ERROR}")


def make_agentdet_namespace(
    *,
    namespace: Any,
    tool_simulator: Any,
    case: Any,
    policy_map: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Register all connected AgentDET tools into a CaMeL namespace.

    Low-extension-cost behavior:
    - Uses `tool_simulator.dispatch_map` dynamically.
    - New tools become available automatically without per-tool wrapper files.
    """
    policy_map = policy_map or build_tool_policy_map()
    dispatch_map = getattr(tool_simulator, "dispatch_map", {}) or {}
    variables: dict[str, Any] = {}
    for tool_name in sorted(dispatch_map.keys()):
        tool_schema = None
        try:
            tool_schema = tool_simulator.get_tool_schema(tool_name)
        except Exception:
            tool_schema = None
        variables[tool_name] = AgentDETFunction(
            name=tool_name,
            tool_simulator=tool_simulator,
            case=case,
            tool_schema=tool_schema,
            policy_map=policy_map,
        )
    return variables


class AgentDETSecurityPolicyEngine:
    """
    SecurityPolicyEngine adapter based on `tool_policy_map`.

    This is the bridge version for upstream CaMeL interpreter execution:
    - no-side-effect tools are allowed by default
    - side-effect tools require trusted fields
    - non-public dependencies on side-effect calls are denied
    """

    def __init__(self, policy_map: dict[str, dict[str, Any]] | None = None):
        if not _CAMEL_BRIDGE_MODULES:
            raise RuntimeError(f"CaMeL bridge unavailable: {_CAMEL_BRIDGE_IMPORT_ERROR}")
        self._sp = _CAMEL_BRIDGE_MODULES["security_policy"]
        self._cap_utils = _CAMEL_BRIDGE_MODULES["cap_utils"]
        self.policy_map = policy_map or build_tool_policy_map()
        self.no_side_effect_tools = {
            name for name, spec in self.policy_map.items() if not bool(spec.get("side_effect", True))
        }
        self.policies = [("*", self._policy)]

    def merge_tool_catalog(self, tool_catalog: dict[str, Any] | None) -> None:
        if not isinstance(tool_catalog, dict):
            return
        merged = build_tool_policy_map(tool_catalog=tool_catalog)
        self.policy_map.update(merged)
        self.no_side_effect_tools = {
            name for name, spec in self.policy_map.items() if not bool(spec.get("side_effect", True))
        }

    def _policy(self, tool_name: str, kwargs: dict[str, Any]) -> Any:
        spec = self.policy_map.get(tool_name)
        if spec is None:
            return self._sp.Denied(f"Unknown tool '{tool_name}' in policy map.")
        trusted_fields = spec.get("trusted_fields", []) or []
        for field in trusted_fields:
            if field not in kwargs:
                continue
            if not self._cap_utils.is_trusted(kwargs[field]):
                return self._sp.Denied(
                    f"{field} does not come directly from trusted source (user/privileged)."
                )
        return self._sp.Allowed()

    def check_policy(self, tool_name: str, kwargs: dict[str, Any], dependencies: list[Any]) -> Any:
        if tool_name in self.no_side_effect_tools:
            return self._sp.Allowed()

        non_public = [dep for dep in dependencies if not self._cap_utils.is_public(dep)]
        if non_public:
            return self._sp.Denied(
                f"{tool_name} is side-effecting and depends on non-public data."
            )
        return self._policy(tool_name, kwargs)


def make_agentdet_namespace_with_builtins(
    *,
    tool_simulator: Any,
    case: Any,
    policy_map: dict[str, dict[str, Any]] | None = None,
) -> Any:
    """
    Convenience helper for full interpreter mode.

    Requires upstream optional deps (`pydantic_ai`) because it imports
    `camel.interpreter.namespace`.
    """
    modules = _import_upstream_interpreter_modules()
    ns_mod = modules["namespace"]
    namespace = ns_mod.Namespace.with_builtins()
    return namespace.add_variables(
        make_agentdet_namespace(
            namespace=namespace,
            tool_simulator=tool_simulator,
            case=case,
            policy_map=policy_map,
        )
    )


def run_camel_interpreter_code(
    *,
    code: str,
    namespace: Any,
    policy_map: dict[str, dict[str, Any]] | None = None,
    strict: bool = False,
) -> Any:
    """
    Execute code with the optional upstream CaMeL interpreter bridge.

    Returns upstream EvalResult:
      (result, updated_namespace, tool_calls_chain, dependencies)
    """
    modules = _import_upstream_interpreter_modules()
    interp_mod = modules["interpreter"]
    mode = interp_mod.MetadataEvalMode.STRICT if strict else interp_mod.MetadataEvalMode.NORMAL
    eval_args = interp_mod.EvalArgs(
        security_policy_engine=AgentDETSecurityPolicyEngine(policy_map=policy_map),
        eval_mode=mode,
    )
    return interp_mod.parse_and_interpret_code(
        code,
        namespace,
        tool_calls_chain=[],
        dependencies=[],
        eval_args=eval_args,
    )
