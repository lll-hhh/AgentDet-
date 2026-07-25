"""
IPIGuard adapter for AgentDET.

This adapter loads the upstream AgentDojo IPIGuard implementation externally:
- `OpenAIConstructLLM` for DAG construction
- `OpenAITraverseLLM` for runtime argument update

Tool allow/deny grouping is loaded from the single active config file:
`config/ipiguard_tool_groups.json`.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import openai
from networkx import topological_sort
from pydantic import BaseModel, ConfigDict, create_model

def _resolve_upstream_source() -> Path:
    raw_path = str(os.getenv("IPIGUARD_UPSTREAM_SRC", "") or "").strip()
    if not raw_path:
        raise RuntimeError(
            "IPIGuard is not bundled with AgentDET. Clone the upstream IPIGuard "
            "artifact and set IPIGUARD_UPSTREAM_SRC to the directory containing "
            "the agentdojo package. See defenses/ipiguard/README.md."
        )

    source_path = Path(raw_path).expanduser().resolve()
    required_files = (
        source_path / "agentdojo" / "functions_runtime.py",
        source_path
        / "agentdojo"
        / "agent_pipeline"
        / "llms"
        / "ipiguard_llm.py",
    )
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise RuntimeError(
            "IPIGUARD_UPSTREAM_SRC does not contain the required upstream files: "
            f"{formatted}"
        )
    return source_path


UPSTREAM_SRC = _resolve_upstream_source()
if str(UPSTREAM_SRC) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_SRC))


def _prepare_agentdojo_namespace() -> None:
    """
    Avoid importing `agentdojo.agent_pipeline.__init__` (which imports optional providers).
    """
    pipeline_path = UPSTREAM_SRC / "agentdojo" / "agent_pipeline"
    llms_path = pipeline_path / "llms"

    pipeline_pkg_name = "agentdojo.agent_pipeline"
    if pipeline_pkg_name not in sys.modules:
        pipeline_pkg = types.ModuleType(pipeline_pkg_name)
        pipeline_pkg.__path__ = [str(pipeline_path)]
        sys.modules[pipeline_pkg_name] = pipeline_pkg

    llms_pkg_name = "agentdojo.agent_pipeline.llms"
    if llms_pkg_name not in sys.modules:
        llms_pkg = types.ModuleType(llms_pkg_name)
        llms_pkg.__path__ = [str(llms_path)]
        sys.modules[llms_pkg_name] = llms_pkg


_prepare_agentdojo_namespace()

from agentdojo.agent_pipeline.llms.ipiguard_llm import OpenAIConstructLLM, OpenAITraverseLLM
from agentdojo.functions_runtime import EmptyEnv, Function, FunctionCall, FunctionsRuntime
from agentdojo.types import ChatSystemMessage, ChatUserMessage


def _json_schema_type_to_python(schema_type: str | None) -> type:
    mapping: dict[str, type] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    if not schema_type:
        return Any
    return mapping.get(schema_type, Any)


def _build_parameter_model(tool_name: str, schema: dict[str, Any]) -> type[BaseModel]:
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required = set(schema.get("required", [])) if isinstance(schema, dict) else set()

    fields: dict[str, tuple[type, Any]] = {}
    for field_name, field_schema in properties.items():
        field_type = _json_schema_type_to_python(field_schema.get("type"))
        default = ... if field_name in required else None
        fields[field_name] = (field_type, default)

    return create_model(
        f"IPIGuardParams_{tool_name}",
        __config__=ConfigDict(extra="allow"),
        **fields,
    )


def _build_function(tool_def: dict[str, Any]) -> Function:
    tool_name = tool_def["name"]
    description = tool_def.get("description", "")
    parameters_schema = tool_def.get("parameters", {"type": "object", "properties": {}})
    parameters_model = _build_parameter_model(tool_name, parameters_schema)

    def _noop_run(**kwargs: Any) -> str:
        return ""

    return Function(
        name=tool_name,
        description=description,
        parameters=parameters_model,
        dependencies={},
        run=_noop_run,
        full_docstring=description or tool_name,
        return_type=None,
    )


@dataclass
class IPIGuardPlan:
    query: str
    runtime: FunctionsRuntime
    dag: Any
    extra_args: dict[str, Any]
    topological_order: list[str]
    cursor: int = 0


class IPIGuardService:
    """
    Strict wrapper over externally installed AgentDojo IPIGuard components.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        base_url: str | None = None,
    ):
        self.api_key = api_key or os.getenv("DEFENSE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("DEFENSE_BASE_URL") or os.getenv("OPENAI_BASE_URL")

        if not self.api_key and self.base_url:
            self.api_key = "EMPTY"
        if not self.api_key:
            raise RuntimeError(
                "IPIGuard requires DEFENSE_API_KEY/OPENAI_API_KEY, or set DEFENSE_BASE_URL "
                "to an OpenAI-compatible endpoint (dummy key will be used)."
            )

        self.model = model
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        self._read_only_tools, self._writable_tools = self._load_tool_type_groups()
        self.constructor = OpenAIConstructLLM(self.client, model=self.model)
        self.traverse_llm = OpenAITraverseLLM(self.client, model=self.model)

    @staticmethod
    def _load_tool_type_groups() -> tuple[set[str], set[str]]:
        """
        Load the single active IPIGuard tool grouping config.
        """
        repo_root = Path(__file__).resolve().parents[2]
        config_path = repo_root / "config" / "ipiguard_tool_groups.json"
        if not config_path.exists():
            raise RuntimeError(
                f"Missing IPIGuard tool grouping config: {config_path}. "
                "Expected a single source of truth at config/ipiguard_tool_groups.json."
            )

        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load IPIGuard tool grouping config: {config_path}"
            ) from exc

        query = {
            str(name).strip()
            for name in (data.get("query") or [])
            if str(name).strip()
        }
        command = {
            str(name).strip()
            for name in (data.get("command") or [])
            if str(name).strip()
        }
        high_risk = {
            str(name).strip()
            for name in (data.get("high_risk_command") or [])
            if str(name).strip()
        }
        writable = command.union(high_risk)
        read_only = query - writable
        return read_only, writable

    def build_runtime(self, tools: Sequence[dict[str, Any]]) -> FunctionsRuntime:
        functions = [_build_function(tool_def) for tool_def in tools]
        return FunctionsRuntime(functions=functions)

    def construct_plan(
        self,
        user_instruction: str,
        tools: Sequence[dict[str, Any]],
        system_message: str = "",
    ) -> IPIGuardPlan:
        runtime = self.build_runtime(tools)
        messages = [
            ChatSystemMessage(role="system", content=system_message),
            ChatUserMessage(role="user", content=user_instruction),
        ]
        extra_args: dict[str, Any] = {}

        _, _, _, _, result_args = self.constructor.query(
            query=user_instruction,
            runtime=runtime,
            env=EmptyEnv(),
            messages=messages,
            extra_args=extra_args,
        )
        dag = result_args.get("dag")
        if dag is None:
            raise RuntimeError("IPIGuard constructor returned no DAG.")
        if dag.number_of_nodes() == 0:
            raise RuntimeError("IPIGuard constructor returned an empty DAG.")

        order = [str(node_id) for node_id in topological_sort(dag)]
        return IPIGuardPlan(
            query=user_instruction,
            runtime=runtime,
            dag=dag,
            extra_args=result_args,
            topological_order=order,
            cursor=0,
        )

    def extract_allowed_tools(self, dag: Any) -> set[str]:
        allowed: set[str] = set()
        for node_id in dag.nodes:
            node = dag.nodes[node_id]
            function_call = node.get("function_call")
            if function_call is not None:
                allowed.add(function_call.function)
        return allowed

    def to_agentdojo_messages(self, messages: Iterable[Any]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        tool_calls_by_id: dict[str, FunctionCall] = {}

        for message in messages:
            role = message.get("role") if isinstance(message, dict) else getattr(message, "role", "")
            content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")

            if role == "system":
                converted.append({"role": "system", "content": content})
                continue
            if role == "user":
                converted.append({"role": "user", "content": content})
                continue

            if role == "assistant":
                raw_calls = (
                    message.get("tool_calls", []) if isinstance(message, dict) else getattr(message, "tool_calls", [])
                ) or []
                converted_calls: list[FunctionCall] = []
                for raw_call in raw_calls:
                    call_name = raw_call.get("name", "") if isinstance(raw_call, dict) else getattr(raw_call, "name", "")
                    call_args = raw_call.get("args", {}) if isinstance(raw_call, dict) else getattr(raw_call, "args", {})
                    call_id = raw_call.get("id", "") if isinstance(raw_call, dict) else getattr(raw_call, "id", "")
                    function_call = FunctionCall(function=call_name, args=call_args, id=call_id or None)
                    if function_call.id:
                        tool_calls_by_id[function_call.id] = function_call
                    converted_calls.append(function_call)

                converted.append(
                    {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": converted_calls if converted_calls else None,
                    }
                )
                continue

            if role == "tool":
                tool_call_id = (
                    message.get("tool_call_id", "") if isinstance(message, dict) else getattr(message, "tool_call_id", "")
                )
                tool_name = message.get("tool_name", "") if isinstance(message, dict) else getattr(message, "tool_name", "")
                tool_call = tool_calls_by_id.get(tool_call_id)
                if tool_call is None:
                    tool_call = FunctionCall(
                        function=tool_name,
                        args={},
                        id=tool_call_id or None,
                    )
                converted.append(
                    {
                        "role": "tool",
                        "content": content,
                        "tool_call_id": tool_call_id or tool_call.id,
                        "tool_call": tool_call,
                        "error": None,
                    }
                )

        return converted

    def get_expected_call(
        self,
        plan: IPIGuardPlan,
        history_messages: Sequence[dict[str, Any]],
    ) -> tuple[str | None, FunctionCall | None]:
        if plan.cursor >= len(plan.topological_order):
            return None, None

        node_id = plan.topological_order[plan.cursor]
        node_data = plan.dag.nodes[node_id]
        function_call = copy.deepcopy(node_data["function_call"])

        extra_args = dict(plan.extra_args)
        extra_args["dag"] = plan.dag
        extra_args["current_node"] = node_id
        extra_args["current_tool_call"] = function_call
        extra_args["new_tool_calls"] = []

        _, _, _, _, updated_args = self.traverse_llm.query_args_update(
            query=plan.query,
            runtime=plan.runtime,
            env=EmptyEnv(),
            messages=history_messages,
            extra_args=extra_args,
        )
        updated_call = updated_args["current_tool_call"]
        plan.dag.nodes[node_id]["function_call"] = updated_call
        plan.extra_args = updated_args

        return node_id, updated_call

    def advance_plan(self, plan: IPIGuardPlan) -> None:
        plan.cursor += 1

    def is_new_tool_call_allowed(self, tool_name: str) -> bool:
        if not tool_name:
            return False
        if tool_name in self._read_only_tools:
            return True
        if tool_name in self._writable_tools:
            return True
        read_only_prefixes = (
            "get_",
            "list_",
            "read_",
            "search_",
            "view_",
            "track_",
            "check_",
        )
        return tool_name.startswith(read_only_prefixes)

    def is_read_only_tool(self, tool_name: str) -> bool:
        if not tool_name:
            return False
        if tool_name in self._read_only_tools:
            return True
        if tool_name in self._writable_tools:
            return False
        read_only_prefixes = (
            "get_",
            "list_",
            "read_",
            "search_",
            "view_",
            "track_",
            "check_",
        )
        return tool_name.startswith(read_only_prefixes)

    def generate_plan(self, user_instruction: str, tools: list[dict[str, Any]]) -> set[str]:
        plan = self.construct_plan(user_instruction=user_instruction, tools=tools)
        return self.extract_allowed_tools(plan.dag)
