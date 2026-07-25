"""
Planner plugins for the unified agent loop.

A planner plugin decides the next assistant step:
- produce tool calls
- or produce a final response
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from core.message_types import Message, ToolCall


@dataclass
class PlannerStep:
    tool_calls: list[ToolCall] = field(default_factory=list)
    assistant_message: Message | None = None
    final_response: str = ""
    is_complete: bool = False


class PlannerPlugin:
    name = "base"

    def start(
        self,
        *,
        case: Any,
        state: Any,
        tools: list[dict[str, Any]],
        llm: Any,
    ) -> None:
        return None

    def next_step(
        self,
        *,
        case: Any,
        state: Any,
        tools: list[dict[str, Any]],
        llm: Any,
    ) -> PlannerStep:
        raise NotImplementedError

    def on_tool_result(
        self,
        *,
        case: Any,
        state: Any,
        tool_call: ToolCall,
        tool_output: Any,
        tool_content: str,
    ) -> None:
        return None

    def finish(self, *, case: Any, state: Any) -> None:
        return None

    def export_artifacts(self, *, state: Any) -> dict[str, Any]:
        return {"name": self.name}


class StandardPlannerPlugin(PlannerPlugin):
    name = "standard"

    def next_step(
        self,
        *,
        case: Any,
        state: Any,
        tools: list[dict[str, Any]],
        llm: Any,
    ) -> PlannerStep:
        response = llm.call(state.messages, tools)
        if response.tool_calls:
            return PlannerStep(
                tool_calls=list(response.tool_calls),
                assistant_message=response,
                is_complete=False,
            )
        return PlannerStep(
            assistant_message=response,
            final_response=response.content or "",
            is_complete=True,
        )


class IPIGuardPlannerPlugin(PlannerPlugin):
    name = "ipiguard"

    def __init__(self, service: Any):
        self.service = service
        self.plan = None
        self.pending_calls: list[ToolCall] = []
        self.awaiting_plan_result_id: str = ""
        self.finished = False
        self.dropped_new_calls: list[dict[str, Any]] = []
        self.planned_tool_names: set[str] = set()

    def _history_messages(self, state: Any) -> list[dict[str, Any]]:
        return self.service.to_agentdojo_messages(state.messages)

    @staticmethod
    def _tool_step(call: ToolCall) -> PlannerStep:
        return PlannerStep(
            tool_calls=[call],
            assistant_message=Message(
                role="assistant",
                content="",
                tool_calls=[call],
            ),
            is_complete=False,
        )

    @staticmethod
    def _empty_env() -> Any:
        try:
            from agentdojo.functions_runtime import EmptyEnv

            return EmptyEnv()
        except Exception:
            class _EmptyEnv:  # pragma: no cover
                pass

            return _EmptyEnv()

    def start(
        self,
        *,
        case: Any,
        state: Any,
        tools: list[dict[str, Any]],
        llm: Any,
    ) -> None:
        self.plan = self.service.construct_plan(
            user_instruction=case.user_instruction,
            tools=tools,
            system_message=case.profile,
        )
        self.pending_calls = []
        self.awaiting_plan_result_id = ""
        self.finished = False
        self.dropped_new_calls = []
        self.planned_tool_names = set(self.service.extract_allowed_tools(self.plan.dag))
        setattr(state, "ipiguard_dropped_new_calls", [])

    def _final_step(self, state: Any) -> PlannerStep:
        if self.finished:
            return PlannerStep(
                final_response=state.final_response or "",
                is_complete=True,
            )

        final_extra_args = dict(self.plan.extra_args)
        final_extra_args["dag"] = self.plan.dag
        _, _, _, final_messages, _ = self.service.traverse_llm.query_response(
            query=self.plan.query,
            runtime=self.plan.runtime,
            env=self._empty_env(),
            messages=self._history_messages(state),
            extra_args=final_extra_args,
        )

        final_content = ""
        if final_messages:
            last_message = final_messages[-1]
            if isinstance(last_message, dict):
                final_content = last_message.get("content", "") or ""

        self.finished = True
        return PlannerStep(
            assistant_message=Message(role="assistant", content=final_content),
            final_response=final_content,
            is_complete=True,
        )

    def next_step(
        self,
        *,
        case: Any,
        state: Any,
        tools: list[dict[str, Any]],
        llm: Any,
    ) -> PlannerStep:
        if self.plan is None:
            raise RuntimeError("IPIGuard planner was not initialized.")

        if self.pending_calls:
            return self._tool_step(self.pending_calls.pop(0))

        if self.plan.cursor >= len(self.plan.topological_order):
            return self._final_step(state)

        history_messages = self._history_messages(state)
        node_id, expected_call = self.service.get_expected_call(
            self.plan,
            history_messages,
        )
        if node_id is None or expected_call is None:
            return self._final_step(state)

        call = ToolCall(
            name=expected_call.function,
            args=dict(expected_call.args),
            id=expected_call.id or str(node_id),
        )

        self.awaiting_plan_result_id = call.id
        return self._tool_step(call)

    def on_tool_result(
        self,
        *,
        case: Any,
        state: Any,
        tool_call: ToolCall,
        tool_output: Any,
        tool_content: str,
    ) -> None:
        if self.plan is None:
            return
        if tool_call.id != self.awaiting_plan_result_id:
            return

        self.service.advance_plan(self.plan)
        self.awaiting_plan_result_id = ""

        extra_args = dict(self.plan.extra_args)
        extra_args["dag"] = self.plan.dag
        extra_args["new_tool_calls"] = []

        _, _, _, _, updated_args = self.service.traverse_llm.query_node_expansion(
            query=self.plan.query,
            runtime=self.plan.runtime,
            env=self._empty_env(),
            messages=self._history_messages(state),
            extra_args=extra_args,
        )
        self.plan.extra_args = updated_args

        for new_call in updated_args.get("new_tool_calls", []) or []:
            new_name = new_call.get("function_name", "")
            if not new_name:
                continue
            if not self.service.is_new_tool_call_allowed(new_name):
                dropped = {
                    "tool_name": new_name,
                    "reason": "runtime expansion not allowed by ipiguard tool policy",
                }
                self.dropped_new_calls.append(dropped)
                setattr(state, "ipiguard_dropped_new_calls", list(self.dropped_new_calls))
                continue
            if self.service.is_read_only_tool(new_name):
                new_args = new_call.get("args", {}) or {}
                self.pending_calls.append(
                    ToolCall(
                        name=new_name,
                        args=dict(new_args),
                        id=str(uuid.uuid4()),
                    )
                )
                continue
            if new_name not in self.planned_tool_names:
                dropped = {
                    "tool_name": new_name,
                    "reason": "runtime-expanded command/high-risk tool not present in initial plan",
                }
                self.dropped_new_calls.append(dropped)
                setattr(state, "ipiguard_dropped_new_calls", list(self.dropped_new_calls))
                continue
            new_args = new_call.get("args", {}) or {}
            self.pending_calls.append(
                ToolCall(
                    name=new_name,
                    args=dict(new_args),
                    id=str(uuid.uuid4()),
                )
            )

    def export_artifacts(self, *, state: Any) -> dict[str, Any]:
        plan_summary: dict[str, Any] = {}
        if self.plan is not None:
            dag = getattr(self.plan, "dag", None)
            if dag is not None:
                plan_summary = {
                    "cursor": int(getattr(self.plan, "cursor", 0) or 0),
                    "topological_order": [
                        str(x) for x in (getattr(self.plan, "topological_order", []) or [])
                    ],
                    "node_count": int(len(list(getattr(dag, "nodes", [])))),
                    "edge_count": int(len(list(getattr(dag, "edges", [])))),
                }
        return {
            "name": self.name,
            "planned_tool_names": sorted(self.planned_tool_names),
            "pending_calls": [
                {"name": tc.name, "args": dict(tc.args), "id": tc.id}
                for tc in (self.pending_calls or [])
            ],
            "awaiting_plan_result_id": self.awaiting_plan_result_id,
            "finished": self.finished,
            "dropped_new_calls": list(self.dropped_new_calls or []),
            "plan_summary": plan_summary,
        }



from defenses.camel.adapter import PrivilegedPlanGenerator, QuarantinedLLM


class CaMeLPlannerPlugin(PlannerPlugin):
    """
    CaMeL dual-LLM planner plugin.

    Upstream reference:
    - ``PrivilegedLLM`` in privileged_llm.py: generates code/plan from user query.
    - ``query_quarantined_llm`` in quarantined_llm.py: parses tool outputs.

    Architecture:
    1. **Privileged LLM** (``start()``):
       Sees ONLY the user instruction + tool schemas. Never sees tool outputs.
       Generates a deterministic plan: list of ``{tool, args, description}``.

    2. **Quarantined LLM** (``on_tool_result()``):
       Sees ONLY tool output + extraction query. Never sees user instruction.
       Parses unstructured tool output into structured data.
       Cannot influence control flow — only produces data.

    3. **Arg Resolution** (``next_step()``):
       Plan args may contain ``$ref:step_N`` placeholders referencing outputs
       of previous steps. These are resolved using quarantined parsing results,
       ensuring injected content in tool outputs cannot alter the plan.
    """

    name = "camel"

    def __init__(
        self,
        *,
        privileged_llm: PrivilegedPlanGenerator | None = None,
        quarantined_llm: QuarantinedLLM | None = None,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ):
        llm_kwargs = dict(
            api_key=api_key, model=model, base_url=base_url, temperature=temperature
        )
        self.privileged_llm = privileged_llm or PrivilegedPlanGenerator(**llm_kwargs)
        self.quarantined_llm = quarantined_llm or QuarantinedLLM(**llm_kwargs)

        self.plan_steps: list[dict[str, Any]] = []
        self.cursor = 0
        self.finished = False
        self.tool_results: dict[int, Any] = {}
        self.parsed_results: dict[int, dict[str, Any]] = {}

    def start(
        self,
        *,
        case: Any,
        state: Any,
        tools: list[dict[str, Any]],
        llm: Any,
    ) -> None:
        """
        Privileged LLM generates plan from user instruction + tool schemas.
        The LLM never sees tool outputs — this is the key CaMeL security property.
        """
        self.plan_steps = self.privileged_llm.generate_plan(
            user_instruction=case.user_instruction,
            tools=tools,
        )
        self.cursor = 0
        self.finished = False
        self.tool_results = {}
        self.parsed_results = {}

        setattr(
            state,
            "camel_privileged_plan",
            [
                {
                    "step": i,
                    "tool": (step.get("tool", "") if isinstance(step, dict) else ""),
                    "args": (step.get("args", {}) if isinstance(step, dict) else {}),
                    "description": (step.get("description", "") if isinstance(step, dict) else ""),
                }
                for i, step in enumerate(self.plan_steps)
            ],
        )
        setattr(state, "camel_quarantined_parses", [])

    def _resolve_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """
        Resolve plan argument references like ``$ref:step_N:field``.

        If a plan step says ``{"to": "$ref:step_0:email"}``, this resolves it
        using the quarantined LLM's parsed output from step 0.
        This ensures that even if tool output contains injection payloads,
        the quarantined LLM's structured parsing (which never sees the user
        instruction) mediates the data flow.
        """
        resolved = {}
        for key, val in args.items():
            if isinstance(val, str) and val.startswith("$ref:step_"):
                try:
                    parts = val.split(":")
                    step_idx = int(parts[1].replace("step_", ""))
                    field_name = parts[2] if len(parts) > 2 else None

                    if step_idx in self.parsed_results:
                        parsed = self.parsed_results[step_idx]
                        if field_name and field_name in parsed:
                            resolved[key] = parsed[field_name]
                        else:
                            resolved[key] = parsed.get("raw", val)
                    elif step_idx in self.tool_results:
                        resolved[key] = self.tool_results[step_idx]
                    else:
                        resolved[key] = val
                except (ValueError, IndexError):
                    resolved[key] = val
            else:
                resolved[key] = val
        return resolved

    def next_step(
        self,
        *,
        case: Any,
        state: Any,
        tools: list[dict[str, Any]],
        llm: Any,
    ) -> PlannerStep:
        if self.finished or self.cursor >= len(self.plan_steps):
            response = llm.call(state.messages, tools)
            return PlannerStep(
                assistant_message=response,
                final_response=response.content or "",
                is_complete=True,
            )

        while self.cursor < len(self.plan_steps):
            step = self.plan_steps[self.cursor]
            if not isinstance(step, dict):
                self.cursor += 1
                continue
            tool_name = str(step.get("tool", "")).strip()
            if tool_name:
                break
            self.cursor += 1

        if self.cursor >= len(self.plan_steps):
            response = llm.call(state.messages, tools)
            return PlannerStep(
                assistant_message=response,
                final_response=response.content or "",
                is_complete=True,
            )

        step = self.plan_steps[self.cursor]
        tool_name = str(step.get("tool", "")).strip()
        raw_args = step.get("args", {}) if isinstance(step, dict) else {}
        if not isinstance(raw_args, dict):
            raw_args = {}
        description = str(step.get("description", "")) if isinstance(step, dict) else ""

        resolved_args = self._resolve_args(raw_args)

        call = ToolCall(
            name=tool_name,
            args=resolved_args,
            id=str(uuid.uuid4()),
        )

        return PlannerStep(
            tool_calls=[call],
            assistant_message=Message(
                role="assistant",
                content=f"[CaMeL plan step {self.cursor}] {description}",
                tool_calls=[call],
            ),
            is_complete=False,
        )

    def on_tool_result(
        self,
        *,
        case: Any,
        state: Any,
        tool_call: ToolCall,
        tool_output: Any,
        tool_content: str,
    ) -> None:
        """
        Process tool result using quarantined LLM.

        The quarantined LLM parses unstructured tool output into structured data.
        It NEVER sees the user instruction — only the tool output and extraction query.
        This prevents prompt injection in tool outputs from influencing the plan.
        """
        step_idx = self.cursor

        self.tool_results[step_idx] = tool_output

        needs_parsing = False
        for future_step in self.plan_steps[self.cursor + 1:]:
            future_args = future_step.get("args", {}) or {}
            for val in future_args.values():
                if isinstance(val, str) and f"$ref:step_{step_idx}" in val:
                    needs_parsing = True
                    break

        if needs_parsing:
            step_desc = self.plan_steps[self.cursor].get("description", "")
            parsed = self.quarantined_llm.parse_tool_output(
                tool_output=tool_content,
                extraction_query=f"Extract structured data from this tool result. Context: {step_desc}",
            )
            self.parsed_results[step_idx] = parsed

            parses = getattr(state, "camel_quarantined_parses", [])
            parses.append({
                "step": step_idx,
                "tool": tool_call.name,
                "parsed_fields": list(parsed.keys()),
                "status": parsed.get("_quarantined_status", "unknown"),
            })
            setattr(state, "camel_quarantined_parses", parses)

        self.cursor += 1

    def export_artifacts(self, *, state: Any) -> dict[str, Any]:
        return {
            "name": self.name,
            "plan_steps": list(self.plan_steps or []),
            "cursor": self.cursor,
            "finished": self.finished,
            "tool_results": {str(k): v for k, v in (self.tool_results or {}).items()},
            "parsed_results": {str(k): v for k, v in (self.parsed_results or {}).items()},
        }
