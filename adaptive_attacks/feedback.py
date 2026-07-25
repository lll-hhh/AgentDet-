"""Convert AgentDET execution artifacts into bounded attacker feedback."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Iterable

from .base import AttackFeedback


def _message_payload(message: Any) -> dict[str, Any] | None:
    if hasattr(message, "__dict__"):
        message = vars(message)
    if not isinstance(message, dict):
        return None

    payload: dict[str, Any] = {
        "role": str(message.get("role", "") or ""),
        "content": str(message.get("content", "") or ""),
    }
    tool_calls = message.get("tool_calls", [])
    if isinstance(tool_calls, list):
        cleaned_calls = []
        for call in tool_calls:
            if hasattr(call, "__dict__"):
                call = vars(call)
            if not isinstance(call, dict):
                continue
            cleaned_calls.append(
                {
                    "name": str(call.get("name", "") or ""),
                    "args": deepcopy(call.get("args", {})),
                    "id": str(call.get("id", "") or ""),
                }
            )
        if cleaned_calls:
            payload["tool_calls"] = cleaned_calls
    if message.get("tool_call_id"):
        payload["tool_call_id"] = str(message.get("tool_call_id"))
    if message.get("tool_name"):
        payload["tool_name"] = str(message.get("tool_name"))
    return payload


def _visible_defense_events(execution: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose application-visible decisions, never GT labels or internal scores."""

    planner = _planner_artifacts(execution)
    judgments = planner.get("leaf_judgments", {})
    if not isinstance(judgments, dict):
        return []

    events: list[dict[str, Any]] = []
    for node_id, judgment in judgments.items():
        if not isinstance(judgment, dict):
            continue
        events.append(
            {
                "node_id": str(judgment.get("node_id", node_id) or node_id),
                "tool_name": str(judgment.get("tool_name", "") or ""),
                "arguments": deepcopy(judgment.get("arguments", {})),
                "threshold_verdict": str(
                    judgment.get("threshold_verdict", "") or ""
                ),
                "pre_policy_verdict": str(
                    judgment.get("pre_policy_verdict", "") or ""
                ),
                "reason": str(judgment.get("reason", "") or ""),
            }
        )
    return events


def visible_trace_from_execution(execution: dict[str, Any]) -> dict[str, Any]:
    """Build the query-visible trace used by PAIR-style gray-box attacks.

    Model reasoning, oracle matches, benchmark GT, confirmation answers, and
    internal uncertainty scores are deliberately excluded.
    """

    trace = execution.get("trace_payload", {})
    trace = trace if isinstance(trace, dict) else {}
    raw_messages = trace.get("messages", [])
    raw_messages = raw_messages if isinstance(raw_messages, list) else []
    messages = []
    for message in raw_messages:
        cleaned = _message_payload(message)
        if cleaned is not None:
            messages.append(cleaned)

    tool_calls = trace.get("tool_calls_made", [])
    if not isinstance(tool_calls, list):
        tool_calls = []
    return {
        "case_id": str(trace.get("case_id", "") or ""),
        "blocked": bool(trace.get("blocked", False)),
        "blocked_by": str(trace.get("blocked_by", "") or ""),
        "blocked_reason": str(trace.get("blocked_reason", "") or ""),
        "runtime_error": bool(trace.get("runtime_error", False)),
        "runtime_error_message": str(trace.get("runtime_error_message", "") or ""),
        "final_response": str(trace.get("final_response", "") or ""),
        "messages": messages,
        "tool_calls_made": deepcopy(tool_calls),
        "defense_events": _visible_defense_events(execution),
    }


def _eval_payload(execution: dict[str, Any]) -> dict[str, Any]:
    result = execution.get("eval_result")
    if result is not None and hasattr(result, "to_dict"):
        payload = result.to_dict()
        if isinstance(payload, dict):
            return payload
    trace = execution.get("trace_payload", {})
    payload = trace.get("eval", {}) if isinstance(trace, dict) else {}
    return payload if isinstance(payload, dict) else {}


def _planner_artifacts(execution: dict[str, Any]) -> dict[str, Any]:
    artifacts = execution.get("defense_artifacts", {})
    if not isinstance(artifacts, dict):
        return {}
    planner = artifacts.get("planner", {})
    return planner if isinstance(planner, dict) else {}


def _confirmation_count(planner: dict[str, Any]) -> int:
    judgments = planner.get("leaf_judgments", {})
    if not isinstance(judgments, dict):
        return 0
    count = 0
    for judgment in judgments.values():
        if not isinstance(judgment, dict):
            continue
        verdicts = (
            judgment.get("threshold_verdict"),
            judgment.get("pre_policy_verdict"),
            judgment.get("verdict"),
        )
        if any(str(value or "").strip().lower() == "ask_user" for value in verdicts):
            count += 1
    return count


def _target_tool_called(
    tool_calls: Iterable[Any],
    *,
    targets: Iterable[str],
) -> bool:
    normalized_targets = {
        str(target).strip().lower() for target in targets if str(target).strip()
    }
    if not normalized_targets:
        return False
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        searchable = json.dumps(call, ensure_ascii=False, sort_keys=True, default=str).lower()
        if any(target in searchable for target in normalized_targets):
            return True
    return False


def feedback_from_execution(
    execution: dict[str, Any],
    *,
    targets: Iterable[str] = (),
) -> AttackFeedback:
    eval_payload = _eval_payload(execution)
    trace = execution.get("trace_payload", {})
    trace = trace if isinstance(trace, dict) else {}
    planner = _planner_artifacts(execution)
    confirmations = _confirmation_count(planner)
    skipped = planner.get("skipped_leaves", [])
    skipped_count = len(skipped) if isinstance(skipped, list) else 0
    tool_calls = eval_payload.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        tool_calls = []

    blocked = bool(trace.get("blocked", False)) or skipped_count > 0
    runtime_error = bool(trace.get("runtime_error", False)) or not bool(
        execution.get("ok", True)
    )
    target_called = _target_tool_called(tool_calls, targets=targets)
    attack_successful = bool(eval_payload.get("attack_successful", False))
    defense_triggered = bool(eval_payload.get("defense_triggered", False)) or bool(
        confirmations or skipped_count or blocked
    )

    notes: list[str] = []
    if attack_successful:
        notes.append("attacker target state reached")
    elif target_called:
        notes.append("a target-associated value appeared in a tool call")
    if confirmations:
        notes.append(f"defense requested {confirmations} confirmation(s)")
    if skipped_count:
        notes.append(f"defense skipped {skipped_count} planned write leaf/leaves")
    if blocked:
        notes.append("agent execution was blocked")
    if runtime_error:
        notes.append("execution ended with a runtime error")

    return AttackFeedback(
        attack_successful=attack_successful,
        attack_success_total=bool(eval_payload.get("attack_success_total", False)),
        task_completed=bool(eval_payload.get("task_completed", False)),
        defense_triggered=defense_triggered,
        blocked=blocked,
        blocked_by=str(
            trace.get("blocked_by", "") or ("gardener" if skipped_count else "")
        ),
        confirmation_count=confirmations,
        skipped_leaf_count=skipped_count,
        target_tool_called=target_called,
        runtime_error=runtime_error,
        notes=tuple(notes),
        visible_trace=visible_trace_from_execution(execution),
    )


__all__ = ["feedback_from_execution", "visible_trace_from_execution"]
