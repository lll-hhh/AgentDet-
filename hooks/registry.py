
import time
from typing import Any

from .base import DefenseHook, HookResult


class HookRegistry:

    def __init__(self):
        self.hooks: dict[str, list[DefenseHook]] = {
            "input": [],
            "plan": [],
            "reasoning": [],
            "action": [],
            "finish": [],
        }
        self.execution_log: list[dict[str, Any]] = []

    @staticmethod
    def _serialize_context(context: dict[str, Any]) -> dict[str, Any]:
        """
        Keep a focused, JSON-friendly snapshot of the hook context.

        Full message history already exists in per-case trace files, so the
        hook log only records the fields that explain *why* a hook ran.
        """
        snapshot: dict[str, Any] = {}

        for key in (
            "source",
            "content",
            "tool_name",
            "tool_args",
            "thought",
            "planner",
            "iteration",
            "final_response",
            "is_complete",
        ):
            if key in context:
                snapshot[key] = context.get(key)

        assistant_message = context.get("assistant_message")
        if assistant_message is not None:
            snapshot["assistant_message"] = {
                "role": str(getattr(assistant_message, "role", "")),
                "content": str(getattr(assistant_message, "content", "")),
                "tool_calls": [
                    {
                        "name": str(getattr(tc, "name", "")),
                        "args": dict(getattr(tc, "args", {}) or {}),
                        "id": str(getattr(tc, "id", "")),
                    }
                    for tc in (getattr(assistant_message, "tool_calls", []) or [])
                ],
            }

        case_obj = context.get("case")
        if case_obj is not None:
            snapshot["case"] = {
                "id": str(getattr(case_obj, "id", "")),
                "label": str(getattr(getattr(case_obj, "label", None), "safety", "")),
            }

        tools = context.get("tools")
        if isinstance(tools, list):
            snapshot["tool_names"] = [
                str(tool.get("name", "")).strip()
                for tool in tools
                if isinstance(tool, dict) and str(tool.get("name", "")).strip()
            ]

        return snapshot

    def register(self, hook: DefenseHook) -> None:
        if hook.layer not in self.hooks:
            raise ValueError(f"Invalid hook layer: {hook.layer}")

        self.hooks[hook.layer].append(hook)

    def register_many(self, hooks: list[DefenseHook]) -> None:
        for hook in hooks:
            self.register(hook)

    def unregister(self, hook_name: str, layer: str | None = None) -> bool:
        layers = [layer] if layer else self.hooks.keys()
        removed = False

        for l in layers:
            original_len = len(self.hooks[l])
            self.hooks[l] = [h for h in self.hooks[l] if h.name != hook_name]
            if len(self.hooks[l]) < original_len:
                removed = True

        return removed

    def clear(self, layer: str | None = None) -> None:
        if layer:
            self.hooks[layer] = []
        else:
            for l in self.hooks:
                self.hooks[l] = []

    def run_layer(
        self,
        layer: str,
        context: dict[str, Any]
    ) -> HookResult:
        if layer not in self.hooks:
            raise ValueError(f"Invalid hook layer: {layer}")


        modified_context = context.copy()

        for hook in self.hooks[layer]:

            hook.ensure_initialized()


            start_time = time.time()
            result = hook.check(modified_context)
            elapsed_ms = (time.time() - start_time) * 1000


            result.hook_name = hook.name
            result.latency_ms = elapsed_ms


            self.execution_log.append({
                "hook": hook.name,
                "layer": layer,
                "passed": result.passed,
                "score": result.score,
                "reason": result.reason,
                "latency_ms": elapsed_ms,
                "modified_content": result.modified_content,
                "metadata": dict(result.metadata or {}),
                "context": self._serialize_context(modified_context),
            })


            if result.modified_content is not None and "content" in modified_context:
                modified_context["content"] = result.modified_content


            if hook.is_blocking and not result.passed:
                return result


        return HookResult(
            passed=True,
            hook_name="",
            modified_content=modified_context.get("content"),
        )

    def get_hooks(self, layer: str) -> list[DefenseHook]:
        return self.hooks.get(layer, [])

    def get_all_hooks(self) -> dict[str, list[DefenseHook]]:
        return self.hooks.copy()

    def get_execution_log(self) -> list[dict[str, Any]]:
        return self.execution_log.copy()

    def clear_log(self) -> None:
        self.execution_log = []

    def export_artifacts(self) -> dict[str, Any]:
        return {
            "registered_hooks": {
                layer: [hook.export_artifacts() for hook in hooks]
                for layer, hooks in self.hooks.items()
                if hooks
            },
            "execution_log": self.get_execution_log(),
        }

    def get_stats(self) -> dict[str, Any]:
        stats = {
            "total": 0,
            "by_layer": {},
            "blocking": 0,
            "transform": 0,
            "requires_model": 0,
        }

        for layer, hooks in self.hooks.items():
            stats["by_layer"][layer] = len(hooks)
            stats["total"] += len(hooks)

            for hook in hooks:
                if hook.is_blocking:
                    stats["blocking"] += 1
                else:
                    stats["transform"] += 1
                if hook.requires_model:
                    stats["requires_model"] += 1

        return stats

    def __repr__(self) -> str:
        counts = {l: len(h) for l, h in self.hooks.items()}
        return f"HookRegistry({counts})"
