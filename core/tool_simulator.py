
import json
import inspect
from typing import Any, Callable

from .message_types import Case
from tools.connected.finance import FinanceTools
from tools.connected.office import OfficeTools
from tools.connected.smarthome import SmartHomeTools
from tools.connected.travel import TravelTools
from tools.connected.shopping import ShoppingTools
from tools.connected.system import SystemTools
from tools.connected.medical import MedicalTools

class ToolSimulator:
    """
    Deterministic in-memory executor for the benchmark tool contracts.
    """

    def __init__(self, tool_library: dict[str, Any] | None = None):
        self.tool_library = tool_library or {}

        self.finance = FinanceTools()
        self.office = OfficeTools()
        self.smarthome = SmartHomeTools()
        self.travel = TravelTools()
        self.shopping = ShoppingTools()
        self.system = SystemTools()
        self.medical = MedicalTools()
        self.dispatch_map: dict[str, Callable] = {}
        self._register_library_dispatch_entries()

    def _iter_connected_domains(self) -> tuple[Any, ...]:
        return (
            self.finance,
            self.office,
            self.smarthome,
            self.travel,
            self.shopping,
            self.system,
            self.medical,
        )

    def _resolve_connected_handler(self, tool_name: str) -> Callable[..., Any] | None:
        """
        Resolve a tool handler by method name across connected tool domains.

        This keeps expansion cost low: when adding a new tool, defining the
        method on a connected domain class and adding schema to tool_library
        is enough for automatic registration.
        """
        for domain in self._iter_connected_domains():
            handler = getattr(domain, tool_name, None)
            if callable(handler):
                return handler
        return None

    def _register_library_dispatch_entries(self) -> None:
        """
        Auto-register tool handlers discovered from tool library.

        Existing explicit mappings remain untouched; this only fills missing
        entries so new tools can be onboarded with minimal glue code.
        """
        raw_tools = self.tool_library.get("tools", {})
        if not isinstance(raw_tools, dict):
            return
        for tool_name in raw_tools.keys():
            handler = self._resolve_connected_handler(tool_name)
            if handler is not None:
                self.dispatch_map[tool_name] = handler

    @staticmethod
    def _prepare_handler_args(
        handler: Callable[..., Any],
        args: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        signature = inspect.signature(handler)
        params = signature.parameters
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

        if accepts_kwargs:
            return dict(args), []

        filtered: dict[str, Any] = {}
        for name, param in params.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if name in args:
                filtered[name] = args[name]

        missing: list[str] = []
        for name, param in params.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param.default is inspect._empty and name not in filtered:
                missing.append(name)
        return filtered, missing

    def execute(
        self,
        tool_name: str,
        args: dict[str, Any],
        case: Case
    ) -> dict[str, Any]:
        """
        Execute a connected tool against the current world-state model.
        """
        if tool_name not in self.dispatch_map:
            return {
                "status": "error",
                "message": (
                    f"Tool '{tool_name}' is not active in the current tool library. "
                    "Update the active contracts/library before calling it."
                ),
            }
        try:
            handler = self.dispatch_map[tool_name]
            call_args, missing = self._prepare_handler_args(handler, args or {})
            if missing:
                return {
                    "status": "error",
                    "message": f"Missing required arguments for {tool_name}: {', '.join(missing)}",
                }
            return handler(**call_args)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Runtime execution failed: {str(e)}"
            }

    def get_tool_schema(self, tool_name: str) -> dict[str, Any]:
        """
        Get tool schema from library.
        """
        if "tools" in self.tool_library:
            tools = self.tool_library["tools"]
            if tool_name in tools:
                return tools[tool_name]

        if tool_name in self.tool_library:
            return self.tool_library[tool_name]

        return {
            "name": tool_name,
            "description": f"Tool: {tool_name}",
            "parameters": {"type": "object", "properties": {}},
            "returns": {"type": "object"},
        }

    def get_tools_for_case(self, case: Case) -> list[dict[str, Any]]:
        base_tools = list(case.tools or [])
        extra_tools = list(getattr(case, "tool_whitelist_extras", []) or [])

        ordered: list[str] = []
        seen: set[str] = set()
        for name in [*base_tools, *extra_tools]:
            tool_name = str(name).strip()
            if not tool_name or tool_name in seen:
                continue
            seen.add(tool_name)
            ordered.append(tool_name)

        return [self.get_tool_schema(t) for t in ordered]

    @classmethod
    def from_library_file(cls, path: str) -> "ToolSimulator":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(tool_library=data)
