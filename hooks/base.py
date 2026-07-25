
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class HookResult:
    passed: bool = True
    reason: str = ""
    score: float = 0.0
    hook_name: str = ""
    latency_ms: float = 0.0
    modified_content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DefenseHook(ABC):


    name: str = "base"
    layer: Literal["input", "plan", "reasoning", "action", "finish"] = "input"


    is_blocking: bool = True
    requires_model: bool = False

    def __init__(self, **config):
        self.config = config
        self._initialized = False

    def initialize(self) -> None:
        self._initialized = True

    def ensure_initialized(self) -> None:
        if not self._initialized:
            self.initialize()

    @abstractmethod
    def check(self, context: dict[str, Any]) -> HookResult:
        pass

    def transform(self, content: str) -> str:
        return content

    def reset(self) -> None:
        pass

    def get_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "layer": self.layer,
            "is_blocking": self.is_blocking,
            "requires_model": self.requires_model,
        }

    def export_artifacts(self) -> dict[str, Any]:
        """
        Export hook-specific audit artifacts.

        Subclasses can override this to expose richer internal state. The
        default export keeps enough metadata to identify how the hook ran
        without leaking heavyweight client objects.
        """
        return {
            "name": self.name,
            "layer": self.layer,
            "is_blocking": self.is_blocking,
            "requires_model": self.requires_model,
            "config": dict(self.config or {}),
        }
