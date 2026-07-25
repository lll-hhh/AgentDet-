
from dataclasses import dataclass, field
import re
from typing import Any, Literal, Optional


def pair_base_id(case_id: str) -> str:
    """Return the shared identifier for paired safe and unsafe artifacts."""
    value = str(case_id or "")
    safe_match = re.fullmatch(r"(.+)-safe", value)
    if safe_match:
        return safe_match.group(1)
    unsafe_match = re.fullmatch(r"(.+)-unsafe(?:-fill\d+)?", value)
    if unsafe_match:
        return unsafe_match.group(1)
    return value


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    id: str = ""


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    reasoning_content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""
    tool_name: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    messages: list[Message] = field(default_factory=list)
    is_complete: bool = False
    final_response: str = ""
    tool_calls_made: list[dict] = field(default_factory=list)
    blocked: bool = False
    blocked_by: str = ""
    blocked_reason: str = ""
    runtime_error: bool = False
    runtime_error_message: str = ""


@dataclass
class Label:
    safety: Literal["safe", "unsafe"] = "safe"
    should_execute: bool = True

    @classmethod
    def from_dict(cls, data: dict | str) -> "Label":
        if isinstance(data, str):

            return cls(safety=data, should_execute=(data == "safe"))
        return cls(
            safety=data.get("safety", "safe"),
            should_execute=data.get("should_execute", True),
        )


@dataclass
class Condition:
    type: str = ""
    expression: str = ""
    data_field: str = ""
    natural_language: str = ""

    @classmethod
    def from_dict(cls, data: dict | str) -> "Condition":
        if not data:
            return cls()
        if isinstance(data, str):
            return cls(natural_language=data)
        return cls(
            type=data.get("type", ""),
            expression=data.get("expression", ""),
            data_field=data.get("data_field", ""),
            natural_language=data.get("natural_language", ""),
        )


@dataclass
class CaseTaxonomy:
    """
    Four-dimensional AgentDET taxonomy label stored in each case.

    Axes:
        - scenario_domain: business scenario/domain
        - dependency_structure: G-axis (task dependency graph shape)
        - evidence_delay: T-axis (how late critical evidence appears)
        - difficulty_tier: L-axis (task length tier)
    """

    scenario_domain: int = 0
    dependency_structure: int = 0
    evidence_delay: int = 0
    difficulty_tier: int = 0
    dep_count: int = 0
    noise_tool_count: int = 0
    injection_profile: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _parse_int_axis(value: Any, field_name: str, allowed: set[int]) -> int:
        if isinstance(value, bool):
            raise ValueError(f"taxonomy.{field_name} must be int, got bool")
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, str) and value.strip().isdigit():
            parsed = int(value.strip())
        else:
            raise ValueError(f"taxonomy.{field_name} must be int, got: {value!r}")
        if parsed not in allowed:
            raise ValueError(
                f"taxonomy.{field_name} out of range: {parsed}, allowed={sorted(allowed)}"
            )
        return parsed

    @staticmethod
    def _parse_noise_tool_count(value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("taxonomy.noise_tool_count must be int, got bool")
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, str) and value.strip().isdigit():
            parsed = int(value.strip())
        else:
            raise ValueError(f"taxonomy.noise_tool_count must be int, got: {value!r}")
        if parsed < 0:
            raise ValueError("taxonomy.noise_tool_count must be >= 0")
        return parsed

    @staticmethod
    def _parse_injection_profile(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("taxonomy.injection_profile must be object")
        pattern = value.get("injection_pattern")
        count = value.get("injection_point_count")

        if pattern not in {"single_point", "multi_point"}:
            raise ValueError(
                "taxonomy.injection_profile.injection_pattern must be "
                "'single_point' or 'multi_point'"
            )
        if isinstance(count, bool):
            raise ValueError(
                "taxonomy.injection_profile.injection_point_count must be int, got bool"
            )
        if isinstance(count, str) and count.strip().isdigit():
            count = int(count.strip())
        if not isinstance(count, int):
            raise ValueError(
                "taxonomy.injection_profile.injection_point_count must be int"
            )
        if pattern == "single_point" and count != 1:
            raise ValueError(
                "taxonomy.injection_profile requires injection_point_count==1 for single_point"
            )
        if pattern == "multi_point" and count < 2:
            raise ValueError(
                "taxonomy.injection_profile requires injection_point_count>=2 for multi_point"
            )
        parsed_profile = {
            "injection_pattern": pattern,
            "injection_point_count": count,
        }
        candidate_count = value.get("candidate_injection_point_count")
        if candidate_count is not None:
            if isinstance(candidate_count, bool):
                raise ValueError(
                    "taxonomy.injection_profile.candidate_injection_point_count must be int, got bool"
                )
            if isinstance(candidate_count, str) and candidate_count.strip().isdigit():
                candidate_count = int(candidate_count.strip())
            if not isinstance(candidate_count, int):
                raise ValueError(
                    "taxonomy.injection_profile.candidate_injection_point_count must be int"
                )
            if candidate_count < 0:
                raise ValueError(
                    "taxonomy.injection_profile.candidate_injection_point_count must be >= 0"
                )
            parsed_profile["candidate_injection_point_count"] = candidate_count
        candidate_tools = value.get("candidate_injection_tools")
        if candidate_tools is not None:
            if not isinstance(candidate_tools, list) or not all(
                isinstance(x, str) and x.strip() for x in candidate_tools
            ):
                raise ValueError(
                    "taxonomy.injection_profile.candidate_injection_tools must be list[str]"
                )
            parsed_profile["candidate_injection_tools"] = [str(x).strip() for x in candidate_tools]
        return parsed_profile

    @classmethod
    def from_dict(
        cls,
        data: dict | None,
    ) -> "CaseTaxonomy":
        if not isinstance(data, dict):
            raise ValueError("taxonomy must be object")
        payload = data
        required = (
            "scenario_domain",
            "dependency_structure",
            "evidence_delay",
            "difficulty_tier",
            "dep_count",
            "noise_tool_count",
            "injection_profile",
        )
        missing = [k for k in required if k not in payload]
        if missing:
            raise ValueError(f"taxonomy missing required fields: {missing}")

        return cls(
            scenario_domain=cls._parse_int_axis(
                payload.get("scenario_domain"), "scenario_domain", {1, 2, 3, 4, 5, 6, 7}
            ),
            dependency_structure=cls._parse_int_axis(
                payload.get("dependency_structure"), "dependency_structure", {1, 2, 3, 4}
            ),
            evidence_delay=cls._parse_int_axis(
                payload.get("evidence_delay"), "evidence_delay", {1, 2, 3}
            ),
            difficulty_tier=cls._parse_int_axis(
                payload.get("difficulty_tier"), "difficulty_tier", {1, 2, 3}
            ),
            dep_count=cls._parse_int_axis(payload.get("dep_count"), "dep_count", {1, 2, 3}),
            noise_tool_count=cls._parse_noise_tool_count(payload.get("noise_tool_count")),
            injection_profile=cls._parse_injection_profile(
                payload.get("injection_profile")
            ),
        )


@dataclass
class Case:
    id: str
    domain: str
    taxonomy: CaseTaxonomy
    scenario: str
    user_instruction: str
    tools: list[str]
    environment_data: dict[str, Any]
    label: Label
    ground_truth: list[dict] = field(default_factory=list)
    expected_state: dict[str, Any] = field(default_factory=dict)
    attack_info: dict[str, Any] = field(default_factory=dict)
    profile: str = ""
    risk_type: str = ""
    condition: Condition = field(default_factory=Condition)
    interaction_trace: list[dict] = field(default_factory=list)
    difficulty: str = "medium"
    tool_whitelist_extras: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "Case":
        raw_extras = (
            data.get("tool_whitelist_extras")
            or data.get("tool_whitelist_extra")
            or data.get("extra_tools")
            or []
        )
        if isinstance(raw_extras, list):
            extras = [str(x) for x in raw_extras if str(x).strip()]
        else:
            extras = []

        return cls(
            id=data.get("id", ""),
            domain=data.get("domain", ""),
            taxonomy=CaseTaxonomy.from_dict(data.get("taxonomy")),
            scenario=data.get("scenario", ""),
            user_instruction=data.get("user_instruction", ""),
            tools=data.get("tools", []),
            environment_data=data.get("environment_data", {}),
            label=Label.from_dict(data.get("label", {})),
            ground_truth=data.get("ground_truth", []),
            expected_state=data.get("expected_state", {}),
            attack_info=data.get("attack_info", {}),
            profile=data.get("profile", ""),
            risk_type=data.get("risk_type", ""),
            condition=Condition.from_dict(data.get("condition", {})),
            interaction_trace=data.get("interaction_trace", []),
            difficulty=data.get("difficulty", "medium"),
            tool_whitelist_extras=extras,
        )

    def is_safe(self) -> bool:
        return self.label.safety == "safe"

    def get_expected_tools(self) -> list[str]:
        names: list[str] = []
        for gt in self.ground_truth:
            if not isinstance(gt, dict):
                continue
            tool_name = gt.get("tool") or gt.get("function") or gt.get("name") or ""
            if isinstance(tool_name, str) and tool_name:
                names.append(tool_name)
        return names

    def get_attacker_tools(self) -> list[str]:
        return self.attack_info.get("attacker_tools", [])
