
import json
from pathlib import Path
from typing import Iterator

from .message_types import Case, CaseTaxonomy

REQUIRED_CASE_FIELDS: tuple[str, ...] = (
    "id",
    "domain",
    "taxonomy",
    "scenario",
    "user_instruction",
    "tools",
    "environment_data",
    "label",
    "ground_truth",
    "expected_state",
    "attack_info",
)
OPTIONAL_CASE_FIELDS: set[str] = {
    "profile",
    "risk_type",
    "condition",
    "interaction_trace",
    "difficulty",
    "tool_whitelist_extras",
    "tool_whitelist_extra",
    "extra_tools",
}
ALLOWED_CASE_FIELDS: set[str] = set(REQUIRED_CASE_FIELDS) | OPTIONAL_CASE_FIELDS


class CaseLoader:

    def __init__(self, cases_dir: str | Path):
        self.cases_dir = Path(cases_dir)

    def _build_case_from_file(self, json_file: Path) -> Case:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        valid, reason = self._validate_case_data(data)
        if not valid:
            raise ValueError(f"Invalid case structure: {json_file} -> {reason}")

        try:
            case = Case.from_dict(data)
        except Exception as e:
            raise ValueError(f"Failed to parse case: {json_file} -> {e}") from e

        setattr(case, "__source_file__", str(json_file.resolve()))
        return case

    def load_case(self, case_id: str) -> Case:




        for json_file in self.cases_dir.rglob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if json_file.stem == case_id or data.get("id") == case_id:
                return self._build_case_from_file(json_file)

        raise FileNotFoundError(f"Case not found: {case_id}")

    def load_all(self) -> list[Case]:
        cases = []
        for json_file in sorted(self.cases_dir.rglob("*.json")):
            if json_file.name == "manifest.json":
                continue

            try:
                cases.append(self._build_case_from_file(json_file))
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON: {json_file} -> {e}") from e
        return cases

    def iter_cases(self) -> Iterator[Case]:
        for json_file in sorted(self.cases_dir.rglob("*.json")):
            if json_file.name == "manifest.json":
                continue
            try:
                yield self._build_case_from_file(json_file)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON: {json_file} -> {e}") from e

    def filter_by_domain(self, domain: str) -> list[Case]:
        target = str(domain).strip().lower()
        return [c for c in self.load_all() if str(c.domain).strip().lower() == target]

    def filter_by_difficulty(self, level: str) -> list[Case]:
        return [
            c for c in self.load_all()
            if self._difficulty_level(c) == level
        ]

    def filter_by_label(self, label: str) -> list[Case]:
        return [c for c in self.load_all() if c.label.safety == label]

    def filter(
        self,
        domains: list[str] | None = None,
        difficulties: list[str] | None = None,
        labels: list[str] | None = None,
        risk_types: list[str] | None = None,
    ) -> list[Case]:
        cases = self.load_all()

        if domains:
            normalized_domains = {str(d).strip().lower() for d in domains}
            cases = [c for c in cases if str(c.domain).strip().lower() in normalized_domains]

        if difficulties:
            cases = [
                c for c in cases
                if self._difficulty_level(c) in difficulties
            ]

        if labels:
            normalized_labels = {str(l).strip().lower() for l in labels}
            cases = [c for c in cases if str(c.label.safety).strip().lower() in normalized_labels]

        if risk_types:
            cases = [c for c in cases if c.risk_type in risk_types]

        return cases

    def count(self) -> int:
        return sum(1 for _ in self.iter_cases())

    def get_stats(self) -> dict:
        cases = self.load_all()

        domains = {}
        difficulties = {}
        labels = {"safe": 0, "unsafe": 0}

        for c in cases:
            domains[c.domain] = domains.get(c.domain, 0) + 1
            diff = self._difficulty_level(c) or "unknown"
            difficulties[diff] = difficulties.get(diff, 0) + 1
            labels[c.label.safety] = labels.get(c.label.safety, 0) + 1

        return {
            "total": len(cases),
            "by_domain": domains,
            "by_difficulty": difficulties,
            "by_label": labels,
        }

    @staticmethod
    def _difficulty_level(case: Case) -> str:
        """
        Resolve legacy difficulty level only.

        Note: taxonomy is intentionally ignored here. Taxonomy is metadata for
        offline analysis, not runtime loading behavior.
        """
        difficulty = getattr(case, "difficulty", "")
        if isinstance(difficulty, dict):
            return str(difficulty.get("level", ""))
        if isinstance(difficulty, str):
            return difficulty
        return ""

    @staticmethod
    def _validate_case_data(data: object) -> tuple[bool, str]:
        """
        Strict schema gate for runtime loading.
        Any invalid file should fail fast with a precise error message.
        """
        if not isinstance(data, dict):
            return False, "root must be an object"

        missing = [field for field in REQUIRED_CASE_FIELDS if field not in data]
        if missing:
            return False, f"missing fields: {missing}"
        unexpected = sorted([k for k in data.keys() if k not in ALLOWED_CASE_FIELDS])
        if unexpected:
            return False, f"unexpected top-level fields (new standard forbids): {unexpected}"

        if not isinstance(data.get("id"), str):
            return False, "field 'id' must be string"
        if not str(data.get("id", "")).strip():
            return False, "field 'id' must be non-empty string"
        if not isinstance(data.get("user_instruction"), str):
            return False, "field 'user_instruction' must be string"
        if not isinstance(data.get("tools"), list):
            return False, "field 'tools' must be list"
        if not all(isinstance(t, str) and str(t).strip() for t in data.get("tools", [])):
            return False, "field 'tools' must contain non-empty strings"
        if not isinstance(data.get("environment_data"), dict):
            return False, "field 'environment_data' must be object"
        if not isinstance(data.get("label"), dict):
            return False, "field 'label' must be object"
        if not isinstance(data.get("taxonomy"), dict):
            return False, "field 'taxonomy' must be object"
        if not isinstance(data.get("domain"), str):
            return False, "field 'domain' must be string"
        if not isinstance(data.get("scenario"), str):
            return False, "field 'scenario' must be string"
        if not isinstance(data.get("ground_truth"), list):
            return False, "field 'ground_truth' must be list"
        if not isinstance(data.get("expected_state"), dict):
            return False, "field 'expected_state' must be object"
        if not isinstance(data.get("attack_info"), dict):
            return False, "field 'attack_info' must be object"

        for field_name in ("profile", "risk_type"):
            if field_name in data and not isinstance(data[field_name], str):
                return False, f"field '{field_name}' must be string"
        if "condition" in data and not isinstance(data["condition"], (dict, str)):
            return False, "field 'condition' must be object or string"
        if "interaction_trace" in data and not isinstance(data["interaction_trace"], list):
            return False, "field 'interaction_trace' must be list"
        if "difficulty" in data and not isinstance(data["difficulty"], (dict, str)):
            return False, "field 'difficulty' must be object or string"
        for field_name in ("tool_whitelist_extras", "tool_whitelist_extra", "extra_tools"):
            if field_name not in data:
                continue
            value = data[field_name]
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item.strip() for item in value
            ):
                return False, f"field '{field_name}' must contain non-empty strings"

        label = data.get("label", {})
        safety = label.get("safety")
        should_execute = label.get("should_execute")
        if safety not in {"safe", "unsafe"}:
            return False, "label.safety must be 'safe' or 'unsafe'"
        if not isinstance(should_execute, bool):
            return False, "label.should_execute must be bool"

        try:
            taxonomy = CaseTaxonomy.from_dict(data.get("taxonomy"))
        except Exception as e:
            return False, f"invalid taxonomy: {e}"

        attack_info = data.get("attack_info", {})
        if safety == "safe":
            if attack_info != {}:
                return False, "safe case requires attack_info == {}"
        else:
            if not attack_info:
                return False, "unsafe case requires non-empty attack_info"
            attacker_states = attack_info.get("attacker_states")
            if not isinstance(attacker_states, list) or len(attacker_states) == 0:
                return False, "unsafe attack_info.attacker_states must be non-empty list"
            if not all(isinstance(x, dict) and x for x in attacker_states):
                return False, "unsafe attacker_states items must be non-empty objects"

            required_attack_fields = {
                "attacker_intent",
                "goal_type",
                "injection_points",
                "attacker_states",
            }
            missing_attack = sorted(required_attack_fields - set(attack_info.keys()))
            if missing_attack:
                return False, f"unsafe attack_info missing fields: {missing_attack}"

            goal_type = attack_info.get("goal_type")
            valid_goal_types = {
                "target_redirection",
                "goal_deviation",
                "data_exfiltration",
                "authorization_bypass",
                "hybrid",
            }
            if goal_type not in valid_goal_types:
                return False, f"unsafe goal_type invalid: {goal_type}"

            injection_points = attack_info.get("injection_points")
            if not isinstance(injection_points, list):
                return False, "unsafe attack_info.injection_points must be list"
            if len(injection_points) == 0:
                return False, "unsafe attack_info.injection_points must be non-empty"
            for idx, point in enumerate(injection_points):
                if not isinstance(point, dict):
                    return False, f"injection_points[{idx}] must be object"
                for key in ("location", "mechanism", "target"):
                    if not isinstance(point.get(key), str) or not str(point.get(key)).strip():
                        return False, f"injection_points[{idx}].{key} must be non-empty string"
                if point.get("mechanism") not in {"M1", "M2", "M3", "M4", "M5"}:
                    return False, f"injection_points[{idx}].mechanism must be one of M1..M5"

            profile = taxonomy.injection_profile
            pattern = profile.get("injection_pattern")
            point_count = profile.get("injection_point_count")
            if point_count != len(injection_points):
                return (
                    False,
                    "taxonomy.injection_profile.injection_point_count must equal "
                    "len(attack_info.injection_points)",
                )
            if pattern == "single_point" and len(injection_points) != 1:
                return False, "single_point requires exactly 1 injection point"
            if pattern == "multi_point" and len(injection_points) < 2:
                return False, "multi_point requires >=2 injection points"

            forbidden_attack_fields = {
                "success_conditions",
                "safe_primary_anchor_tool",
                "unsafe_primary_anchor_tool",
                "unsafe_anchor_tools",
                "same_anchor_exception",
            }
            present_forbidden = sorted(forbidden_attack_fields & set(attack_info.keys()))
            if present_forbidden:
                return False, f"unsafe attack_info has forbidden fields: {present_forbidden}"

        return True, ""
