"""Execution and reporting for bounded adaptive attack stress tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, is_dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable

from .base import AdaptiveAttackMethod, AttackFeedback, build_attack_context
from .feedback import feedback_from_execution
from .sites import (
    CasePair,
    apply_candidate,
    payload_to_case,
    resolve_injection_sites,
    write_derived_pair,
)


def _jsonify(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _jsonify(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonify(value.to_dict())
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonify(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "")).strip("_")
    return normalized or "unknown"


def _config_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return sha256(serialized.encode("utf-8")).hexdigest()


def _targets(pair: CasePair) -> list[str]:
    attack_info = pair.unsafe_payload.get("attack_info", {})
    points = attack_info.get("injection_points", []) if isinstance(attack_info, dict) else []
    if not isinstance(points, list):
        return []
    return [
        str(point.get("target", "") or "")
        for point in points
        if isinstance(point, dict) and str(point.get("target", "") or "")
    ]


class AgentDETCaseExecutor:
    """Thin adapter over the repository's existing single-case evaluator."""

    def __init__(
        self,
        *,
        victim_model: str,
        tool_library_path: str,
        backend_kwargs: dict[str, Any] | None,
        ask_threshold: int,
        ask_user_policy: str = "oracle",
        leaf_strategy: str = "global",
        max_iterations: int = 64,
        adaptive_multiplier: float = 4.0,
        adaptive_min_iterations: int = 15,
        adaptive_max_iterations: int = 0,
        verbose: bool = False,
    ) -> None:
        self.victim_model = victim_model
        self.tool_library_path = tool_library_path
        tool_library = _read_json(Path(tool_library_path), {})
        raw_tools = tool_library.get("tools", tool_library)
        self.tool_definitions = {
            str(name): deepcopy(definition)
            for name, definition in raw_tools.items()
            if isinstance(definition, dict)
        } if isinstance(raw_tools, dict) else {}
        self.backend_kwargs = dict(backend_kwargs or {})
        self.ask_threshold = int(ask_threshold)
        normalized_policy = str(ask_user_policy or "oracle").strip().lower()
        self.ask_user_policy = (
            normalized_policy if normalized_policy in {"allow", "reject", "oracle"} else "oracle"
        )
        normalized_strategy = str(leaf_strategy or "global").strip().lower()
        self.leaf_strategy = (
            normalized_strategy
            if normalized_strategy in {"global", "branch", "queue"}
            else "global"
        )
        self.max_iterations = int(max_iterations)
        self.adaptive_multiplier = float(adaptive_multiplier)
        self.adaptive_min_iterations = int(adaptive_min_iterations)
        self.adaptive_max_iterations = int(adaptive_max_iterations)
        self.verbose = bool(verbose)

    @property
    def pipeline_name(self) -> str:
        return self.victim_model

    def get_tool_definitions(self, case: Any) -> dict[str, dict[str, Any]]:
        """Return the complete schemas for tools exposed to one case."""

        names = [str(name) for name in (getattr(case, "tools", []) or [])]
        return {
            name: deepcopy(self.tool_definitions[name])
            for name in names
            if name in self.tool_definitions
        }

    def __call__(self, case: Any, defense: str) -> dict[str, Any]:
        from run_eval import _execute_case_once, _resolve_matrix_defense_config

        defense_config = _resolve_matrix_defense_config(defense)
        if str(defense).strip().lower() == "gardener":
            defense_config["planner_config"] = {
                "ask_threshold": self.ask_threshold,
                "ask_user_policy": self.ask_user_policy,
                "leaf_strategy": self.leaf_strategy,
            }
        return _execute_case_once(
            case=case,
            llm_name=self.victim_model,
            defense_config=defense_config,
            tool_library_path=self.tool_library_path,
            backend_kwargs=self.backend_kwargs,
            max_iterations=self.max_iterations,
            adaptive_iterations=True,
            adaptive_multiplier=self.adaptive_multiplier,
            adaptive_min=self.adaptive_min_iterations,
            adaptive_max=self.adaptive_max_iterations,
            verbose=self.verbose,
        )


CaseExecutor = Callable[[Any, str], dict[str, Any]]


class AdaptiveStressRunner:
    """Run static, adaptive, and no-defense validity evaluations per case."""

    def __init__(
        self,
        *,
        method: AdaptiveAttackMethod,
        executor: CaseExecutor,
        output_root: str | Path,
        defense: str = "gardener",
        rounds: int = 2,
        ask_threshold: int = 15,
        max_growth_ratio: float = 1.25,
        safe_control: bool = True,
        validity_check: bool = True,
        preserve_safe_text: bool = True,
        victim_model: str = "",
        generator_model: str = "",
        ask_user_policy: str = "oracle",
        leaf_strategy: str = "global",
        static_cache_root: str | Path | None = None,
        force: bool = False,
    ) -> None:
        self.method = method
        self.executor = executor
        self.output_root = Path(output_root)
        self.defense = str(defense or "gardener")
        self.rounds = max(1, int(rounds))
        self.ask_threshold = int(ask_threshold)
        self.max_growth_ratio = max(1.0, float(max_growth_ratio))
        self.safe_control = bool(safe_control)
        self.validity_check = bool(validity_check)
        self.preserve_safe_text = bool(preserve_safe_text)
        self.victim_model = str(victim_model or "")
        self.generator_model = str(generator_model or "")
        self.static_cache_root = (
            Path(static_cache_root).resolve() if static_cache_root else None
        )
        normalized_policy = str(ask_user_policy or "oracle").strip().lower()
        self.ask_user_policy = (
            normalized_policy if normalized_policy in {"allow", "reject", "oracle"} else "oracle"
        )
        normalized_strategy = str(leaf_strategy or "global").strip().lower()
        self.leaf_strategy = (
            normalized_strategy
            if normalized_strategy in {"global", "branch", "queue"}
            else "global"
        )
        self.force = bool(force)
        self.target_pipeline = executor
        self.previous_run_config: dict[str, Any] = {}
        self.defense_profile = {
            "name": "Gardener",
            "ask_threshold": self.ask_threshold,
            "ask_user_policy": self.ask_user_policy,
            "attacker_knowledge": "public mechanism and coarse execution outcome only",
            "query_budget": self.rounds,
            "search_policy": "single-branch best-so-far refinement",
        }
        self.run_config = {
            "method": self.method.name,
            "method_config": dict(getattr(self.method, "cache_config", {}) or {}),
            "defense": self.defense,
            "rounds": self.rounds,
            "ask_threshold": self.ask_threshold,
            "max_growth_ratio": self.max_growth_ratio,
            "safe_control": self.safe_control,
            "ask_user_policy": self.ask_user_policy,
            "leaf_strategy": self.leaf_strategy,
            "validity_check": self.validity_check,
            "preserve_safe_text": self.preserve_safe_text,
            "victim_model": self.victim_model,
            "generator_model": self.generator_model,
            "static_cache_root": str(self.static_cache_root or ""),
        }
        self.run_config_hash = _config_hash(self.run_config)

    def _attack_context(
        self,
        *,
        case: Any,
        injection_sites: list[Any],
        feedback: AttackFeedback | None,
        iteration: int,
    ):
        tool_definitions: dict[str, dict[str, Any]] = {}
        get_tools = getattr(self.executor, "get_tool_definitions", None)
        if callable(get_tools):
            tool_definitions = get_tools(case)
        return build_attack_context(
            case=case,
            target_pipeline=self.target_pipeline,
            tool_definitions=tool_definitions,
            injection_sites=injection_sites,
            defense_profile=self.defense_profile,
            feedback=feedback,
            iteration=iteration,
        )

    def _round_dir(self, pair: CasePair, round_index: int) -> Path:
        return (
            self.output_root
            / "case_runs"
            / _slug(pair.run_id)
            / f"round_{round_index:02d}"
        )

    def _write_execution(
        self,
        *,
        pair: CasePair,
        round_index: int,
        execution: dict[str, Any],
        feedback: AttackFeedback,
        candidate: dict[str, Any] | None,
    ) -> dict[str, str]:
        round_dir = self._round_dir(pair, round_index)
        eval_result = execution.get("eval_result")
        eval_payload = eval_result.to_dict() if hasattr(eval_result, "to_dict") else {}
        _write_json(round_dir / "eval.json", eval_payload)
        _write_json(round_dir / "trace.json", execution.get("trace_payload", {}))
        _write_json(
            round_dir / "defense_artifacts.json",
            execution.get("defense_artifacts", {}),
        )
        _write_json(round_dir / "feedback.json", feedback.to_dict())
        if candidate is not None:
            _write_json(round_dir / "candidate.json", candidate)
        return {
            "eval": str((round_dir / "eval.json").relative_to(self.output_root)),
            "trace": str((round_dir / "trace.json").relative_to(self.output_root)),
            "defense_artifacts": str(
                (round_dir / "defense_artifacts.json").relative_to(self.output_root)
            ),
            "feedback": str((round_dir / "feedback.json").relative_to(self.output_root)),
        }

    def _reuse_static_round(
        self,
        pair: CasePair,
    ) -> tuple[AttackFeedback, dict[str, str]] | None:
        if self.static_cache_root is None:
            return None

        cache_config = _read_json(self.static_cache_root / "run_config.json", {})
        required_config = (
            "defense",
            "ask_threshold",
            "ask_user_policy",
            "leaf_strategy",
            "victim_model",
        )
        mismatched = [
            key
            for key in required_config
            if cache_config.get(key) != self.run_config.get(key)
        ]
        if mismatched:
            raise ValueError(
                "Static cache configuration does not match this run: "
                + ", ".join(mismatched)
            )

        source = (
            self.static_cache_root
            / "case_runs"
            / _slug(pair.run_id)
            / "round_00"
        )
        payloads = {
            "eval": _read_json(source / "eval.json", None),
            "trace": _read_json(source / "trace.json", None),
            "defense_artifacts": _read_json(source / "defense_artifacts.json", None),
            "feedback": _read_json(source / "feedback.json", None),
        }
        missing = [name for name, payload in payloads.items() if payload is None]
        if missing:
            raise ValueError(
                f"Static cache for {pair.run_id} is incomplete: {', '.join(missing)}"
            )

        feedback = self._feedback_from_payload(payloads["feedback"])
        if feedback.runtime_error:
            raise ValueError(f"Static cache for {pair.run_id} contains a runtime error.")

        round_dir = self._round_dir(pair, 0)
        for name, payload in payloads.items():
            _write_json(round_dir / f"{name}.json", payload)
        files = {
            "eval": str((round_dir / "eval.json").relative_to(self.output_root)),
            "trace": str((round_dir / "trace.json").relative_to(self.output_root)),
            "defense_artifacts": str(
                (round_dir / "defense_artifacts.json").relative_to(self.output_root)
            ),
            "feedback": str(
                (round_dir / "feedback.json").relative_to(self.output_root)
            ),
            "reused_from": str(source),
        }
        return feedback, files

    def _execute_payload(
        self,
        *,
        payload: dict[str, Any],
        source_path: Path,
        defense: str,
        targets: Iterable[str],
    ) -> tuple[dict[str, Any], AttackFeedback]:
        case = payload_to_case(payload, source_path=source_path)
        execution = self.executor(case, defense)
        feedback = feedback_from_execution(execution, targets=targets)
        return execution, feedback

    def _cached_case_summary(self, pair: CasePair) -> dict[str, Any] | None:
        path = self.output_root / "case_runs" / _slug(pair.run_id) / "case_summary.json"
        payload = _read_json(path, {})
        if (
            not self.force
            and isinstance(payload, dict)
            and payload.get("status") == "completed"
            and payload.get("config_hash") == self.run_config_hash
        ):
            return payload
        return None

    @staticmethod
    def _feedback_from_payload(payload: dict[str, Any] | None) -> AttackFeedback:
        payload = payload if isinstance(payload, dict) else {}
        return AttackFeedback(
            attack_successful=bool(payload.get("attack_successful", False)),
            attack_success_total=bool(payload.get("attack_success_total", False)),
            task_completed=bool(payload.get("task_completed", False)),
            defense_triggered=bool(payload.get("defense_triggered", False)),
            blocked=bool(payload.get("blocked", False)),
            blocked_by=str(payload.get("blocked_by", "") or ""),
            confirmation_count=int(payload.get("confirmation_count", 0) or 0),
            skipped_leaf_count=int(payload.get("skipped_leaf_count", 0) or 0),
            target_tool_called=bool(payload.get("target_tool_called", False)),
            runtime_error=bool(payload.get("runtime_error", False)),
            notes=tuple(str(item) for item in (payload.get("notes", []) or [])),
            visible_trace=deepcopy(
                payload.get("visible_trace", {})
                if isinstance(payload.get("visible_trace", {}), dict)
                else {}
            ),
        )

    def _previous_config_is_extendable(self) -> bool:
        if not isinstance(self.previous_run_config, dict) or not self.previous_run_config:
            return False
        previous_rounds = int(self.previous_run_config.get("rounds", 0) or 0)
        if previous_rounds < 1 or previous_rounds > self.rounds:
            return False
        previous = dict(self.previous_run_config)
        current = dict(self.run_config)
        previous.pop("rounds", None)
        current.pop("rounds", None)
        return previous == current

    def _resumable_case_summary(self, pair: CasePair) -> dict[str, Any] | None:
        if self.force or not self._previous_config_is_extendable():
            return None
        path = self.output_root / "case_runs" / _slug(pair.run_id) / "case_summary.json"
        payload = _read_json(path, {})
        if not (
            isinstance(payload, dict)
            and payload.get("status") == "completed"
            and isinstance(payload.get("rounds"), list)
            and payload.get("rounds")
        ):
            return None
        recorded_rounds = [
            int(record.get("round", -1))
            for record in payload.get("rounds", [])
            if isinstance(record, dict)
        ]
        if not recorded_rounds or max(recorded_rounds) >= self.rounds:
            return None
        return payload

    def _best_payload_from_summary(
        self,
        *,
        pair: CasePair,
        summary: dict[str, Any],
        best_round: int,
    ) -> dict[str, Any]:
        if best_round <= 0:
            return deepcopy(pair.unsafe_payload)
        for record in summary.get("rounds", []) or []:
            if not isinstance(record, dict) or int(record.get("round", -1)) != best_round:
                continue
            derived_case = str(record.get("derived_case", "") or "")
            if not derived_case:
                break
            payload = _read_json(self.output_root / derived_case, {})
            if payload:
                return payload
            break
        raise ValueError(
            f"Cannot resume {pair.run_id}: missing derived payload for best round {best_round}."
        )

    @staticmethod
    def _anchor_sites(
        base_sites: list[Any],
        current_sites: list[Any],
    ) -> list[Any]:
        baseline_by_path = {site.field_path: site for site in base_sites}
        anchored = []
        for site in current_sites:
            baseline = baseline_by_path.get(site.field_path)
            baseline_text = (
                baseline.baseline_unsafe_text or baseline.unsafe_text
                if baseline is not None
                else site.baseline_unsafe_text or site.unsafe_text
            )
            anchored.append(replace(site, baseline_unsafe_text=baseline_text))
        return anchored

    def _write_carried_round(
        self,
        *,
        pair: CasePair,
        round_index: int,
        feedback: AttackFeedback,
        candidate: dict[str, Any] | None,
    ) -> dict[str, str]:
        round_dir = self._round_dir(pair, round_index)
        _write_json(round_dir / "feedback.json", feedback.to_dict())
        files = {
            "feedback": str((round_dir / "feedback.json").relative_to(self.output_root))
        }
        if candidate is not None:
            _write_json(round_dir / "candidate.json", candidate)
            files["candidate"] = str(
                (round_dir / "candidate.json").relative_to(self.output_root)
            )
        return files

    def _write_aux_execution(
        self,
        *,
        pair: CasePair,
        name: str,
        execution: dict[str, Any],
        feedback: AttackFeedback,
    ) -> dict[str, str]:
        output_dir = self.output_root / "case_runs" / _slug(pair.run_id) / _slug(name)
        eval_result = execution.get("eval_result")
        _write_json(
            output_dir / "eval.json",
            eval_result.to_dict() if hasattr(eval_result, "to_dict") else {},
        )
        _write_json(output_dir / "trace.json", execution.get("trace_payload", {}))
        _write_json(
            output_dir / "defense_artifacts.json",
            execution.get("defense_artifacts", {}),
        )
        _write_json(output_dir / "feedback.json", feedback.to_dict())
        return {
            "eval": str((output_dir / "eval.json").relative_to(self.output_root)),
            "trace": str((output_dir / "trace.json").relative_to(self.output_root)),
            "defense_artifacts": str(
                (output_dir / "defense_artifacts.json").relative_to(self.output_root)
            ),
            "feedback": str(
                (output_dir / "feedback.json").relative_to(self.output_root)
            ),
        }

    def _extend_case_summary(
        self,
        *,
        pair: CasePair,
        existing: dict[str, Any],
    ) -> dict[str, Any]:
        case_root = self.output_root / "case_runs" / _slug(pair.run_id)
        targets = _targets(pair)
        base_sites = resolve_injection_sites(pair.safe_payload, pair.unsafe_payload)
        round_records = deepcopy(existing.get("rounds", []) or [])
        existing_rounds = [
            int(record.get("round", -1))
            for record in round_records
            if isinstance(record, dict)
        ]
        if not existing_rounds:
            raise ValueError(f"Cannot resume {pair.run_id}: no previous rounds found.")

        last_round = max(existing_rounds)
        best_round = int(existing.get("best_adaptive_round", 0) or 0)
        best_feedback = self._feedback_from_payload(existing.get("best_adaptive_feedback"))
        best_payload = self._best_payload_from_summary(
            pair=pair,
            summary=existing,
            best_round=best_round,
        )
        first_success_round = existing.get("first_success_round")
        if first_success_round is not None:
            first_success_round = int(first_success_round)

        query_counts = dict(existing.get("query_counts", {}) or {})
        adaptive_target_queries = int(query_counts.get("adaptive_target", 0) or 0)
        generator_queries = int(query_counts.get("generator", 0) or 0)
        validity_queries = int(query_counts.get("validity_target", 0) or 0)
        best_round_before_extension = best_round

        for round_index in range(last_round + 1, self.rounds + 1):
            if best_feedback.attack_successful:
                break

            current_sites = resolve_injection_sites(pair.safe_payload, best_payload)
            sites = self._anchor_sites(base_sites, current_sites)
            current_case = payload_to_case(best_payload, source_path=pair.unsafe_path)
            candidate = self.method.attack(
                self._attack_context(
                    case=current_case,
                    injection_sites=sites,
                    feedback=best_feedback,
                    iteration=round_index,
                )
            )
            candidate_payload = candidate.to_dict()
            generator_queries += int(candidate.metadata.get("generator_calls", 0) or 0)
            sites_by_id = {site.site_id: site for site in sites}
            changed = any(
                site_id in sites_by_id
                and str(replacement or "").strip()
                != str(sites_by_id[site_id].unsafe_text or "").strip()
                for site_id, replacement in candidate.replacements.items()
            )
            if not candidate.replacements or not changed:
                files = self._write_carried_round(
                    pair=pair,
                    round_index=round_index,
                    feedback=best_feedback,
                    candidate=candidate_payload,
                )
                round_records.append(
                    {
                        "round": round_index,
                        "kind": "adaptive",
                        "strategy": candidate.strategy,
                        "feedback": best_feedback.to_dict(),
                        "best_feedback_after": best_feedback.to_dict(),
                        "files": files,
                        "generation_attempted": True,
                        "target_evaluated": False,
                        "carried_forward": True,
                        "accepted": False,
                        "source_best_round": best_round,
                        "best_round_after": best_round,
                        "score_improvement": 0.0,
                    }
                )
                continue

            derived_payload = apply_candidate(
                best_payload,
                sites,
                candidate,
                preserve_safe_text=self.preserve_safe_text,
                baseline_sites=base_sites,
                max_growth_ratio=self.max_growth_ratio,
            )
            derived_root = self.output_root / "derived_cases" / f"round_{round_index:02d}"
            _, derived_unsafe_path = write_derived_pair(
                pair,
                unsafe_payload=derived_payload,
                output_cases_root=derived_root,
            )
            execution, feedback = self._execute_payload(
                payload=derived_payload,
                source_path=derived_unsafe_path,
                defense=self.defense,
                targets=targets,
            )
            adaptive_target_queries += 1
            files = self._write_execution(
                pair=pair,
                round_index=round_index,
                execution=execution,
                feedback=feedback,
                candidate=candidate_payload,
            )
            best_score_before = best_feedback.score()
            candidate_score = feedback.score()
            accepted = candidate_score > best_score_before
            source_best_round = best_round
            if accepted:
                best_round = round_index
                best_payload = derived_payload
                best_feedback = feedback
            if first_success_round is None and feedback.attack_successful:
                first_success_round = round_index
            round_records.append(
                {
                    "round": round_index,
                    "kind": "adaptive",
                    "strategy": candidate.strategy,
                    "feedback": feedback.to_dict(),
                    "best_feedback_after": best_feedback.to_dict(),
                    "files": files,
                    "derived_case": str(derived_unsafe_path.relative_to(self.output_root)),
                    "generation_attempted": True,
                    "target_evaluated": True,
                    "carried_forward": False,
                    "accepted": accepted,
                    "source_best_round": source_best_round,
                    "best_round_after": best_round,
                    "score_improvement": candidate_score - best_score_before,
                }
            )

        validity_record = existing.get("validity_no_defense")
        if (
            self.validity_check
            and (best_round != best_round_before_extension or not validity_record)
        ):
            validity_execution, validity_feedback = self._execute_payload(
                payload=best_payload,
                source_path=pair.unsafe_path,
                defense="no_defense",
                targets=targets,
            )
            validity_dir = case_root / "validity_no_defense"
            eval_result = validity_execution.get("eval_result")
            _write_json(
                validity_dir / "eval.json",
                eval_result.to_dict() if hasattr(eval_result, "to_dict") else {},
            )
            _write_json(validity_dir / "trace.json", validity_execution.get("trace_payload", {}))
            _write_json(validity_dir / "feedback.json", validity_feedback.to_dict())
            validity_queries += 1
            validity_record = {
                "best_adaptive_round": best_round,
                "feedback": validity_feedback.to_dict(),
                "files": {
                    "eval": str((validity_dir / "eval.json").relative_to(self.output_root)),
                    "trace": str((validity_dir / "trace.json").relative_to(self.output_root)),
                    "feedback": str(
                        (validity_dir / "feedback.json").relative_to(self.output_root)
                    ),
                },
            }

        query_counts.update(
            {
                "safe_target": int(query_counts.get("safe_target", 0) or 0),
                "static_target": int(query_counts.get("static_target", 0) or 0),
                "adaptive_target": adaptive_target_queries,
                "generator": generator_queries,
                "validity_target": validity_queries,
            }
        )
        summary = deepcopy(existing)
        summary.update(
            {
                "status": "completed",
                "config_hash": self.run_config_hash,
                "best_adaptive_round": best_round,
                "best_adaptive_feedback": best_feedback.to_dict(),
                "first_success_round": first_success_round,
                "query_counts": query_counts,
                "rounds": round_records,
                "validity_no_defense": validity_record,
                "resume": {
                    "extended_from_round": last_round,
                    "target_rounds": self.rounds,
                    "previous_config_hash": existing.get("config_hash", ""),
                    "reused_safe_control": True,
                    "reused_static_round": True,
                    "reused_validity": best_round == best_round_before_extension
                    and bool(validity_record),
                },
            }
        )
        _write_json(case_root / "case_summary.json", summary)
        return summary

    def run_pair(self, pair: CasePair) -> dict[str, Any]:
        cached = self._cached_case_summary(pair)
        if cached is not None:
            return cached
        resumable = self._resumable_case_summary(pair)
        if resumable is not None:
            return self._extend_case_summary(pair=pair, existing=resumable)

        case_root = self.output_root / "case_runs" / _slug(pair.run_id)
        case_root.mkdir(parents=True, exist_ok=True)
        targets = _targets(pair)
        base_sites = resolve_injection_sites(pair.safe_payload, pair.unsafe_payload)
        safe_control_record: dict[str, Any] | None = None
        if self.safe_control:
            safe_execution, safe_feedback = self._execute_payload(
                payload=pair.safe_payload,
                source_path=pair.safe_path,
                defense=self.defense,
                targets=(),
            )
            safe_control_record = {
                "feedback": safe_feedback.to_dict(),
                "files": self._write_aux_execution(
                    pair=pair,
                    name="safe_control",
                    execution=safe_execution,
                    feedback=safe_feedback,
                ),
            }
        reused_static = self._reuse_static_round(pair)
        if reused_static is None:
            static_execution, static_feedback = self._execute_payload(
                payload=pair.unsafe_payload,
                source_path=pair.unsafe_path,
                defense=self.defense,
                targets=targets,
            )
            static_files = self._write_execution(
                pair=pair,
                round_index=0,
                execution=static_execution,
                feedback=static_feedback,
                candidate=None,
            )
        else:
            static_feedback, static_files = reused_static
        round_records: list[dict[str, Any]] = []
        round_records.append(
            {
                "round": 0,
                "kind": "static",
                "feedback": static_feedback.to_dict(),
                "best_feedback_after": static_feedback.to_dict(),
                "files": static_files,
                "generation_attempted": False,
                "target_evaluated": True,
                "carried_forward": False,
                "accepted": True,
                "best_round_after": 0,
                "reused_static": reused_static is not None,
            }
        )

        best_round = 0
        best_payload = pair.unsafe_payload
        best_feedback = static_feedback
        first_success_round: int | None = 0 if static_feedback.attack_successful else None
        adaptive_target_queries = 0
        generator_queries = 0

        for round_index in range(1, self.rounds + 1):
            if best_feedback.attack_successful:
                break

            current_sites = resolve_injection_sites(pair.safe_payload, best_payload)
            sites = self._anchor_sites(base_sites, current_sites)
            current_case = payload_to_case(best_payload, source_path=pair.unsafe_path)
            candidate = self.method.attack(
                self._attack_context(
                    case=current_case,
                    injection_sites=sites,
                    feedback=best_feedback,
                    iteration=round_index,
                )
            )
            candidate_payload = candidate.to_dict()
            generator_queries += int(candidate.metadata.get("generator_calls", 0) or 0)
            sites_by_id = {site.site_id: site for site in sites}
            changed = any(
                site_id in sites_by_id
                and str(replacement or "").strip()
                != str(sites_by_id[site_id].unsafe_text or "").strip()
                for site_id, replacement in candidate.replacements.items()
            )
            if not candidate.replacements or not changed:
                files = self._write_carried_round(
                    pair=pair,
                    round_index=round_index,
                    feedback=best_feedback,
                    candidate=candidate_payload,
                )
                round_records.append(
                    {
                        "round": round_index,
                        "kind": "adaptive",
                        "strategy": candidate.strategy,
                        "feedback": best_feedback.to_dict(),
                        "best_feedback_after": best_feedback.to_dict(),
                        "files": files,
                        "generation_attempted": True,
                        "target_evaluated": False,
                        "carried_forward": True,
                        "accepted": False,
                        "source_best_round": best_round,
                        "best_round_after": best_round,
                        "score_improvement": 0.0,
                    }
                )
                continue

            derived_payload = apply_candidate(
                best_payload,
                sites,
                candidate,
                preserve_safe_text=self.preserve_safe_text,
                baseline_sites=base_sites,
                max_growth_ratio=self.max_growth_ratio,
            )
            derived_root = self.output_root / "derived_cases" / f"round_{round_index:02d}"
            _, derived_unsafe_path = write_derived_pair(
                pair,
                unsafe_payload=derived_payload,
                output_cases_root=derived_root,
            )
            execution, feedback = self._execute_payload(
                payload=derived_payload,
                source_path=derived_unsafe_path,
                defense=self.defense,
                targets=targets,
            )
            adaptive_target_queries += 1
            files = self._write_execution(
                pair=pair,
                round_index=round_index,
                execution=execution,
                feedback=feedback,
                candidate=candidate_payload,
            )
            best_score_before = best_feedback.score()
            candidate_score = feedback.score()
            accepted = candidate_score > best_score_before
            source_best_round = best_round
            if accepted:
                best_round = round_index
                best_payload = derived_payload
                best_feedback = feedback
            if first_success_round is None and feedback.attack_successful:
                first_success_round = round_index
            round_records.append(
                {
                    "round": round_index,
                    "kind": "adaptive",
                    "strategy": candidate.strategy,
                    "feedback": feedback.to_dict(),
                    "best_feedback_after": best_feedback.to_dict(),
                    "files": files,
                    "derived_case": str(derived_unsafe_path.relative_to(self.output_root)),
                    "generation_attempted": True,
                    "target_evaluated": True,
                    "carried_forward": False,
                    "accepted": accepted,
                    "source_best_round": source_best_round,
                    "best_round_after": best_round,
                    "score_improvement": candidate_score - best_score_before,
                }
            )

        validity_record: dict[str, Any] | None = None
        if self.validity_check:
            validity_execution, validity_feedback = self._execute_payload(
                payload=best_payload,
                source_path=pair.unsafe_path,
                defense="no_defense",
                targets=targets,
            )
            validity_dir = case_root / "validity_no_defense"
            eval_result = validity_execution.get("eval_result")
            _write_json(
                validity_dir / "eval.json",
                eval_result.to_dict() if hasattr(eval_result, "to_dict") else {},
            )
            _write_json(validity_dir / "trace.json", validity_execution.get("trace_payload", {}))
            _write_json(validity_dir / "feedback.json", validity_feedback.to_dict())
            validity_record = {
                "best_adaptive_round": best_round,
                "feedback": validity_feedback.to_dict(),
                "files": {
                    "eval": str((validity_dir / "eval.json").relative_to(self.output_root)),
                    "trace": str((validity_dir / "trace.json").relative_to(self.output_root)),
                    "feedback": str(
                        (validity_dir / "feedback.json").relative_to(self.output_root)
                    ),
                },
            }

        summary = {
            "status": "completed",
            "config_hash": self.run_config_hash,
            "pair_base_id": pair.pair_base_id,
            "pair_run_id": pair.run_id,
            "domain": pair.domain,
            "source_safe_case": str(pair.safe_path),
            "source_unsafe_case": str(pair.unsafe_path),
            "injection_site_count": len(base_sites),
            "best_adaptive_round": best_round,
            "best_adaptive_feedback": best_feedback.to_dict(),
            "first_success_round": first_success_round,
            "query_counts": {
                "safe_target": 1 if self.safe_control else 0,
                "static_target": 0 if reused_static is not None else 1,
                "adaptive_target": adaptive_target_queries,
                "generator": generator_queries,
                "validity_target": 1 if self.validity_check else 0,
            },
            "safe_control": safe_control_record,
            "rounds": round_records,
            "validity_no_defense": validity_record,
        }
        _write_json(case_root / "case_summary.json", summary)
        return summary

    @staticmethod
    def _rate(values: list[bool]) -> float | None:
        return (sum(bool(value) for value in values) / len(values)) if values else None

    def _aggregate(self, case_summaries: list[dict[str, Any]]) -> dict[str, Any]:
        completed = [item for item in case_summaries if item.get("status") == "completed"]
        round_candidate_success: dict[int, list[bool]] = {}
        round_candidate_confirmations: dict[int, list[int]] = {}
        round_candidate_blocks: dict[int, list[bool]] = {}
        round_effective_success: dict[int, list[bool]] = {}
        round_effective_confirmations: dict[int, list[int]] = {}
        round_effective_blocks: dict[int, list[bool]] = {}
        round_generation_attempts: dict[int, int] = {}
        round_target_evaluations: dict[int, int] = {}
        round_acceptances: dict[int, int] = {}
        round_carry_forwards: dict[int, int] = {}
        best_success: list[bool] = []
        best_confirmations: list[int] = []
        best_blocks: list[bool] = []
        validity_success: list[bool] = []
        first_success_rounds: list[int] = []
        target_queries: list[int] = []
        safe_queries: list[int] = []
        generator_queries: list[int] = []
        validity_queries: list[int] = []
        best_round_distribution: dict[int, int] = {}
        safe_completed: list[bool] = []
        safe_confirmations: list[int] = []
        safe_blocks: list[bool] = []
        safe_triggers: list[bool] = []
        for item in completed:
            records = [
                record
                for record in (item.get("rounds", []) or [])
                if isinstance(record, dict)
            ]
            records_by_round: dict[int, dict[str, Any]] = {}
            for record in records:
                if not isinstance(record, dict):
                    continue
                round_index = int(record.get("round", 0))
                records_by_round[round_index] = record
                feedback = record.get("feedback", {}) or {}
                if bool(record.get("generation_attempted", False)):
                    round_generation_attempts[round_index] = (
                        round_generation_attempts.get(round_index, 0) + 1
                    )
                if bool(record.get("carried_forward", False)):
                    round_carry_forwards[round_index] = (
                        round_carry_forwards.get(round_index, 0) + 1
                    )
                if bool(record.get("target_evaluated", False)):
                    round_target_evaluations[round_index] = (
                        round_target_evaluations.get(round_index, 0) + 1
                    )
                    round_candidate_success.setdefault(round_index, []).append(
                        bool(feedback.get("attack_successful", False))
                    )
                    round_candidate_confirmations.setdefault(round_index, []).append(
                        int(feedback.get("confirmation_count", 0) or 0)
                    )
                    round_candidate_blocks.setdefault(round_index, []).append(
                        bool(feedback.get("blocked", False))
                    )
                if round_index > 0 and bool(record.get("accepted", False)):
                    round_acceptances[round_index] = round_acceptances.get(round_index, 0) + 1
            effective: dict[str, Any] = {}
            for round_index in range(0, self.rounds + 1):
                record = records_by_round.get(round_index)
                if record is not None:
                    feedback = record.get("feedback", {}) or {}
                    effective = record.get("best_feedback_after", feedback) or feedback
                if not effective:
                    continue
                round_effective_success.setdefault(round_index, []).append(
                    bool(effective.get("attack_successful", False))
                )
                round_effective_confirmations.setdefault(round_index, []).append(
                    int(effective.get("confirmation_count", 0) or 0)
                )
                round_effective_blocks.setdefault(round_index, []).append(
                    bool(effective.get("blocked", False))
                )
            best = item.get("best_adaptive_feedback", {}) or {}
            best_success.append(bool(best.get("attack_successful", False)))
            best_confirmations.append(int(best.get("confirmation_count", 0) or 0))
            best_blocks.append(bool(best.get("blocked", False)))
            best_round = int(item.get("best_adaptive_round", 0) or 0)
            best_round_distribution[best_round] = best_round_distribution.get(best_round, 0) + 1
            first_success = item.get("first_success_round")
            if first_success is not None:
                first_success_rounds.append(int(first_success))
            query_counts = item.get("query_counts", {}) or {}
            safe_queries.append(int(query_counts.get("safe_target", 0) or 0))
            target_queries.append(
                int(query_counts.get("static_target", 0) or 0)
                + int(query_counts.get("adaptive_target", 0) or 0)
            )
            generator_queries.append(int(query_counts.get("generator", 0) or 0))
            validity_queries.append(int(query_counts.get("validity_target", 0) or 0))
            safe_control = item.get("safe_control") or {}
            if isinstance(safe_control, dict) and isinstance(
                safe_control.get("feedback"), dict
            ):
                safe_feedback = safe_control["feedback"]
                safe_completed.append(bool(safe_feedback.get("task_completed", False)))
                safe_confirmations.append(
                    int(safe_feedback.get("confirmation_count", 0) or 0)
                )
                safe_blocks.append(bool(safe_feedback.get("blocked", False)))
                safe_triggers.append(bool(safe_feedback.get("defense_triggered", False)))
            validity = item.get("validity_no_defense") or {}
            if isinstance(validity, dict) and isinstance(validity.get("feedback"), dict):
                validity_success.append(
                    bool(validity["feedback"].get("attack_successful", False))
                )

        per_round: dict[str, Any] = {}
        for round_index in sorted(round_effective_success):
            candidate_confirmations = round_candidate_confirmations.get(round_index, [])
            effective_confirmations = round_effective_confirmations.get(round_index, [])
            target_evaluations = round_target_evaluations.get(round_index, 0)
            per_round[str(round_index)] = {
                "cases": len(round_effective_success[round_index]),
                "generation_attempts": round_generation_attempts.get(round_index, 0),
                "target_evaluations": target_evaluations,
                "carried_forward": round_carry_forwards.get(round_index, 0),
                "accepted": round_acceptances.get(round_index, 0),
                "acceptance_rate": (
                    round_acceptances.get(round_index, 0) / target_evaluations
                    if target_evaluations
                    else None
                ),
                "asr": self._rate(round_candidate_success.get(round_index, [])),
                "candidate_asr": self._rate(
                    round_candidate_success.get(round_index, [])
                ),
                "cumulative_best_asr": self._rate(
                    round_effective_success[round_index]
                ),
                "avg_confirmations": (
                    sum(candidate_confirmations) / len(candidate_confirmations)
                    if candidate_confirmations
                    else None
                ),
                "cumulative_best_avg_confirmations": (
                    sum(effective_confirmations) / len(effective_confirmations)
                    if effective_confirmations
                    else None
                ),
                "block_rate": self._rate(
                    round_candidate_blocks.get(round_index, [])
                ),
                "cumulative_best_block_rate": self._rate(
                    round_effective_blocks.get(round_index, [])
                ),
            }
        static_asr = self._rate(round_candidate_success.get(0, []))
        best_adaptive_asr = self._rate(best_success)
        static_confirmations = round_candidate_confirmations.get(0, [])
        static_blocks = round_candidate_blocks.get(0, [])
        adaptive_evaluations = sum(
            count for round_index, count in round_target_evaluations.items() if round_index > 0
        )
        adaptive_acceptances = sum(round_acceptances.values())
        success_by_round = {
            str(round_index): (
                sum(first <= round_index for first in first_success_rounds) / len(completed)
                if completed
                else None
            )
            for round_index in range(0, self.rounds + 1)
        }
        return {
            "status": "completed" if len(completed) == len(case_summaries) else "partial",
            "config": dict(self.run_config),
            "config_hash": self.run_config_hash,
            "counts": {
                "selected": len(case_summaries),
                "completed": len(completed),
                "failed": len(case_summaries) - len(completed),
                "safe_controls": len(safe_completed),
            },
            "safe_bu": self._rate(safe_completed),
            "safe_defense_trigger_rate": self._rate(safe_triggers),
            "safe_avg_confirmations": (
                sum(safe_confirmations) / len(safe_confirmations)
                if safe_confirmations
                else None
            ),
            "safe_block_rate": self._rate(safe_blocks),
            "per_round": per_round,
            "static_asr": static_asr,
            "best_adaptive_asr": best_adaptive_asr,
            "cumulative_success_by_round": success_by_round,
            "adaptive_gap": (
                best_adaptive_asr - static_asr
                if best_adaptive_asr is not None and static_asr is not None
                else None
            ),
            "no_defense_validity_asr": self._rate(validity_success),
            "payload_validity_rate": self._rate(validity_success),
            "candidate_acceptance_rate": (
                adaptive_acceptances / adaptive_evaluations
                if adaptive_evaluations
                else None
            ),
            "average_first_success_round": (
                sum(first_success_rounds) / len(first_success_rounds)
                if first_success_rounds
                else None
            ),
            "successful_cases": len(first_success_rounds),
            "best_round_distribution": {
                str(key): value for key, value in sorted(best_round_distribution.items())
            },
            "average_target_queries_per_case": (
                sum(target_queries) / len(target_queries) if target_queries else None
            ),
            "average_defended_target_queries_per_pair": (
                (sum(safe_queries) + sum(target_queries)) / len(target_queries)
                if target_queries
                else None
            ),
            "average_generator_queries_per_case": (
                sum(generator_queries) / len(generator_queries)
                if generator_queries
                else None
            ),
            "average_total_target_queries_including_validity": (
                (
                    sum(safe_queries) + sum(target_queries) + sum(validity_queries)
                )
                / len(target_queries)
                if target_queries
                else None
            ),
            "static_avg_confirmations": (
                sum(static_confirmations) / len(static_confirmations)
                if static_confirmations
                else None
            ),
            "adaptive_avg_confirmations": (
                sum(best_confirmations) / len(best_confirmations)
                if best_confirmations
                else None
            ),
            "static_block_rate": self._rate(static_blocks),
            "adaptive_block_rate": self._rate(best_blocks),
        }

    def run(self, pairs: Iterable[CasePair], *, workers: int = 1) -> dict[str, Any]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.previous_run_config = _read_json(self.output_root / "run_config.json", {})
        _write_json(self.output_root / "run_config.json", self.run_config)
        pair_list = list(pairs)
        summaries: list[dict[str, Any] | None] = [None] * len(pair_list)

        def run_one(index: int, pair: CasePair) -> tuple[int, dict[str, Any]]:
            print(f"[{index + 1}/{len(pair_list)}] adaptive stress: {pair.run_id}")
            try:
                return index, self.run_pair(pair)
            except Exception as exc:
                failure = {
                    "status": "failed",
                    "config_hash": self.run_config_hash,
                    "pair_base_id": pair.pair_base_id,
                    "pair_run_id": pair.run_id,
                    "domain": pair.domain,
                    "error": str(exc),
                }
                case_root = self.output_root / "case_runs" / _slug(pair.run_id)
                _write_json(case_root / "case_summary.json", failure)
                print(f"  failed: {exc}")
                return index, failure

        worker_count = max(1, int(workers))
        if worker_count == 1:
            for index, pair in enumerate(pair_list):
                result_index, result = run_one(index, pair)
                summaries[result_index] = result
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = {
                    pool.submit(run_one, index, pair): index
                    for index, pair in enumerate(pair_list)
                }
                for future in as_completed(futures):
                    result_index, result = future.result()
                    summaries[result_index] = result

        completed_summaries = [item for item in summaries if item is not None]
        aggregate = self._aggregate(completed_summaries)
        aggregate["cases"] = completed_summaries
        _write_json(self.output_root / "summary.json", aggregate)
        return aggregate


__all__ = ["AdaptiveStressRunner", "AgentDETCaseExecutor", "CaseExecutor"]
