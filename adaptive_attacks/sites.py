"""Resolve injection fields and write isolated derived case trees."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import re
import shutil
from typing import Any, Iterable

from core.case_loader import CaseLoader
from core.message_types import Case, pair_base_id

from .base import AttackCandidate, InjectionSite, JsonPathPart


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Case file must contain one JSON object: {path}")
    return payload


@dataclass(frozen=True)
class CasePair:
    source_root: Path
    safe_path: Path
    unsafe_path: Path
    safe_payload: dict[str, Any]
    unsafe_payload: dict[str, Any]

    @property
    def pair_base_id(self) -> str:
        return pair_base_id(str(self.unsafe_payload.get("id", "")))

    @property
    def domain(self) -> str:
        return str(self.unsafe_payload.get("domain", "Unknown") or "Unknown")

    @property
    def unsafe_variant(self) -> str:
        parent = self.unsafe_path.parent.name
        return parent if parent.lower().startswith("u") else self.unsafe_path.stem

    @property
    def run_id(self) -> str:
        return f"{self.pair_base_id}__{self.unsafe_variant}"


def discover_case_pairs(cases_root: str | Path) -> list[CasePair]:
    root = Path(cases_root).resolve()
    safe_index: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    unsafe_entries: list[tuple[Path, dict[str, Any]]] = []

    for path in sorted(root.rglob("*.json")):
        if path.name == "manifest.json":
            continue
        try:
            payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        label = payload.get("label", {})
        if not isinstance(label, dict):
            continue
        safety = str(label.get("safety", "") or "").lower()
        case_id = str(payload.get("id", "") or "")
        domain = str(payload.get("domain", "") or "")
        if not case_id or safety not in {"safe", "unsafe"}:
            continue
        key = (domain.lower(), pair_base_id(case_id))
        if safety == "safe":
            safe_index[key] = (path, payload)
        else:
            unsafe_entries.append((path, payload))

    pairs: list[CasePair] = []
    for unsafe_path, unsafe_payload in unsafe_entries:
        key = (
            str(unsafe_payload.get("domain", "") or "").lower(),
            pair_base_id(str(unsafe_payload.get("id", "") or "")),
        )
        safe_entry = safe_index.get(key)
        if safe_entry is None:
            continue
        safe_path, safe_payload = safe_entry
        pairs.append(
            CasePair(
                source_root=root,
                safe_path=safe_path,
                unsafe_path=unsafe_path,
                safe_payload=safe_payload,
                unsafe_payload=unsafe_payload,
            )
        )
    return pairs


def stratified_sample_pairs(
    pairs: Iterable[CasePair],
    *,
    sample_size: int,
    seed: int,
) -> list[CasePair]:
    all_pairs = list(pairs)
    if sample_size <= 0 or sample_size >= len(all_pairs):
        return sorted(all_pairs, key=lambda item: (item.domain, item.run_id))

    rng = random.Random(seed)
    buckets: dict[str, list[CasePair]] = {}
    for pair in all_pairs:
        buckets.setdefault(pair.domain, []).append(pair)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: list[CasePair] = []
    domain_names = sorted(buckets)
    while len(selected) < sample_size:
        progressed = False
        for domain in domain_names:
            bucket = buckets[domain]
            if bucket and len(selected) < sample_size:
                selected.append(bucket.pop())
                progressed = True
        if not progressed:
            break
    return selected


def _scalar_identity_values(record: Any, *, id_only: bool) -> set[str]:
    if not isinstance(record, dict):
        return set()
    values: set[str] = set()
    for key, value in record.items():
        key_text = str(key).lower()
        if id_only and not (key_text == "id" or key_text.endswith("_id")):
            continue
        if isinstance(value, (str, int, float)):
            values.add(str(value))
    return values


def _find_record_index(records: Any, selectors: Iterable[str]) -> int | None:
    if not isinstance(records, list):
        return None
    normalized = list(dict.fromkeys(str(value) for value in selectors if str(value)))
    for id_only in (True, False):
        for selector in normalized:
            for index, record in enumerate(records):
                if selector in _scalar_identity_values(record, id_only=id_only):
                    return index
    return None


def _parse_location(
    location: str,
    environment: dict[str, Any],
) -> tuple[str, list[str], tuple[JsonPathPart, ...] | None]:
    """Parse the location variants present across AgentDET case generations."""

    raw = str(location or "").strip()
    collection = ""
    for candidate in sorted(environment, key=len, reverse=True):
        prefixes = tuple(candidate + suffix for suffix in (":", "[", ".", " "))
        if raw == candidate or raw.startswith(prefixes):
            collection = candidate
            break
    text_field_names = {
        "body",
        "body_text",
        "content",
        "description",
        "message",
        "note",
        "subject",
        "text",
        "title",
    }
    if not collection:
        tokens = [
            token.strip("[](),.;:")
            for token in raw.split()
            if token.strip("[](),.;:")
        ]
        matching_fields = [token for token in tokens if token in text_field_names]
        explicit = (matching_fields[-1],) if matching_fields else None
        return "", tokens, explicit

    remainder = raw[len(collection) :].strip()
    selectors: list[str] = []
    explicit_field: tuple[JsonPathPart, ...] | None = None

    bracket_match = re.fullmatch(r"\[([^\]]+)\](?:\.(.+))?", remainder)
    if bracket_match:
        bracket_selector = bracket_match.group(1).strip()
        if "=" in bracket_selector:
            _, bracket_selector = bracket_selector.split("=", 1)
        selectors.append(bracket_selector.strip())
        if bracket_match.group(2):
            explicit_field = tuple(
                part for part in bracket_match.group(2).split(".") if part
            )
        return collection, selectors, explicit_field

    if remainder.startswith(":"):
        body = remainder[1:].strip()
        collection_value = environment.get(collection)
        if isinstance(collection_value, dict) and body in {
            str(key) for key in collection_value
        }:
            selectors.append(body)
            return collection, selectors, None
        if " " in body:
            tokens = [token.strip("[](),.;:") for token in body.split() if token.strip()]
            if tokens:
                selectors.append(tokens[0])
            matching_fields = [token for token in tokens if token in text_field_names]
            explicit_field = (matching_fields[-1],) if matching_fields else None
            return collection, selectors, explicit_field
        selector, dot, field_name = body.partition(".")
        if selector:
            selectors.append(selector.strip())
        if dot and field_name:
            explicit_field = tuple(part for part in field_name.split(".") if part)
        return collection, selectors, explicit_field

    if remainder.startswith(".") and "=" in remainder:
        identity_field, selector_and_field = remainder[1:].split("=", 1)
        selector, dot, field_name = selector_and_field.partition(".")
        if selector.strip():
            selectors.append(selector.strip())
        if dot and field_name:
            explicit_field = tuple(part for part in field_name.split(".") if part)
        if not identity_field.strip():
            return "", [], None
        return collection, selectors, explicit_field

    if remainder.startswith("."):
        body = remainder[1:].strip()
        field_match = re.fullmatch(r"([a-zA-Z0-9_]+)\s+(?:on|for|at)\s+(.+)", body)
        if field_match:
            explicit_field = (field_match.group(1),)
            selector_text = field_match.group(2).strip()
            selectors.extend(
                token.strip("[](),.;:")
                for token in selector_text.split()
                if token.strip("[](),.;:")
            )
            return collection, selectors, explicit_field
        if body:
            selectors.append(body)
        return collection, selectors, None

    if remainder:
        tokens = [
            token.strip("[](),.;:")
            for token in remainder.split()
            if token.strip("[](),.;:")
        ]
        selectors.extend(tokens)
        matching_fields = [token for token in tokens if token in text_field_names]
        explicit_field = (matching_fields[-1],) if matching_fields else None
    return collection, selectors, explicit_field


def _changed_string_paths(
    safe_value: Any,
    unsafe_value: Any,
    *,
    prefix: tuple[JsonPathPart, ...] = (),
) -> list[tuple[tuple[JsonPathPart, ...], str, str]]:
    changes: list[tuple[tuple[JsonPathPart, ...], str, str]] = []
    if isinstance(safe_value, str) and isinstance(unsafe_value, str):
        if safe_value != unsafe_value:
            changes.append((prefix, safe_value, unsafe_value))
        return changes
    if isinstance(safe_value, dict) and isinstance(unsafe_value, dict):
        for key in sorted(set(safe_value) & set(unsafe_value), key=str):
            changes.extend(
                _changed_string_paths(
                    safe_value[key],
                    unsafe_value[key],
                    prefix=(*prefix, str(key)),
                )
            )
        return changes
    if isinstance(safe_value, list) and isinstance(unsafe_value, list):
        for index in range(min(len(safe_value), len(unsafe_value))):
            changes.extend(
                _changed_string_paths(
                    safe_value[index],
                    unsafe_value[index],
                    prefix=(*prefix, index),
                )
            )
    return changes


def _find_mapping_key(mapping: Any, selectors: Iterable[str]) -> str | None:
    if not isinstance(mapping, dict):
        return None
    normalized = list(dict.fromkeys(str(value) for value in selectors if str(value)))
    string_keys = {str(key): key for key in mapping}
    for selector in normalized:
        if selector in string_keys:
            return str(string_keys[selector])
    return None


def _normalized_words(value: str) -> set[str]:
    words = {
        word
        for word in re.split(r"[^a-zA-Z0-9]+", str(value or "").lower())
        if word and word not in {"read", "get", "fetch", "load", "tool"}
    }
    normalized: set[str] = set()
    for word in words:
        normalized.add(word)
        if word.endswith("ies") and len(word) > 3:
            normalized.add(word[:-3] + "y")
        elif word.endswith("s") and len(word) > 3:
            normalized.add(word[:-1])
    return normalized


def _collection_similarity(location: str, collection: str) -> int:
    return len(_normalized_words(location) & _normalized_words(collection))


def _first_changed_record_indices(
    safe_records: Any,
    unsafe_records: Any,
) -> tuple[int, int] | None:
    if not isinstance(unsafe_records, list):
        return None
    safe_list = safe_records if isinstance(safe_records, list) else []
    for unsafe_index, unsafe_record in enumerate(unsafe_records):
        identity_values = _scalar_identity_values(unsafe_record, id_only=True)
        safe_index = _find_record_index(safe_list, identity_values)
        if safe_index is None and unsafe_index < len(safe_list):
            safe_index = unsafe_index
        if safe_index is None:
            if _added_record_text_paths(unsafe_record):
                return unsafe_index, -1
            continue
        if _changed_string_paths(safe_list[safe_index], unsafe_record):
            return unsafe_index, safe_index
    return None


def _changed_record_indices_by_content(
    safe_records: Any,
    unsafe_records: Any,
    selectors: Iterable[str],
) -> tuple[int, int] | None:
    if not isinstance(unsafe_records, list):
        return None
    needles = [
        str(value).strip().lower()
        for value in selectors
        if len(str(value).strip()) >= 4
    ]
    if not needles:
        return None
    safe_list = safe_records if isinstance(safe_records, list) else []
    for unsafe_index, unsafe_record in enumerate(unsafe_records):
        identity_values = _scalar_identity_values(unsafe_record, id_only=True)
        safe_index = _find_record_index(safe_list, identity_values)
        if safe_index is None and unsafe_index < len(safe_list):
            safe_index = unsafe_index
        if safe_index is None:
            changes = _added_record_text_paths(unsafe_record)
            safe_marker = -1
        else:
            changes = _changed_string_paths(safe_list[safe_index], unsafe_record)
            safe_marker = safe_index
        changed_text = " ".join(change[2] for change in changes).lower()
        if changed_text and any(needle in changed_text for needle in needles):
            return unsafe_index, safe_marker
    return None


def _added_record_text_paths(
    value: Any,
    *,
    prefix: tuple[JsonPathPart, ...] = (),
) -> list[tuple[tuple[JsonPathPart, ...], str, str]]:
    """Identify text payload fields when an entire untrusted record was added."""

    changes: list[tuple[tuple[JsonPathPart, ...], str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            is_identity = key_text == "id" or key_text.endswith("_id")
            if isinstance(item, str) and not is_identity and any(
                marker in key_text
                for marker in (
                    "body",
                    "content",
                    "description",
                    "message",
                    "note",
                    "subject",
                    "text",
                    "title",
                )
            ):
                changes.append(((*prefix, str(key)), "", item))
            elif isinstance(item, (dict, list)):
                changes.extend(
                    _added_record_text_paths(item, prefix=(*prefix, str(key)))
                )
        return changes
    if isinstance(value, list):
        for index, item in enumerate(value):
            changes.extend(
                _added_record_text_paths(item, prefix=(*prefix, index))
            )
    return changes


def resolve_injection_sites(
    safe_payload: dict[str, Any],
    unsafe_payload: dict[str, Any],
    *,
    strict: bool = True,
) -> list[InjectionSite]:
    safe_environment = safe_payload.get("environment_data", {})
    unsafe_environment = unsafe_payload.get("environment_data", {})
    attack_info = unsafe_payload.get("attack_info", {})
    points = attack_info.get("injection_points", []) if isinstance(attack_info, dict) else []
    if not isinstance(safe_environment, dict) or not isinstance(unsafe_environment, dict):
        raise ValueError("Both cases must contain object-valued environment_data.")
    if not isinstance(points, list) or not points:
        raise ValueError("Unsafe case has no declared injection points.")

    sites_by_path: dict[tuple[JsonPathPart, ...], InjectionSite] = {}
    unresolved: list[str] = []
    for point_index, point in enumerate(points):
        if not isinstance(point, dict):
            unresolved.append(f"point[{point_index}]")
            continue
        location = str(point.get("location", "") or "")
        collection, location_selectors, explicit_field = _parse_location(
            location,
            unsafe_environment,
        )
        target = str(point.get("target", "") or "")
        selectors = [*location_selectors, target]
        candidate_collections = (
            [collection]
            if collection and collection in unsafe_environment
            else sorted(
                set(safe_environment) | set(unsafe_environment),
                key=lambda value: (-_collection_similarity(location, value), value),
            )
        )
        resolved_point = False
        for candidate_collection in candidate_collections:
            unsafe_records = unsafe_environment.get(candidate_collection)
            safe_records = safe_environment.get(candidate_collection)

            if isinstance(unsafe_records, dict):
                mapping_key = _find_mapping_key(unsafe_records, selectors)
                if mapping_key is None:
                    continue
                unsafe_value = unsafe_records[mapping_key]
                safe_mapping = safe_records if isinstance(safe_records, dict) else {}
                if mapping_key in safe_mapping:
                    safe_value = safe_mapping[mapping_key]
                    changes = _changed_string_paths(safe_value, unsafe_value)
                    if (
                        isinstance(safe_value, str)
                        and isinstance(unsafe_value, str)
                        and safe_value != unsafe_value
                    ):
                        changes = [((), safe_value, unsafe_value)]
                else:
                    if isinstance(unsafe_value, str):
                        changes = [((), "", unsafe_value)]
                    else:
                        changes = _added_record_text_paths(unsafe_value)
                if not changes:
                    continue
                for relative_path, safe_text, unsafe_text in changes:
                    field_path = (candidate_collection, mapping_key, *relative_path)
                    if field_path in sites_by_path:
                        continue
                    field_label = ".".join(str(part) for part in relative_path) or "value"
                    site_id = (
                        f"site-{len(sites_by_path) + 1:02d}:"
                        f"{candidate_collection}:{mapping_key}:{field_label}"
                    )
                    sites_by_path[field_path] = InjectionSite(
                        site_id=site_id,
                        location=location,
                        mechanism=str(point.get("mechanism", "") or ""),
                        target=target,
                        field_path=field_path,
                        safe_text=safe_text,
                        unsafe_text=unsafe_text,
                        baseline_unsafe_text=unsafe_text,
                    )
                resolved_point = True
                break

            unsafe_index = _find_record_index(unsafe_records, selectors)
            safe_index = _find_record_index(safe_records, selectors)

            if unsafe_index is None and not collection:
                content_pair = _changed_record_indices_by_content(
                    safe_records,
                    unsafe_records,
                    selectors,
                )
                if content_pair is not None:
                    unsafe_index, inferred_safe_index = content_pair
                    safe_index = None if inferred_safe_index < 0 else inferred_safe_index
                else:
                    similarity = _collection_similarity(location, candidate_collection)
                if unsafe_index is None and similarity > 0:
                    changed_pair = _first_changed_record_indices(
                        safe_records,
                        unsafe_records,
                    )
                    if changed_pair is not None:
                        unsafe_index, inferred_safe_index = changed_pair
                        safe_index = None if inferred_safe_index < 0 else inferred_safe_index

            if (
                unsafe_index is None
                and collection
                and isinstance(unsafe_records, list)
                and len(unsafe_records) == 1
            ):
                unsafe_index = 0
            if safe_index is None and isinstance(safe_records, list):
                if unsafe_index is not None and unsafe_index < len(safe_records):
                    safe_index = unsafe_index
                elif collection and len(safe_records) == 1:
                    safe_index = 0
            if unsafe_index is None:
                continue

            unsafe_record = unsafe_records[unsafe_index]
            if safe_index is None:
                changes = _added_record_text_paths(unsafe_record)
            else:
                safe_record = safe_records[safe_index]
                changes = _changed_string_paths(safe_record, unsafe_record)
            if explicit_field is not None:
                filtered_changes = [
                    change
                    for change in changes
                    if change[0][: len(explicit_field)] == explicit_field
                ]
                if filtered_changes:
                    changes = filtered_changes
            if not changes:
                continue

            for relative_path, safe_text, unsafe_text in changes:
                field_path: tuple[JsonPathPart, ...] = (
                    candidate_collection,
                    unsafe_index,
                    *relative_path,
                )
                if field_path in sites_by_path:
                    continue
                field_label = ".".join(str(part) for part in relative_path)
                selector_label = next(
                    (value for value in location_selectors if value),
                    target or str(unsafe_index),
                )
                site_id = (
                    f"site-{len(sites_by_path) + 1:02d}:"
                    f"{candidate_collection}:{selector_label}:{field_label}"
                )
                sites_by_path[field_path] = InjectionSite(
                    site_id=site_id,
                    location=location,
                    mechanism=str(point.get("mechanism", "") or ""),
                    target=target,
                    field_path=field_path,
                    safe_text=safe_text,
                    unsafe_text=unsafe_text,
                    baseline_unsafe_text=unsafe_text,
                )
            resolved_point = True
            break
        if not resolved_point:
            unresolved.append(location or f"point[{point_index}]")

    if strict and unresolved:
        raise ValueError(
            "Could not resolve text fields for injection points: "
            + ", ".join(sorted(set(unresolved)))
        )
    if not sites_by_path:
        raise ValueError("No mutable injection text fields were resolved.")
    return list(sites_by_path.values())


def _set_path(root: Any, path: tuple[JsonPathPart, ...], value: Any) -> None:
    cursor = root
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = value


def apply_candidate(
    unsafe_payload: dict[str, Any],
    injection_sites: list[InjectionSite],
    candidate: AttackCandidate,
    *,
    preserve_safe_text: bool = True,
    baseline_sites: list[InjectionSite] | None = None,
    max_growth_ratio: float | None = None,
) -> dict[str, Any]:
    known_sites = {site.site_id: site for site in injection_sites}
    unknown = sorted(set(candidate.replacements) - set(known_sites))
    if unknown:
        raise ValueError(f"Candidate references unknown injection sites: {unknown}")

    baseline_by_path = {
        site.field_path: site for site in (baseline_sites or injection_sites)
    }
    derived = deepcopy(unsafe_payload)
    for site_id, replacement in candidate.replacements.items():
        site = known_sites[site_id]
        text = str(replacement or "").strip()
        if not text:
            raise ValueError(f"Replacement for {site_id} is empty.")
        if preserve_safe_text and site.safe_text not in text:
            raise ValueError(
                f"Replacement for {site_id} must preserve the original safe text verbatim."
            )
        if max_growth_ratio is not None:
            baseline = baseline_by_path.get(site.field_path, site)
            baseline_text = baseline.baseline_unsafe_text or baseline.unsafe_text
            maximum = max(
                len(baseline.safe_text),
                math.ceil(max(1, len(baseline_text)) * max(1.0, max_growth_ratio)),
            )
            if len(text) > maximum:
                raise ValueError(
                    f"Replacement for {site_id} exceeds the original payload growth budget."
                )
        _set_path(derived, ("environment_data", *site.field_path), text)

    valid, reason = CaseLoader._validate_case_data(derived)
    if not valid:
        raise ValueError(f"Derived adaptive case is invalid: {reason}")
    return derived


def payload_to_case(payload: dict[str, Any], *, source_path: Path | None = None) -> Case:
    valid, reason = CaseLoader._validate_case_data(payload)
    if not valid:
        raise ValueError(f"Adaptive case payload is invalid: {reason}")
    case = Case.from_dict(payload)
    if source_path is not None:
        setattr(case, "__source_file__", str(source_path.resolve()))
    return case


def write_derived_pair(
    pair: CasePair,
    *,
    unsafe_payload: dict[str, Any],
    output_cases_root: str | Path,
) -> tuple[Path, Path]:
    destination_root = Path(output_cases_root)
    safe_relative = pair.safe_path.relative_to(pair.source_root)
    unsafe_relative = pair.unsafe_path.relative_to(pair.source_root)
    safe_destination = destination_root / safe_relative
    unsafe_destination = destination_root / unsafe_relative
    safe_destination.parent.mkdir(parents=True, exist_ok=True)
    unsafe_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pair.safe_path, safe_destination)
    unsafe_destination.write_text(
        json.dumps(unsafe_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return safe_destination, unsafe_destination


__all__ = [
    "CasePair",
    "apply_candidate",
    "discover_case_pairs",
    "pair_base_id",
    "payload_to_case",
    "resolve_injection_sites",
    "stratified_sample_pairs",
    "write_derived_pair",
]
