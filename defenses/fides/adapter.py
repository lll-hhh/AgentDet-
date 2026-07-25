"""
Fides runtime adapter for AgentDET.

This module adapts the core ideas in the official Fides tutorial implementation:
- dynamic taint tracking with labels
- policy checks over a trace of runtime actions
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable

from core.defense_config import load_json_config


class PolicyViolation(Exception):
    def __init__(self, reason: str):
        super().__init__(f"Policy violation: {reason}")
        self.reason = reason


class IntegrityLevel(Enum):
    TRUSTED = 0
    UNTRUSTED = 1


@dataclass(frozen=True)
class IntegrityLabel:
    level: IntegrityLevel

    def join(self, other: "IntegrityLabel") -> "IntegrityLabel":
        if self.level.value <= other.level.value:
            return other
        return self

    @classmethod
    def trusted(cls) -> "IntegrityLabel":
        return cls(IntegrityLevel.TRUSTED)

    @classmethod
    def untrusted(cls) -> "IntegrityLabel":
        return cls(IntegrityLevel.UNTRUSTED)


@dataclass(frozen=True)
class FidesLabel:
    integrity: IntegrityLabel
    readers: frozenset[str]

    def join(self, other: "FidesLabel") -> "FidesLabel":
        return FidesLabel(
            integrity=self.integrity.join(other.integrity),
            readers=frozenset(set(self.readers).union(other.readers)),
        )


@dataclass(frozen=True)
class FidesTraceEntry:
    tool_name: str
    tool_args: dict[str, Any]
    label: FidesLabel


Policy = Callable[[list[FidesTraceEntry]], None]


@dataclass(frozen=True)
class FidesToolPolicyProfile:
    tool_count: int
    read_only_tools: frozenset[str]
    sensitive_tools: frozenset[str]
    sink_tools: frozenset[str]
    tool_labels: dict[str, str]


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, dict):
        for v in value.values():
            yield from _iter_strings(v)
        return
    if isinstance(value, list):
        for v in value:
            yield from _iter_strings(v)


_URL_PATTERN = re.compile(
    r"http[s]?://(?:[a-zA-Z0-9]|[$\-_.+!*\\(\\),]|(?:%[0-9a-fA-F]{2}))+"
)
_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def contains_url(text: str) -> bool:
    return bool(_URL_PATTERN.search(text))


def _extract_emails(value: Any) -> frozenset[str]:
    emails: set[str] = set()
    for text in _iter_strings(value):
        for match in _EMAIL_PATTERN.findall(text):
            emails.add(match.lower())
    return frozenset(emails)


def _extract_targets_from_args(
    args: dict[str, Any],
    candidate_keys: set[str],
) -> frozenset[str]:
    targets: set[str] = set()
    for key, value in args.items():
        if key not in candidate_keys:
            continue
        for item in _iter_strings(value):
            if _EMAIL_PATTERN.fullmatch(item.strip()):
                targets.add(item.strip().lower())
    return frozenset(targets)


class FidesRuntimeService:
    """
    Dynamic IFC engine inspired by the Fides notebook implementation.
    """

    DEFAULT_SENSITIVE_TOOLS = {
        "transfer_money",
        "add_recipient",
        "pay_bill",
        "send_email",
        "delete_file",
        "write_file",
        "update_device_state",
        "set_temperature",
    }

    DEFAULT_SINK_TOOLS = {
        "send_email",
        "write_file",
        "transfer_money",
        "add_recipient",
        "pay_bill",
    }
    DEFAULT_TARGET_KEYS = {
        "to",
        "recipient",
        "recipients",
        "email",
        "emails",
        "cc",
        "bcc",
        "channel",
        "target_email",
    }

    def __init__(
        self,
        trusted_sources: set[str] | None = None,
        trusted_tool_sources: set[str] | None = None,
        sensitive_tools: set[str] | None = None,
        sink_tools: set[str] | None = None,
        target_keys: set[str] | None = None,
        policy: Policy | None = None,
        enforce_url_policy: bool = True,
        enforce_sensitive_policy: bool = True,
        enforce_declassification_policy: bool = True,
    ):
        self.trusted_sources = trusted_sources or {"system", "user"}
        self.trusted_tool_sources = trusted_tool_sources or set()
        self.sensitive_tools = sensitive_tools or set(self.DEFAULT_SENSITIVE_TOOLS)
        self.sink_tools = sink_tools or set(self.DEFAULT_SINK_TOOLS)
        self.target_keys = target_keys or set(self.DEFAULT_TARGET_KEYS)
        self.policy = policy
        self.enforce_url_policy = enforce_url_policy
        self.enforce_sensitive_policy = enforce_sensitive_policy
        self.enforce_declassification_policy = enforce_declassification_policy

    def initial_label(self) -> FidesLabel:
        return FidesLabel(
            integrity=IntegrityLabel.trusted(),
            readers=frozenset({"user"}),
        )

    def label_from_source(self, source: str, payload: Any) -> FidesLabel:
        if source in self.trusted_sources or source in self.trusted_tool_sources:
            integrity = IntegrityLabel.trusted()
        else:
            integrity = IntegrityLabel.untrusted()

        readers = _extract_emails(payload)
        if not readers:
            readers = frozenset({"user"})

        return FidesLabel(integrity=integrity, readers=readers)

    def update_label_with_tool_output(
        self,
        current_label: FidesLabel,
        tool_name: str,
        tool_output: Any,
    ) -> FidesLabel:
        tool_source = f"tool:{tool_name}"
        next_label = self.label_from_source(tool_source, tool_output)
        return current_label.join(next_label)

    def check_tool_call(
        self,
        trace: list[FidesTraceEntry],
        tool_name: str,
        tool_args: dict[str, Any],
        label: FidesLabel,
    ) -> None:
        candidate_trace = [*trace, FidesTraceEntry(tool_name=tool_name, tool_args=tool_args, label=label)]
        if self.policy is not None:
            self.policy(candidate_trace)
            return
        self._default_policy(candidate_trace)

    def _default_policy(self, trace: list[FidesTraceEntry]) -> None:
        entry = trace[-1]
        if entry.label.integrity.level != IntegrityLevel.UNTRUSTED:
            return

        tool_name = entry.tool_name
        tool_args = entry.tool_args

        if self.enforce_sensitive_policy and tool_name in self.sensitive_tools:
            raise PolicyViolation(
                f"Attempted sensitive action '{tool_name}' in untrusted context"
            )

        if self.enforce_url_policy and tool_name in self.sink_tools:
            for text in _iter_strings(tool_args):
                if contains_url(text):
                    raise PolicyViolation(
                        f"Attempted sink action '{tool_name}' with untrusted URL content"
                    )

        if self.enforce_declassification_policy and tool_name in self.sink_tools:
            targets = _extract_targets_from_args(tool_args, self.target_keys)
            if targets and not set(targets).issubset(entry.label.readers):
                raise PolicyViolation(
                    f"Attempted declassification via '{tool_name}' to unauthorized targets {sorted(targets)}"
                )


FIDES_READ_ONLY_TOOLS = {
    "check_compliance_status",
    "check_drug_interaction",
    "check_firmware_version",
    "check_network_connections",
    "check_supplier_rating",
    "check_visa_requirements",
    "classify_transaction_risk",
    "compare_products",
    "estimate_shipping",
    "extract_action_items_from_email",
    "get_account_balance",
    "get_account_holds",
    "get_activity_log",
    "get_alarm_history",
    "get_approval_status",
    "get_attachment_manifest",
    "get_audit_trail",
    "get_auth_events",
    "get_baggage_rules",
    "get_battery_health",
    "get_beneficiary_bank_profile",
    "get_booking_details",
    "get_calendar_event_details",
    "get_calendar_events",
    "get_cart_constraints",
    "get_case_timeline",
    "get_checkin_window",
    "get_connection_risk",
    "get_contact_details",
    "get_coupon_eligibility",
    "get_coverage_limits",
    "get_delivery_window",
    "get_device_capabilities",
    "get_device_status",
    "get_disk_usage",
    "get_document_manifest",
    "get_email_metadata",
    "get_email_thread_state",
    "get_energy_consumption_report",
    "get_fare_rules",
    "get_fee_breakdown",
    "get_file_version_history",
    "get_flight_status",
    "get_host_inventory",
    "get_hotel_policy",
    "get_inventory_status",
    "get_invoice_status",
    "get_job_history",
    "get_lab_results",
    "get_loyalty_status",
    "get_medical_record",
    "get_network_status",
    "get_open_ports",
    "get_order_details",
    "get_pending_transfers",
    "get_prescription",
    "get_prior_authorization_status",
    "get_process_tree",
    "get_product_details",
    "get_provider_directory_entry",
    "get_recent_automation_failures",
    "get_recent_deployments",
    "get_recent_interactions",
    "get_referral_status",
    "get_refill_eligibility",
    "get_restock_eta",
    "get_return_eligibility",
    "get_running_processes",
    "get_scene_members",
    "get_schedule_details",
    "get_seller_policy",
    "get_sensor_data",
    "get_service_status",
    "get_spending_limit",
    "get_system_logs",
    "get_task_details",
    "get_transaction_history",
    "get_transfer_policy",
    "get_travel_advisory",
    "get_user_access_list",
    "get_weather_forecast",
    "list_attendees",
    "list_available_slots",
    "list_devices",
    "list_email_attachments",
    "list_emails",
    "list_files",
    "list_recipients",
    "list_tasks",
    "list_tickets",
    "read_access_case_ticket",
    "read_access_history_note",
    "read_access_override_note",
    "read_access_request_thread",
    "read_account_flag_note",
    "read_airline_followup_message",
    "read_alarm_rule_note",
    "read_alert_detail",
    "read_approval_guideline",
    "read_approval_memo",
    "read_attachment_text",
    "read_attendee_followup_thread",
    "read_audit_note",
    "read_authorization_case_ticket",
    "read_bank_letter_text",
    "read_board_packet_text",
    "read_booking_agent_note",
    "read_booking_message_thread",
    "read_camera_caption_text",
    "read_case_message_thread",
    "read_case_note",
    "read_change_control_rule",
    "read_change_log_note",
    "read_checkin_requirement_note",
    "read_clinician_remark",
    "read_config_bundle_text",
    "read_contact_remark",
    "read_contract_attachment_text",
    "read_coupon_rule_note",
    "read_coverage_policy_note",
    "read_deployment_history_note",
    "read_device_incident_ticket",
    "read_device_note",
    "read_device_packet_text",
    "read_email",
    "read_email_thread",
    "read_escalation_message",
    "read_event_history_note",
    "read_event_note",
    "read_exception_approval",
    "read_file",
    "read_firmware_advisory_text",
    "read_flight_disruption_case",
    "read_fulfillment_history_note",
    "read_guest_message",
    "read_host_note",
    "read_incident_history_note",
    "read_incident_postmortem_note",
    "read_incident_severity_rule",
    "read_insurer_message",
    "read_internal_request_ticket",
    "read_invoice_or_receipt_text",
    "read_invoice_packet_text",
    "read_itinerary_attachment",
    "read_lab_packet_text",
    "read_log_excerpt_text",
    "read_message_thread",
    "read_operator_chat_thread",
    "read_operator_followup_message",
    "read_operator_remark",
    "read_ops_escalation_case",
    "read_ops_followup_message",
    "read_order_history_note",
    "read_order_message_thread",
    "read_order_note",
    "read_order_support_ticket",
    "read_packing_slip_text",
    "read_payee_message",
    "read_policy_note",
    "read_prescription_attachment_text",
    "read_previous_alarm_note",
    "read_previous_outage_note",
    "read_previous_referral_note",
    "read_prior_authorization_note",
    "read_prior_case_note",
    "read_prior_disruption_note",
    "read_prior_incident_note",
    "read_prior_return_note",
    "read_prior_review_note",
    "read_provider_message",
    "read_rebooking_history_note",
    "read_recipient_profile_note",
    "read_reconciliation_note",
    "read_referral_agent_note",
    "read_referral_case_ticket",
    "read_referral_packet_text",
    "read_referral_requirement_note",
    "read_refill_rule_note",
    "read_release_packet_text",
    "read_return_agent_note",
    "read_return_case_ticket",
    "read_return_policy_note",
    "read_runbook_policy",
    "read_schedule_rule_note",
    "read_security_policy_note",
    "read_seller_attachment_text",
    "read_seller_message",
    "read_seller_remark",
    "read_service_annotation",
    "read_settlement_ticket",
    "read_shipping_policy_note",
    "read_support_followup_message",
    "read_task_comment",
    "read_ticket",
    "read_ticket_rule_note",
    "read_ticket_thread",
    "read_transfer_policy_note",
    "read_transfer_thread",
    "read_travel_desk_message",
    "read_travel_policy_note",
    "read_travel_support_ticket",
    "read_traveler_profile_note",
    "read_trip_history_note",
    "read_trip_override_note",
    "read_vendor_handling_rule",
    "read_visa_packet_text",
    "read_wire_escalation_case",
    "read_wire_packet_text",
    "scan_for_vulnerabilities",
    "search_contacts",
    "search_doctors",
    "search_email",
    "summarize_email_thread",
    "track_order",
    "verify_insurance_coverage",
    "verify_recipient_identity",
}

FIDES_SENSITIVE_TOOLS = {
    "acknowledge_alert",
    "add_email_label",
    "add_recipient",
    "apply_coupon",
    "apply_loyalty_points",
    "archive_email",
    "book_appointment",
    "cancel_flight",
    "cancel_order",
    "cancel_scheduled_transfer",
    "change_flight",
    "check_in",
    "complete_task",
    "compress_folder",
    "copy_file",
    "create_file",
    "create_folder",
    "delete_file",
    "extract_archive",
    "forward_email",
    "freeze_account",
    "lock_door",
    "move_file",
    "place_order",
    "purchase_travel_insurance",
    "remove_coupon",
    "remove_device_schedule",
    "remove_from_cart",
    "remove_recipient",
    "rename_file",
    "reply_email",
    "request_prescription_refill",
    "request_specialist_referral",
    "request_travel_invoice",
    "reschedule_appointment",
    "reschedule_event",
    "restart_service",
    "return_item",
    "schedule_maintenance",
    "schedule_transfer",
    "send_email",
    "send_instant_message",
    "set_alarm_mode",
    "set_camera_privacy_mode",
    "set_device_schedule",
    "transfer_money",
    "unfreeze_account",
    "unlock_door",
    "update_task",
    "update_ticket_status",
    "upload_medical_document",
    "write_file",
}


def _normalize_tool_map(tool_catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(tool_catalog, dict):
        return {}
    if "tools" in tool_catalog and isinstance(tool_catalog["tools"], dict):
        raw = tool_catalog["tools"]
    else:
        raw = tool_catalog
    normalized: dict[str, dict[str, Any]] = {}
    for name, entry in raw.items():
        if not isinstance(name, str):
            continue
        if isinstance(entry, dict):
            normalized[name] = entry
        else:
            normalized[name] = {}
    return normalized


def _classify_tool(tool_name: str, tool_def: dict[str, Any]) -> tuple[str, str]:
    del tool_def
    normalized = str(tool_name or "").strip()
    if normalized in FIDES_READ_ONLY_TOOLS:
        return "read_only", "explicit query group"
    if normalized in FIDES_SENSITIVE_TOOLS:
        return "sensitive", "explicit command/high_risk_command group"
    return "sensitive", "unknown tool fallback"


def derive_fides_tool_policy(
    tool_catalog: dict[str, Any],
    extra_tool_names: Iterable[str] | None = None,
    force_sensitive_tools: Iterable[str] | None = None,
    force_read_only_tools: Iterable[str] | None = None,
    force_sink_tools: Iterable[str] | None = None,
) -> FidesToolPolicyProfile:
    default_cfg = load_json_config("fides_tool_policy.json", default={}) or {}
    if force_sensitive_tools is None:
        force_sensitive_tools = default_cfg.get("sensitive_tools", [])
    if force_read_only_tools is None:
        force_read_only_tools = default_cfg.get("read_only_tools", [])
    if force_sink_tools is None:
        force_sink_tools = default_cfg.get("sink_tools", [])

    normalized = _normalize_tool_map(tool_catalog)

    if extra_tool_names:
        for tool_name in extra_tool_names:
            if isinstance(tool_name, str) and tool_name and tool_name not in normalized:
                normalized[tool_name] = {
                    "name": tool_name,
                    "description": f"Tool: {tool_name}",
                    "category": "Unknown",
                }

    read_only_tools: set[str] = set()
    sensitive_tools: set[str] = set()
    tool_labels: dict[str, str] = {}

    for tool_name in sorted(normalized.keys()):
        label, _ = _classify_tool(tool_name, normalized[tool_name])
        tool_labels[tool_name] = label
        if label == "read_only":
            read_only_tools.add(tool_name)
        else:
            sensitive_tools.add(tool_name)

    forced_read_only = {t for t in (force_read_only_tools or []) if isinstance(t, str) and t}
    forced_sensitive = {t for t in (force_sensitive_tools or []) if isinstance(t, str) and t}

    for tool_name in forced_read_only:
        read_only_tools.add(tool_name)
        sensitive_tools.discard(tool_name)
        tool_labels[tool_name] = "read_only"

    for tool_name in forced_sensitive:
        sensitive_tools.add(tool_name)
        read_only_tools.discard(tool_name)
        tool_labels[tool_name] = "sensitive"

    sink_tools = {t for t in (force_sink_tools or []) if isinstance(t, str) and t}

    return FidesToolPolicyProfile(
        tool_count=len(normalized),
        read_only_tools=frozenset(read_only_tools),
        sensitive_tools=frozenset(sensitive_tools),
        sink_tools=frozenset(sink_tools),
        tool_labels=tool_labels,
    )


def build_fides_runtime_service(
    tool_catalog: dict[str, Any],
    extra_tool_names: Iterable[str] | None = None,
    force_sensitive_tools: Iterable[str] | None = None,
    force_read_only_tools: Iterable[str] | None = None,
    force_sink_tools: Iterable[str] | None = None,
) -> tuple[FidesRuntimeService, FidesToolPolicyProfile]:
    profile = derive_fides_tool_policy(
        tool_catalog=tool_catalog,
        extra_tool_names=extra_tool_names,
        force_sensitive_tools=force_sensitive_tools,
        force_read_only_tools=force_read_only_tools,
        force_sink_tools=force_sink_tools,
    )
    service = FidesRuntimeService(
        sensitive_tools=set(profile.sensitive_tools),
        sink_tools=set(profile.sink_tools),
    )
    return service, profile
