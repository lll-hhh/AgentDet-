"""
CaMeL (Contextual Agentic Machine-Learning) — Tool Security Policy Map.

Policy entries:
- side_effect: whether the tool mutates world state.
- trusted_fields: args that must originate from trusted/user intent.
- readers: args that represent information recipients/audience.
- policy_note: human-readable rationale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.defense_config import load_json_config


_DEFAULT_TOOL_LIBRARY_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "tool_library.json"
)

_EXPLICIT_READ_ONLY_TOOLS = {
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

_EXPLICIT_SIDE_EFFECT_TOOLS = {
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

_READER_KEYS = {
    "to",
    "cc",
    "bcc",
    "recipient",
    "recipients",
    "user",
    "users",
    "email",
    "emails",
    "channel",
    "target_email",
}

_HIGH_RISK_TRUSTED_KEYS = {
    "from_account",
    "to_account",
    "account_id",
    "amount",
    "quantity",
    "price",
    "action",
    "mode",
    "state",
    "destination",
    "dropoff",
    "path",
    "file_path",
    "src",
    "dst",
    "source",
}


def _load_default_tool_catalog() -> dict[str, Any]:
    if not _DEFAULT_TOOL_LIBRARY_PATH.exists():
        return {}
    try:
        return json.loads(_DEFAULT_TOOL_LIBRARY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _normalize_tool_catalog(tool_catalog: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(tool_catalog, dict):
        return {}
    raw = tool_catalog.get("tools", tool_catalog)
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for name, entry in raw.items():
        if not isinstance(name, str) or not name:
            continue
        normalized[name] = entry if isinstance(entry, dict) else {}
    return normalized


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _is_read_only_tool_name(tool_name: str) -> bool:
    normalized = str(tool_name or "").strip()
    return normalized in _EXPLICIT_READ_ONLY_TOOLS


def _is_side_effect_tool_name(tool_name: str) -> bool:
    normalized = str(tool_name or "").strip()
    if normalized in _EXPLICIT_SIDE_EFFECT_TOOLS:
        return True
    if normalized in _EXPLICIT_READ_ONLY_TOOLS:
        return False
    return True


def _infer_policy_from_schema(tool_name: str, tool_def: dict[str, Any]) -> dict[str, Any]:
    params = tool_def.get("parameters", {}) if isinstance(tool_def, dict) else {}
    if not isinstance(params, dict):
        params = {}
    properties = params.get("properties", {}) if isinstance(params.get("properties", {}), dict) else {}
    required = params.get("required", [])
    if not isinstance(required, list):
        required = []
    prop_names = [name for name in properties.keys() if isinstance(name, str)]

    side_effect = _is_side_effect_tool_name(tool_name)
    readers = [name for name in prop_names if name in _READER_KEYS]

    trusted_fields: list[str] = []
    if side_effect:
        trusted_fields.extend([name for name in required if isinstance(name, str)])
        trusted_fields.extend(
            [
                name
                for name in prop_names
                if name in _HIGH_RISK_TRUSTED_KEYS or name.endswith("_id")
            ]
        )
        trusted_fields = _dedupe_keep_order(trusted_fields)

    if side_effect:
        note = "Auto-inferred side-effect policy from explicit tool class + schema."
    else:
        note = "Auto-inferred read-only policy from explicit tool class + schema."

    return {
        "side_effect": side_effect,
        "trusted_fields": trusted_fields,
        "readers": readers,
        "policy_note": note,
    }


def _sanitize_policy_for_schema(
    tool_name: str,
    policy: dict[str, Any],
    tool_def: dict[str, Any],
) -> dict[str, Any]:
    params = tool_def.get("parameters", {}) if isinstance(tool_def, dict) else {}
    if not isinstance(params, dict):
        return policy
    properties = params.get("properties", {})
    if not isinstance(properties, dict) or not properties:
        return policy

    prop_names = set(name for name in properties.keys() if isinstance(name, str))
    trusted_fields = [
        field for field in (policy.get("trusted_fields", []) or []) if field in prop_names
    ]
    readers = [field for field in (policy.get("readers", []) or []) if field in prop_names]

    side_effect = bool(policy.get("side_effect", True))
    if side_effect and not trusted_fields:
        inferred = _infer_policy_from_schema(tool_name, tool_def)
        trusted_fields = inferred["trusted_fields"]
        if not readers:
            readers = inferred["readers"]

    return {
        "side_effect": side_effect,
        "trusted_fields": trusted_fields,
        "readers": readers,
        "policy_note": str(policy.get("policy_note", "")),
    }


def build_tool_policy_map(tool_catalog: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """
    Build the security policy map for AgentDET tools.

    Behavior:
    - Provides curated policies for existing known tools.
    - Auto-infers policies for newly added tools found in tool_library.
    - Sanitizes trusted_fields/readers against tool schema properties.
    """
    read_only = {
        "get_account_balance": "Query account balance — no mutation.",
        "get_calendar_events": "List calendar events — read only.",
        "get_device_status": "Query device status — read only.",
        "get_exchange_rate": "Currency rate lookup — read only.",
        "get_flight_status": "Flight status query — read only.",
        "get_medical_record": "Read medical record — read only.",
        "get_prescription": "Read prescription — read only.",
        "get_product_details": "Product info query — read only.",
        "get_recommendations": "Get recommendations — read only.",
        "get_security_camera_image": "View camera feed — read only.",
        "get_sensor_data": "Read sensor data — read only.",
        "get_stock_price": "Stock price query — read only.",
        "get_system_time": "System time query — read only.",
        "get_transaction_history": "Transaction history — read only.",
        "list_attendees": "List event attendees — read only.",
        "list_devices": "List smart devices — read only.",
        "list_emails": "List emails — read only.",
        "list_files": "List files — read only.",
        "read_email": "Read email content — read only.",
        "read_file": "Read file content — read only.",
        "search_contacts": "Search contacts — read only.",
        "search_email": "Search emails — read only.",
        "search_flights": "Search flights — read only.",
        "search_hotels": "Search hotels — read only.",
        "search_product": "Search products — read only.",
        "track_order": "Track order status — read only.",
        "view_cart": "View shopping cart — read only.",
    }

    side_effect = {
        "add_calendar_event": {
            "trusted_fields": ["title", "start_time", "end_time"],
            "readers": ["attendees"],
            "policy_note": "Calendar event creation — attendees are audience for event data.",
        },
        "add_recipient": {
            "trusted_fields": ["account_id", "recipient_name", "recipient_account"],
            "readers": [],
            "policy_note": "Adding payment recipient — all fields must be user-specified.",
        },
        "add_to_cart": {
            "trusted_fields": ["product_id"],
            "readers": [],
            "policy_note": "Adding to cart — product must be user-chosen.",
        },
        "arm_system": {
            "trusted_fields": [],
            "readers": [],
            "policy_note": "Arm security system — no args to validate.",
        },
        "book_appointment": {
            "trusted_fields": ["doctor", "time"],
            "readers": [],
            "policy_note": "Book medical appointment — doctor/time from user.",
        },
        "book_flight": {
            "trusted_fields": ["flight_number"],
            "readers": [],
            "policy_note": "Book flight — flight must be user-specified.",
        },
        "book_hotel": {
            "trusted_fields": ["hotel_name"],
            "readers": [],
            "policy_note": "Book hotel — hotel must be user-specified.",
        },
        "buy_fund": {
            "trusted_fields": ["account_id", "symbol", "amount"],
            "readers": [],
            "policy_note": "Buy fund — all financial params must be user-specified.",
        },
        "buy_stock": {
            "trusted_fields": ["account_id", "symbol", "quantity"],
            "readers": [],
            "policy_note": "Buy stock — all financial params must be user-specified.",
        },
        "call_taxi": {
            "trusted_fields": ["pickup", "dropoff"],
            "readers": [],
            "policy_note": "Request taxi — pickup/dropoff must be user-specified.",
        },
        "cancel_appointment": {
            "trusted_fields": ["appointment_id"],
            "readers": [],
            "policy_note": "Cancel appointment — ID must come from user.",
        },
        "cancel_calendar_event": {
            "trusted_fields": ["event_id"],
            "readers": [],
            "policy_note": "Cancel event — event ID must come from user.",
        },
        "cancel_flight": {
            "trusted_fields": ["booking_id"],
            "readers": [],
            "policy_note": "Cancel flight — booking ID must come from user.",
        },
        "cancel_hotel": {
            "trusted_fields": ["booking_id"],
            "readers": [],
            "policy_note": "Cancel hotel — booking ID must come from user.",
        },
        "check_in": {
            "trusted_fields": ["booking_id"],
            "readers": [],
            "policy_note": "Check in — booking ID must come from user.",
        },
        "control_device": {
            "trusted_fields": ["device_id", "action"],
            "readers": [],
            "policy_note": "IoT control — device and action must be user-specified.",
        },
        "create_contact": {
            "trusted_fields": ["name"],
            "readers": [],
            "policy_note": "Create contact — name from user.",
        },
        "create_folder": {
            "trusted_fields": ["path"],
            "readers": [],
            "policy_note": "Create folder — path from user.",
        },
        "delete_contact": {
            "trusted_fields": ["name"],
            "readers": [],
            "policy_note": "Delete contact — name must come from user.",
        },
        "delete_email": {
            "trusted_fields": ["email_id"],
            "readers": [],
            "policy_note": "Delete email — email ID must come from user directly.",
        },
        "delete_file": {
            "trusted_fields": ["path", "file_path"],
            "readers": [],
            "policy_note": "Delete file — path must come from user directly.",
        },
        "disarm_system": {
            "trusted_fields": [],
            "readers": [],
            "policy_note": "Disarm security system — no args to validate.",
        },
        "edit_contact": {
            "trusted_fields": ["name"],
            "readers": [],
            "policy_note": "Edit contact — contact name from user.",
        },
        "freeze_account": {
            "trusted_fields": ["account_id"],
            "readers": [],
            "policy_note": "Freeze account — account ID must come from user.",
        },
        "lock_door": {
            "trusted_fields": ["device_id"],
            "readers": [],
            "policy_note": "Lock door — device ID from user.",
        },
        "mark_as_spam": {
            "trusted_fields": ["email_id"],
            "readers": [],
            "policy_note": "Mark as spam — email ID from user.",
        },
        "move_file": {
            "trusted_fields": ["src", "dst", "source", "destination"],
            "readers": [],
            "policy_note": "Move file — source and destination must come from user.",
        },
        "pay_bill": {
            "trusted_fields": ["bill_id", "amount"],
            "readers": [],
            "policy_note": "Pay bill — bill ID and amount must be user-specified.",
        },
        "place_order": {
            "trusted_fields": [],
            "readers": [],
            "policy_note": "Place order from cart — relies on cart contents.",
        },
        "remove_from_cart": {
            "trusted_fields": ["product_id"],
            "readers": [],
            "policy_note": "Remove from cart — product must be user-specified.",
        },
        "reschedule_event": {
            "trusted_fields": ["event_id", "new_start_time"],
            "readers": [],
            "policy_note": "Reschedule — event ID and new time from user.",
        },
        "return_item": {
            "trusted_fields": ["order_id"],
            "readers": [],
            "policy_note": "Return item — order ID from user.",
        },
        "sell_fund": {
            "trusted_fields": ["account_id", "symbol", "amount"],
            "readers": [],
            "policy_note": "Sell fund — all financial params must be user-specified.",
        },
        "sell_stock": {
            "trusted_fields": ["account_id", "symbol", "quantity"],
            "readers": [],
            "policy_note": "Sell stock — all financial params must be user-specified.",
        },
        "send_email": {
            "trusted_fields": ["to"],
            "readers": ["to", "cc", "bcc"],
            "policy_note": (
                "Send email — recipient must be user-specified. "
                "Body/subject readable by recipients."
            ),
        },
        "send_instant_message": {
            "trusted_fields": ["user"],
            "readers": ["user"],
            "policy_note": "Send IM — recipient must be user-specified.",
        },
        "set_thermostat_mode": {
            "trusted_fields": ["mode"],
            "readers": [],
            "policy_note": "Thermostat control — mode from user.",
        },
        "transfer_money": {
            "trusted_fields": ["from_account", "to_account", "amount"],
            "readers": [],
            "policy_note": "Transfer money — all params must be user-specified.",
        },
        "unlock_door": {
            "trusted_fields": ["device_id"],
            "readers": [],
            "policy_note": "Unlock door — device ID from user.",
        },
        "update_device_state": {
            "trusted_fields": ["device_id", "state"],
            "readers": [],
            "policy_note": "Update device state — device and state from user.",
        },
        "update_recurring_payment": {
            "trusted_fields": ["payment_id", "amount"],
            "readers": [],
            "policy_note": "Update recurring payment — payment ID and amount from user.",
        },
        "write_file": {
            "trusted_fields": ["file_path"],
            "readers": [],
            "policy_note": "Write file — file path from user. Content may come from processing.",
        },
    }

    policy_map: dict[str, dict[str, Any]] = {}
    for tool_name, note in read_only.items():
        policy_map[tool_name] = {
            "side_effect": False,
            "trusted_fields": [],
            "readers": [],
            "policy_note": note,
        }
    for tool_name, spec in side_effect.items():
        policy_map[tool_name] = {
            "side_effect": True,
            **spec,
        }

    external = load_json_config("camel_tool_policy_map.json", default={}) or {}
    external_policies = external.get("policies", external)
    if isinstance(external_policies, dict):
        for tool_name, policy in external_policies.items():
            if not isinstance(tool_name, str) or not tool_name or not isinstance(policy, dict):
                continue
            policy_map[tool_name] = {
                "side_effect": bool(policy.get("side_effect", True)),
                "trusted_fields": list(policy.get("trusted_fields", []) or []),
                "readers": list(policy.get("readers", []) or []),
                "policy_note": str(policy.get("policy_note", "")),
            }

    normalized_catalog = _normalize_tool_catalog(tool_catalog or _load_default_tool_catalog())
    for tool_name, tool_def in normalized_catalog.items():
        if tool_name not in policy_map:
            policy_map[tool_name] = {
                "side_effect": _is_side_effect_tool_name(tool_name),
                "trusted_fields": [],
                "readers": [],
                "policy_note": "Explicit fallback policy for uncatalogued tool.",
            }
            policy_map[tool_name] = _sanitize_policy_for_schema(
                tool_name=tool_name,
                policy=policy_map[tool_name],
                tool_def=tool_def,
            )

    return policy_map
