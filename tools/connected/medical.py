from typing import Any, Dict, List
import uuid
import datetime
from core.world.environment import WorldState, Appointment, MedicalRecord

class MedicalTools:
    """
    Medical tools connected to the World Model.
    Handles Appointment booking and Record access.
    """

    def __init__(self):
        self.current_user = "user@example.com"

    def book_appointment(
        self,
        doctor: str | None = None,
        time: str | None = None,
        department: str | None = None,
        specialty: str | None = None,
        datetime: str | None = None,
        date_time: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Book a medical appointment."""
        if not doctor:
            doctor = kwargs.get("physician")
        if time is None:
            time = datetime
        if time is None:
            time = date_time or kwargs.get("start_time")
        if department is None:
            department = specialty
        if department is None:
            department = kwargs.get("dept")
        if not doctor or not time or not department:
            return {
                "status": "error",
                "message": "Missing required fields: doctor, time, department.",
            }

        apt_id = f"apt_{uuid.uuid4().hex[:8]}"
        appointment = Appointment(
            id=apt_id,
            patient=self.current_user,
            doctor=doctor,
            time=time,
            department=department,
            status="Scheduled"
        )
        WorldState.get_instance().medical.appointments.append(appointment)
        return {
            "status": "success",
            "message": "Appointment booked.",
            "appointment_id": apt_id
        }

    def cancel_appointment(self, appointment_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Cancel an appointment."""
        if not appointment_id:
            appointment_id = kwargs.get("id")
        if not appointment_id:
            return {"status": "error", "message": "Missing required argument: appointment_id"}

        apt = next((a for a in WorldState.get_instance().medical.appointments if a.id == appointment_id), None)
        if not apt:
            return {"status": "error", "message": "Appointment not found."}

        apt.status = "Cancelled"
        return {"status": "success", "message": f"Appointment {appointment_id} cancelled."}

    def get_medical_record(self, patient_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """
        Get medical record.
        Highly sensitive tool.
        """
        if not patient_id:
            patient_id = kwargs.get("patient") or self.current_user

        records = WorldState.get_instance().medical.records

        record = records.get(patient_id)

        if not record:
            for stored_key, stored_record in records.items():
                if (stored_key.endswith(patient_id) or
                    patient_id.endswith(stored_key) or
                    stored_key.replace("-", "").endswith(patient_id.replace("-", "")) or
                    patient_id.replace("-", "").endswith(stored_key.replace("-", ""))):
                    record = stored_record
                    break

        if not record:
            parts = patient_id.split("-")
            for i in range(1, len(parts)):
                alt_id = "-".join(parts[i:])
                record = records.get(alt_id)
                if record:
                    break

        if not record:
            return {"status": "error", "message": "Medical record not found."}

        return {
            "status": "success",
            "history": record.history,
            "prescriptions": record.prescriptions
        }

    def get_prescription(self, prescription_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Get details of a prescription."""
        if not prescription_id:
            prescription_id = kwargs.get("id")
        if not prescription_id:
            return {"status": "error", "message": "Missing required argument: prescription_id"}
        return {"status": "success", "detail": f"Prescription {prescription_id} details..."}

    def search_doctors(
        self,
        department: str | None = None,
        location: str | None = None,
        date: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Search doctors from catalog with basic filters."""
        if not department:
            department = kwargs.get("specialty")
        if not location:
            location = kwargs.get("city")
        if not date:
            date = kwargs.get("day")
        if not department:
            return {"status": "error", "message": "Missing required field: department."}

        catalog = WorldState.get_instance().context.get("doctor_catalog", [])
        if not isinstance(catalog, list) or not catalog:
            catalog = [
                {
                    "doctor": "Dr. Alice Lee",
                    "department": department,
                    "location": location or "New York",
                    "available_date": date or "2026-03-01",
                },
                {
                    "doctor": "Dr. Brian Wu",
                    "department": department,
                    "location": location or "New York",
                    "available_date": date or "2026-03-02",
                },
            ]

        results = [
            x
            for x in catalog
            if str(x.get("department", "")).lower() == str(department).lower()
            and (not location or str(x.get("location", "")).lower() == str(location).lower())
            and (not date or str(x.get("available_date", "")).startswith(str(date)))
        ]
        return {"status": "success", "doctors": results, "count": len(results)}

    def list_available_slots(
        self,
        doctor: str | None = None,
        date: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """List available slots for a doctor/date pair."""
        if not doctor:
            doctor = kwargs.get("physician")
        if not date:
            date = kwargs.get("day")
        if not doctor or not date:
            return {"status": "error", "message": "Missing required fields: doctor, date."}

        slots_db = WorldState.get_instance().context.setdefault("doctor_slots", {})
        key = f"{doctor}|{date}"
        slots = slots_db.get(key)
        if slots is None:
            slots = [
                f"{date}T09:00:00",
                f"{date}T11:00:00",
                f"{date}T14:30:00",
            ]
            slots_db[key] = slots
        return {"status": "success", "doctor": doctor, "date": date, "slots": slots}

    def reschedule_appointment(
        self,
        appointment_id: str | None = None,
        new_time: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Reschedule an existing appointment."""
        if not appointment_id:
            appointment_id = kwargs.get("id")
        if not new_time:
            new_time = kwargs.get("time")
        if not appointment_id or not new_time:
            return {"status": "error", "message": "Missing required fields: appointment_id, new_time."}
        apt = next((a for a in WorldState.get_instance().medical.appointments if a.id == appointment_id), None)
        if not apt:
            return {"status": "error", "message": "Appointment not found."}
        old_time = apt.time
        apt.time = new_time
        return {
            "status": "success",
            "appointment_id": appointment_id,
            "old_time": old_time,
            "new_time": new_time,
        }

    def upload_medical_document(
        self,
        patient_id: str | None = None,
        document_name: str | None = None,
        content: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Upload medical document and link it to patient records."""
        if not patient_id:
            patient_id = kwargs.get("patient") or self.current_user
        if not document_name:
            document_name = kwargs.get("filename")
        if content is None:
            content = kwargs.get("body")
        if not patient_id or not document_name or content is None:
            return {
                "status": "error",
                "message": "Missing required fields: patient_id, document_name, content.",
            }
        safe_name = str(document_name).replace("\\", "_").replace("/", "_")
        path = f"/medical/{patient_id}/{safe_name}"
        WorldState.get_instance().fs.write_file(path, str(content))
        docs = WorldState.get_instance().context.setdefault("medical_documents", {})
        docs.setdefault(patient_id, []).append(
            {
                "path": path,
                "document_name": safe_name,
                "uploaded_at": datetime.datetime.now().isoformat(),
            }
        )
        if patient_id not in WorldState.get_instance().medical.records:
            WorldState.get_instance().medical.records[patient_id] = MedicalRecord(
                patient=patient_id,
                history="",
                prescriptions=[],
            )
        record = WorldState.get_instance().medical.records[patient_id]
        history_line = f"[{datetime.datetime.now().isoformat()}] uploaded {safe_name}"
        record.history = (record.history + "\n" + history_line).strip()
        return {"status": "success", "patient_id": patient_id, "path": path}

    def request_prescription_refill(
        self,
        patient_id: str | None = None,
        prescription_id: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a refill request for a prescription."""
        if not patient_id:
            patient_id = kwargs.get("patient") or self.current_user
        if not prescription_id:
            prescription_id = kwargs.get("id")
        if not patient_id or not prescription_id:
            return {
                "status": "error",
                "message": "Missing required fields: patient_id, prescription_id.",
            }
        request_id = f"ref_{uuid.uuid4().hex[:8]}"
        entry = {
            "request_id": request_id,
            "patient_id": patient_id,
            "prescription_id": prescription_id,
            "status_text": "submitted",
            "created_at": datetime.datetime.now().isoformat(),
        }
        WorldState.get_instance().context.setdefault("refill_requests", []).append(entry)
        return {"status": "success", **entry}

    def check_drug_interaction(
        self,
        patient_id: str | None = None,
        medication_list: List[str] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run lightweight drug interaction checks."""
        if not patient_id:
            patient_id = kwargs.get("patient") or self.current_user
        if medication_list is None and isinstance(kwargs.get("medications"), list):
            medication_list = kwargs.get("medications")
        if not patient_id or not isinstance(medication_list, list):
            return {
                "status": "error",
                "message": "Missing required fields: patient_id, medication_list(list).",
            }
        normalized = [str(m).strip().lower() for m in medication_list if str(m).strip()]
        known_pairs = {
            tuple(sorted(["warfarin", "aspirin"])): "Increased bleeding risk.",
            tuple(sorted(["ibuprofen", "prednisone"])): "Higher GI irritation risk.",
            tuple(sorted(["metformin", "contrast dye"])): "Potential lactic acidosis risk.",
        }
        findings = []
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                key = tuple(sorted([normalized[i], normalized[j]]))
                if key in known_pairs:
                    findings.append(
                        {
                            "drug_a": key[0],
                            "drug_b": key[1],
                            "severity": "warning",
                            "note": known_pairs[key],
                        }
                    )
        report = {
            "report_id": f"di_{uuid.uuid4().hex[:8]}",
            "patient_id": patient_id,
            "medications": normalized,
            "findings": findings,
            "risk_level": "high" if findings else "low",
            "created_at": datetime.datetime.now().isoformat(),
        }
        WorldState.get_instance().context.setdefault("drug_interaction_reports", []).append(report)
        return {"status": "success", **report}


    def verify_insurance_coverage(
        self,
        patient_id: str | None = None,
        procedure: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Check whether a procedure is covered by the patient's insurance."""
        if not patient_id:
            patient_id = kwargs.get("patient") or self.current_user
        if not procedure:
            procedure = kwargs.get("treatment") or kwargs.get("service")
        if not patient_id or not procedure:
            return {"status": "error", "message": "Missing required fields: patient_id, procedure."}
        insurance = WorldState.get_instance().context.get("insurance_coverage", {})
        patient_ins = insurance.get(patient_id)
        if not patient_ins:
            return {"status": "error", "message": f"No insurance record for patient {patient_id}."}
        covered = patient_ins.get("covered_procedures", [])
        is_covered = procedure.lower() in [c.lower() for c in covered]
        return {
            "status": "success",
            "patient_id": patient_id,
            "procedure": procedure,
            "is_covered": is_covered,
            "plan": patient_ins.get("plan_name", "unknown"),
            "copay": patient_ins.get("copay", 0) if is_covered else None,
            "notes": patient_ins.get("notes", ""),
        }

    def get_lab_results(
        self,
        patient_id: str | None = None,
        test_id: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Retrieve laboratory test results for a patient."""
        if not patient_id:
            patient_id = kwargs.get("patient") or self.current_user
        if not test_id:
            test_id = kwargs.get("lab_id") or kwargs.get("order_id")
        if not patient_id:
            return {"status": "error", "message": "Missing required field: patient_id."}
        lab_db = WorldState.get_instance().context.get("lab_results", {})
        if test_id:
            result = None
            if isinstance(lab_db, dict):
                direct = lab_db.get(test_id)
                if isinstance(direct, dict):
                    result = direct
                else:
                    for maybe_results in lab_db.values():
                        if isinstance(maybe_results, list):
                            for item in maybe_results:
                                if isinstance(item, dict) and item.get("test_id") == test_id:
                                    result = item
                                    break
                        if result is not None:
                            break
            if not result:
                return {"status": "error", "message": f"Lab result {test_id} not found."}
            return {"status": "success", "test_id": test_id, **result}
        patient_results: Any
        if isinstance(lab_db, dict):
            direct_results = lab_db.get(patient_id)
            if isinstance(direct_results, list):
                patient_results = direct_results
            else:
                patient_results = {
                    k: v for k, v in lab_db.items()
                    if isinstance(v, dict) and v.get("patient_id") == patient_id
                }
        else:
            patient_results = []
        return {"status": "success", "patient_id": patient_id, "results": patient_results}

    def request_specialist_referral(
        self,
        patient_id: str | None = None,
        specialty: str | None = None,
        reason: str | None = None,
        referring_doctor: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a specialist referral for a patient."""
        if not patient_id:
            patient_id = kwargs.get("patient") or self.current_user
        if not specialty:
            specialty = kwargs.get("department")
        if not reason:
            reason = kwargs.get("notes") or kwargs.get("clinical_reason")
        if not patient_id or not specialty or not reason:
            return {
                "status": "error",
                "message": "Missing required fields: patient_id, specialty, reason.",
            }
        ref_id = f"ref_{uuid.uuid4().hex[:8]}"
        entry = {
            "referral_id": ref_id,
            "patient_id": patient_id,
            "specialty": specialty,
            "reason": reason,
            "referring_doctor": referring_doctor or "N/A",
            "status": "submitted",
            "created_at": datetime.datetime.now().isoformat(),
        }
        WorldState.get_instance().context.setdefault("referrals", []).append(entry)
        return {"status": "success", **entry}


from tools.connected._v3_roots import ensure_list_root, find_many, find_one, lookup_by_arg, make_id


def _medical_v3_read_case_message_thread(self, case_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    case_id = case_id or kwargs.get("id")
    if not case_id:
        return {"status": "error", "message": "Missing required argument: case_id"}
    items = find_many(ensure_list_root(self, "medical_case_threads"), case_id=case_id)
    if not items:
        return {"status": "error", "message": f"No case thread found for {case_id}."}
    return {"status": "success", "case_id": case_id, "messages": items, "count": len(items)}


def _medical_v3_read_referral_note(self, case_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    case_id = case_id or kwargs.get("id")
    if not case_id:
        return {"status": "error", "message": "Missing required argument: case_id"}
    items = find_many(ensure_list_root(self, "medical_referral_notes"), case_id=case_id)
    if not items:
        return {"status": "error", "message": "Referral note not found."}
    return {"status": "success", "case_id": case_id, "notes": items, "count": len(items)}


def _medical_v3_read_clinician_remark(self, case_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    case_id = case_id or kwargs.get("id")
    if not case_id:
        return {"status": "error", "message": "Missing required argument: case_id"}
    items = find_many(ensure_list_root(self, "medical_clinician_remarks"), case_id=case_id)
    if not items:
        return {"status": "error", "message": "Clinician remark not found."}
    return {"status": "success", "case_id": case_id, "remarks": items, "count": len(items)}


def _medical_v3_get_medical_record(self, case_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    case_id = case_id or kwargs.get("id")
    if not case_id:
        return {"status": "error", "message": "Missing required argument: case_id"}
    item = find_one(ensure_list_root(self, "medical_records"), case_id=case_id)
    if not item:
        return {"status": "error", "message": "Medical record not found."}
    return {"status": "success", **item}


def _medical_v3_get_prescription(self, case_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    case_id = case_id or kwargs.get("id")
    if not case_id:
        return {"status": "error", "message": "Missing required argument: case_id"}
    items = find_many(ensure_list_root(self, "medical_prescriptions"), case_id=case_id)
    if not items:
        return {"status": "error", "message": "Prescription not found."}
    return {"status": "success", "case_id": case_id, "prescriptions": items, "count": len(items)}


def _medical_v3_verify_insurance_coverage(self, case_id: str | None = None, procedure: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    case_id = case_id or kwargs.get("id")
    procedure = procedure or kwargs.get("service")
    if not case_id or not procedure:
        return {"status": "error", "message": "Missing required fields: case_id, procedure."}
    item = find_one(ensure_list_root(self, "medical_coverage_records"), case_id=case_id)
    if not item:
        return {"status": "error", "message": "Coverage record not found."}
    return {
        "status": "success",
        "case_id": case_id,
        "procedure": procedure,
        "coverage_status": item.get("coverage_status"),
        "limits_text": item.get("limits_text"),
    }


def _medical_v3_get_lab_results(self, case_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    case_id = case_id or kwargs.get("id")
    if not case_id:
        return {"status": "error", "message": "Missing required argument: case_id"}
    item = find_one(ensure_list_root(self, "medical_lab_results"), case_id=case_id)
    if not item:
        return {"status": "error", "message": "Lab result not found."}
    return {"status": "success", **item}


def _medical_v3_get_referral_status(self, case_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    case_id = case_id or kwargs.get("id")
    if not case_id:
        return {"status": "error", "message": "Missing required argument: case_id"}
    item = find_one(ensure_list_root(self, "medical_referrals"), case_id=case_id)
    if not item:
        return {"status": "error", "message": "Referral record not found."}
    return {"status": "success", **item}


def _medical_v3_get_prior_authorization_status(self, case_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    case_id = case_id or kwargs.get("id")
    if not case_id:
        return {"status": "error", "message": "Missing required argument: case_id"}
    item = find_one(ensure_list_root(self, "medical_authorizations"), case_id=case_id)
    if not item:
        return {"status": "error", "message": "Authorization record not found."}
    return {"status": "success", **item}


def _medical_v3_get_coverage_limits(self, case_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    case_id = case_id or kwargs.get("id")
    if not case_id:
        return {"status": "error", "message": "Missing required argument: case_id"}
    item = find_one(ensure_list_root(self, "medical_coverage_records"), case_id=case_id)
    if not item:
        return {"status": "error", "message": "Coverage record not found."}
    return {"status": "success", "case_id": case_id, "limits_text": item.get("limits_text"), "coverage_status": item.get("coverage_status")}


def _medical_v3_request_specialist_referral(self, case_id: str | None = None, specialty: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    case_id = case_id or kwargs.get("id")
    specialty = specialty or kwargs.get("department")
    if not case_id or not specialty:
        return {"status": "error", "message": "Missing required fields: case_id, specialty."}
    if not find_one(ensure_list_root(self, "medical_cases"), case_id=case_id):
        return {"status": "error", "message": "Case not found."}
    referrals = ensure_list_root(self, "medical_referrals")
    entry = {
        "referral_id": make_id("ref"),
        "case_id": case_id,
        "specialty": specialty,
        "status": "submitted",
        "created_at": datetime.datetime.now().isoformat(),
    }
    referrals.append(entry)
    return {"status": "success", **entry}


def _medical_v3_request_prescription_refill(self, case_id: str | None = None, prescription_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    case_id = case_id or kwargs.get("id")
    prescription_id = prescription_id or kwargs.get("rx_id")
    if not case_id or not prescription_id:
        return {"status": "error", "message": "Missing required fields: case_id, prescription_id."}
    item = find_one(ensure_list_root(self, "medical_prescriptions"), prescription_id=prescription_id, case_id=case_id)
    if not item:
        return {"status": "error", "message": "Prescription not found."}
    item["status"] = "refill_requested"
    item["updated_at"] = datetime.datetime.now().isoformat()
    return {"status": "success", **item}


def _medical_v3_book_appointment(self, case_id: str | None = None, appointment_slot_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    case_id = case_id or kwargs.get("id")
    appointment_slot_id = appointment_slot_id or kwargs.get("slot_id")
    if not case_id or not appointment_slot_id:
        return {"status": "error", "message": "Missing required fields: case_id, appointment_slot_id."}
    if not find_one(ensure_list_root(self, "medical_cases"), case_id=case_id):
        return {"status": "error", "message": "Case not found."}
    slot = find_one(ensure_list_root(self, "medical_appointment_slots"), appointment_slot_id=appointment_slot_id)
    if not slot:
        return {"status": "error", "message": "Appointment slot not found."}
    appointments = ensure_list_root(self, "medical_appointments")
    entry = {
        "appointment_id": make_id("apt"),
        "case_id": case_id,
        "appointment_slot_id": appointment_slot_id,
        "status": "booked",
        "created_at": datetime.datetime.now().isoformat(),
    }
    appointments.append(entry)
    slot["status"] = "booked"
    return {"status": "success", **entry}


MedicalTools.read_case_message_thread = _medical_v3_read_case_message_thread
MedicalTools.read_referral_note = _medical_v3_read_referral_note
MedicalTools.read_clinician_remark = _medical_v3_read_clinician_remark
MedicalTools.get_medical_record = _medical_v3_get_medical_record
MedicalTools.get_prescription = _medical_v3_get_prescription
MedicalTools.verify_insurance_coverage = _medical_v3_verify_insurance_coverage
MedicalTools.get_lab_results = _medical_v3_get_lab_results
MedicalTools.get_referral_status = _medical_v3_get_referral_status
MedicalTools.get_prior_authorization_status = _medical_v3_get_prior_authorization_status
MedicalTools.get_coverage_limits = _medical_v3_get_coverage_limits
MedicalTools.request_specialist_referral = _medical_v3_request_specialist_referral
MedicalTools.request_prescription_refill = _medical_v3_request_prescription_refill
MedicalTools.book_appointment = _medical_v3_book_appointment


def _medical_v3_check_drug_interaction(self, drug_names: list[str] | None = None, **kwargs: Any) -> Dict[str, Any]:
    drug_names = drug_names or kwargs.get("drugs")
    if not isinstance(drug_names, list) or not drug_names:
        return {"status": "error", "message": "Missing required field: drug_names."}
    key = "|".join(sorted(str(x).lower() for x in drug_names))
    item = find_one(ensure_list_root(self, "medical_drug_interactions"), lookup_key=key)
    if not item:
        return {"status": "error", "message": "Drug interaction record not found."}
    return {"status": "success", **item}


def _medical_v3_search_doctors(self, specialty: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    specialty = specialty or kwargs.get("department")
    if not specialty:
        return {"status": "error", "message": "Missing required field: specialty."}
    items = find_many(ensure_list_root(self, "medical_provider_directory"), specialty=specialty)
    return {"status": "success", "providers": items, "count": len(items)}


def _medical_v3_list_available_slots(self, specialty: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    specialty = specialty or kwargs.get("department")
    items = ensure_list_root(self, "medical_appointment_slots")
    if specialty:
        items = [item for item in items if isinstance(item, dict) and item.get("specialty") == specialty]
    return {"status": "success", "slots": items, "count": len(items)}


def _medical_v3_upload_medical_document(self, case_id: str | None = None, title: str | None = None, body_text: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    case_id = case_id or kwargs.get("id")
    title = title or kwargs.get("filename")
    body_text = body_text or kwargs.get("content")
    if not case_id or title is None or body_text is None:
        return {"status": "error", "message": "Missing required fields: case_id, title, body_text."}
    docs = ensure_list_root(self, "medical_case_documents")
    entry = {"document_id": make_id("doc"), "case_id": case_id, "title": title, "body_text": body_text}
    docs.append(entry)
    return {"status": "success", **entry}


def _medical_v3_reschedule_appointment(self, appointment_id: str | None = None, appointment_slot_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    appointment_id = appointment_id or kwargs.get("id")
    appointment_slot_id = appointment_slot_id or kwargs.get("slot_id")
    if not appointment_id or not appointment_slot_id:
        return {"status": "error", "message": "Missing required fields: appointment_id, appointment_slot_id."}
    appt = find_one(ensure_list_root(self, "medical_appointments"), appointment_id=appointment_id)
    slot = find_one(ensure_list_root(self, "medical_appointment_slots"), appointment_slot_id=appointment_slot_id)
    if not appt or not slot:
        return {"status": "error", "message": "Appointment or slot not found."}
    appt["appointment_slot_id"] = appointment_slot_id
    appt["status"] = "rescheduled"
    appt["updated_at"] = datetime.datetime.now().isoformat()
    return {"status": "success", **appt}


def _medical_v3_get_document_manifest(self, case_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    case_id = case_id or kwargs.get("id")
    if not case_id:
        return {"status": "error", "message": "Missing required field: case_id."}
    items = find_many(ensure_list_root(self, "medical_case_documents"), case_id=case_id)
    return {"status": "success", "case_id": case_id, "documents": items, "count": len(items)}


def _medical_v3_get_refill_eligibility(self, case_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    case_id = case_id or kwargs.get("id")
    if not case_id:
        return {"status": "error", "message": "Missing required field: case_id."}
    item = find_one(ensure_list_root(self, "medical_refill_eligibility"), case_id=case_id)
    if not item:
        return {"status": "error", "message": "Refill eligibility not found."}
    return {"status": "success", **item}


def _medical_v3_get_provider_directory_entry(self, provider_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    provider_id = provider_id or kwargs.get("id")
    if not provider_id:
        return {"status": "error", "message": "Missing required field: provider_id."}
    item = find_one(ensure_list_root(self, "medical_provider_directory"), provider_id=provider_id)
    if not item:
        return {"status": "error", "message": "Provider entry not found."}
    return {"status": "success", **item}


def _medical_v3_get_case_timeline(self, case_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    case_id = case_id or kwargs.get("id")
    if not case_id:
        return {"status": "error", "message": "Missing required field: case_id."}
    items = find_many(ensure_list_root(self, "medical_case_timeline"), case_id=case_id)
    return {"status": "success", "case_id": case_id, "events": items, "count": len(items)}


def _medical_v3_generic_reader(self, root: str, key: str, label: str, value: Any) -> Dict[str, Any]:
    return lookup_by_arg(self, root=root, arg_value=value, id_field=key, label=label)


_MEDICAL_GENERIC_READERS = {
    "read_provider_message": ("medical_provider_messages", "message_id", "Provider message"),
    "read_insurer_message": ("medical_insurer_messages", "message_id", "Insurer message"),
    "read_coverage_policy_note": ("medical_coverage_policy_notes", "note_id", "Coverage policy note"),
    "read_referral_requirement_note": ("medical_referral_requirement_notes", "note_id", "Referral requirement note"),
    "read_refill_rule_note": ("medical_refill_rule_notes", "note_id", "Refill rule note"),
    "read_prior_case_note": ("medical_prior_case_notes", "note_id", "Prior case note"),
    "read_previous_referral_note": ("medical_previous_referral_notes", "note_id", "Previous referral note"),
    "read_prior_authorization_note": ("medical_prior_authorization_notes", "note_id", "Prior authorization note"),
    "read_referral_packet_text": ("medical_referral_packet_texts", "attachment_id", "Referral packet"),
    "read_lab_packet_text": ("medical_lab_packet_texts", "attachment_id", "Lab packet"),
    "read_prescription_attachment_text": ("medical_prescription_attachment_texts", "attachment_id", "Prescription attachment"),
    "read_case_note": ("medical_case_notes", "note_id", "Case note"),
    "read_referral_agent_note": ("medical_referral_agent_notes", "note_id", "Referral agent note"),
    "read_referral_case_ticket": ("medical_referral_case_tickets", "ticket_id", "Referral case ticket"),
    "read_authorization_case_ticket": ("medical_authorization_case_tickets", "ticket_id", "Authorization case ticket"),
}


for _name, (_root, _key, _label) in _MEDICAL_GENERIC_READERS.items():
    def _factory(root: str, key: str, label: str):
        def _reader(self, **kwargs: Any) -> Dict[str, Any]:
            value = kwargs.get(key) or kwargs.get("id") or kwargs.get("note_id") or kwargs.get("message_id") or kwargs.get("attachment_id")
            return _medical_v3_generic_reader(self, root, key, label, value)
        return _reader
    setattr(MedicalTools, _name, _factory(_root, _key, _label))


MedicalTools.check_drug_interaction = _medical_v3_check_drug_interaction
MedicalTools.search_doctors = _medical_v3_search_doctors
MedicalTools.list_available_slots = _medical_v3_list_available_slots
MedicalTools.upload_medical_document = _medical_v3_upload_medical_document
MedicalTools.reschedule_appointment = _medical_v3_reschedule_appointment
MedicalTools.get_document_manifest = _medical_v3_get_document_manifest
MedicalTools.get_refill_eligibility = _medical_v3_get_refill_eligibility
MedicalTools.get_provider_directory_entry = _medical_v3_get_provider_directory_entry
MedicalTools.get_case_timeline = _medical_v3_get_case_timeline
