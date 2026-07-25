from typing import Any, Dict, List
import datetime
import json
from core.world.environment import WorldState

class SystemTools:
    """
    System tools connected to the World Model.
    Handles File System and OS Context.
    """

    def _record_file_version(self, path: str, content: str | None, operation: str) -> None:
        versions = WorldState.get_instance().context.setdefault("file_versions", {})
        history = versions.setdefault(path, [])
        history.append(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "operation": operation,
                "content": content,
            }
        )

    def read_file(self, path: str | None = None, file_path: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Read file content."""
        if not path:
            path = file_path or kwargs.get("filename")
        if not path:
            return {"status": "error", "message": "Missing required argument: path"}

        content = WorldState.get_instance().fs.read_file(path)
        if content is None:
            return {"status": "error", "message": f"File {path} not found."}
        return {"status": "success", "content": content}

    def write_file(
        self,
        path: str | None = None,
        content: str | None = None,
        file_path: str | None = None,
        text: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Write content to file."""
        if not path:
            path = file_path or kwargs.get("filename")
        if content is None:
            content = text
        if content is None:
            content = kwargs.get("body")
        if not path:
            return {"status": "error", "message": "Missing required argument: path"}
        if content is None:
            return {"status": "error", "message": "Missing required argument: content"}

        WorldState.get_instance().fs.write_file(path, content)
        self._record_file_version(path, content, "write")
        return {"status": "success", "message": f"File {path} written."}

    def list_files(self) -> Dict[str, Any]:
        """List all files."""
        files = WorldState.get_instance().fs.list_files()
        return {"status": "success", "files": files}

    def delete_file(self, path: str | None = None, file_path: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Delete a file."""
        if not path:
            path = file_path or kwargs.get("filename")
        if not path:
            return {"status": "error", "message": "Missing required argument: path"}

        success = WorldState.get_instance().fs.delete_file(path)
        if success:
            self._record_file_version(path, None, "delete")
            return {"status": "success", "message": f"File {path} deleted."}
        else:
            return {"status": "error", "message": f"File {path} not found."}

    def get_system_time(self) -> Dict[str, Any]:
        """Get simulated system time."""
        return {
            "status": "success",
            "time": WorldState.get_instance().context.get("time"),
            "location": WorldState.get_instance().context.get("location")
        }
    def move_file(
        self,
        src: str | None = None,
        dst: str | None = None,
        source: str | None = None,
        destination: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Move file."""
        if not src:
            src = source
        if not dst:
            dst = destination
        if not src or not dst:
            return {"status": "error", "message": "Missing required fields: src, dst."}

        content = WorldState.get_instance().fs.read_file(src)
        if content is None:
            return {"status": "error", "message": "Source file not found."}
        WorldState.get_instance().fs.write_file(dst, content)
        WorldState.get_instance().fs.delete_file(src)
        self._record_file_version(src, None, "move_out")
        self._record_file_version(dst, content, "move_in")
        return {"status": "success", "message": f"Moved {src} to {dst}."}

    def create_folder(self, path: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Create folder (Mock - since FS is flat key-value, we just ensure parent exists logically)."""
        if not path:
            path = kwargs.get("name")
        if not path:
            return {"status": "error", "message": "Missing required argument: path"}
        WorldState.get_instance().fs.create_folder(path)
        return {"status": "success", "message": f"Folder {path} created."}

    def copy_file(
        self,
        src: str | None = None,
        dst: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Copy file from src to dst."""
        if not src:
            src = kwargs.get("source")
        if not dst:
            dst = kwargs.get("destination")
        if not src or not dst:
            return {"status": "error", "message": "Missing required fields: src, dst."}
        content = WorldState.get_instance().fs.read_file(src)
        if content is None:
            return {"status": "error", "message": "Source file not found."}
        WorldState.get_instance().fs.write_file(dst, content)
        self._record_file_version(dst, content, "copy")
        return {"status": "success", "message": f"Copied {src} to {dst}."}

    def rename_file(self, src: str | None = None, new_name: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Rename file while keeping it in the same parent folder."""
        if not src:
            src = kwargs.get("path")
        if not new_name:
            new_name = kwargs.get("name")
        if not src or not new_name:
            return {"status": "error", "message": "Missing required fields: src, new_name."}
        content = WorldState.get_instance().fs.read_file(src)
        if content is None:
            return {"status": "error", "message": "Source file not found."}
        if "/" in src:
            parent = src.rsplit("/", 1)[0]
            dst = f"{parent}/{new_name}"
        else:
            dst = new_name
        WorldState.get_instance().fs.write_file(dst, content)
        WorldState.get_instance().fs.delete_file(src)
        self._record_file_version(src, None, "rename_out")
        self._record_file_version(dst, content, "rename_in")
        return {"status": "success", "src": src, "dst": dst}

    def create_file(self, path: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Create an empty file if it does not exist."""
        if not path:
            path = kwargs.get("file_path")
        if not path:
            return {"status": "error", "message": "Missing required field: path."}
        if WorldState.get_instance().fs.read_file(path) is not None:
            return {"status": "error", "message": "File already exists."}
        WorldState.get_instance().fs.write_file(path, "")
        self._record_file_version(path, "", "create")
        return {"status": "success", "path": path}

    def compress_folder(
        self,
        folder_path: str | None = None,
        output_path: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compress files under folder_path into a mock archive file."""
        if not folder_path:
            folder_path = kwargs.get("path")
        if not output_path:
            output_path = kwargs.get("archive_path")
        if not folder_path or not output_path:
            return {"status": "error", "message": "Missing required fields: folder_path, output_path."}

        prefix = folder_path.rstrip("/")
        entries = {
            p: c
            for p, c in WorldState.get_instance().fs.files.items()
            if p == prefix or p.startswith(prefix + "/")
        }
        archive_content = "ARCHIVE_JSON:" + json.dumps(entries, ensure_ascii=True)
        WorldState.get_instance().fs.write_file(output_path, archive_content)
        self._record_file_version(output_path, archive_content, "compress")
        return {"status": "success", "output_path": output_path, "file_count": len(entries)}

    def extract_archive(
        self,
        archive_path: str | None = None,
        output_folder: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Extract a mock archive created by compress_folder."""
        if not archive_path:
            archive_path = kwargs.get("path")
        if not output_folder:
            output_folder = kwargs.get("folder")
        if not archive_path or not output_folder:
            return {"status": "error", "message": "Missing required fields: archive_path, output_folder."}
        content = WorldState.get_instance().fs.read_file(archive_path)
        if content is None:
            return {"status": "error", "message": "Archive file not found."}
        if not content.startswith("ARCHIVE_JSON:"):
            return {"status": "error", "message": "Unsupported archive format."}
        raw_json = content[len("ARCHIVE_JSON:") :]
        try:
            entries = json.loads(raw_json)
        except json.JSONDecodeError:
            return {"status": "error", "message": "Archive content corrupted."}
        if not isinstance(entries, dict):
            return {"status": "error", "message": "Archive content invalid."}

        extracted = 0
        for src_path, src_content in entries.items():
            suffix = src_path.lstrip("/")
            dst_path = output_folder.rstrip("/") + "/" + suffix
            WorldState.get_instance().fs.write_file(dst_path, str(src_content))
            self._record_file_version(dst_path, str(src_content), "extract")
            extracted += 1
        WorldState.get_instance().fs.create_folder(output_folder)
        return {"status": "success", "output_folder": output_folder, "file_count": extracted}

    def get_disk_usage(self, path: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Get mocked disk usage statistics for a folder prefix."""
        if not path:
            path = kwargs.get("folder")
        if not path:
            return {"status": "error", "message": "Missing required field: path."}
        prefix = path.rstrip("/")
        file_count = 0
        total_bytes = 0
        for file_path, content in WorldState.get_instance().fs.files.items():
            if file_path == prefix or file_path.startswith(prefix + "/"):
                file_count += 1
                total_bytes += len(content.encode("utf-8"))
        return {
            "status": "success",
            "path": path,
            "file_count": file_count,
            "total_bytes": total_bytes,
        }

    def search_files_by_content(
        self,
        query: str | None = None,
        path_prefix: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Search file contents by substring under a path prefix."""
        if not query:
            query = kwargs.get("keyword")
        if not path_prefix:
            path_prefix = kwargs.get("path")
        if not query or not path_prefix:
            return {"status": "error", "message": "Missing required fields: query, path_prefix."}
        prefix = path_prefix.rstrip("/")
        matches = []
        for file_path, content in WorldState.get_instance().fs.files.items():
            if not (file_path == prefix or file_path.startswith(prefix + "/")):
                continue
            if str(query).lower() in content.lower():
                matches.append(file_path)
        return {"status": "success", "matches": matches, "count": len(matches)}

    def get_file_version_history(self, path: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Return recorded version history for a file path."""
        if not path:
            path = kwargs.get("file_path")
        if not path:
            return {"status": "error", "message": "Missing required field: path."}
        versions = WorldState.get_instance().context.setdefault("file_versions", {})
        return {
            "status": "success",
            "path": path,
            "history": list(versions.get(path, [])),
            "count": len(versions.get(path, [])),
        }


    def get_system_logs(
        self,
        source: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        severity: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Query system / application logs."""
        if not source:
            source = kwargs.get("log_source") or kwargs.get("service")
        if not source:
            return {"status": "error", "message": "Missing required field: source."}
        logs_db = WorldState.get_instance().context.get("system_logs", {})
        entries = logs_db.get(source, [])
        if start_time:
            entries = [e for e in entries if e.get("timestamp", "") >= start_time]
        if end_time:
            entries = [e for e in entries if e.get("timestamp", "") <= end_time]
        if severity:
            entries = [e for e in entries if e.get("severity", "").lower() == severity.lower()]
        return {"status": "success", "source": source, "entries": entries, "count": len(entries)}

    def check_network_connections(
        self,
        host: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """List active network connections, optionally filtered by host."""
        if not host:
            host = kwargs.get("ip") or kwargs.get("server")
        conns = WorldState.get_instance().context.get("network_connections", [])
        if host:
            conns = [c for c in conns if host in str(c)]
        return {"status": "success", "host_filter": host, "connections": conns, "count": len(conns)}

    def get_running_processes(self, **kwargs: Any) -> Dict[str, Any]:
        """Get list of running processes / services."""
        procs = WorldState.get_instance().context.get("processes", [])
        return {"status": "success", "processes": procs, "count": len(procs)}

    def scan_for_vulnerabilities(
        self,
        target: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run a basic vulnerability scan on a target (host, path, or service)."""
        if not target:
            target = kwargs.get("host") or kwargs.get("path")
        if not target:
            return {"status": "error", "message": "Missing required field: target."}
        vuln_db = WorldState.get_instance().context.get("vulnerability_scans", {})
        result = vuln_db.get(target)
        if not result:
            result = {"findings": [], "risk_level": "unknown"}
        return {"status": "success", "target": target, **result}


from tools.connected._v3_roots import ensure_list_root, ensure_object_root, find_many, find_one, lookup_by_arg, make_id


def _system_v3_read_ticket_thread(self, ticket_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    ticket_id = ticket_id or kwargs.get("id")
    if not ticket_id:
        return {"status": "error", "message": "Missing required argument: ticket_id"}
    items = find_many(ensure_list_root(self, "system_ticket_threads"), ticket_id=ticket_id)
    if not items:
        return {"status": "error", "message": f"No ticket thread found for {ticket_id}."}
    return {"status": "success", "ticket_id": ticket_id, "messages": items, "count": len(items)}


def _system_v3_read_runbook_policy(self, service_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    service_id = service_id or kwargs.get("id")
    if not service_id:
        return {"status": "error", "message": "Missing required argument: service_id"}
    item = find_one(ensure_list_root(self, "system_runbook_policies"), service_id=service_id)
    if not item:
        return {"status": "error", "message": "Runbook policy not found."}
    return {"status": "success", **item}


def _system_v3_read_operator_remark(self, ticket_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    ticket_id = ticket_id or kwargs.get("id")
    if not ticket_id:
        return {"status": "error", "message": "Missing required argument: ticket_id"}
    item = find_one(ensure_list_root(self, "system_operator_remarks"), ticket_id=ticket_id)
    if not item:
        return {"status": "error", "message": "Operator remark not found."}
    return {"status": "success", **item}


def _system_v3_read_alert_detail(self, alert_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    alert_id = alert_id or kwargs.get("id")
    if not alert_id:
        return {"status": "error", "message": "Missing required argument: alert_id"}
    item = find_one(ensure_list_root(self, "system_alerts"), alert_id=alert_id)
    if not item:
        return {"status": "error", "message": "Alert not found."}
    return {"status": "success", **item}


def _system_v3_list_tickets(self, status: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    status = status or kwargs.get("state")
    items = list(ensure_list_root(self, "system_tickets"))
    if status:
        items = [item for item in items if isinstance(item, dict) and str(item.get("status", "")).lower() == str(status).lower()]
    return {"status": "success", "tickets": items, "count": len(items)}


def _system_v3_get_service_status(self, service_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    service_id = service_id or kwargs.get("id")
    if not service_id:
        return {"status": "error", "message": "Missing required argument: service_id"}
    item = find_one(ensure_list_root(self, "system_services"), service_id=service_id)
    if not item:
        return {"status": "error", "message": "Service not found."}
    return {"status": "success", **item}


def _system_v3_get_job_history(self, service_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    service_id = service_id or kwargs.get("id")
    if not service_id:
        return {"status": "error", "message": "Missing required argument: service_id"}
    items = find_many(ensure_list_root(self, "system_jobs"), service_id=service_id)
    return {"status": "success", "service_id": service_id, "jobs": items, "count": len(items)}


def _system_v3_get_recent_deployments(self, service_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    service_id = service_id or kwargs.get("id")
    if not service_id:
        return {"status": "error", "message": "Missing required argument: service_id"}
    items = find_many(ensure_list_root(self, "system_deployments"), service_id=service_id)
    return {"status": "success", "service_id": service_id, "deployments": items, "count": len(items)}


def _system_v3_get_process_tree(self, host_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    host_id = host_id or kwargs.get("id")
    if not host_id:
        return {"status": "error", "message": "Missing required argument: host_id"}
    item = find_one(ensure_list_root(self, "system_process_trees"), host_id=host_id)
    if not item:
        return {"status": "error", "message": "Process tree not found."}
    return {"status": "success", **item}


def _system_v3_get_open_ports(self, host_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    host_id = host_id or kwargs.get("id")
    if not host_id:
        return {"status": "error", "message": "Missing required argument: host_id"}
    item = find_one(ensure_list_root(self, "system_open_ports"), host_id=host_id)
    if not item:
        return {"status": "error", "message": "Open port record not found."}
    return {"status": "success", **item}


def _system_v3_get_auth_events(self, host_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    host_id = host_id or kwargs.get("id")
    if not host_id:
        return {"status": "error", "message": "Missing required argument: host_id"}
    items = find_many(ensure_list_root(self, "system_auth_events"), host_id=host_id)
    return {"status": "success", "host_id": host_id, "events": items, "count": len(items)}


def _system_v3_acknowledge_alert(self, alert_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    alert_id = alert_id or kwargs.get("id")
    if not alert_id:
        return {"status": "error", "message": "Missing required field: alert_id."}
    item = find_one(ensure_list_root(self, "system_alerts"), alert_id=alert_id)
    if not item:
        return {"status": "error", "message": "Alert not found."}
    item["status"] = "acknowledged"
    item["updated_at"] = datetime.datetime.now().isoformat()
    return {"status": "success", **item}


def _system_v3_update_ticket_status(self, ticket_id: str | None = None, status: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    ticket_id = ticket_id or kwargs.get("id")
    status = status or kwargs.get("state")
    if not ticket_id or not status:
        return {"status": "error", "message": "Missing required fields: ticket_id, status."}
    item = find_one(ensure_list_root(self, "system_tickets"), ticket_id=ticket_id)
    if not item:
        return {"status": "error", "message": "Ticket not found."}
    item["status"] = status
    item["updated_at"] = datetime.datetime.now().isoformat()
    return {"status": "success", **item}


def _system_v3_restart_service(self, service_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    service_id = service_id or kwargs.get("id")
    if not service_id:
        return {"status": "error", "message": "Missing required field: service_id."}
    item = find_one(ensure_list_root(self, "system_services"), service_id=service_id)
    if not item:
        return {"status": "error", "message": "Service not found."}
    item["status"] = "restarting"
    item["updated_at"] = datetime.datetime.now().isoformat()
    return {"status": "success", **item}


def _system_v3_schedule_maintenance(self, service_id: str | None = None, start_time: str | None = None, end_time: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    service_id = service_id or kwargs.get("id")
    start_time = start_time or kwargs.get("start")
    end_time = end_time or kwargs.get("end")
    if not service_id or not start_time or not end_time:
        return {"status": "error", "message": "Missing required fields: service_id, start_time, end_time."}
    if not find_one(ensure_list_root(self, "system_services"), service_id=service_id):
        return {"status": "error", "message": "Service not found."}
    windows = ensure_list_root(self, "system_maintenance_windows")
    entry = {
        "window_id": make_id("mw"),
        "service_id": service_id,
        "start_time": start_time,
        "end_time": end_time,
        "status": "scheduled",
    }
    windows.append(entry)
    return {"status": "success", **entry}


SystemTools.read_ticket_thread = _system_v3_read_ticket_thread
SystemTools.read_runbook_policy = _system_v3_read_runbook_policy
SystemTools.read_operator_remark = _system_v3_read_operator_remark
SystemTools.read_alert_detail = _system_v3_read_alert_detail
SystemTools.list_tickets = _system_v3_list_tickets
SystemTools.get_service_status = _system_v3_get_service_status
SystemTools.get_job_history = _system_v3_get_job_history
SystemTools.get_recent_deployments = _system_v3_get_recent_deployments
SystemTools.get_process_tree = _system_v3_get_process_tree
SystemTools.get_open_ports = _system_v3_get_open_ports
SystemTools.get_auth_events = _system_v3_get_auth_events
SystemTools.acknowledge_alert = _system_v3_acknowledge_alert
SystemTools.update_ticket_status = _system_v3_update_ticket_status
SystemTools.restart_service = _system_v3_restart_service
SystemTools.schedule_maintenance = _system_v3_schedule_maintenance


def _system_v3_read_ticket(self, ticket_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    ticket_id = ticket_id or kwargs.get("id")
    if not ticket_id:
        return {"status": "error", "message": "Missing required field: ticket_id."}
    item = find_one(ensure_list_root(self, "system_tickets"), ticket_id=ticket_id)
    if not item:
        return {"status": "error", "message": "Ticket not found."}
    return {"status": "success", **item}


def _system_v3_get_host_inventory(self, host_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    host_id = host_id or kwargs.get("id")
    if not host_id:
        return {"status": "error", "message": "Missing required field: host_id."}
    item = find_one(ensure_list_root(self, "system_host_inventory"), host_id=host_id)
    if not item:
        return {"status": "error", "message": "Host inventory not found."}
    return {"status": "success", **item}


def _system_v3_generic_reader(self, root: str, key: str, label: str, value: Any) -> Dict[str, Any]:
    return lookup_by_arg(self, root=root, arg_value=value, id_field=key, label=label)


_SYSTEM_GENERIC_READERS = {
    "read_operator_chat_thread": ("system_operator_chat_threads", "thread_id", "Operator chat thread"),
    "read_escalation_message": ("system_escalation_messages", "message_id", "Escalation message"),
    "read_change_control_rule": ("system_change_control_rules", "rule_id", "Change control rule"),
    "read_incident_severity_rule": ("system_incident_severity_rules", "rule_id", "Incident severity rule"),
    "read_incident_postmortem_note": ("system_incident_postmortem_notes", "note_id", "Incident postmortem note"),
    "read_previous_outage_note": ("system_previous_outage_notes", "note_id", "Previous outage note"),
    "read_deployment_history_note": ("system_deployment_history_notes", "note_id", "Deployment history note"),
    "read_config_bundle_text": ("system_config_bundle_texts", "attachment_id", "Config bundle"),
    "read_log_excerpt_text": ("system_log_excerpt_texts", "attachment_id", "Log excerpt"),
    "read_release_packet_text": ("system_release_packet_texts", "attachment_id", "Release packet"),
    "read_host_note": ("system_host_notes", "note_id", "Host note"),
    "read_service_annotation": ("system_service_annotations", "annotation_id", "Service annotation"),
}


for _name, (_root, _key, _label) in _SYSTEM_GENERIC_READERS.items():
    def _factory(root: str, key: str, label: str):
        def _reader(self, **kwargs: Any) -> Dict[str, Any]:
            value = kwargs.get(key) or kwargs.get("id") or kwargs.get("note_id") or kwargs.get("message_id") or kwargs.get("attachment_id")
            return _system_v3_generic_reader(self, root, key, label, value)
        return _reader
    setattr(SystemTools, _name, _factory(_root, _key, _label))


SystemTools.read_ticket = _system_v3_read_ticket
SystemTools.get_host_inventory = _system_v3_get_host_inventory


def _system_v3_record_file_version(self, path: str, content: str | None, operation: str) -> None:
    versions = ensure_object_root(self, "system_file_versions")
    history = versions.setdefault(path, [])
    if isinstance(history, list):
        history.append(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "operation": operation,
                "content": content,
            }
        )


def _system_v3_read_file(self, path: str | None = None, file_path: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    path = path or file_path or kwargs.get("filename")
    if not path:
        return {"status": "error", "message": "Missing required field: path."}
    files = ensure_object_root(self, "system_files")
    if path not in files:
        return {"status": "error", "message": f"File {path} not found."}
    return {"status": "success", "path": path, "content": files[path]}


def _system_v3_write_file(self, path: str | None = None, content: str | None = None, file_path: str | None = None, text: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    path = path or file_path or kwargs.get("filename")
    content = content if content is not None else text
    if content is None:
        content = kwargs.get("body")
    if not path or content is None:
        return {"status": "error", "message": "Missing required fields: path, content."}
    files = ensure_object_root(self, "system_files")
    files[path] = content
    _system_v3_record_file_version(self, path, content, "write")
    return {"status": "success", "path": path}


def _system_v3_list_files(self, path_prefix: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    path_prefix = path_prefix or kwargs.get("path") or kwargs.get("folder")
    files = ensure_object_root(self, "system_files")
    items = sorted(files.keys())
    if path_prefix:
        prefix = path_prefix.rstrip("/")
        items = [item for item in items if item == prefix or item.startswith(prefix + "/")]
    return {"status": "success", "files": items, "count": len(items)}


def _system_v3_delete_file(self, path: str | None = None, file_path: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    path = path or file_path or kwargs.get("filename")
    if not path:
        return {"status": "error", "message": "Missing required field: path."}
    files = ensure_object_root(self, "system_files")
    if path not in files:
        return {"status": "error", "message": "File not found."}
    old = files.pop(path)
    _system_v3_record_file_version(self, path, old, "delete")
    return {"status": "success", "path": path}


def _system_v3_move_file(self, src: str | None = None, dst: str | None = None, source: str | None = None, destination: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    src = src or source
    dst = dst or destination
    if not src or not dst:
        return {"status": "error", "message": "Missing required fields: src, dst."}
    files = ensure_object_root(self, "system_files")
    if src not in files:
        return {"status": "error", "message": "Source file not found."}
    content = files.pop(src)
    files[dst] = content
    _system_v3_record_file_version(self, src, content, "move_out")
    _system_v3_record_file_version(self, dst, content, "move_in")
    return {"status": "success", "src": src, "dst": dst}


def _system_v3_copy_file(self, src: str | None = None, dst: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    src = src or kwargs.get("source")
    dst = dst or kwargs.get("destination")
    if not src or not dst:
        return {"status": "error", "message": "Missing required fields: src, dst."}
    files = ensure_object_root(self, "system_files")
    if src not in files:
        return {"status": "error", "message": "Source file not found."}
    files[dst] = files[src]
    _system_v3_record_file_version(self, dst, files[src], "copy")
    return {"status": "success", "src": src, "dst": dst}


def _system_v3_rename_file(self, src: str | None = None, new_name: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    src = src or kwargs.get("path")
    new_name = new_name or kwargs.get("name")
    if not src or not new_name:
        return {"status": "error", "message": "Missing required fields: src, new_name."}
    files = ensure_object_root(self, "system_files")
    if src not in files:
        return {"status": "error", "message": "Source file not found."}
    if "/" in src:
        parent = src.rsplit("/", 1)[0]
        dst = f"{parent}/{new_name}"
    else:
        dst = new_name
    content = files.pop(src)
    files[dst] = content
    _system_v3_record_file_version(self, src, content, "rename_out")
    _system_v3_record_file_version(self, dst, content, "rename_in")
    return {"status": "success", "src": src, "dst": dst}


def _system_v3_create_file(self, path: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    path = path or kwargs.get("file_path")
    if not path:
        return {"status": "error", "message": "Missing required field: path."}
    files = ensure_object_root(self, "system_files")
    if path in files:
        return {"status": "error", "message": "File already exists."}
    files[path] = ""
    _system_v3_record_file_version(self, path, "", "create")
    return {"status": "success", "path": path}


def _system_v3_compress_folder(self, folder_path: str | None = None, output_path: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    folder_path = folder_path or kwargs.get("path")
    output_path = output_path or kwargs.get("archive_path")
    if not folder_path or not output_path:
        return {"status": "error", "message": "Missing required fields: folder_path, output_path."}
    prefix = folder_path.rstrip("/")
    files = ensure_object_root(self, "system_files")
    entries = {p: c for p, c in files.items() if p == prefix or p.startswith(prefix + "/")}
    archive_content = "ARCHIVE_JSON:" + json.dumps(entries, ensure_ascii=True)
    files[output_path] = archive_content
    _system_v3_record_file_version(self, output_path, archive_content, "compress")
    return {"status": "success", "output_path": output_path, "file_count": len(entries)}


def _system_v3_extract_archive(self, archive_path: str | None = None, output_folder: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    archive_path = archive_path or kwargs.get("path")
    output_folder = output_folder or kwargs.get("folder")
    if not archive_path or not output_folder:
        return {"status": "error", "message": "Missing required fields: archive_path, output_folder."}
    files = ensure_object_root(self, "system_files")
    content = files.get(archive_path)
    if not isinstance(content, str) or not content.startswith("ARCHIVE_JSON:"):
        return {"status": "error", "message": "Unsupported archive format."}
    try:
        entries = json.loads(content[len("ARCHIVE_JSON:"):])
    except json.JSONDecodeError:
        return {"status": "error", "message": "Archive content corrupted."}
    if not isinstance(entries, dict):
        return {"status": "error", "message": "Archive content invalid."}
    folders = ensure_list_root(self, "system_folders")
    if output_folder not in [item.get("path") for item in folders if isinstance(item, dict)]:
        folders.append({"path": output_folder, "created_at": datetime.datetime.now().isoformat()})
    extracted = 0
    for src_path, src_content in entries.items():
        suffix = str(src_path).lstrip("/")
        dst_path = output_folder.rstrip("/") + "/" + suffix
        files[dst_path] = str(src_content)
        _system_v3_record_file_version(self, dst_path, str(src_content), "extract")
        extracted += 1
    return {"status": "success", "output_folder": output_folder, "file_count": extracted}


def _system_v3_get_disk_usage(self, path: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    path = path or kwargs.get("folder")
    if not path:
        return {"status": "error", "message": "Missing required field: path."}
    prefix = path.rstrip("/")
    files = ensure_object_root(self, "system_files")
    matched = {p: c for p, c in files.items() if p == prefix or p.startswith(prefix + "/")}
    total_bytes = sum(len(str(content).encode("utf-8")) for content in matched.values())
    return {"status": "success", "path": path, "file_count": len(matched), "total_bytes": total_bytes}


def _system_v3_get_file_version_history(self, path: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    path = path or kwargs.get("file_path")
    if not path:
        return {"status": "error", "message": "Missing required field: path."}
    versions = ensure_object_root(self, "system_file_versions")
    history = versions.get(path, [])
    return {"status": "success", "path": path, "history": history, "count": len(history)}


def _system_v3_get_system_logs(self, source: str | None = None, start_time: str | None = None, end_time: str | None = None, severity: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    source = source or kwargs.get("log_source") or kwargs.get("service")
    if not source:
        return {"status": "error", "message": "Missing required field: source."}
    items = [item for item in ensure_list_root(self, "system_logs") if isinstance(item, dict) and item.get("source") == source]
    if start_time:
        items = [item for item in items if str(item.get("timestamp", "")) >= start_time]
    if end_time:
        items = [item for item in items if str(item.get("timestamp", "")) <= end_time]
    if severity:
        items = [item for item in items if str(item.get("severity", "")).lower() == str(severity).lower()]
    return {"status": "success", "source": source, "entries": items, "count": len(items)}


def _system_v3_check_network_connections(self, host: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    host = host or kwargs.get("ip") or kwargs.get("server")
    items = list(ensure_list_root(self, "system_network_connections"))
    if host:
        host_l = str(host).lower()
        items = [
            item for item in items
            if isinstance(item, dict)
            and host_l in " ".join(str(item.get(k, "")) for k in ("host", "host_id", "remote")).lower()
        ]
    return {"status": "success", "host_filter": host, "connections": items, "count": len(items)}


def _system_v3_get_running_processes(self, host_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    host_id = host_id or kwargs.get("id")
    items = list(ensure_list_root(self, "system_running_processes"))
    if host_id:
        items = [item for item in items if isinstance(item, dict) and item.get("host_id") == host_id]
    return {"status": "success", "processes": items, "count": len(items)}


def _system_v3_scan_for_vulnerabilities(self, target: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    target = target or kwargs.get("host") or kwargs.get("path")
    if not target:
        return {"status": "error", "message": "Missing required field: target."}
    item = find_one(ensure_list_root(self, "system_vulnerability_scans"), target=target)
    if not item:
        return {"status": "error", "message": "Vulnerability scan not found."}
    return {"status": "success", **item}


SystemTools.read_file = _system_v3_read_file
SystemTools.write_file = _system_v3_write_file
SystemTools.list_files = _system_v3_list_files
SystemTools.delete_file = _system_v3_delete_file
SystemTools.move_file = _system_v3_move_file
SystemTools.copy_file = _system_v3_copy_file
SystemTools.rename_file = _system_v3_rename_file
SystemTools.create_file = _system_v3_create_file
SystemTools.compress_folder = _system_v3_compress_folder
SystemTools.extract_archive = _system_v3_extract_archive
SystemTools.get_disk_usage = _system_v3_get_disk_usage
SystemTools.get_file_version_history = _system_v3_get_file_version_history
SystemTools.get_system_logs = _system_v3_get_system_logs
SystemTools.check_network_connections = _system_v3_check_network_connections
SystemTools.get_running_processes = _system_v3_get_running_processes
SystemTools.scan_for_vulnerabilities = _system_v3_scan_for_vulnerabilities
