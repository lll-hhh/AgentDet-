from typing import Any, Dict, List
import uuid
import datetime
from core.world.environment import WorldState

class SmartHomeTools:
    """
    SmartHome tools connected to the World Model.
    Handles IoT devices like locks, lights, thermostats.
    """

    def _find_security_system_id(self) -> str | None:
        for dev_id, state in WorldState.get_instance().iot.devices.items():
            if state.get("type") == "security_system":
                return dev_id
        return None

    def list_devices(self) -> Dict[str, Any]:
        """
        List all available IoT devices and their types.
        """
        devices = []
        for dev_id, state in WorldState.get_instance().iot.devices.items():
            devices.append({
                "id": dev_id,
                "type": state.get("type", "unknown"),
                "status": state.get("status", "unknown")
            })

        return {
            "status": "success",
            "devices": devices,
            "count": len(devices)
        }

    def get_device_status(
        self,
        device_id: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Get full status of a specific device.
        """
        if not device_id:
            device_id = kwargs.get("id")
        if not device_id:
            return {"status": "error", "message": "Missing required argument: device_id"}

        state = WorldState.get_instance().iot.get_state(device_id)
        if not state:
            return {"status": "error", "message": f"Device {device_id} not found."}

        return {
            "status": "success",
            "device_id": device_id,
            "state": state
        }

    def update_device_state(
        self,
        device_id: str | None = None,
        property: str | None = None,
        value: Any = None,
        action: str | None = None,
        state: Any = None,
        status: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Update a property of a device.
        CRITICAL: This simulates Physical Reality.
        E.g., Cannot 'unlock' if 'system_armed' is True (logic can be extended here).
        """
        if not device_id:
            device_id = kwargs.get("id")
        if property is None:
            property = kwargs.get("key")

        if action and property is None:
            if action in {"on", "off"}:
                property = "status"
                value = "on" if action == "on" else "off"
            elif action == "set":
                property = "value"
            elif action in {"lock", "unlock"}:
                property = "status"
                value = "locked" if action == "lock" else "unlocked"

        if property is None and status is not None:
            property = "status"
            value = status
        if property is None and state is not None:
            property = "status"
            value = state

        if property == "state" and isinstance(value, dict) and "locked" in value:
            property = "status"
            value = "locked" if value["locked"] else "unlocked"

        if isinstance(value, dict) and len(value) == 1:
            inner_key, inner_val = next(iter(value.items()))
            if inner_key in (property, "value", "state", "data"):
                value = inner_val

        if not device_id or property is None:
            return {
                "status": "error",
                "message": "Missing required fields: device_id, property.",
            }

        state = WorldState.get_instance().iot.get_state(device_id)
        if not state:
            return {"status": "error", "message": f"Device {device_id} not found."}

        if property == "status" and value == "unlocked" and state.get("type") == "lock":
            for d_id, d_state in WorldState.get_instance().iot.devices.items():
                if d_state.get("type") == "security_system" and d_state.get("status") == "armed":
                     return {
                         "status": "error",
                         "message": "Security System prevented unlocking. Disarm system first."
                     }

        WorldState.get_instance().iot.update_state(device_id, {property: value})

        return {
            "status": "success",
            "message": f"Device {device_id} updated.",
            "new_state": WorldState.get_instance().iot.get_state(device_id)
        }

    def lock_door(self, device_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Secure entry."""
        if not device_id:
            device_id = kwargs.get("id")
        if not device_id:
            return {"status": "error", "message": "Missing required argument: device_id"}
        return self.update_device_state(device_id, "status", "locked")

    def unlock_door(self, device_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Unsecure entry."""
        if not device_id:
            device_id = kwargs.get("id")
        if not device_id:
            return {"status": "error", "message": "Missing required argument: device_id"}
        return self.update_device_state(device_id, "status", "unlocked")

    def arm_system(self) -> Dict[str, Any]:
        """Activate security."""
        for d_id, state in WorldState.get_instance().iot.devices.items():
            if state.get("type") == "security_system":
                WorldState.get_instance().iot.update_state(d_id, {"status": "armed"})
                return {"status": "success", "message": "System armed."}
        return {"status": "error", "message": "No security system found."}

    def disarm_system(self) -> Dict[str, Any]:
        """Deactivate security."""
        for d_id, state in WorldState.get_instance().iot.devices.items():
            if state.get("type") == "security_system":
                WorldState.get_instance().iot.update_state(d_id, {"status": "disarmed"})
                return {"status": "success", "message": "System disarmed."}
        return {"status": "error", "message": "No security system found."}

    def get_security_camera_image(self, camera_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Get feed snapshot."""
        if not camera_id:
            camera_id = kwargs.get("device_id")
        if not camera_id:
            return {"status": "error", "message": "Missing required argument: camera_id"}

        state = WorldState.get_instance().iot.get_state(camera_id)
        if not state or state.get("type") != "camera":
            return {"status": "error", "message": "Camera not found."}
        return {"status": "success", "image_url": f"http://mock-cam/{camera_id}/snapshot.jpg"}

    def set_thermostat_mode(self, mode: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """HVAC control."""
        if not mode:
            mode = kwargs.get("value")
        if not mode:
            return {"status": "error", "message": "Missing required argument: mode"}

        for d_id, state in WorldState.get_instance().iot.devices.items():
            if state.get("type") == "thermostat":
                WorldState.get_instance().iot.update_state(d_id, {"mode": mode})
                return {"status": "success", "message": f"Thermostat set to {mode}."}
        return {"status": "error", "message": "Thermostat not found."}

    def get_sensor_data(self, sensor_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Get sensor data from a sensor-like device."""
        if not sensor_id:
            sensor_id = kwargs.get("device_id")
        if not sensor_id:
            return {"status": "error", "message": "Missing required argument: sensor_id"}

        state = WorldState.get_instance().iot.get_state(sensor_id)
        if not state:
            return {"status": "error", "message": f"Sensor {sensor_id} not found."}
        reading = state.get("value", state.get("reading", state.get("status")))
        return {
            "status": "success",
            "sensor_id": sensor_id,
            "value": reading,
            "raw_state": state,
        }

    def control_device(
        self,
        device_id: str | None = None,
        action: str | None = None,
        value: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compatibility wrapper for action-based smart home controls."""
        if not device_id:
            device_id = kwargs.get("id")
        if not action:
            action = kwargs.get("operation")
        if value is None:
            value = kwargs.get("state")
        if not device_id or not action:
            return {
                "status": "error",
                "message": "Missing required fields: device_id, action.",
            }

        lowered = str(action).lower()
        if lowered == "lock":
            return self.lock_door(device_id=device_id)
        if lowered == "unlock":
            return self.unlock_door(device_id=device_id)
        if lowered == "arm":
            return self.arm_system()
        if lowered == "disarm":
            return self.disarm_system()
        if lowered in {"on", "off", "toggle", "set"}:
            return self.update_device_state(
                device_id=device_id,
                action=lowered,
                value=value,
            )
        return self.update_device_state(
            device_id=device_id,
            property="status",
            value=lowered,
        )

    def add_device(
        self,
        device_id: str | None = None,
        type: str | None = None,
        room: str | None = None,
        state: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Add a new IoT device."""
        if not device_id:
            device_id = kwargs.get("id")
        if not type:
            type = kwargs.get("device_type")
        if state is None:
            raw_state = kwargs.get("status")
            if isinstance(raw_state, dict):
                state = raw_state
        if not device_id or not type:
            return {"status": "error", "message": "Missing required fields: device_id, type."}
        if device_id in WorldState.get_instance().iot.devices:
            return {"status": "error", "message": "Device already exists."}
        payload: Dict[str, Any] = {"type": type, "status": "off"}
        if room:
            payload["room"] = room
        if isinstance(state, dict):
            payload.update(state)
        WorldState.get_instance().iot.devices[device_id] = payload
        return {"status": "success", "device_id": device_id, "state": payload}

    def remove_device(self, device_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Remove an IoT device."""
        if not device_id:
            device_id = kwargs.get("id")
        if not device_id:
            return {"status": "error", "message": "Missing required field: device_id."}
        if device_id not in WorldState.get_instance().iot.devices:
            return {"status": "error", "message": "Device not found."}
        del WorldState.get_instance().iot.devices[device_id]
        return {"status": "success", "device_id": device_id}

    def set_device_schedule(
        self,
        device_id: str | None = None,
        action: str | None = None,
        cron: str | None = None,
        value: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Set a scheduled action for a device."""
        if not device_id:
            device_id = kwargs.get("id")
        if not action:
            action = kwargs.get("operation")
        if not cron:
            cron = kwargs.get("schedule")
        if not device_id or not action or not cron:
            return {
                "status": "error",
                "message": "Missing required fields: device_id, action, cron.",
            }
        if device_id not in WorldState.get_instance().iot.devices:
            return {"status": "error", "message": "Device not found."}
        schedule_id = f"sch_{uuid.uuid4().hex[:8]}"
        schedules = WorldState.get_instance().context.setdefault("device_schedules", {})
        schedules[schedule_id] = {
            "schedule_id": schedule_id,
            "device_id": device_id,
            "action": action,
            "cron": cron,
            "value": value,
            "status_text": "active",
            "created_at": datetime.datetime.now().isoformat(),
        }
        return {"status": "success", "schedule_id": schedule_id}

    def remove_device_schedule(self, schedule_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Remove or deactivate a device schedule."""
        if not schedule_id:
            schedule_id = kwargs.get("id")
        if not schedule_id:
            return {"status": "error", "message": "Missing required field: schedule_id."}
        schedules = WorldState.get_instance().context.setdefault("device_schedules", {})
        item = schedules.get(schedule_id)
        if not item:
            return {"status": "error", "message": "Schedule not found."}
        item["status_text"] = "removed"
        return {"status": "success", "schedule_id": schedule_id}

    def set_alarm_mode(self, mode: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Set alarm mode on security system."""
        if not mode:
            mode = kwargs.get("value")
        if not mode:
            return {"status": "error", "message": "Missing required field: mode."}
        sec_id = self._find_security_system_id()
        if not sec_id:
            sec_id = "security_system"
            WorldState.get_instance().iot.devices[sec_id] = {"type": "security_system", "status": "disarmed"}
        WorldState.get_instance().iot.update_state(sec_id, {"mode": mode})
        return {"status": "success", "device_id": sec_id, "mode": mode}

    def trigger_siren(self, duration_sec: int | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Trigger siren device for a duration."""
        if duration_sec is None:
            duration_sec = kwargs.get("duration")
        if duration_sec is None:
            duration_sec = 30
        try:
            duration_sec = int(duration_sec)
        except (TypeError, ValueError):
            return {"status": "error", "message": "Invalid duration_sec format."}
        duration_sec = max(1, min(duration_sec, 3600))
        siren_id = next(
            (d_id for d_id, st in WorldState.get_instance().iot.devices.items() if st.get("type") == "siren"),
            None,
        )
        if not siren_id:
            siren_id = "siren"
            WorldState.get_instance().iot.devices[siren_id] = {"type": "siren"}
        WorldState.get_instance().iot.update_state(
            siren_id,
            {
                "status": "on",
                "duration_sec": duration_sec,
                "triggered_at": datetime.datetime.now().isoformat(),
            },
        )
        return {"status": "success", "device_id": siren_id, "duration_sec": duration_sec}

    def mute_siren(self, **kwargs: Any) -> Dict[str, Any]:
        """Mute active siren."""
        siren_id = next(
            (d_id for d_id, st in WorldState.get_instance().iot.devices.items() if st.get("type") == "siren"),
            None,
        )
        if not siren_id:
            return {"status": "error", "message": "Siren not found."}
        WorldState.get_instance().iot.update_state(siren_id, {"status": "muted", "muted_at": datetime.datetime.now().isoformat()})
        return {"status": "success", "device_id": siren_id}

    def create_scene(
        self,
        scene_name: str | None = None,
        actions: List[Dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a smart-home scene with a list of device actions."""
        if not scene_name:
            scene_name = kwargs.get("name")
        if actions is None and isinstance(kwargs.get("steps"), list):
            actions = kwargs.get("steps")
        if not scene_name or not isinstance(actions, list):
            return {"status": "error", "message": "Missing required fields: scene_name, actions(list)."}
        scenes = WorldState.get_instance().context.setdefault("iot_scenes", {})
        scenes[scene_name] = {
            "scene_name": scene_name,
            "actions": actions,
            "created_at": datetime.datetime.now().isoformat(),
        }
        return {"status": "success", "scene_name": scene_name, "action_count": len(actions)}

    def run_scene(self, scene_name: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Execute a previously defined smart-home scene."""
        if not scene_name:
            scene_name = kwargs.get("name")
        if not scene_name:
            return {"status": "error", "message": "Missing required field: scene_name."}
        scenes = WorldState.get_instance().context.setdefault("iot_scenes", {})
        scene = scenes.get(scene_name)
        if not scene:
            return {"status": "error", "message": "Scene not found."}
        applied = 0
        for action in scene.get("actions", []):
            if not isinstance(action, dict):
                continue
            device_id = action.get("device_id")
            if not device_id:
                continue
            if device_id not in WorldState.get_instance().iot.devices:
                WorldState.get_instance().iot.devices[device_id] = {}
            if "property" in action:
                WorldState.get_instance().iot.update_state(device_id, {action["property"]: action.get("value")})
                applied += 1
                continue
            if "status" in action:
                WorldState.get_instance().iot.update_state(device_id, {"status": action.get("status")})
                applied += 1
                continue
            if "action" in action:
                lowered = str(action["action"]).lower()
                if lowered == "lock":
                    WorldState.get_instance().iot.update_state(device_id, {"status": "locked"})
                    applied += 1
                elif lowered == "unlock":
                    WorldState.get_instance().iot.update_state(device_id, {"status": "unlocked"})
                    applied += 1
                elif lowered in {"on", "off"}:
                    WorldState.get_instance().iot.update_state(device_id, {"status": lowered})
                    applied += 1
        WorldState.get_instance().context.setdefault("iot_scene_runs", []).append(
            {
                "scene_name": scene_name,
                "applied_actions": applied,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )
        return {"status": "success", "scene_name": scene_name, "applied_actions": applied}

    def set_camera_privacy_mode(
        self,
        camera_id: str | None = None,
        enabled: bool | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Enable/disable camera privacy mode."""
        if not camera_id:
            camera_id = kwargs.get("device_id")
        if enabled is None:
            enabled = kwargs.get("value")
        if not camera_id or enabled is None:
            return {"status": "error", "message": "Missing required fields: camera_id, enabled."}
        if camera_id not in WorldState.get_instance().iot.devices:
            return {"status": "error", "message": "Camera not found."}
        WorldState.get_instance().iot.update_state(camera_id, {"privacy_mode": bool(enabled)})
        return {"status": "success", "camera_id": camera_id, "privacy_mode": bool(enabled)}

    def set_geofence_rule(
        self,
        rule_id: str | None = None,
        center: Dict[str, Any] | str | None = None,
        radius: float | None = None,
        trigger_action: Dict[str, Any] | str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Set a geofence automation rule."""
        if not rule_id:
            rule_id = kwargs.get("id") or f"geo_{uuid.uuid4().hex[:8]}"
        if center is None:
            center = kwargs.get("location")
        if radius is None:
            radius = kwargs.get("distance")
        if trigger_action is None:
            trigger_action = kwargs.get("action")
        if center is None or radius is None or trigger_action is None:
            return {
                "status": "error",
                "message": "Missing required fields: center, radius, trigger_action.",
            }
        try:
            radius = float(radius)
        except (TypeError, ValueError):
            return {"status": "error", "message": "Invalid radius format."}
        rules = WorldState.get_instance().context.setdefault("geofence_rules", {})
        rules[rule_id] = {
            "rule_id": rule_id,
            "center": center,
            "radius": radius,
            "trigger_action": trigger_action,
            "updated_at": datetime.datetime.now().isoformat(),
        }
        return {"status": "success", "rule_id": rule_id}

    def get_energy_consumption_report(
        self,
        start_time: str | None = None,
        end_time: str | None = None,
        device_id: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a simple mocked energy report."""
        if not start_time:
            start_time = kwargs.get("start")
        if not end_time:
            end_time = kwargs.get("end")
        if not device_id:
            device_id = kwargs.get("id")
        if not start_time or not end_time:
            return {"status": "error", "message": "Missing required fields: start_time, end_time."}

        target_ids = [device_id] if device_id else list(WorldState.get_instance().iot.devices.keys())
        total_wh = 0.0
        by_device: Dict[str, float] = {}
        for d_id in target_ids:
            state = WorldState.get_instance().iot.devices.get(d_id, {})
            base = 5.0
            if str(state.get("status", "")).lower() in {"on", "armed", "active"}:
                base = 22.0
            if state.get("type") in {"thermostat", "heater", "ac"}:
                base += 10.0
            by_device[d_id] = round(base, 3)
            total_wh += base
        report = {
            "report_id": f"energy_{uuid.uuid4().hex[:8]}",
            "start_time": start_time,
            "end_time": end_time,
            "device_id": device_id,
            "consumption_wh": round(total_wh, 3),
            "by_device": by_device,
            "generated_at": datetime.datetime.now().isoformat(),
        }
        WorldState.get_instance().context.setdefault("energy_reports", []).append(report)
        return {"status": "success", **report}


    def get_activity_log(
        self,
        device_id: str | None = None,
        period: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Get event/activity history for a device or the whole home."""
        if not device_id:
            device_id = kwargs.get("id")
        if not period:
            period = kwargs.get("timeframe") or "24h"
        logs = WorldState.get_instance().context.get("activity_logs", {})
        if device_id:
            entries = logs.get(device_id, [])
        else:
            entries = []
            for v in logs.values():
                entries.extend(v)
        return {"status": "success", "device_id": device_id, "period": period,
                "entries": entries, "count": len(entries)}

    def check_firmware_version(
        self,
        device_id: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Check current firmware version and whether update is available."""
        if not device_id:
            device_id = kwargs.get("id")
        if not device_id:
            return {"status": "error", "message": "Missing required field: device_id."}
        fw_db = WorldState.get_instance().context.get("firmware_info", {})
        info = fw_db.get(device_id)
        if not info:
            state = WorldState.get_instance().iot.devices.get(device_id)
            if not state:
                return {"status": "error", "message": f"Device {device_id} not found."}
            info = {"current_version": state.get("firmware", "1.0.0"),
                    "latest_version": "1.0.0", "update_available": False}
        return {"status": "success", "device_id": device_id, **info}

    def get_network_status(self, **kwargs: Any) -> Dict[str, Any]:
        """Get home network status and connected device count."""
        net = WorldState.get_instance().context.get("network_status")
        if net:
            return {"status": "success", **net}
        device_count = len(WorldState.get_instance().iot.devices)
        return {
            "status": "success",
            "connected_devices": device_count,
            "network_type": "WiFi 6",
            "signal_strength": "good",
            "external_ip": "203.0.113.42",
        }


from tools.connected._v3_roots import ensure_list_root, find_many, find_one, lookup_by_arg


def _smarthome_v3_read_access_request_thread(self, device_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    device_id = device_id or kwargs.get("id")
    if not device_id:
        return {"status": "error", "message": "Missing required argument: device_id"}
    items = find_many(ensure_list_root(self, "smarthome_access_threads"), device_id=device_id)
    if not items:
        return {"status": "error", "message": f"No access thread found for {device_id}."}
    return {"status": "success", "device_id": device_id, "messages": items, "count": len(items)}


def _smarthome_v3_read_security_policy_note(self, policy_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    policy_id = policy_id or kwargs.get("id")
    if not policy_id:
        return {"status": "error", "message": "Missing required argument: policy_id"}
    item = find_one(ensure_list_root(self, "smarthome_security_policy_notes"), policy_id=policy_id)
    if not item:
        return {"status": "error", "message": "Security policy note not found."}
    return {"status": "success", **item}


def _smarthome_v3_read_incident_history_note(self, device_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    device_id = device_id or kwargs.get("id")
    if not device_id:
        return {"status": "error", "message": "Missing required argument: device_id"}
    items = find_many(ensure_list_root(self, "smarthome_incident_history_notes"), device_id=device_id)
    if not items:
        return {"status": "error", "message": "Incident history note not found."}
    return {"status": "success", "device_id": device_id, "notes": items, "count": len(items)}


def _smarthome_v3_list_devices(self, room: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    room = room or kwargs.get("zone")
    items = list(ensure_list_root(self, "smarthome_devices"))
    if room:
        items = [item for item in items if isinstance(item, dict) and item.get("room") == room]
    return {"status": "success", "devices": items, "count": len(items)}


def _smarthome_v3_get_device_status(self, device_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    device_id = device_id or kwargs.get("id")
    if not device_id:
        return {"status": "error", "message": "Missing required argument: device_id"}
    item = find_one(ensure_list_root(self, "smarthome_devices"), device_id=device_id)
    if not item:
        return {"status": "error", "message": "Device not found."}
    return {"status": "success", **item}


def _smarthome_v3_get_activity_log(self, device_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    device_id = device_id or kwargs.get("id")
    if not device_id:
        return {"status": "error", "message": "Missing required argument: device_id"}
    item = find_one(ensure_list_root(self, "smarthome_activity_logs"), device_id=device_id)
    if not item:
        return {"status": "error", "message": "Activity log not found."}
    return {"status": "success", **item}


def _smarthome_v3_get_sensor_data(self, device_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    device_id = device_id or kwargs.get("id")
    if not device_id:
        return {"status": "error", "message": "Missing required argument: device_id"}
    item = find_one(ensure_list_root(self, "smarthome_sensor_data"), device_id=device_id)
    if not item:
        return {"status": "error", "message": "Sensor data not found."}
    return {"status": "success", **item}


def _smarthome_v3_get_alarm_history(self, device_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    device_id = device_id or kwargs.get("id")
    if not device_id:
        return {"status": "error", "message": "Missing required argument: device_id"}
    items = find_many(ensure_list_root(self, "smarthome_alarm_history"), device_id=device_id)
    if not items:
        return {"status": "error", "message": "Alarm history not found."}
    return {"status": "success", "device_id": device_id, "events": items, "count": len(items)}


def _smarthome_v3_get_schedule_details(self, device_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    device_id = device_id or kwargs.get("id")
    if not device_id:
        return {"status": "error", "message": "Missing required argument: device_id"}
    items = find_many(ensure_list_root(self, "smarthome_schedules"), device_id=device_id)
    if not items:
        return {"status": "error", "message": "Schedule not found."}
    return {"status": "success", "device_id": device_id, "schedules": items, "count": len(items)}


def _smarthome_v3_get_user_access_list(self, device_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    device_id = device_id or kwargs.get("id")
    if not device_id:
        return {"status": "error", "message": "Missing required argument: device_id"}
    item = find_one(ensure_list_root(self, "smarthome_access_lists"), device_id=device_id)
    if not item:
        return {"status": "error", "message": "Access list not found."}
    return {"status": "success", **item}


def _smarthome_v3_lock_door(self, device_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    device_id = device_id or kwargs.get("id")
    if not device_id:
        return {"status": "error", "message": "Missing required field: device_id."}
    item = find_one(ensure_list_root(self, "smarthome_devices"), device_id=device_id)
    if not item:
        return {"status": "error", "message": "Device not found."}
    item["status"] = "locked"
    item["updated_at"] = datetime.datetime.now().isoformat()
    return {"status": "success", **item}


def _smarthome_v3_unlock_door(self, device_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    device_id = device_id or kwargs.get("id")
    if not device_id:
        return {"status": "error", "message": "Missing required field: device_id."}
    item = find_one(ensure_list_root(self, "smarthome_devices"), device_id=device_id)
    if not item:
        return {"status": "error", "message": "Device not found."}
    item["status"] = "unlocked"
    item["updated_at"] = datetime.datetime.now().isoformat()
    return {"status": "success", **item}


def _smarthome_v3_set_alarm_mode(self, device_id: str | None = None, mode: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    device_id = device_id or kwargs.get("id")
    mode = mode or kwargs.get("alarm_mode")
    if not device_id or mode is None:
        return {"status": "error", "message": "Missing required fields: device_id, mode."}
    item = find_one(ensure_list_root(self, "smarthome_devices"), device_id=device_id)
    if not item:
        return {"status": "error", "message": "Device not found."}
    item["alarm_mode"] = mode
    item["updated_at"] = datetime.datetime.now().isoformat()
    return {"status": "success", **item}


def _smarthome_v3_remove_device_schedule(self, schedule_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    schedule_id = schedule_id or kwargs.get("id")
    if not schedule_id:
        return {"status": "error", "message": "Missing required field: schedule_id."}
    schedules = ensure_list_root(self, "smarthome_schedules")
    item = find_one(schedules, schedule_id=schedule_id)
    if not item:
        return {"status": "error", "message": "Schedule not found."}
    item["status"] = "removed"
    item["updated_at"] = datetime.datetime.now().isoformat()
    return {"status": "success", **item}


SmartHomeTools.read_access_request_thread = _smarthome_v3_read_access_request_thread
SmartHomeTools.read_security_policy_note = _smarthome_v3_read_security_policy_note
SmartHomeTools.read_incident_history_note = _smarthome_v3_read_incident_history_note
SmartHomeTools.list_devices = _smarthome_v3_list_devices
SmartHomeTools.get_device_status = _smarthome_v3_get_device_status
SmartHomeTools.get_activity_log = _smarthome_v3_get_activity_log
SmartHomeTools.get_sensor_data = _smarthome_v3_get_sensor_data
SmartHomeTools.get_alarm_history = _smarthome_v3_get_alarm_history
SmartHomeTools.get_schedule_details = _smarthome_v3_get_schedule_details
SmartHomeTools.get_user_access_list = _smarthome_v3_get_user_access_list
SmartHomeTools.lock_door = _smarthome_v3_lock_door
SmartHomeTools.unlock_door = _smarthome_v3_unlock_door
SmartHomeTools.set_alarm_mode = _smarthome_v3_set_alarm_mode
SmartHomeTools.remove_device_schedule = _smarthome_v3_remove_device_schedule


def _smarthome_v3_get_network_status(self, **kwargs: Any) -> Dict[str, Any]:
    item = find_one(ensure_list_root(self, "smarthome_network_status"), network_id="default")
    if not item:
        return {"status": "error", "message": "Network status not found."}
    return {"status": "success", **item}


def _smarthome_v3_check_firmware_version(self, device_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    device_id = device_id or kwargs.get("id")
    if not device_id:
        return {"status": "error", "message": "Missing required field: device_id."}
    item = find_one(ensure_list_root(self, "smarthome_firmware_versions"), device_id=device_id)
    if not item:
        return {"status": "error", "message": "Firmware version not found."}
    return {"status": "success", **item}


def _smarthome_v3_get_energy_consumption_report(self, device_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    device_id = device_id or kwargs.get("id")
    if not device_id:
        return {"status": "error", "message": "Missing required field: device_id."}
    item = find_one(ensure_list_root(self, "smarthome_energy_reports"), device_id=device_id)
    if not item:
        return {"status": "error", "message": "Energy report not found."}
    return {"status": "success", **item}


def _smarthome_v3_set_camera_privacy_mode(self, device_id: str | None = None, privacy_mode: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    device_id = device_id or kwargs.get("id")
    privacy_mode = privacy_mode or kwargs.get("mode")
    if not device_id or privacy_mode is None:
        return {"status": "error", "message": "Missing required fields: device_id, privacy_mode."}
    item = find_one(ensure_list_root(self, "smarthome_devices"), device_id=device_id)
    if not item:
        return {"status": "error", "message": "Device not found."}
    item["privacy_mode"] = privacy_mode
    item["updated_at"] = datetime.datetime.now().isoformat()
    return {"status": "success", **item}


def _smarthome_v3_set_device_schedule(self, device_id: str | None = None, window: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    device_id = device_id or kwargs.get("id")
    window = window or kwargs.get("schedule_window")
    if not device_id or window is None:
        return {"status": "error", "message": "Missing required fields: device_id, window."}
    schedules = ensure_list_root(self, "smarthome_schedules")
    entry = {"schedule_id": f"sch_{datetime.datetime.now().strftime('%H%M%S%f')[:10]}", "device_id": device_id, "window": window, "status": "active"}
    schedules.append(entry)
    return {"status": "success", **entry}


def _smarthome_v3_get_device_capabilities(self, device_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    device_id = device_id or kwargs.get("id")
    if not device_id:
        return {"status": "error", "message": "Missing required field: device_id."}
    item = find_one(ensure_list_root(self, "smarthome_device_capabilities"), device_id=device_id)
    if not item:
        return {"status": "error", "message": "Device capabilities not found."}
    return {"status": "success", **item}


def _smarthome_v3_get_scene_members(self, scene_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    scene_id = scene_id or kwargs.get("id")
    if not scene_id:
        return {"status": "error", "message": "Missing required field: scene_id."}
    items = find_many(ensure_list_root(self, "smarthome_scene_members"), scene_id=scene_id)
    return {"status": "success", "scene_id": scene_id, "members": items, "count": len(items)}


def _smarthome_v3_get_battery_health(self, device_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    device_id = device_id or kwargs.get("id")
    if not device_id:
        return {"status": "error", "message": "Missing required field: device_id."}
    item = find_one(ensure_list_root(self, "smarthome_battery_health"), device_id=device_id)
    if not item:
        return {"status": "error", "message": "Battery health not found."}
    return {"status": "success", **item}


def _smarthome_v3_get_recent_automation_failures(self, device_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    device_id = device_id or kwargs.get("id")
    if not device_id:
        return {"status": "error", "message": "Missing required field: device_id."}
    items = find_many(ensure_list_root(self, "smarthome_recent_automation_failures"), device_id=device_id)
    return {"status": "success", "device_id": device_id, "failures": items, "count": len(items)}


def _smarthome_v3_generic_reader(self, root: str, key: str, label: str, value: Any) -> Dict[str, Any]:
    return lookup_by_arg(self, root=root, arg_value=value, id_field=key, label=label)


_SMARTHOME_GENERIC_READERS = {
    "read_guest_message": ("smarthome_guest_messages", "message_id", "Guest message"),
    "read_operator_followup_message": ("smarthome_operator_followup_messages", "message_id", "Operator follow-up message"),
    "read_alarm_rule_note": ("smarthome_alarm_rule_notes", "note_id", "Alarm rule note"),
    "read_schedule_rule_note": ("smarthome_schedule_rule_notes", "note_id", "Schedule rule note"),
    "read_previous_alarm_note": ("smarthome_previous_alarm_notes", "note_id", "Previous alarm note"),
    "read_access_history_note": ("smarthome_access_history_notes", "note_id", "Access history note"),
    "read_device_packet_text": ("smarthome_device_packet_texts", "attachment_id", "Device packet"),
    "read_camera_caption_text": ("smarthome_camera_caption_texts", "attachment_id", "Camera caption"),
    "read_firmware_advisory_text": ("smarthome_firmware_advisory_texts", "attachment_id", "Firmware advisory"),
    "read_device_note": ("smarthome_device_notes", "note_id", "Device note"),
    "read_access_override_note": ("smarthome_access_override_notes", "note_id", "Access override note"),
    "read_device_incident_ticket": ("smarthome_device_incident_tickets", "ticket_id", "Device incident ticket"),
    "read_access_case_ticket": ("smarthome_access_case_tickets", "ticket_id", "Access case ticket"),
}


for _name, (_root, _key, _label) in _SMARTHOME_GENERIC_READERS.items():
    def _factory(root: str, key: str, label: str):
        def _reader(self, **kwargs: Any) -> Dict[str, Any]:
            value = kwargs.get(key) or kwargs.get("id") or kwargs.get("note_id") or kwargs.get("message_id") or kwargs.get("attachment_id")
            return _smarthome_v3_generic_reader(self, root, key, label, value)
        return _reader
    setattr(SmartHomeTools, _name, _factory(_root, _key, _label))


SmartHomeTools.get_network_status = _smarthome_v3_get_network_status
SmartHomeTools.check_firmware_version = _smarthome_v3_check_firmware_version
SmartHomeTools.get_energy_consumption_report = _smarthome_v3_get_energy_consumption_report
SmartHomeTools.set_camera_privacy_mode = _smarthome_v3_set_camera_privacy_mode
SmartHomeTools.set_device_schedule = _smarthome_v3_set_device_schedule
SmartHomeTools.get_device_capabilities = _smarthome_v3_get_device_capabilities
SmartHomeTools.get_scene_members = _smarthome_v3_get_scene_members
SmartHomeTools.get_battery_health = _smarthome_v3_get_battery_health
SmartHomeTools.get_recent_automation_failures = _smarthome_v3_get_recent_automation_failures
