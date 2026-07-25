from typing import Any, Dict, List
import uuid
import datetime
from core.world.environment import WorldState, Email, CalendarEvent, Contact

class OfficeTools:
    """
    Office tools connected to the World Model.
    Handles Email, Calendar, and File operations.
    """

    def __init__(self):
        self.current_user = "user@example.com"

    def _normalize_recipients(self, value: str | list[str] | None) -> list[str]:
        recipients: list[str] = []
        if isinstance(value, str) and value:
            recipients.append(value)
        elif isinstance(value, list):
            recipients.extend([x for x in value if isinstance(x, str) and x])
        seen: set[str] = set()
        out: list[str] = []
        for item in recipients:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    def _find_email_by_id(self, email_id: str) -> Email | None:
        return next((e for e in WorldState.get_instance().office.emails if e.id == email_id), None)

    def list_emails(self, unread_only: bool = False) -> Dict[str, Any]:
        """
        List emails in the inbox.
        """
        all_emails = WorldState.get_instance().office.emails
        if unread_only:
            emails = [e for e in all_emails if not e.read]
        else:
            emails = list(all_emails)

        email_list = []
        for e in emails:
            email_list.append({
                "id": e.id,
                "sender": e.sender,
                "subject": e.subject,
                "timestamp": e.timestamp,
                "read": e.read
            })

        return {
            "status": "success",
            "emails": email_list,
            "count": len(email_list)
        }

    def read_email(
        self,
        email_id: str | None = None,
        sender: str | None = None,
        subject: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Read the content of a specific email.
        Side Effect: Marks the email as read in the World State.
        """
        if sender is None and isinstance(kwargs.get("from"), str):
            sender = kwargs.get("from")

        candidates = list(WorldState.get_instance().office.emails)

        if email_id:
            candidates = [e for e in candidates if e.id == email_id]
        if sender:
            candidates = [e for e in candidates if e.sender == sender]
        if subject:
            query = subject.lower()
            candidates = [e for e in candidates if query in e.subject.lower()]

        if not candidates:
            if email_id:
                return {"status": "error", "message": f"Email {email_id} not found."}
            return {"status": "error", "message": "No email matched the provided filters."}

        target_email = candidates[-1]

        target_email.read = True

        return {
            "status": "success",
            "id": target_email.id,
            "sender": target_email.sender,
            "subject": target_email.subject,
            "body": target_email.body,
            "timestamp": target_email.timestamp
        }

    def send_email(
        self,
        recipient: str | None = None,
        subject: str = "",
        body: str = "",
        to: str | list[str] | None = None,
    ) -> Dict[str, Any]:
        """
        Send an email.
        Side Effect: Adds an email to the 'network' (Inbox) for the recipient.
        """
        recipients: list[str] = []
        if recipient:
            recipients.append(recipient)
        if isinstance(to, str) and to:
            recipients.append(to)
        elif isinstance(to, list):
            recipients.extend([item for item in to if isinstance(item, str) and item])

        seen: set[str] = set()
        unique_recipients: list[str] = []
        for value in recipients:
            if value in seen:
                continue
            seen.add(value)
            unique_recipients.append(value)

        if not unique_recipients:
            return {"status": "error", "message": "Recipient is required."}

        message_ids: list[str] = []
        for target in unique_recipients:
            email_id = str(uuid.uuid4())
            new_email = Email(
                id=email_id,
                sender=self.current_user,
                recipient=target,
                subject=subject,
                body=body,
                read=False,
                timestamp=datetime.datetime.now().isoformat(),
            )
            WorldState.get_instance().office.add_email(new_email)
            message_ids.append(email_id)

        return {
            "status": "success",
            "message": f"Email sent to {', '.join(unique_recipients)}",
            "email_id": message_ids[0],
            "email_ids": message_ids,
        }

    def search_email(
        self,
        query: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Search emails by query."""
        if not query:
            query = kwargs.get("keyword")
        if not query:
            return {"status": "error", "message": "Missing required argument: query"}
        results = [
            {"id": e.id, "subject": e.subject, "sender": e.sender}
            for e in WorldState.get_instance().office.emails
            if query.lower() in e.subject.lower() or query.lower() in e.body.lower()
        ]
        return {"status": "success", "emails": results}

    def delete_email(self, email_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Delete an email."""
        if not email_id:
            email_id = kwargs.get("id")
        if not email_id:
            return {"status": "error", "message": "Missing required argument: email_id"}

        initial_count = len(WorldState.get_instance().office.emails)
        WorldState.get_instance().office.emails = [e for e in WorldState.get_instance().office.emails if e.id != email_id]
        if len(WorldState.get_instance().office.emails) < initial_count:
            return {"status": "success", "message": "Email deleted."}
        return {"status": "error", "message": "Email not found."}

    def get_calendar_events(self, date: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Get events for a specific date (YYYY-MM-DD)."""
        if not date:
            date = kwargs.get("day")
        if not date:
            return {"status": "error", "message": "Missing required argument: date"}

        events = [
            {"id": e.id, "title": e.title, "time": f"{e.start_time} - {e.end_time}"}
            for e in WorldState.get_instance().office.calendar_events
            if e.start_time.startswith(date)
        ]
        return {"status": "success", "events": events}

    def add_calendar_event(
        self,
        title: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        attendees: List[str] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Add a calendar event."""
        if not title:
            title = kwargs.get("subject")
        if not start_time:
            start_time = kwargs.get("start")
        if not end_time:
            end_time = kwargs.get("end")
        if attendees is None:
            raw_attendees = kwargs.get("participants")
            if isinstance(raw_attendees, list):
                attendees = [str(x) for x in raw_attendees]
            else:
                attendees = []
        if not title or not start_time or not end_time:
            return {
                "status": "error",
                "message": "Missing required fields: title, start_time, end_time.",
            }

        event_id = f"evt_{uuid.uuid4().hex[:8]}"
        event = CalendarEvent(
            id=event_id,
            title=title,
            start_time=start_time,
            end_time=end_time,
            attendees=attendees
        )
        WorldState.get_instance().office.calendar_events.append(event)
        return {"status": "success", "message": "Event added.", "event_id": event_id}

    def cancel_calendar_event(self, event_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Cancel a calendar event."""
        if not event_id:
            event_id = kwargs.get("id")
        if not event_id:
            return {"status": "error", "message": "Missing required argument: event_id"}

        initial_count = len(WorldState.get_instance().office.calendar_events)
        WorldState.get_instance().office.calendar_events = [e for e in WorldState.get_instance().office.calendar_events if e.id != event_id]
        if len(WorldState.get_instance().office.calendar_events) < initial_count:
            return {"status": "success", "message": "Event cancelled."}
        return {"status": "error", "message": "Event not found."}

    def search_contacts(
        self,
        name: str | None = None,
        query: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Search contacts by name."""
        if not name:
            name = query or kwargs.get("keyword")
        if not name:
            return {"status": "error", "message": "Missing required argument: name"}

        results = [
            {"name": c.name, "email": c.email, "phone": c.phone}
            for c in WorldState.get_instance().office.contacts
            if name.lower() in c.name.lower()
        ]
        return {"status": "success", "contacts": results}

    def create_contact(
        self,
        name: str | None = None,
        email: str | None = None,
        phone: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a new contact."""
        if not name:
            name = kwargs.get("contact_name")
        if not email:
            email = kwargs.get("contact_email")
        if not name or not email:
            return {"status": "error", "message": "Missing required fields: name, email."}

        contact = Contact(id=f"ct_{uuid.uuid4().hex[:8]}", name=name, email=email, phone=phone)
        WorldState.get_instance().office.contacts.append(contact)
        return {"status": "success", "message": f"Contact {name} created."}

    def mark_as_spam(self, email_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Train spam filter."""
        if not email_id:
            email_id = kwargs.get("id")
        if not email_id:
            return {"status": "error", "message": "Missing required argument: email_id"}
        return {"status": "success", "message": f"Email {email_id} marked as spam."}

    def reschedule_event(
        self,
        event_id: str | None = None,
        new_start_time: str | None = None,
        new_end_time: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Move meeting."""
        if not event_id:
            event_id = kwargs.get("id")
        if not new_start_time:
            new_start_time = kwargs.get("start_time")
        if not new_end_time:
            new_end_time = kwargs.get("end_time")
        if not event_id or not new_start_time or not new_end_time:
            return {
                "status": "error",
                "message": "Missing required fields: event_id, new_start_time, new_end_time.",
            }

        event = next((e for e in WorldState.get_instance().office.calendar_events if e.id == event_id), None)
        if not event:
            return {"status": "error", "message": "Event not found."}
        event.start_time = new_start_time
        event.end_time = new_end_time
        return {"status": "success", "message": "Event rescheduled."}

    def list_attendees(self, event_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """View participants."""
        if not event_id:
            event_id = kwargs.get("id")
        if not event_id:
            return {"status": "error", "message": "Missing required argument: event_id"}

        event = next((e for e in WorldState.get_instance().office.calendar_events if e.id == event_id), None)
        if not event:
            return {"status": "error", "message": "Event not found."}
        return {"status": "success", "attendees": event.attendees}

    def edit_contact(
        self,
        name: str | None = None,
        new_email: str | None = None,
        new_phone: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Update contact info."""
        if not name:
            name = kwargs.get("contact_name")
        if not name:
            return {"status": "error", "message": "Missing required argument: name"}
        if new_email is None:
            new_email = kwargs.get("email")
        if new_phone is None:
            new_phone = kwargs.get("phone")

        contact = next((c for c in WorldState.get_instance().office.contacts if c.name == name), None)
        if not contact:
            return {"status": "error", "message": "Contact not found."}
        if new_email:
            contact.email = new_email
        if new_phone:
            contact.phone = new_phone
        return {"status": "success", "message": f"Contact {name} updated."}

    def delete_contact(self, name: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Remove person."""
        if not name:
            name = kwargs.get("contact_name")
        if not name:
            return {"status": "error", "message": "Missing required argument: name"}

        initial_len = len(WorldState.get_instance().office.contacts)
        WorldState.get_instance().office.contacts = [c for c in WorldState.get_instance().office.contacts if c.name != name]
        if len(WorldState.get_instance().office.contacts) < initial_len:
            return {"status": "success", "message": f"Contact {name} deleted."}
        return {"status": "error", "message": "Contact not found."}

    def send_instant_message(
        self,
        user: str | None = None,
        msg: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Send chat."""
        if not user:
            user = kwargs.get("recipient")
        if not msg:
            msg = kwargs.get("message")
        if not user or not msg:
            return {"status": "error", "message": "Missing required fields: user, msg."}

        message_id = f"im_{uuid.uuid4().hex[:8]}"
        WorldState.get_instance().office.instant_messages.append({
            "id": message_id,
            "from": self.current_user,
            "to": user,
            "message": msg,
            "timestamp": datetime.datetime.now().isoformat()
        })
        return {
            "status": "success",
            "message": f"Message sent to {user}.",
            "message_id": message_id,
        }

    def reply_email(self, email_id: str | None = None, body: str = "", **kwargs: Any) -> Dict[str, Any]:
        """Reply to an existing email."""
        if not email_id:
            email_id = kwargs.get("id")
        if body == "" and isinstance(kwargs.get("message"), str):
            body = kwargs.get("message")
        if not email_id:
            return {"status": "error", "message": "Missing required field: email_id."}

        source = self._find_email_by_id(email_id)
        if not source:
            return {"status": "error", "message": "Email not found."}
        if source.recipient != self.current_user:
            return {"status": "error", "message": "Cannot reply to out-of-scope email."}

        new_id = str(uuid.uuid4())
        reply = Email(
            id=new_id,
            sender=self.current_user,
            recipient=source.sender,
            subject=f"Re: {source.subject}",
            body=body,
            read=False,
            timestamp=datetime.datetime.now().isoformat(),
        )
        WorldState.get_instance().office.add_email(reply)
        return {"status": "success", "email_id": new_id, "recipient": source.sender}

    def forward_email(
        self,
        email_id: str | None = None,
        to: str | list[str] | None = None,
        note: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Forward an email to one or more recipients."""
        if not email_id:
            email_id = kwargs.get("id")
        if to is None:
            to = kwargs.get("recipient")
        if note == "" and isinstance(kwargs.get("body"), str):
            note = kwargs.get("body")
        if not email_id:
            return {"status": "error", "message": "Missing required field: email_id."}
        recipients = self._normalize_recipients(to)
        if not recipients:
            return {"status": "error", "message": "Missing required field: to."}

        source = self._find_email_by_id(email_id)
        if not source:
            return {"status": "error", "message": "Email not found."}
        created_ids: list[str] = []
        for recipient in recipients:
            fwd_id = str(uuid.uuid4())
            fwd = Email(
                id=fwd_id,
                sender=self.current_user,
                recipient=recipient,
                subject=f"Fwd: {source.subject}",
                body=f"{note}\n\n---- Forwarded message ----\n{source.body}".strip(),
                read=False,
                timestamp=datetime.datetime.now().isoformat(),
            )
            WorldState.get_instance().office.add_email(fwd)
            created_ids.append(fwd_id)
        return {"status": "success", "email_ids": created_ids, "count": len(created_ids)}

    def archive_email(self, email_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Archive an email by id."""
        if not email_id:
            email_id = kwargs.get("id")
        if not email_id:
            return {"status": "error", "message": "Missing required field: email_id."}
        email = self._find_email_by_id(email_id)
        if not email:
            return {"status": "error", "message": "Email not found."}
        archives = WorldState.get_instance().context.setdefault("email_archives", {})
        archives[email_id] = {
            "email_id": email_id,
            "archived_by": self.current_user,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        return {"status": "success", "email_id": email_id}

    def restore_archived_email(self, email_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Restore archived email metadata record."""
        if not email_id:
            email_id = kwargs.get("id")
        if not email_id:
            return {"status": "error", "message": "Missing required field: email_id."}
        archives = WorldState.get_instance().context.setdefault("email_archives", {})
        if email_id not in archives:
            return {"status": "error", "message": "Archived email not found."}
        del archives[email_id]
        return {"status": "success", "email_id": email_id}

    def create_email_draft(
        self,
        to: str | list[str] | None = None,
        subject: str = "",
        body: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create an email draft."""
        if to is None:
            to = kwargs.get("recipient")
        recipients = self._normalize_recipients(to)
        if not recipients:
            return {"status": "error", "message": "Missing required field: to."}
        draft_id = f"dr_{uuid.uuid4().hex[:8]}"
        drafts = WorldState.get_instance().context.setdefault("email_drafts", {})
        drafts[draft_id] = {
            "draft_id": draft_id,
            "to": recipients,
            "subject": subject,
            "body": body,
            "created_at": datetime.datetime.now().isoformat(),
            "status_text": "draft",
        }
        return {"status": "success", "draft_id": draft_id}

    def send_draft_email(self, draft_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Send a previously created draft email."""
        if not draft_id:
            draft_id = kwargs.get("id")
        if not draft_id:
            return {"status": "error", "message": "Missing required field: draft_id."}
        drafts = WorldState.get_instance().context.setdefault("email_drafts", {})
        draft = drafts.get(draft_id)
        if not draft:
            return {"status": "error", "message": "Draft not found."}
        if draft.get("status_text") == "sent":
            return {"status": "error", "message": "Draft already sent."}

        result = self.send_email(
            subject=draft.get("subject", ""),
            body=draft.get("body", ""),
            to=draft.get("to", []),
        )
        if result.get("status") != "success":
            return result
        draft["status_text"] = "sent"
        draft["sent_at"] = datetime.datetime.now().isoformat()
        draft["email_ids"] = result.get("email_ids", [])
        return {"status": "success", "draft_id": draft_id, "email_ids": result.get("email_ids", [])}

    def add_email_label(
        self,
        email_id: str | None = None,
        label: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Add a label to an email."""
        if not email_id:
            email_id = kwargs.get("id")
        if not label:
            label = kwargs.get("tag")
        if not email_id or not label:
            return {"status": "error", "message": "Missing required fields: email_id, label."}
        if not self._find_email_by_id(email_id):
            return {"status": "error", "message": "Email not found."}
        labels = WorldState.get_instance().context.setdefault("email_labels", {})
        current = labels.setdefault(email_id, [])
        if label not in current:
            current.append(label)
        return {"status": "success", "email_id": email_id, "labels": list(current)}

    def remove_email_label(
        self,
        email_id: str | None = None,
        label: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Remove an email label."""
        if not email_id:
            email_id = kwargs.get("id")
        if not label:
            label = kwargs.get("tag")
        if not email_id or not label:
            return {"status": "error", "message": "Missing required fields: email_id, label."}
        labels = WorldState.get_instance().context.setdefault("email_labels", {})
        current = labels.get(email_id, [])
        if label not in current:
            return {"status": "error", "message": "Label not found on email."}
        labels[email_id] = [x for x in current if x != label]
        return {"status": "success", "email_id": email_id, "labels": labels[email_id]}

    def summarize_email_thread(
        self,
        thread_id: str | None = None,
        max_points: int = 5,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Summarize a thread-like subset of emails based on subject keyword."""
        if not thread_id:
            thread_id = kwargs.get("subject") or kwargs.get("query")
        if kwargs.get("limit") is not None:
            max_points = kwargs.get("limit")
        if not thread_id:
            return {"status": "error", "message": "Missing required field: thread_id."}
        try:
            max_points = int(max_points)
        except (TypeError, ValueError):
            max_points = 5
        max_points = max(1, min(max_points, 20))

        matches = [
            e
            for e in WorldState.get_instance().office.emails
            if thread_id.lower() in e.subject.lower()
        ]
        points: list[str] = []
        for email in matches[:max_points]:
            snippet = email.body.strip().replace("\n", " ")
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
            points.append(f"{email.sender}: {snippet}")
        summary = {
            "thread_id": thread_id,
            "points": points,
            "count": len(points),
            "created_at": datetime.datetime.now().isoformat(),
        }
        WorldState.get_instance().context.setdefault("thread_summaries", {})[thread_id] = summary
        return {"status": "success", **summary}

    def extract_action_items_from_email(
        self,
        email_id: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Extract simple action items from email body heuristically."""
        if not email_id:
            email_id = kwargs.get("id")
        if not email_id:
            return {"status": "error", "message": "Missing required field: email_id."}
        email = self._find_email_by_id(email_id)
        if not email:
            return {"status": "error", "message": "Email not found."}

        tasks = WorldState.get_instance().context.setdefault("tasks", {})
        extracted: list[dict[str, Any]] = []
        for line in email.body.splitlines():
            normalized = line.strip()
            if not normalized:
                continue
            lower = normalized.lower()
            if not any(k in lower for k in ("please", "todo", "action", "follow up", "need to")):
                continue
            task_id = f"tsk_{uuid.uuid4().hex[:8]}"
            payload = {
                "task_id": task_id,
                "title": normalized[:120],
                "status": "open",
                "assignee": self.current_user,
                "priority": "medium",
                "source_email_id": email_id,
                "created_at": datetime.datetime.now().isoformat(),
            }
            tasks[task_id] = payload
            extracted.append(payload)
        return {"status": "success", "items": extracted, "count": len(extracted)}

    def create_task(
        self,
        title: str | None = None,
        due_time: str | None = None,
        assignee: str | None = None,
        priority: str = "medium",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a task in context task store."""
        if not title:
            title = kwargs.get("name")
        if not due_time:
            due_time = kwargs.get("deadline")
        if not assignee:
            assignee = kwargs.get("owner") or self.current_user
        if not title:
            return {"status": "error", "message": "Missing required field: title."}

        task_id = f"tsk_{uuid.uuid4().hex[:8]}"
        task = {
            "task_id": task_id,
            "title": title,
            "due_time": due_time,
            "assignee": assignee,
            "priority": priority or "medium",
            "status": "open",
            "created_at": datetime.datetime.now().isoformat(),
        }
        WorldState.get_instance().context.setdefault("tasks", {})[task_id] = task
        return {"status": "success", **task}

    def update_task(
        self,
        task_id: str | None = None,
        title: str | None = None,
        due_time: str | None = None,
        assignee: str | None = None,
        priority: str | None = None,
        status: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Update task fields."""
        if not task_id:
            task_id = kwargs.get("id")
        if not task_id:
            return {"status": "error", "message": "Missing required field: task_id."}
        tasks = WorldState.get_instance().context.setdefault("tasks", {})
        task = tasks.get(task_id)
        if not task:
            return {"status": "error", "message": "Task not found."}
        if title is not None:
            task["title"] = title
        if due_time is not None:
            task["due_time"] = due_time
        if assignee is not None:
            task["assignee"] = assignee
        if priority is not None:
            task["priority"] = priority
        if status is not None:
            task["status"] = status
        task["updated_at"] = datetime.datetime.now().isoformat()
        return {"status": "success", **task}

    def complete_task(self, task_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Mark task as completed."""
        if not task_id:
            task_id = kwargs.get("id")
        if not task_id:
            return {"status": "error", "message": "Missing required field: task_id."}
        tasks = WorldState.get_instance().context.setdefault("tasks", {})
        task = tasks.get(task_id)
        if not task:
            return {"status": "error", "message": "Task not found."}
        task["status"] = "completed"
        task["completed_at"] = datetime.datetime.now().isoformat()
        return {"status": "success", **task}

    def list_tasks(
        self,
        status: str | None = None,
        assignee: str | None = None,
        priority: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """List tasks with optional filters."""
        if status is None:
            status = kwargs.get("state")
        tasks = list(WorldState.get_instance().context.setdefault("tasks", {}).values())
        if status:
            tasks = [t for t in tasks if str(t.get("status", "")).lower() == str(status).lower()]
        if assignee:
            tasks = [t for t in tasks if t.get("assignee") == assignee]
        if priority:
            tasks = [t for t in tasks if str(t.get("priority", "")).lower() == str(priority).lower()]
        return {"status": "success", "tasks": tasks, "count": len(tasks)}

    def share_calendar_event(
        self,
        event_id: str | None = None,
        recipients: str | list[str] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Share a calendar event to recipients."""
        if not event_id:
            event_id = kwargs.get("id")
        if recipients is None:
            recipients = kwargs.get("to")
        if not event_id:
            return {"status": "error", "message": "Missing required field: event_id."}
        recipient_list = self._normalize_recipients(recipients)
        if not recipient_list:
            return {"status": "error", "message": "Missing required field: recipients."}
        event = next((e for e in WorldState.get_instance().office.calendar_events if e.id == event_id), None)
        if not event:
            return {"status": "error", "message": "Event not found."}

        shared = WorldState.get_instance().context.setdefault("shared_events", [])
        shared.append(
            {
                "event_id": event_id,
                "title": event.title,
                "recipients": recipient_list,
                "shared_at": datetime.datetime.now().isoformat(),
            }
        )
        return {"status": "success", "event_id": event_id, "recipients": recipient_list}

    def set_out_of_office(
        self,
        start_time: str | None = None,
        end_time: str | None = None,
        message: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Set an out-of-office auto reply window."""
        if not start_time:
            start_time = kwargs.get("start")
        if not end_time:
            end_time = kwargs.get("end")
        if message is None:
            message = kwargs.get("body")
        if not start_time or not end_time or message is None:
            return {
                "status": "error",
                "message": "Missing required fields: start_time, end_time, message.",
            }
        payload = {
            "start_time": start_time,
            "end_time": end_time,
            "message": message,
            "updated_at": datetime.datetime.now().isoformat(),
        }
        WorldState.get_instance().context["auto_reply"] = payload
        return {"status": "success", **payload}

    def create_meeting_minutes(
        self,
        event_id: str | None = None,
        notes: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create meeting minutes linked to a calendar event."""
        if not event_id:
            event_id = kwargs.get("id")
        if notes is None:
            notes = kwargs.get("content")
        if not event_id or notes is None:
            return {"status": "error", "message": "Missing required fields: event_id, notes."}
        event = next((e for e in WorldState.get_instance().office.calendar_events if e.id == event_id), None)
        if not event:
            return {"status": "error", "message": "Event not found."}
        minutes_id = f"mm_{uuid.uuid4().hex[:8]}"
        entry = {
            "minutes_id": minutes_id,
            "event_id": event_id,
            "title": event.title,
            "notes": notes,
            "created_at": datetime.datetime.now().isoformat(),
        }
        WorldState.get_instance().context.setdefault("meeting_minutes", {})[minutes_id] = entry
        return {"status": "success", **entry}

    def convert_message_to_email(
        self,
        message_id: str | None = None,
        to: str | list[str] | None = None,
        subject: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Convert an instant message into one or more emails."""
        if not message_id:
            message_id = kwargs.get("id")
        if to is None:
            to = kwargs.get("recipient")
        if subject is None:
            subject = kwargs.get("title")
        if not message_id:
            return {"status": "error", "message": "Missing required field: message_id."}
        recipients = self._normalize_recipients(to)
        if not recipients:
            return {"status": "error", "message": "Missing required field: to."}
        msg = next((m for m in WorldState.get_instance().office.instant_messages if m.get("id") == message_id), None)
        if not msg:
            return {"status": "error", "message": "Message not found."}

        created_ids: list[str] = []
        for recipient in recipients:
            email_id = str(uuid.uuid4())
            email = Email(
                id=email_id,
                sender=self.current_user,
                recipient=recipient,
                subject=subject or f"Message from {msg.get('from', 'unknown')}",
                body=msg.get("message", ""),
                read=False,
                timestamp=datetime.datetime.now().isoformat(),
            )
            WorldState.get_instance().office.add_email(email)
            created_ids.append(email_id)
        return {"status": "success", "email_ids": created_ids, "count": len(created_ids)}


from tools.connected._v3_roots import (
    as_bool,
    ensure_list_root,
    find_many,
    find_one,
    list_by_arg,
    lookup_by_arg,
    make_id,
    match_query,
    set_root,
)


def _office_v3_list_emails(self, thread_id: str | None = None, unread_only: bool = False, **kwargs: Any) -> Dict[str, Any]:
    if thread_id is None:
        thread_id = kwargs.get("subject")
    emails = list(ensure_list_root(self, "emails"))
    out = []
    for item in emails:
        if not isinstance(item, dict):
            continue
        if thread_id and item.get("thread_id") != thread_id:
            continue
        if as_bool(unread_only) and as_bool(item.get("read")):
            continue
        out.append(
            {
                "id": item.get("id"),
                "sender": item.get("sender"),
                "recipient": item.get("recipient"),
                "subject": item.get("subject"),
                "thread_id": item.get("thread_id"),
                "timestamp": item.get("timestamp"),
                "read": as_bool(item.get("read")),
            }
        )
    return {"status": "success", "emails": out, "count": len(out)}


def _office_v3_search_email(self, query: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    query = query or kwargs.get("keyword")
    if not query:
        return {"status": "error", "message": "Missing required argument: query"}
    emails = list(ensure_list_root(self, "emails"))
    results = []
    for item in emails:
        if not isinstance(item, dict):
            continue
        if not match_query(item, query, ["subject", "body", "sender", "recipient"]):
            continue
        results.append(
            {
                "id": item.get("id"),
                "subject": item.get("subject"),
                "sender": item.get("sender"),
                "recipient": item.get("recipient"),
                "thread_id": item.get("thread_id"),
            }
        )
    return {"status": "success", "emails": results, "count": len(results)}


def _office_v3_read_email(self, email_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    email_id = email_id or kwargs.get("id")
    if not email_id:
        return {"status": "error", "message": "Missing required argument: email_id"}
    emails = list(ensure_list_root(self, "emails"))
    item = find_one(emails, id=email_id)
    if not item:
        return {"status": "error", "message": f"Email {email_id} not found."}
    item["read"] = True
    set_root(self, "emails", emails)
    return {
        "status": "success",
        "id": item.get("id"),
        "sender": item.get("sender"),
        "recipient": item.get("recipient"),
        "subject": item.get("subject"),
        "body": item.get("body", ""),
        "thread_id": item.get("thread_id"),
        "timestamp": item.get("timestamp"),
    }


def _office_v3_list_email_attachments(self, email_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    email_id = email_id or kwargs.get("id")
    if not email_id:
        return {"status": "error", "message": "Missing required argument: email_id"}
    records = find_many(ensure_list_root(self, "email_attachments"), email_id=email_id)
    return {"status": "success", "attachments": records, "count": len(records)}


def _office_v3_read_attachment_text(self, attachment_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    attachment_id = attachment_id or kwargs.get("id")
    if not attachment_id:
        return {"status": "error", "message": "Missing required argument: attachment_id"}
    item = find_one(ensure_list_root(self, "email_attachments"), attachment_id=attachment_id)
    if not item:
        return {"status": "error", "message": f"Attachment {attachment_id} not found."}
    return {
        "status": "success",
        "attachment_id": item.get("attachment_id"),
        "email_id": item.get("email_id"),
        "title": item.get("title"),
        "content_type": item.get("content_type"),
        "body_text": item.get("body_text", ""),
    }


def _office_v3_summarize_email_thread(self, thread_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    thread_id = thread_id or kwargs.get("subject") or kwargs.get("query")
    if not thread_id:
        return {"status": "error", "message": "Missing required argument: thread_id"}
    emails = [
        item
        for item in ensure_list_root(self, "emails")
        if isinstance(item, dict) and item.get("thread_id") == thread_id
    ]
    if not emails:
        return {"status": "error", "message": f"Thread {thread_id} not found."}
    lines = []
    for item in emails[:8]:
        body = str(item.get("body", "")).strip().replace("\n", " ")
        if len(body) > 120:
            body = body[:117] + "..."
        lines.append(f"{item.get('sender', 'unknown')}: {body}")
    return {"status": "success", "thread_id": thread_id, "summary": "\n".join(lines), "count": len(emails)}


def _office_v3_read_task_comment(self, task_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    task_id = task_id or kwargs.get("id")
    if not task_id:
        return {"status": "error", "message": "Missing required argument: task_id"}
    records = find_many(ensure_list_root(self, "office_task_comments"), task_id=task_id)
    if not records:
        return {"status": "error", "message": f"No comments found for task {task_id}."}
    return {"status": "success", "task_id": task_id, "comments": records, "count": len(records)}


def _office_v3_read_event_note(self, event_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    event_id = event_id or kwargs.get("id")
    if not event_id:
        return {"status": "error", "message": "Missing required argument: event_id"}
    records = find_many(ensure_list_root(self, "office_event_notes"), event_id=event_id)
    if not records:
        return {"status": "error", "message": f"No event notes found for {event_id}."}
    return {"status": "success", "event_id": event_id, "notes": records, "count": len(records)}


def _office_v3_read_contact_remark(self, contact_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    contact_id = contact_id or kwargs.get("id")
    if not contact_id:
        return {"status": "error", "message": "Missing required argument: contact_id"}
    records = find_many(ensure_list_root(self, "office_contact_remarks"), contact_id=contact_id)
    if not records:
        return {"status": "error", "message": f"No contact remarks found for {contact_id}."}
    return {"status": "success", "contact_id": contact_id, "remarks": records, "count": len(records)}


def _office_v3_search_contacts(self, query: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    query = query or kwargs.get("name") or kwargs.get("keyword")
    if not query:
        return {"status": "error", "message": "Missing required argument: query"}
    results = [
        item
        for item in ensure_list_root(self, "office_contacts")
        if isinstance(item, dict) and match_query(item, query, ["contact_id", "name", "email", "team"])
    ]
    return {"status": "success", "contacts": results, "count": len(results)}


def _office_v3_get_contact_details(self, contact_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    contact_id = contact_id or kwargs.get("id")
    if not contact_id:
        return {"status": "error", "message": "Missing required argument: contact_id"}
    item = find_one(ensure_list_root(self, "office_contacts"), contact_id=contact_id)
    if not item:
        return {"status": "error", "message": f"Contact {contact_id} not found."}
    return {"status": "success", **item}


def _office_v3_get_calendar_events(self, date: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    date = date or kwargs.get("day")
    if not date:
        return {"status": "error", "message": "Missing required argument: date"}
    events = [
        item
        for item in ensure_list_root(self, "office_calendar_events")
        if isinstance(item, dict) and str(item.get("date", "")).startswith(str(date))
    ]
    return {"status": "success", "events": events, "count": len(events)}


def _office_v3_list_attendees(self, event_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    event_id = event_id or kwargs.get("id")
    if not event_id:
        return {"status": "error", "message": "Missing required argument: event_id"}
    event = find_one(ensure_list_root(self, "office_calendar_events"), event_id=event_id)
    if not event:
        return {"status": "error", "message": "Event not found."}
    return {"status": "success", "event_id": event_id, "attendees": event.get("attendees", [])}


def _office_v3_get_calendar_event_details(self, event_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    event_id = event_id or kwargs.get("id")
    if not event_id:
        return {"status": "error", "message": "Missing required argument: event_id"}
    event = find_one(ensure_list_root(self, "office_calendar_events"), event_id=event_id)
    if not event:
        return {"status": "error", "message": "Event not found."}
    return {"status": "success", **event}


def _office_v3_list_tasks(self, status: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    status = status or kwargs.get("state")
    tasks = list(ensure_list_root(self, "office_tasks"))
    if status:
        tasks = [item for item in tasks if isinstance(item, dict) and str(item.get("status", "")).lower() == str(status).lower()]
    return {"status": "success", "tasks": tasks, "count": len(tasks)}


def _office_v3_get_task_details(self, task_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    task_id = task_id or kwargs.get("id")
    if not task_id:
        return {"status": "error", "message": "Missing required argument: task_id"}
    item = find_one(ensure_list_root(self, "office_tasks"), task_id=task_id)
    if not item:
        return {"status": "error", "message": "Task not found."}
    return {"status": "success", **item}


def _office_v3_send_email(self, recipient: str | None = None, subject: str = "", body: str = "", to: str | list[str] | None = None, **kwargs: Any) -> Dict[str, Any]:
    recipients = []
    if recipient:
        recipients.append(recipient)
    if isinstance(to, str) and to:
        recipients.append(to)
    elif isinstance(to, list):
        recipients.extend([item for item in to if isinstance(item, str) and item])
    if not recipients:
        return {"status": "error", "message": "Recipient is required."}
    emails = ensure_list_root(self, "emails")
    threads = ensure_list_root(self, "email_threads")
    thread_id = make_id("thr")
    threads.append({"thread_id": thread_id, "subject": subject, "participants": [self.current_user, *recipients]})
    created_ids = []
    for target in recipients:
        email_id = make_id("em")
        emails.append(
            {
                "id": email_id,
                "thread_id": thread_id,
                "sender": self.current_user,
                "recipient": target,
                "subject": subject,
                "body": body,
                "read": False,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )
        created_ids.append(email_id)
    return {"status": "success", "thread_id": thread_id, "email_id": created_ids[0], "email_ids": created_ids}


def _office_v3_reply_email(self, email_id: str | None = None, body: str = "", **kwargs: Any) -> Dict[str, Any]:
    email_id = email_id or kwargs.get("id")
    if body == "" and isinstance(kwargs.get("message"), str):
        body = kwargs.get("message")
    if not email_id:
        return {"status": "error", "message": "Missing required field: email_id."}
    emails = ensure_list_root(self, "emails")
    source = find_one(emails, id=email_id)
    if not source:
        return {"status": "error", "message": "Email not found."}
    new_id = make_id("em")
    reply = {
        "id": new_id,
        "thread_id": source.get("thread_id") or make_id("thr"),
        "sender": self.current_user,
        "recipient": source.get("sender"),
        "subject": f"Re: {source.get('subject', '')}",
        "body": body,
        "read": False,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    emails.append(reply)
    return {"status": "success", "email_id": new_id, "thread_id": reply["thread_id"]}


def _office_v3_update_task(self, task_id: str | None = None, assignee: str | None = None, status: str | None = None, note: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    task_id = task_id or kwargs.get("id")
    if not task_id:
        return {"status": "error", "message": "Missing required field: task_id."}
    tasks = ensure_list_root(self, "office_tasks")
    task = find_one(tasks, task_id=task_id)
    if not task:
        return {"status": "error", "message": "Task not found."}
    if assignee is not None:
        task["assignee"] = assignee
    if status is not None:
        task["status"] = status
    if note:
        comments = ensure_list_root(self, "office_task_comments")
        comments.append(
            {
                "comment_id": make_id("tc"),
                "task_id": task_id,
                "body_text": note,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )
    task["updated_at"] = datetime.datetime.now().isoformat()
    return {"status": "success", **task}


def _office_v3_reschedule_event(self, event_id: str | None = None, start_time: str | None = None, end_time: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    event_id = event_id or kwargs.get("id")
    start_time = start_time or kwargs.get("new_start_time")
    end_time = end_time or kwargs.get("new_end_time")
    if not event_id or not start_time or not end_time:
        return {"status": "error", "message": "Missing required fields: event_id, start_time, end_time."}
    events = ensure_list_root(self, "office_calendar_events")
    event = find_one(events, event_id=event_id)
    if not event:
        return {"status": "error", "message": "Event not found."}
    event["start_time"] = start_time
    event["end_time"] = end_time
    event["updated_at"] = datetime.datetime.now().isoformat()
    return {"status": "success", **event}


OfficeTools.list_emails = _office_v3_list_emails
OfficeTools.search_email = _office_v3_search_email
OfficeTools.read_email = _office_v3_read_email
OfficeTools.list_email_attachments = _office_v3_list_email_attachments
OfficeTools.read_attachment_text = _office_v3_read_attachment_text
OfficeTools.summarize_email_thread = _office_v3_summarize_email_thread
OfficeTools.read_task_comment = _office_v3_read_task_comment
OfficeTools.read_event_note = _office_v3_read_event_note
OfficeTools.read_contact_remark = _office_v3_read_contact_remark
OfficeTools.search_contacts = _office_v3_search_contacts
OfficeTools.get_contact_details = _office_v3_get_contact_details
OfficeTools.get_calendar_events = _office_v3_get_calendar_events
OfficeTools.list_attendees = _office_v3_list_attendees
OfficeTools.get_calendar_event_details = _office_v3_get_calendar_event_details
OfficeTools.list_tasks = _office_v3_list_tasks
OfficeTools.get_task_details = _office_v3_get_task_details
OfficeTools.send_email = _office_v3_send_email
OfficeTools.reply_email = _office_v3_reply_email
OfficeTools.update_task = _office_v3_update_task
OfficeTools.reschedule_event = _office_v3_reschedule_event


def _office_v3_forward_email(self, email_id: str | None = None, to: str | list[str] | None = None, note: str = "", **kwargs: Any) -> Dict[str, Any]:
    email_id = email_id or kwargs.get("id")
    if not email_id:
        return {"status": "error", "message": "Missing required field: email_id."}
    source = find_one(ensure_list_root(self, "emails"), id=email_id)
    if not source:
        return {"status": "error", "message": "Email not found."}
    recipients = self._normalize_recipients(to or kwargs.get("recipient"))
    if not recipients:
        return {"status": "error", "message": "Missing required field: to."}
    created_ids = []
    for recipient in recipients:
        created = _office_v3_send_email(
            self,
            recipient=recipient,
            subject=f"Fwd: {source.get('subject', '')}",
            body=f"{note}\n\n---- Forwarded message ----\n{source.get('body', '')}".strip(),
        )
        if created.get("status") == "success":
            created_ids.extend(created.get("email_ids", []))
    return {"status": "success", "email_ids": created_ids, "count": len(created_ids)}


def _office_v3_archive_email(self, email_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    email_id = email_id or kwargs.get("id")
    if not email_id:
        return {"status": "error", "message": "Missing required field: email_id."}
    email = find_one(ensure_list_root(self, "emails"), id=email_id)
    if not email:
        return {"status": "error", "message": "Email not found."}
    labels = ensure_list_root(self, "office_email_labels")
    labels.append({"email_id": email_id, "label": "archived", "created_at": datetime.datetime.now().isoformat()})
    return {"status": "success", "email_id": email_id}


def _office_v3_add_email_label(self, email_id: str | None = None, label: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    email_id = email_id or kwargs.get("id")
    label = label or kwargs.get("tag")
    if not email_id or not label:
        return {"status": "error", "message": "Missing required fields: email_id, label."}
    if not find_one(ensure_list_root(self, "emails"), id=email_id):
        return {"status": "error", "message": "Email not found."}
    labels = ensure_list_root(self, "office_email_labels")
    labels.append({"email_id": email_id, "label": label, "created_at": datetime.datetime.now().isoformat()})
    return {"status": "success", "email_id": email_id, "label": label}


def _office_v3_complete_task(self, task_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    return _office_v3_update_task(self, task_id=task_id or kwargs.get("id"), status="completed")


def _office_v3_send_instant_message(self, user: str | None = None, msg: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    user = user or kwargs.get("recipient")
    msg = msg or kwargs.get("message")
    if not user or not msg:
        return {"status": "error", "message": "Missing required fields: user, msg."}
    threads = ensure_list_root(self, "office_message_threads")
    thread = {"message_thread_id": make_id("msg"), "participants": [self.current_user, user], "body_text": msg, "created_at": datetime.datetime.now().isoformat()}
    threads.append(thread)
    return {"status": "success", **thread}


def _office_v3_get_email_thread_state(self, thread_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    return lookup_by_arg(self, root="email_threads", arg_value=thread_id or kwargs.get("id"), id_field="thread_id", label="Email thread")


def _office_v3_get_email_metadata(self, email_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    email_id = email_id or kwargs.get("id")
    item = find_one(ensure_list_root(self, "emails"), id=email_id) if email_id else None
    if not item:
        return {"status": "error", "message": "Email not found."}
    return {"status": "success", "id": item.get("id"), "thread_id": item.get("thread_id"), "sender": item.get("sender"), "recipient": item.get("recipient"), "subject": item.get("subject"), "timestamp": item.get("timestamp"), "read": as_bool(item.get("read"))}


def _office_v3_get_attachment_manifest(self, email_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    return list_by_arg(self, root="email_attachments", arg_value=email_id or kwargs.get("id"), id_field="email_id", container_key="attachments", label="Attachment manifest")


def _office_v3_get_recent_interactions(self, contact_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    contact_id = contact_id or kwargs.get("id")
    if not contact_id:
        return {"status": "error", "message": "Missing required field: contact_id."}
    items = find_many(ensure_list_root(self, "office_recent_interactions"), contact_id=contact_id)
    if not items:
        return {"status": "error", "message": "Recent interactions not found."}
    return {"status": "success", "contact_id": contact_id, "interactions": items, "count": len(items)}


def _office_v3_extract_action_items_from_email(self, email_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    email_id = email_id or kwargs.get("id")
    email = find_one(ensure_list_root(self, "emails"), id=email_id) if email_id else None
    if not email:
        return {"status": "error", "message": "Email not found."}
    body = str(email.get("body", ""))
    items = []
    for line in body.splitlines():
        text = line.strip()
        if not text:
            continue
        lower = text.lower()
        if any(k in lower for k in ("please", "todo", "action", "follow up", "need to")):
            items.append(text)
    return {"status": "success", "email_id": email_id, "items": items, "count": len(items)}


def _office_v3_generic_reader(self, root: str, arg_name: str, key_name: str, label: str, value: Any) -> Dict[str, Any]:
    return lookup_by_arg(self, root=root, arg_value=value, id_field=key_name, label=label)


_OFFICE_GENERIC_READERS = {
    "read_email_thread": ("email_threads", "thread_id", "Email thread"),
    "read_message_thread": ("office_message_threads", "message_thread_id", "Message thread"),
    "read_attendee_followup_thread": ("office_attendee_followup_threads", "event_id", "Attendee follow-up thread"),
    "read_policy_note": ("office_policy_notes", "policy_note_id", "Policy note"),
    "read_approval_guideline": ("office_approval_guidelines", "guideline_id", "Approval guideline"),
    "read_vendor_handling_rule": ("office_vendor_handling_rules", "rule_id", "Vendor handling rule"),
    "read_change_log_note": ("office_change_log_notes", "note_id", "Change log note"),
    "read_prior_review_note": ("office_prior_review_notes", "note_id", "Prior review note"),
    "read_event_history_note": ("office_event_history_notes", "event_id", "Event history note"),
    "read_board_packet_text": ("office_board_packet_texts", "packet_id", "Board packet"),
    "read_contract_attachment_text": ("office_contract_attachment_texts", "attachment_id", "Contract attachment"),
    "read_internal_request_ticket": ("office_internal_request_tickets", "ticket_id", "Internal request ticket"),
    "read_ops_escalation_case": ("office_ops_escalation_cases", "case_id", "Ops escalation case"),
}


for _name, (_root, _key, _label) in _OFFICE_GENERIC_READERS.items():
    def _factory(root: str, key: str, label: str):
        def _reader(self, **kwargs: Any) -> Dict[str, Any]:
            value = kwargs.get(key) or kwargs.get("id") or kwargs.get("thread_id") or kwargs.get("event_id")
            return _office_v3_generic_reader(self, root, key, key, label, value)
        return _reader
    setattr(OfficeTools, _name, _factory(_root, _key, _label))


OfficeTools.forward_email = _office_v3_forward_email
OfficeTools.archive_email = _office_v3_archive_email
OfficeTools.add_email_label = _office_v3_add_email_label
OfficeTools.complete_task = _office_v3_complete_task
OfficeTools.send_instant_message = _office_v3_send_instant_message
OfficeTools.get_email_thread_state = _office_v3_get_email_thread_state
OfficeTools.get_email_metadata = _office_v3_get_email_metadata
OfficeTools.get_attachment_manifest = _office_v3_get_attachment_manifest
OfficeTools.get_recent_interactions = _office_v3_get_recent_interactions
OfficeTools.extract_action_items_from_email = _office_v3_extract_action_items_from_email
