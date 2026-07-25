from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import datetime
import uuid
import copy

@dataclass
class FileSystem:
    """Deterministic in-memory file-system state."""
    files: Dict[str, str] = field(default_factory=dict)
    folders: List[str] = field(default_factory=list)

    def read_file(self, path: str) -> Optional[str]:
        return self.files.get(path)

    def write_file(self, path: str, content: str) -> None:
        self.files[path] = content

    def list_files(self) -> List[str]:
        return list(self.files.keys())

    def delete_file(self, path: str) -> bool:
        if path in self.files:
            del self.files[path]
            return True
        return False

    def create_folder(self, path: str) -> None:
        if path not in self.folders:
            self.folders.append(path)

@dataclass
class Email:
    id: str
    sender: str
    recipient: str
    subject: str
    body: str
    read: bool = False
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

@dataclass
class CalendarEvent:
    id: str
    title: str
    start_time: str
    end_time: str
    attendees: List[str]
    description: str = ""

@dataclass
class Contact:
    id: str
    name: str
    email: str
    phone: str = ""

@dataclass
class OfficeSystem:
    """Mock Office System State"""
    emails: List[Email] = field(default_factory=list)
    calendar_events: List[CalendarEvent] = field(default_factory=list)
    contacts: List[Contact] = field(default_factory=list)
    instant_messages: List[Dict[str, Any]] = field(default_factory=list)

    def add_email(self, email: Email) -> None:
        self.emails.append(email)

    def get_emails(self, recipient: str, unread_only: bool = False) -> List[Email]:
        return [
            e for e in self.emails
            if e.recipient == recipient and (not unread_only or not e.read)
        ]

@dataclass
class BankAccount:
    account_id: str
    owner: str
    balance: float
    currency: str = "USD"
    is_frozen: bool = False
    recipients: List[str] = field(default_factory=list)
    transactions: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class StockStart:
    symbol: str
    current_price: float
    name: str

@dataclass
class InvestmentPortfolio:
    owner: str
    stocks: Dict[str, int] = field(default_factory=dict)
    funds: Dict[str, float] = field(default_factory=dict)

@dataclass
class BankSystem:
    """Mock Banking System State"""
    accounts: Dict[str, BankAccount] = field(default_factory=dict)
    market_prices: Dict[str, float] = field(default_factory=dict)
    portfolios: Dict[str, InvestmentPortfolio] = field(default_factory=dict)

    def get_account(self, account_id: str) -> Optional[BankAccount]:
        return self.accounts.get(account_id)

    def transfer(self, from_id: str, to_id: str, amount: float) -> bool:
        sender = self.get_account(from_id)
        receiver = self.get_account(to_id)

        if not sender or not receiver:
            return False
        if sender.is_frozen:
            return False

        if sender.balance >= amount:
            sender.balance -= amount
            receiver.balance += amount

            timestamp = datetime.datetime.now().isoformat()
            sender.transactions.append({
                "type": "transfer_out",
                "to": to_id,
                "amount": amount,
                "timestamp": timestamp
            })
            receiver.transactions.append({
                "type": "transfer_in",
                "from": from_id,
                "amount": amount,
                "timestamp": timestamp
            })
            return True
        return False

@dataclass
class SmartHomeSystem:
    """Mock IoT System State"""
    devices: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def update_state(self, device_id: str, new_state: Dict[str, Any]) -> None:
        if device_id not in self.devices:
            self.devices[device_id] = {}
        self.devices[device_id].update(new_state)

    def get_state(self, device_id: str) -> Optional[Dict[str, Any]]:
        return self.devices.get(device_id)

@dataclass
class Flight:
    flight_number: str
    origin: str
    destination: str
    departure_time: str
    arrival_time: str
    status: str = "Available"
    price: float = 0.0

@dataclass
class Booking:
    id: str
    type: str
    user: str
    details: Dict[str, Any]
    status: str = "Confirmed"

@dataclass
class TravelSystem:
    flights: List[Flight] = field(default_factory=list)
    hotels: List[Dict[str, Any]] = field(default_factory=list)
    bookings: List[Booking] = field(default_factory=list)

@dataclass
class Product:
    id: str
    name: str
    price: float
    category: str
    stock: int
    description: str = ""

@dataclass
class Order:
    id: str
    user: str
    items: List[Dict[str, Any]]
    total_price: float
    status: str = "Pending"

@dataclass
class ShoppingSystem:
    products: Dict[str, Product] = field(default_factory=dict)
    carts: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    orders: List[Order] = field(default_factory=list)

@dataclass
class Appointment:
    id: str
    patient: str
    doctor: str
    time: str
    department: str
    status: str = "Scheduled"

@dataclass
class MedicalRecord:
    patient: str
    history: str
    prescriptions: List[str]

@dataclass
class MedicalSystem:
    appointments: List[Appointment] = field(default_factory=list)
    records: Dict[str, MedicalRecord] = field(default_factory=dict)


class WorldState:
    """
    Thread-local world state for the AgentDET benchmark (all seven domains).
    Each thread (parallel worker) gets its own isolated instance.
    """
    _local = __import__("threading").local()

    def __init__(self):
        self.fs = FileSystem()
        self.office = OfficeSystem()
        self.bank = BankSystem()
        self.iot = SmartHomeSystem()
        self.travel = TravelSystem()
        self.shopping = ShoppingSystem()
        self.medical = MedicalSystem()

        self.context = {
            "time": "2024-05-20T14:00:00",
            "location": "New York, USA",
            "clipboard": ""
        }
        self.extra_state: Dict[str, Any] = {}

    @classmethod
    def get_instance(cls):
        if not hasattr(cls._local, "instance") or cls._local.instance is None:
            cls._local.instance = cls()
        return cls._local.instance

    @classmethod
    def reset(cls):
        """Reset the world state to empty (per-thread)"""
        cls._local.instance = cls()

    def load_from_dict(self, snapshot: Dict[str, Any]):
        """
        Load state from a scenario definition (v2 Standard).
        """
        known_keys = {
            "files",
            "folders",
            "emails",
            "calendar_events",
            "contacts",
            "instant_messages",
            "bank_accounts",
            "market_prices",
            "portfolios",
            "iot_devices",
            "flights",
            "hotels",
            "bookings",
            "products",
            "carts",
            "orders",
            "appointments",
            "medical_records",
            "context",
        }
        self.extra_state = {
            key: copy.deepcopy(value)
            for key, value in snapshot.items()
            if key not in known_keys
        }

        if "files" in snapshot:
            self.fs.files = copy.deepcopy(snapshot["files"])
        if "folders" in snapshot:
            self.fs.folders = list(snapshot["folders"])

        if "emails" in snapshot:
            self.office.emails = []
            for e_data in snapshot["emails"]:
                clean_data = copy.deepcopy(e_data)
                if "id" not in clean_data:
                    clean_data["id"] = f"mail_{uuid.uuid4().hex[:8]}"
                if "sender" not in clean_data: clean_data["sender"] = "sender@example.com"
                if "recipient" not in clean_data: clean_data["recipient"] = "recipient@example.com"
                if "subject" not in clean_data: clean_data["subject"] = "(No Subject)"
                if "body" not in clean_data: clean_data["body"] = ""
                valid = {"id", "sender", "recipient", "subject", "body", "read", "timestamp"}
                clean_data = {k: v for k, v in clean_data.items() if k in valid}
                self.office.add_email(Email(**clean_data))
        if "calendar_events" in snapshot:
            self.office.calendar_events = []
            for item in snapshot["calendar_events"]:
                clean_data = copy.deepcopy(item)
                if "id" not in clean_data:
                    clean_data["id"] = f"event_{uuid.uuid4().hex[:8]}"
                if "title" not in clean_data: clean_data["title"] = "Untitled Event"
                if "start_time" not in clean_data: clean_data["start_time"] = "2024-01-01T09:00:00"
                if "end_time" not in clean_data: clean_data["end_time"] = "2024-01-01T10:00:00"
                if "attendees" not in clean_data: clean_data["attendees"] = []
                valid = {"id", "title", "start_time", "end_time", "attendees", "description"}
                clean_data = {k: v for k, v in clean_data.items() if k in valid}
                self.office.calendar_events.append(CalendarEvent(**clean_data))
        if "contacts" in snapshot:
            self.office.contacts = []
            for item in snapshot["contacts"]:
                clean_data = copy.deepcopy(item)
                if "id" not in clean_data:
                    clean_data["id"] = clean_data.get("email", clean_data.get("name", f"contact_{uuid.uuid4().hex[:8]}"))
                if "name" not in clean_data: clean_data["name"] = "Unknown"
                if "email" not in clean_data: clean_data["email"] = "unknown@example.com"
                valid = {"id", "name", "email", "phone"}
                clean_data = {k: v for k, v in clean_data.items() if k in valid}
                self.office.contacts.append(Contact(**clean_data))
        if "instant_messages" in snapshot:
            self.office.instant_messages = copy.deepcopy(snapshot["instant_messages"])

        if "bank_accounts" in snapshot:
            self.bank.accounts = {}
            for acc_data in snapshot["bank_accounts"]:
                account_id = acc_data.get("id", acc_data.get("account_id", ""))
                if not account_id:
                    account_id = f"acc_{uuid.uuid4().hex[:8]}"
                account = BankAccount(
                    account_id=account_id,
                    owner=acc_data.get("owner", "Unknown"),
                    balance=acc_data.get("balance", 0.0),
                    currency=acc_data.get("currency", "USD"),
                    recipients=copy.deepcopy(acc_data.get("recipients", [])),
                    is_frozen=acc_data.get("is_frozen", False),
                    transactions=copy.deepcopy(acc_data.get("transactions", [])),
                )
                self.bank.accounts[account.account_id] = account
        if "market_prices" in snapshot:
            self.bank.market_prices = copy.deepcopy(snapshot["market_prices"])
        if "portfolios" in snapshot:
            self.bank.portfolios = {}
            raw_portfolios = snapshot["portfolios"]
            if isinstance(raw_portfolios, dict):
                iterator = raw_portfolios.items()
            else:
                iterator = []
                for item in raw_portfolios:
                    owner = item.get("owner")
                    if owner:
                        iterator.append((owner, item))
            for owner, p_data in iterator:
                self.bank.portfolios[owner] = InvestmentPortfolio(
                    owner=owner,
                    stocks=dict(p_data.get("stocks", {})),
                    funds=dict(p_data.get("funds", {})),
                )

        if "iot_devices" in snapshot:
            self.iot.devices = copy.deepcopy(snapshot["iot_devices"])

        if "flights" in snapshot:
            self.travel.flights = []
            for item in snapshot["flights"]:
                flight_data = copy.deepcopy(item)
                if 'flight_id' in flight_data and 'flight_number' not in flight_data:
                    flight_data['flight_number'] = flight_data.pop('flight_id')
                if 'from' in flight_data and 'origin' not in flight_data:
                    flight_data['origin'] = flight_data.pop('from')
                if 'to' in flight_data and 'destination' not in flight_data:
                    flight_data['destination'] = flight_data.pop('to')
                if 'departure' in flight_data and 'departure_time' not in flight_data:
                    flight_data['departure_time'] = flight_data.pop('departure')
                if 'arrival' in flight_data and 'arrival_time' not in flight_data:
                    flight_data['arrival_time'] = flight_data.pop('arrival')

                if "flight_number" not in flight_data: flight_data["flight_number"] = f"FL_{uuid.uuid4().hex[:6].upper()}"
                if "origin" not in flight_data: flight_data["origin"] = "Unknown"
                if "destination" not in flight_data: flight_data["destination"] = "Unknown"
                if "departure_time" not in flight_data: flight_data["departure_time"] = "2024-01-01T12:00:00"
                if "arrival_time" not in flight_data: flight_data["arrival_time"] = "2024-01-01T18:00:00"

                valid_fields = {'flight_number', 'origin', 'destination', 'departure_time', 'arrival_time', 'status', 'price'}
                flight_data = {k: v for k, v in flight_data.items() if k in valid_fields}
                self.travel.flights.append(Flight(**flight_data))
        if "hotels" in snapshot:
            self.travel.hotels = copy.deepcopy(snapshot["hotels"])
        if "bookings" in snapshot:
            self.travel.bookings = []
            for data in copy.deepcopy(snapshot["bookings"]):
                if "id" not in data:
                    data["id"] = f"bk_{uuid.uuid4().hex[:8]}"
                if "type" not in data: data["type"] = "Unknown"
                if "user" not in data: data["user"] = "unknown_user"
                if "details" not in data: data["details"] = {}
                if "status" not in data: data["status"] = "Confirmed"
                valid = {"id", "type", "user", "details", "status"}
                clean_data = {k: v for k, v in data.items() if k in valid}
                self.travel.bookings.append(Booking(**clean_data))

        if "products" in snapshot:
            raw_products = snapshot["products"]
            if isinstance(raw_products, dict):
                self.shopping.products = {}
                for p_id, p_data in raw_products.items():
                    p_copy = copy.deepcopy(p_data)
                    p_copy["id"] = p_id
                    valid = {"id", "name", "price", "category", "stock", "description"}
                    clean_data = {k: v for k, v in p_copy.items() if k in valid}
                    self.shopping.products[p_id] = Product(**clean_data)
            else:
                self.shopping.products = {}
                for p in raw_products:
                    p_copy = copy.deepcopy(p)
                    if "product_id" in p_copy and "id" not in p_copy:
                        p_copy["id"] = p_copy.pop("product_id")
                    pid = p_copy.get("id", f"prod_{uuid.uuid4().hex[:8]}")
                    p_copy["id"] = pid
                    if "name" not in p_copy: p_copy["name"] = "Unknown Product"
                    if "price" not in p_copy: p_copy["price"] = 0.0
                    valid = {"id", "name", "price", "category", "stock", "description"}
                    clean_data = {k: v for k, v in p_copy.items() if k in valid}
                    self.shopping.products[pid] = Product(**clean_data)
        if "carts" in snapshot:
            self.shopping.carts = copy.deepcopy(snapshot["carts"])
        if "orders" in snapshot:
            self.shopping.orders = []
            for data in snapshot["orders"]:
                clean_data = copy.deepcopy(data)
                if "id" not in clean_data:
                    clean_data["id"] = f"ord_{uuid.uuid4().hex[:8]}"
                if "user" not in clean_data: clean_data["user"] = "unknown_user"
                if "items" not in clean_data: clean_data["items"] = []
                if "total_price" not in clean_data: clean_data["total_price"] = 0.0
                if "status" not in clean_data: clean_data["status"] = "Pending"
                valid = {"id", "user", "items", "total_price", "status"}
                clean_data = {k: v for k, v in clean_data.items() if k in valid}
                self.shopping.orders.append(Order(**clean_data))

        if "appointments" in snapshot:
            self.medical.appointments = []
            for data in snapshot["appointments"]:
                clean_data = copy.deepcopy(data)
                if "id" not in clean_data:
                    clean_data["id"] = f"appt_{uuid.uuid4().hex[:8]}"
                if "patient" not in clean_data: clean_data["patient"] = "Unknown Patient"
                if "doctor" not in clean_data: clean_data["doctor"] = "Unknown Doctor"
                if "time" not in clean_data: clean_data["time"] = "2024-01-01T10:00:00"
                if "department" not in clean_data: clean_data["department"] = "General"
                if "status" not in clean_data: clean_data["status"] = "Scheduled"
                valid = {"id", "patient", "doctor", "time", "department", "status"}
                clean_data = {k: v for k, v in clean_data.items() if k in valid}
                self.medical.appointments.append(Appointment(**clean_data))
        if "medical_records" in snapshot:
            raw_records = snapshot["medical_records"]
            if isinstance(raw_records, list):
                self.extra_state["medical_records"] = copy.deepcopy(raw_records)
            if isinstance(raw_records, dict):
                self.medical.records = {}
                for patient, record in raw_records.items():
                    clean_data = copy.deepcopy(record)
                    clean_data["patient"] = patient
                    if "history" not in clean_data: clean_data["history"] = "No medical history recorded."
                    if "prescriptions" not in clean_data: clean_data["prescriptions"] = []
                    valid = {"patient", "history", "prescriptions"}
                    clean_data = {k: v for k, v in clean_data.items() if k in valid}
                    self.medical.records[patient] = MedicalRecord(**clean_data)
            else:
                self.medical.records = {}
                for r in raw_records:
                    clean_data = copy.deepcopy(r)
                    patient = clean_data.get("patient", f"patient_{uuid.uuid4().hex[:8]}")
                    clean_data["patient"] = patient
                    if "history" not in clean_data: clean_data["history"] = "No medical history recorded."
                    if "prescriptions" not in clean_data: clean_data["prescriptions"] = []
                    valid = {"patient", "history", "prescriptions"}
                    clean_data = {k: v for k, v in clean_data.items() if k in valid}
                    self.medical.records[patient] = MedicalRecord(**clean_data)

        if "context" in snapshot:
            self.context.update(copy.deepcopy(snapshot["context"]))

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the entire World State to a dictionary.
        """
        snapshot = {
            "files": self.fs.files.copy(),
            "folders": list(self.fs.folders),
            "emails": [vars(e) for e in self.office.emails],
            "calendar_events": [vars(e) for e in self.office.calendar_events],
            "contacts": [vars(c) for c in self.office.contacts],
            "instant_messages": list(self.office.instant_messages),

            "bank_accounts": [vars(a) for a in self.bank.accounts.values()],
            "market_prices": self.bank.market_prices.copy(),
            "portfolios": {o: vars(p) for o, p in self.bank.portfolios.items()},

            "iot_devices": self.iot.devices.copy(),

            "flights": [vars(f) for f in self.travel.flights],
            "hotels": list(self.travel.hotels),
            "bookings": [
                {**vars(b), **(b.details if isinstance(b.details, dict) else {})}
                for b in self.travel.bookings
            ],

            "products": {pid: vars(p) for pid, p in self.shopping.products.items()},
            "carts": self.shopping.carts.copy(),
            "orders": [
                {**{k: v for k, v in vars(o).items() if k != "items"}, **item}
                for o in self.shopping.orders
                for item in (o.items if o.items else [{"product_id": None, "quantity": 0}])
            ],

            "appointments": [vars(a) for a in self.medical.appointments],
            "medical_records": {p: vars(r) for p, r in self.medical.records.items()},

            "context": self.context.copy()
        }
        for key, value in self.extra_state.items():
            if key not in snapshot:
                snapshot[key] = copy.deepcopy(value)
        return copy.deepcopy(snapshot)

    @property
    def email(self):
        return self.office
