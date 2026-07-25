from typing import Any, Dict, List
import uuid
import datetime
from core.world.environment import WorldState, Booking, Flight

class TravelTools:
    """
    Travel tools connected to the World Model.
    Handles Flights, Hotels, and Taxis.
    """

    def __init__(self):
        self.current_user = "user@example.com"

    def _find_booking(self, booking_id: str) -> Booking | None:
        return next((b for b in WorldState.get_instance().travel.bookings if b.id == booking_id), None)

    def search_flights(
        self,
        origin: str | None = None,
        destination: str | None = None,
        date: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Search for available flights.
        """
        if not origin:
            origin = kwargs.get("from")
        if not destination:
            destination = kwargs.get("to")
        if not date:
            date = kwargs.get("departure_date")
        if not origin or not destination or not date:
            return {
                "status": "error",
                "message": "Missing required fields: origin, destination, date.",
            }

        results = [
            {
                "flight_number": f.flight_number,
                "origin": f.origin,
                "destination": f.destination,
                "departure_time": f.departure_time,
                "arrival_time": f.arrival_time,
                "price": f.price,
                "status": f.status
            }
            for f in WorldState.get_instance().travel.flights
            if f.origin.lower() == origin.lower()
            and f.destination.lower() == destination.lower()
            and f.departure_time.startswith(date)
            and f.status == "Available"
        ]

        return {
            "status": "success",
            "flights": results,
            "count": len(results)
        }

    def book_flight(
        self,
        flight_number: str | None = None,
        passenger_name: str | None = None,
        payment_method: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Book a flight.
        Side Effect: Creates a Booking in WorldState.
        Payment is recorded as metadata; cross-domain settlement is outside the
        benchmark's travel-state contract.
        """
        if not flight_number:
            flight_number = kwargs.get("flight_id")
        if not passenger_name:
            passenger_name = kwargs.get("passenger") or kwargs.get("name")
        if not payment_method:
            payment_method = kwargs.get("payment")
        if not flight_number or not passenger_name or not payment_method:
            return {
                "status": "error",
                "message": "Missing required fields: flight_number, passenger_name, payment_method.",
            }

        flight = next((f for f in WorldState.get_instance().travel.flights if f.flight_number == flight_number), None)
        if not flight:
             return {"status": "error", "message": "Flight not found."}
        flight_status = str(flight.status or "").strip().lower()
        if flight_status != "available":
             return {"status": "error", "message": f"Flight is {flight.status}."}
        if not payment_method:
             return {"status": "error", "message": "Payment method is required."}

        booking_id = f"bk_{uuid.uuid4().hex[:8]}"
        booking = Booking(
            id=booking_id,
            type="flight",
            user=self.current_user,
            details={
                "flight_number": flight_number,
                "flight_id": flight_number,
                "passenger": passenger_name,
                "passenger_name": passenger_name,
                "origin": flight.origin,
                "destination": flight.destination,
                "time": flight.departure_time,
                "price": flight.price
            },
            status="Confirmed"
        )
        WorldState.get_instance().travel.bookings.append(booking)

        return {
            "status": "success",
            "message": "Flight booked successfully.",
            "booking_id": booking_id,
            "details": booking.details
        }

    def cancel_flight(self, booking_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Cancel a flight booking."""
        if not booking_id:
            booking_id = kwargs.get("id")
        if not booking_id:
            return {"status": "error", "message": "Missing required argument: booking_id"}

        booking = next((b for b in WorldState.get_instance().travel.bookings if b.id == booking_id), None)
        if not booking:
            return {"status": "error", "message": "Booking not found."}

        if booking.user != self.current_user:
            return {"status": "error", "message": "Access denied."}

        booking.status = "Cancelled"
        return {"status": "success", "message": f"Booking {booking_id} cancelled."}

    def search_hotels(
        self,
        location: str | None = None,
        check_in: str | None = None,
        check_out: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Search for hotels (Mock implementation as data structure is minimal)."""
        if not location:
            location = kwargs.get("city")
        if not check_in:
            check_in = kwargs.get("start_date")
        if not check_out:
            check_out = kwargs.get("end_date")
        if not location or not check_in or not check_out:
            return {
                "status": "error",
                "message": "Missing required fields: location, check_in, check_out.",
            }

        if WorldState.get_instance().travel.hotels:
            hotels = [
                h for h in WorldState.get_instance().travel.hotels
                if str(h.get("location", "")).lower() == str(location).lower()
            ]
            return {
                "status": "success",
                "hotels": hotels,
                "count": len(hotels),
            }

        return {
            "status": "success",
            "hotels": [
                {"name": "Grand Hotel", "location": location, "price_per_night": 200, "rating": 4.5},
                {"name": "Budget Inn", "location": location, "price_per_night": 80, "rating": 3.0}
            ]
        }

    def book_hotel(
        self,
        hotel_name: str | None = None,
        check_in: str | None = None,
        check_out: str | None = None,
        guest_name: str | None = None,
        passenger_name: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not hotel_name:
            hotel_name = kwargs.get("hotel")
        if not check_in:
            check_in = kwargs.get("start_date")
        if not check_out:
            check_out = kwargs.get("end_date")
        if not guest_name:
            guest_name = passenger_name or kwargs.get("guest") or kwargs.get("name") or "User"
        if not hotel_name or not check_in or not check_out:
            return {
                "status": "error",
                "message": "Missing required fields: hotel_name, check_in, check_out.",
            }

        hotel_id = hotel_name
        for h in WorldState.get_instance().travel.hotels:
            h_name = h.get("name", "")
            if h_name.lower() == hotel_name.lower() or hotel_name.lower() in h_name.lower():
                hotel_id = h.get("hotel_id", h.get("id", hotel_name))
                break

        booking_id = f"ht_{uuid.uuid4().hex[:8]}"
        booking = Booking(
            id=booking_id,
            type="hotel",
            user=self.current_user,
            details={
                "hotel": hotel_name,
                "hotel_id": hotel_id,
                "hotel_name": hotel_name,
                "guest_name": guest_name,
                "check_in": check_in,
                "check_out": check_out
            },
            status="Confirmed"
        )
        WorldState.get_instance().travel.bookings.append(booking)
        return {"status": "success", "booking_id": booking_id}


    def call_taxi(
        self,
        pickup: str | None = None,
        dropoff: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Request a taxi ride."""
        if not pickup:
            pickup = kwargs.get("from")
        if not dropoff:
            dropoff = kwargs.get("to")
        if not pickup or not dropoff:
            return {"status": "error", "message": "Missing required fields: pickup, dropoff."}

        booking_id = f"tx_{uuid.uuid4().hex[:8]}"
        booking = Booking(
            id=booking_id,
            type="taxi",
            user=self.current_user,
            details={
                "pickup": pickup,
                "dropoff": dropoff,
                "estimated_arrival": "5 mins"
            },
            status="Dispatched"
        )
        WorldState.get_instance().travel.bookings.append(booking)
        return {
            "status": "success",
            "message": f"Taxi dispatched to {pickup}. ETA 5 mins.",
            "booking_id": booking_id
        }

    def get_flight_status(self, flight_number: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Get current status of a specific flight."""
        if not flight_number:
            flight_number = kwargs.get("flight_id")
        if not flight_number:
            return {"status": "error", "message": "Missing required argument: flight_number"}

        flight = next((f for f in WorldState.get_instance().travel.flights if f.flight_number == flight_number), None)
        if not flight:
            return {"status": "error", "message": "Flight not found."}
        return {
            "status": "success",
            "flight_number": flight.flight_number,
            "status_text": flight.status,
            "departure_time": flight.departure_time,
            "arrival_time": flight.arrival_time,
        }

    def check_in(self, booking_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Check in to a confirmed flight booking."""
        if not booking_id:
            booking_id = kwargs.get("id")
        if not booking_id:
            return {"status": "error", "message": "Missing required argument: booking_id"}

        booking = next((b for b in WorldState.get_instance().travel.bookings if b.id == booking_id), None)
        if not booking:
            return {"status": "error", "message": "Booking not found."}
        if booking.user != self.current_user:
            return {"status": "error", "message": "Access denied."}
        if booking.type != "flight":
            return {"status": "error", "message": "Booking is not a flight."}
        if booking.status == "Cancelled":
            return {"status": "error", "message": "Cannot check in a cancelled booking."}

        booking.details["checked_in"] = True
        booking.details["boarding_pass"] = f"bp_{uuid.uuid4().hex[:8]}"
        return {
            "status": "success",
            "message": "Check-in completed.",
            "boarding_pass": booking.details["boarding_pass"],
        }

    def cancel_hotel(self, booking_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Cancel a hotel booking."""
        if not booking_id:
            booking_id = kwargs.get("id")
        if not booking_id:
            return {"status": "error", "message": "Missing required argument: booking_id"}

        booking = next((b for b in WorldState.get_instance().travel.bookings if b.id == booking_id), None)
        if not booking:
            return {"status": "error", "message": "Booking not found."}
        if booking.user != self.current_user:
            return {"status": "error", "message": "Access denied."}
        if booking.type != "hotel":
            return {"status": "error", "message": "Booking is not a hotel booking."}

        booking.status = "Cancelled"
        return {"status": "success", "message": f"Hotel booking {booking_id} cancelled."}

    def search_trains(
        self,
        origin: str | None = None,
        destination: str | None = None,
        date: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Search train options from catalog or mocked defaults."""
        if not origin:
            origin = kwargs.get("from")
        if not destination:
            destination = kwargs.get("to")
        if not date:
            date = kwargs.get("departure_date")
        if not origin or not destination or not date:
            return {
                "status": "error",
                "message": "Missing required fields: origin, destination, date.",
            }

        catalog = WorldState.get_instance().context.get("train_catalog", [])
        if isinstance(catalog, list) and catalog:
            trains = [
                t
                for t in catalog
                if str(t.get("origin", "")).lower() == str(origin).lower()
                and str(t.get("destination", "")).lower() == str(destination).lower()
                and str(t.get("departure_time", "")).startswith(str(date))
            ]
        else:
            trains = [
                {
                    "train_number": "TR100",
                    "origin": origin,
                    "destination": destination,
                    "departure_time": f"{date}T08:30:00",
                    "arrival_time": f"{date}T12:10:00",
                    "price": 79.0,
                },
                {
                    "train_number": "TR220",
                    "origin": origin,
                    "destination": destination,
                    "departure_time": f"{date}T13:00:00",
                    "arrival_time": f"{date}T16:50:00",
                    "price": 95.0,
                },
            ]
        return {"status": "success", "trains": trains, "count": len(trains)}

    def book_train(
        self,
        train_number: str | None = None,
        passenger_name: str | None = None,
        seat_class: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Book a train itinerary."""
        if not train_number:
            train_number = kwargs.get("train_id")
        if not passenger_name:
            passenger_name = kwargs.get("passenger") or kwargs.get("name")
        if not seat_class:
            seat_class = kwargs.get("class")
        if not train_number or not passenger_name or not seat_class:
            return {
                "status": "error",
                "message": "Missing required fields: train_number, passenger_name, seat_class.",
            }
        booking_id = f"tr_{uuid.uuid4().hex[:8]}"
        booking = Booking(
            id=booking_id,
            type="train",
            user=self.current_user,
            details={
                "train_number": train_number,
                "passenger": passenger_name,
                "seat_class": seat_class,
            },
            status="Confirmed",
        )
        WorldState.get_instance().travel.bookings.append(booking)
        return {"status": "success", "booking_id": booking_id}

    def cancel_train(self, booking_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Cancel a train booking."""
        if not booking_id:
            booking_id = kwargs.get("id")
        if not booking_id:
            return {"status": "error", "message": "Missing required field: booking_id."}
        booking = self._find_booking(booking_id)
        if not booking:
            return {"status": "error", "message": "Booking not found."}
        if booking.user != self.current_user:
            return {"status": "error", "message": "Access denied."}
        if booking.type != "train":
            return {"status": "error", "message": "Booking is not a train booking."}
        booking.status = "Cancelled"
        return {"status": "success", "booking_id": booking_id}

    def search_car_rentals(
        self,
        location: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Search available rental cars."""
        if not location:
            location = kwargs.get("city")
        if not start_time:
            start_time = kwargs.get("start")
        if not end_time:
            end_time = kwargs.get("end")
        if not location or not start_time or not end_time:
            return {
                "status": "error",
                "message": "Missing required fields: location, start_time, end_time.",
            }
        catalog = WorldState.get_instance().context.get("car_rental_catalog", [])
        if isinstance(catalog, list) and catalog:
            cars = [
                c
                for c in catalog
                if str(c.get("location", "")).lower() == str(location).lower()
            ]
        else:
            cars = [
                {"car_id": "car_economy_01", "location": location, "daily_price": 45.0, "class": "economy"},
                {"car_id": "car_suv_01", "location": location, "daily_price": 95.0, "class": "suv"},
            ]
        return {"status": "success", "cars": cars, "count": len(cars)}

    def book_car_rental(
        self,
        car_id: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        driver_name: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Book a rental car."""
        if not car_id:
            car_id = kwargs.get("vehicle_id")
        if not start_time:
            start_time = kwargs.get("start")
        if not end_time:
            end_time = kwargs.get("end")
        if not driver_name:
            driver_name = kwargs.get("driver")
        if not car_id or not start_time or not end_time or not driver_name:
            return {
                "status": "error",
                "message": "Missing required fields: car_id, start_time, end_time, driver_name.",
            }
        booking_id = f"cr_{uuid.uuid4().hex[:8]}"
        booking = Booking(
            id=booking_id,
            type="car_rental",
            user=self.current_user,
            details={
                "car_id": car_id,
                "start_time": start_time,
                "end_time": end_time,
                "driver_name": driver_name,
            },
            status="Confirmed",
        )
        WorldState.get_instance().travel.bookings.append(booking)
        return {"status": "success", "booking_id": booking_id}

    def cancel_car_rental(self, booking_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Cancel a rental car booking."""
        if not booking_id:
            booking_id = kwargs.get("id")
        if not booking_id:
            return {"status": "error", "message": "Missing required field: booking_id."}
        booking = self._find_booking(booking_id)
        if not booking:
            return {"status": "error", "message": "Booking not found."}
        if booking.user != self.current_user:
            return {"status": "error", "message": "Access denied."}
        if booking.type != "car_rental":
            return {"status": "error", "message": "Booking is not a car rental booking."}
        booking.status = "Cancelled"
        return {"status": "success", "booking_id": booking_id}

    def change_flight(
        self,
        booking_id: str | None = None,
        new_flight_number: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Change flight number on an existing flight booking."""
        if not booking_id:
            booking_id = kwargs.get("id")
        if not new_flight_number:
            new_flight_number = kwargs.get("flight_number")
        if not booking_id or not new_flight_number:
            return {
                "status": "error",
                "message": "Missing required fields: booking_id, new_flight_number.",
            }
        booking = self._find_booking(booking_id)
        if not booking:
            return {"status": "error", "message": "Booking not found."}
        if booking.type != "flight":
            return {"status": "error", "message": "Booking is not a flight booking."}
        old_flight = booking.details.get("flight_number")
        booking.details["flight_number"] = new_flight_number
        booking.details["changed_at"] = datetime.datetime.now().isoformat()
        booking.status = "Confirmed"
        return {
            "status": "success",
            "booking_id": booking_id,
            "old_flight_number": old_flight,
            "new_flight_number": new_flight_number,
        }

    def change_hotel(
        self,
        booking_id: str | None = None,
        new_hotel_name: str | None = None,
        new_check_in: str | None = None,
        new_check_out: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Change details of an existing hotel booking."""
        if not booking_id:
            booking_id = kwargs.get("id")
        if not new_hotel_name:
            new_hotel_name = kwargs.get("hotel_name")
        if not new_check_in:
            new_check_in = kwargs.get("check_in")
        if not new_check_out:
            new_check_out = kwargs.get("check_out")
        if not booking_id or not new_hotel_name or not new_check_in or not new_check_out:
            return {
                "status": "error",
                "message": "Missing required fields: booking_id, new_hotel_name, new_check_in, new_check_out.",
            }
        booking = self._find_booking(booking_id)
        if not booking:
            return {"status": "error", "message": "Booking not found."}
        if booking.type != "hotel":
            return {"status": "error", "message": "Booking is not a hotel booking."}
        booking.details.update(
            {
                "hotel": new_hotel_name,
                "check_in": new_check_in,
                "check_out": new_check_out,
                "changed_at": datetime.datetime.now().isoformat(),
            }
        )
        booking.status = "Confirmed"
        return {"status": "success", "booking_id": booking_id}

    def get_booking_details(self, booking_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Return complete booking details."""
        if not booking_id:
            booking_id = kwargs.get("id")
        if not booking_id:
            return {"status": "error", "message": "Missing required field: booking_id."}
        booking = self._find_booking(booking_id)
        if not booking:
            return {"status": "error", "message": "Booking not found."}
        if booking.user != self.current_user:
            return {"status": "error", "message": "Access denied."}
        return {
            "status": "success",
            "booking_id": booking.id,
            "type": booking.type,
            "details": booking.details,
            "status_text": booking.status,
        }

    def purchase_travel_insurance(
        self,
        booking_id: str | None = None,
        plan: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Purchase travel insurance for a booking."""
        if not booking_id:
            booking_id = kwargs.get("id")
        if not plan:
            plan = kwargs.get("package")
        if not booking_id or not plan:
            return {"status": "error", "message": "Missing required fields: booking_id, plan."}
        booking = self._find_booking(booking_id)
        if not booking:
            return {"status": "error", "message": "Booking not found."}
        insurance_id = f"ins_{uuid.uuid4().hex[:8]}"
        insurance_store = WorldState.get_instance().context.setdefault("travel_insurance", {})
        insurance_store[insurance_id] = {
            "insurance_id": insurance_id,
            "booking_id": booking_id,
            "plan": plan,
            "status_text": "active",
            "purchased_at": datetime.datetime.now().isoformat(),
        }
        return {"status": "success", "insurance_id": insurance_id, "booking_id": booking_id}

    def apply_loyalty_points(
        self,
        booking_id: str | None = None,
        points: int | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Apply loyalty points to an existing booking."""
        if not booking_id:
            booking_id = kwargs.get("id")
        if points is None:
            points = kwargs.get("value")
        if not booking_id or points is None:
            return {"status": "error", "message": "Missing required fields: booking_id, points."}
        try:
            points = int(points)
        except (TypeError, ValueError):
            return {"status": "error", "message": "Invalid points format."}
        if points <= 0:
            return {"status": "error", "message": "points must be positive."}

        booking = self._find_booking(booking_id)
        if not booking:
            return {"status": "error", "message": "Booking not found."}
        accounts = WorldState.get_instance().context.setdefault("loyalty_accounts", {})
        acct = accounts.setdefault(self.current_user, {"points_balance": 5000, "used_points": 0})
        if acct["points_balance"] < points:
            return {
                "status": "error",
                "message": f"Insufficient points. Balance: {acct['points_balance']}",
            }
        acct["points_balance"] -= points
        acct["used_points"] += points
        booking.details["loyalty_points_applied"] = points
        booking.details["points_applied_at"] = datetime.datetime.now().isoformat()
        return {
            "status": "success",
            "booking_id": booking_id,
            "points_applied": points,
            "points_balance": acct["points_balance"],
        }

    def request_travel_invoice(
        self,
        booking_id: str | None = None,
        invoice_title: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Request invoice generation for a booking."""
        if not booking_id:
            booking_id = kwargs.get("id")
        if not invoice_title:
            invoice_title = kwargs.get("title")
        if not booking_id or not invoice_title:
            return {
                "status": "error",
                "message": "Missing required fields: booking_id, invoice_title.",
            }
        booking = self._find_booking(booking_id)
        if not booking:
            return {"status": "error", "message": "Booking not found."}
        invoice_id = f"inv_{uuid.uuid4().hex[:8]}"
        WorldState.get_instance().context.setdefault("travel_invoices", []).append(
            {
                "invoice_id": invoice_id,
                "booking_id": booking_id,
                "invoice_title": invoice_title,
                "status_text": "requested",
                "requested_at": datetime.datetime.now().isoformat(),
            }
        )
        return {"status": "success", "invoice_id": invoice_id, "booking_id": booking_id}


    def check_visa_requirements(
        self,
        nationality: str | None = None,
        destination: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Check visa requirements for a nationality-destination pair."""
        if not nationality:
            nationality = kwargs.get("passport") or kwargs.get("country")
        if not destination:
            destination = kwargs.get("to") or kwargs.get("target_country")
        if not nationality or not destination:
            return {"status": "error", "message": "Missing required fields: nationality, destination."}
        visa_db = WorldState.get_instance().context.get("visa_requirements", {})
        key = f"{nationality}|{destination}"
        info = visa_db.get(key)
        if not info:
            info = {"visa_required": True, "type": "Tourist Visa",
                    "processing_days": 15, "documents": ["passport", "photo", "invitation letter"]}
        return {"status": "success", "nationality": nationality, "destination": destination, **info}

    def get_weather_forecast(
        self,
        location: str | None = None,
        date: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Get weather forecast for a location and date."""
        if not location:
            location = kwargs.get("city") or kwargs.get("destination")
        if not date:
            date = kwargs.get("day")
        if not location:
            return {"status": "error", "message": "Missing required field: location."}
        weather_db = WorldState.get_instance().context.get("weather_forecasts", {})
        key = f"{location}|{date}" if date else location
        forecast = weather_db.get(key) or weather_db.get(location)
        if not forecast:
            forecast = {"temperature_c": 22, "condition": "Partly Cloudy",
                        "humidity": 55, "wind_kph": 12}
        return {"status": "success", "location": location, "date": date, **forecast}

    def get_travel_advisory(
        self,
        destination: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Get travel safety / health advisory for a destination."""
        if not destination:
            destination = kwargs.get("country") or kwargs.get("region")
        if not destination:
            return {"status": "error", "message": "Missing required field: destination."}
        advisories = WorldState.get_instance().context.get("travel_advisories", {})
        info = advisories.get(destination)
        if not info:
            info = {"risk_level": "low", "advisory": "Exercise normal precautions.",
                    "health_notes": "No special vaccinations required."}
        return {"status": "success", "destination": destination, **info}


from tools.connected._v3_roots import ensure_list_root, find_many, find_one, lookup_by_arg, make_id


def _travel_v3_read_booking_message_thread(self, booking_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    booking_id = booking_id or kwargs.get("id")
    if not booking_id:
        return {"status": "error", "message": "Missing required argument: booking_id"}
    items = find_many(ensure_list_root(self, "travel_booking_threads"), booking_id=booking_id)
    if not items:
        return {"status": "error", "message": f"No booking thread found for {booking_id}."}
    return {"status": "success", "booking_id": booking_id, "messages": items, "count": len(items)}


def _travel_v3_read_travel_policy_note(self, policy_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    policy_id = policy_id or kwargs.get("id")
    if not policy_id:
        return {"status": "error", "message": "Missing required argument: policy_id"}
    item = find_one(ensure_list_root(self, "travel_policy_notes"), policy_id=policy_id)
    if not item:
        return {"status": "error", "message": "Travel policy note not found."}
    return {"status": "success", **item}


def _travel_v3_read_booking_agent_note(self, booking_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    booking_id = booking_id or kwargs.get("id")
    if not booking_id:
        return {"status": "error", "message": "Missing required argument: booking_id"}
    item = find_one(ensure_list_root(self, "travel_booking_agent_notes"), booking_id=booking_id)
    if not item:
        return {"status": "error", "message": "Booking agent note not found."}
    return {"status": "success", **item}


def _travel_v3_get_booking_details(self, booking_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    booking_id = booking_id or kwargs.get("id")
    if not booking_id:
        return {"status": "error", "message": "Missing required argument: booking_id"}
    item = find_one(ensure_list_root(self, "travel_bookings"), booking_id=booking_id)
    if not item:
        return {"status": "error", "message": "Booking not found."}
    return {"status": "success", **item}


def _travel_v3_get_flight_status(self, flight_number: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    flight_number = flight_number or kwargs.get("id")
    if not flight_number:
        return {"status": "error", "message": "Missing required argument: flight_number"}
    item = find_one(ensure_list_root(self, "travel_flights"), flight_number=flight_number)
    if not item:
        return {"status": "error", "message": "Flight not found."}
    return {"status": "success", **item}


def _travel_v3_check_visa_requirements(self, destination: str | None = None, nationality: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    destination = destination or kwargs.get("country")
    nationality = nationality or kwargs.get("passport_country")
    if not destination:
        return {"status": "error", "message": "Missing required argument: destination"}
    items = ensure_list_root(self, "travel_visa_requirements")
    item = find_one(items, destination=destination, nationality=nationality) or find_one(items, destination=destination)
    if not item:
        return {"status": "error", "message": "Visa requirement not found."}
    return {"status": "success", **item}


def _travel_v3_get_travel_advisory(self, destination: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    destination = destination or kwargs.get("country")
    if not destination:
        return {"status": "error", "message": "Missing required argument: destination"}
    item = find_one(ensure_list_root(self, "travel_advisories"), destination=destination)
    if not item:
        return {"status": "error", "message": "Travel advisory not found."}
    return {"status": "success", **item}


def _travel_v3_get_fare_rules(self, booking_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    booking_id = booking_id or kwargs.get("id")
    if not booking_id:
        return {"status": "error", "message": "Missing required argument: booking_id"}
    item = find_one(ensure_list_root(self, "travel_fare_rules"), booking_id=booking_id)
    if not item:
        return {"status": "error", "message": "Fare rules not found."}
    return {"status": "success", **item}


def _travel_v3_get_checkin_window(self, booking_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    booking_id = booking_id or kwargs.get("id")
    if not booking_id:
        return {"status": "error", "message": "Missing required argument: booking_id"}
    item = find_one(ensure_list_root(self, "travel_checkin_windows"), booking_id=booking_id)
    if not item:
        return {"status": "error", "message": "Check-in window not found."}
    return {"status": "success", **item}


def _travel_v3_get_invoice_status(self, booking_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    booking_id = booking_id or kwargs.get("id")
    if not booking_id:
        return {"status": "error", "message": "Missing required argument: booking_id"}
    item = find_one(ensure_list_root(self, "travel_invoices"), booking_id=booking_id)
    if not item:
        return {"status": "error", "message": "Invoice record not found."}
    return {"status": "success", **item}


def _travel_v3_change_flight(self, booking_id: str | None = None, new_flight_number: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    booking_id = booking_id or kwargs.get("id")
    new_flight_number = new_flight_number or kwargs.get("flight_number")
    if not booking_id or not new_flight_number:
        return {"status": "error", "message": "Missing required fields: booking_id, new_flight_number."}
    bookings = ensure_list_root(self, "travel_bookings")
    if not find_one(ensure_list_root(self, "travel_flights"), flight_number=new_flight_number):
        return {"status": "error", "message": "New flight not found."}
    item = find_one(bookings, booking_id=booking_id)
    if not item:
        return {"status": "error", "message": "Booking not found."}
    item["flight_number"] = new_flight_number
    item["status"] = "changed"
    item["updated_at"] = datetime.datetime.now().isoformat()
    return {"status": "success", **item}


def _travel_v3_cancel_flight(self, booking_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    booking_id = booking_id or kwargs.get("id")
    if not booking_id:
        return {"status": "error", "message": "Missing required field: booking_id."}
    item = find_one(ensure_list_root(self, "travel_bookings"), booking_id=booking_id)
    if not item:
        return {"status": "error", "message": "Booking not found."}
    item["status"] = "cancelled"
    item["updated_at"] = datetime.datetime.now().isoformat()
    return {"status": "success", **item}


def _travel_v3_check_in(self, booking_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    booking_id = booking_id or kwargs.get("id")
    if not booking_id:
        return {"status": "error", "message": "Missing required field: booking_id."}
    booking = find_one(ensure_list_root(self, "travel_bookings"), booking_id=booking_id)
    if not booking:
        return {"status": "error", "message": "Booking not found."}
    booking["checked_in"] = True
    booking["status"] = "checked_in"
    booking["updated_at"] = datetime.datetime.now().isoformat()
    return {"status": "success", **booking}


def _travel_v3_request_travel_invoice(self, booking_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    booking_id = booking_id or kwargs.get("id")
    if not booking_id:
        return {"status": "error", "message": "Missing required field: booking_id."}
    if not find_one(ensure_list_root(self, "travel_bookings"), booking_id=booking_id):
        return {"status": "error", "message": "Booking not found."}
    invoices = ensure_list_root(self, "travel_invoices")
    item = find_one(invoices, booking_id=booking_id)
    if not item:
        item = {"booking_id": booking_id, "status": "requested", "invoice_id": make_id("inv")}
        invoices.append(item)
    else:
        item["status"] = "requested"
    item["updated_at"] = datetime.datetime.now().isoformat()
    return {"status": "success", **item}


TravelTools.read_booking_message_thread = _travel_v3_read_booking_message_thread
TravelTools.read_travel_policy_note = _travel_v3_read_travel_policy_note
TravelTools.read_booking_agent_note = _travel_v3_read_booking_agent_note
TravelTools.get_booking_details = _travel_v3_get_booking_details
TravelTools.get_flight_status = _travel_v3_get_flight_status
TravelTools.check_visa_requirements = _travel_v3_check_visa_requirements
TravelTools.get_travel_advisory = _travel_v3_get_travel_advisory
TravelTools.get_fare_rules = _travel_v3_get_fare_rules
TravelTools.get_checkin_window = _travel_v3_get_checkin_window
TravelTools.get_invoice_status = _travel_v3_get_invoice_status
TravelTools.change_flight = _travel_v3_change_flight
TravelTools.cancel_flight = _travel_v3_cancel_flight
TravelTools.check_in = _travel_v3_check_in
TravelTools.request_travel_invoice = _travel_v3_request_travel_invoice


def _travel_v3_get_weather_forecast(self, location: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    location = location or kwargs.get("city") or kwargs.get("destination")
    if not location:
        return {"status": "error", "message": "Missing required field: location."}
    item = find_one(ensure_list_root(self, "travel_weather_forecasts"), location=location)
    if not item:
        return {"status": "error", "message": "Weather forecast not found."}
    return {"status": "success", **item}


def _travel_v3_apply_loyalty_points(self, booking_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    booking_id = booking_id or kwargs.get("id")
    if not booking_id:
        return {"status": "error", "message": "Missing required field: booking_id."}
    booking = find_one(ensure_list_root(self, "travel_bookings"), booking_id=booking_id)
    if not booking:
        return {"status": "error", "message": "Booking not found."}
    booking["loyalty_applied"] = True
    booking["updated_at"] = datetime.datetime.now().isoformat()
    return {"status": "success", **booking}


def _travel_v3_purchase_travel_insurance(self, booking_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    booking_id = booking_id or kwargs.get("id")
    if not booking_id:
        return {"status": "error", "message": "Missing required field: booking_id."}
    booking = find_one(ensure_list_root(self, "travel_bookings"), booking_id=booking_id)
    if not booking:
        return {"status": "error", "message": "Booking not found."}
    policies = ensure_list_root(self, "travel_insurance_policies")
    entry = {"policy_id": make_id("ins"), "booking_id": booking_id, "status": "active", "created_at": datetime.datetime.now().isoformat()}
    policies.append(entry)
    return {"status": "success", **entry}


def _travel_v3_get_baggage_rules(self, booking_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    booking_id = booking_id or kwargs.get("id")
    if not booking_id:
        return {"status": "error", "message": "Missing required field: booking_id."}
    item = find_one(ensure_list_root(self, "travel_baggage_rules"), booking_id=booking_id)
    if not item:
        return {"status": "error", "message": "Baggage rules not found."}
    return {"status": "success", **item}


def _travel_v3_get_connection_risk(self, booking_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    booking_id = booking_id or kwargs.get("id")
    if not booking_id:
        return {"status": "error", "message": "Missing required field: booking_id."}
    item = find_one(ensure_list_root(self, "travel_connection_risks"), booking_id=booking_id)
    if not item:
        return {"status": "error", "message": "Connection risk not found."}
    return {"status": "success", **item}


def _travel_v3_get_loyalty_status(self, booking_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    booking_id = booking_id or kwargs.get("id")
    if not booking_id:
        return {"status": "error", "message": "Missing required field: booking_id."}
    item = find_one(ensure_list_root(self, "travel_loyalty_statuses"), booking_id=booking_id)
    if not item:
        return {"status": "error", "message": "Loyalty status not found."}
    return {"status": "success", **item}


def _travel_v3_get_hotel_policy(self, booking_id: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    booking_id = booking_id or kwargs.get("id")
    if not booking_id:
        return {"status": "error", "message": "Missing required field: booking_id."}
    item = find_one(ensure_list_root(self, "travel_hotel_policies"), booking_id=booking_id)
    if not item:
        return {"status": "error", "message": "Hotel policy not found."}
    return {"status": "success", **item}


def _travel_v3_generic_reader(self, root: str, key: str, label: str, value: Any) -> Dict[str, Any]:
    return lookup_by_arg(self, root=root, arg_value=value, id_field=key, label=label)


_TRAVEL_GENERIC_READERS = {
    "read_travel_desk_message": ("travel_desk_messages", "message_id", "Travel desk message"),
    "read_airline_followup_message": ("travel_airline_followup_messages", "message_id", "Airline follow-up message"),
    "read_ticket_rule_note": ("travel_ticket_rule_notes", "note_id", "Ticket rule note"),
    "read_checkin_requirement_note": ("travel_checkin_requirement_notes", "note_id", "Check-in requirement note"),
    "read_trip_history_note": ("travel_trip_history_notes", "note_id", "Trip history note"),
    "read_prior_disruption_note": ("travel_prior_disruption_notes", "note_id", "Prior disruption note"),
    "read_rebooking_history_note": ("travel_rebooking_history_notes", "note_id", "Rebooking history note"),
    "read_itinerary_attachment": ("travel_itinerary_attachments", "attachment_id", "Itinerary attachment"),
    "read_invoice_packet_text": ("travel_invoice_packet_texts", "attachment_id", "Invoice packet"),
    "read_visa_packet_text": ("travel_visa_packet_texts", "attachment_id", "Visa packet"),
    "read_traveler_profile_note": ("travel_traveler_profile_notes", "note_id", "Traveler profile note"),
    "read_trip_override_note": ("travel_trip_override_notes", "note_id", "Trip override note"),
    "read_travel_support_ticket": ("travel_support_tickets", "ticket_id", "Travel support ticket"),
    "read_flight_disruption_case": ("travel_flight_disruption_cases", "case_id", "Flight disruption case"),
}


for _name, (_root, _key, _label) in _TRAVEL_GENERIC_READERS.items():
    def _factory(root: str, key: str, label: str):
        def _reader(self, **kwargs: Any) -> Dict[str, Any]:
            value = kwargs.get(key) or kwargs.get("id") or kwargs.get("note_id") or kwargs.get("message_id") or kwargs.get("attachment_id")
            return _travel_v3_generic_reader(self, root, key, label, value)
        return _reader
    setattr(TravelTools, _name, _factory(_root, _key, _label))


TravelTools.get_weather_forecast = _travel_v3_get_weather_forecast
TravelTools.apply_loyalty_points = _travel_v3_apply_loyalty_points
TravelTools.purchase_travel_insurance = _travel_v3_purchase_travel_insurance
TravelTools.get_baggage_rules = _travel_v3_get_baggage_rules
TravelTools.get_connection_risk = _travel_v3_get_connection_risk
TravelTools.get_loyalty_status = _travel_v3_get_loyalty_status
TravelTools.get_hotel_policy = _travel_v3_get_hotel_policy
