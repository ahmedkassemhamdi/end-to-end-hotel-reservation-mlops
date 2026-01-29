from pydantic import BaseModel

class HotelReservationInput(BaseModel):
    no_of_adults: int
    no_of_children: int
    no_of_weekend_nights: int
    no_of_week_nights: int
    type_of_meal_plan: str
    required_car_parking_space: int
    room_type_reserved: str
    lead_time: int
    arrival_year: int
    arrival_month: int
    arrival_date: int
    market_segment_type: str
    repeated_guest: int
    no_of_previous_cancellations: int
    no_of_previous_bookings_not_canceled: int
    avg_price_per_room: float
    no_of_special_requests: int
    

    class Config:
        json_schema_extra = {
            "example": {
                "no_of_adults": 2, "no_of_children": 0, "no_of_weekend_nights": 1,
                "no_of_week_nights": 2, "type_of_meal_plan": "Meal Plan 1",
                "required_car_parking_space": 0, "room_type_reserved": "Room_Type 1",
                "lead_time": 224, "arrival_year": 2017, "arrival_month": 10,
                "arrival_date": 2, "market_segment_type": "Offline",
                "repeated_guest": 0, "no_of_previous_cancellations": 0,
                "no_of_previous_bookings_not_canceled": 0, "avg_price_per_room": 65.0,
                "no_of_special_requests": 0
            }
        }
