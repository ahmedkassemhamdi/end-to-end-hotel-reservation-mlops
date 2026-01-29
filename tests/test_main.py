from fastapi.testclient import TestClient
from src.api.app import app  

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_endpoint():
    test_data = {
        "no_of_adults": 2,
        "no_of_children": 0,
        "no_of_weekend_nights": 1,
        "no_of_week_nights": 2,
        "type_of_meal_plan": "Meal Plan 1",
        "required_car_parking_space": 0,
        "room_type_reserved": "Room_Type 1",
        "lead_time": 224,
        "arrival_year": 2017,
        "arrival_month": 10,
        "arrival_date": 2,
        "market_segment_type": "Offline",
        "repeated_guest": 0,
        "no_of_previous_cancellations": 0,
        "no_of_previous_bookings_not_canceled": 0,
        "avg_price_per_room": 65.0,
        "no_of_special_requests": 0
    }


    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    json_response = response.json()
    assert "prediction_label" in json_response
    assert "prediction_code" in json_response
    print("Prediction Response:", json_response)
