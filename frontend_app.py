import streamlit as st
import requests

# 1. Page Configuration
st.set_page_config(page_title="Booking Intelligence Pro", page_icon="🏨", layout="wide")

st.title("🏨 Hotel Reservation Intelligence System")
st.markdown("Predicting booking reliability using optimized XGBoost Engine.")
st.markdown("---")

# 2. Input Form
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👥 Guest Profile")
        no_of_adults = st.number_input("Adults", min_value=1, max_value=10, value=2)
        no_of_children = st.number_input("Children", min_value=0, max_value=10, value=0)
        repeated_guest = st.checkbox("Is Repeated Guest?")
        no_of_special_requests = st.number_input("Special Requests", min_value=0, value=1)
        required_car_parking_space = st.selectbox("Parking Space Required?", [0, 1])

    with col2:
        st.subheader("📅 Booking Timeline")
        lead_time = st.number_input("Lead Time (Days)", min_value=0, value=10)
        arrival_year = st.selectbox("Arrival Year", [2017, 2018])
        arrival_month = st.slider("Arrival Month", 1, 12, 6)
        arrival_date = st.slider("Arrival Date", 1, 31, 15)
        market_segment_type = st.selectbox("Market Segment", ["Online", "Offline", "Corporate", "Aviation", "Complementary"])

    with col3:
        st.subheader("💰 Room & Finance")
        avg_price_per_room = st.number_input("Avg Price per Room ($)", min_value=0.0, value=100.0)
        room_type_reserved = st.selectbox("Room Type", ["Room_Type 1", "Room_Type 2", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
        type_of_meal_plan = st.selectbox("Meal Plan", ["Meal Plan 1", "Meal Plan 2", "Not Selected"])
        no_of_weekend_nights = st.number_input("Weekend Nights", min_value=0, value=1)
        no_of_week_nights = st.number_input("Week Nights", min_value=0, value=2)

    st.markdown("---")
    with st.expander("📊 Historical Data (Optional)"):
        c1, c2 = st.columns(2)
        no_of_previous_cancellations = c1.number_input("Previous Cancellations", value=0)
        no_of_previous_bookings_not_canceled = c2.number_input("Previous Success Bookings", value=0)

    submit = st.form_submit_button("🔍 Run AI Analysis")

# 3. Logic & API Connection
if submit:
    payload = {
        "no_of_adults": no_of_adults,
        "no_of_children": no_of_children,
        "no_of_weekend_nights": no_of_weekend_nights,
        "no_of_week_nights": no_of_week_nights,
        "type_of_meal_plan": type_of_meal_plan,
        "required_car_parking_space": int(required_car_parking_space),
        "room_type_reserved": room_type_reserved,
        "lead_time": lead_time,
        "arrival_year": arrival_year,
        "arrival_month": arrival_month,
        "arrival_date": arrival_date,
        "market_segment_type": market_segment_type,
        "repeated_guest": int(repeated_guest),
        "no_of_previous_cancellations": no_of_previous_cancellations,
        "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
        "avg_price_per_room": avg_price_per_room,
        "no_of_special_requests": no_of_special_requests
    }

    try:
        with st.spinner('Analyzing booking patterns...'):
            response = requests.post("http://localhost:8000/predict", json=payload)
            response.raise_for_status() 
            res = response.json()

        if "probability" in res:
            label = res["prediction_label"]
            proba = res["probability"]

            st.markdown("### 📊 Prediction Results")
            
            if label == "Not Canceled":
                st.success(f"### ✅ Result: {label}")
                st.metric("Confidence Score (Stay)", f"{proba:.1%}")
                st.progress(min(max(float(proba), 0.0), 1.0))
            else:
                st.error(f"### ⚠️ Result: {label}")
                conf_cancel = 1 - proba
                st.metric("Confidence Score (Cancel)", f"{conf_cancel:.1%}")
                st.progress(min(max(float(conf_cancel), 0.0), 1.0))

            if 0.45 < proba < 0.60:
                st.warning("🧐 **Moderate Uncertainty:** The model is not 100% sure. Manual verification is recommended.")
        
        else:
            st.error(f"⚠️ API Response Issue: The server didn't return a probability. Response: {res}")

    except Exception as e:
        st.error(f"❌ Connection Error: Ensure the FastAPI server is running on http://localhost:8000. Error details: {e}")
