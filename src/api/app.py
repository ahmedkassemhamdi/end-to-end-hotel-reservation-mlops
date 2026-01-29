from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import uvicorn
from src.schemas.input_schema import HotelReservationInput

app = FastAPI(
    title="Hotel Reservation Prediction API",
    version="1.0"
)
model = None

try:
    model = joblib.load("models/xgb_model.pkl")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

@app.get("/")
async def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict_cancellation(data: HotelReservationInput):
    try:
        df = pd.DataFrame([data.model_dump()])
        
        df['total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
        df['total_guests'] = df['no_of_adults'] + df['no_of_children']
        df['price_per_night'] = df['avg_price_per_room']
        df['is_weekend_only'] = (df['no_of_week_nights'] == 0).astype(int)

        y_prob = model.predict_proba(df)[0, 1]

        threshold = 0.55 
        custom_pred = 1 if y_prob >= threshold else 0

        prediction_label = "Not Canceled" if custom_pred == 1 else "Canceled"
        
        return {
            "prediction_code": int(custom_pred),
            "prediction_label": prediction_label,
            "probability": round(float(y_prob), 3)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
