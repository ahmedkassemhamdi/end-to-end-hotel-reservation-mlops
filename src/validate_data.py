from src.logger import get_logger
from src.custom_exception import CustomException
import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema

logger = get_logger(__name__)

hotel_schema = DataFrameSchema(
    columns={
        "Booking_ID": Column(str, nullable=False),
        "no_of_adults": Column(int, Check.between(0, 10)),
        "no_of_children": Column(float, Check.between(0, 10), nullable=True),
        "no_of_weekend_nights": Column(int, Check.between(0, 15)),
        "no_of_week_nights": Column(int, Check.between(0, 30)),
        "type_of_meal_plan": Column(str, nullable=True),
        "required_car_parking_space": Column(int, Check.isin([0, 1])),
        "room_type_reserved": Column(str, nullable=True),
        "lead_time": Column(int, Check.between(0, 500)),
        "arrival_year": Column(int, Check.between(2015, 2030)),
        "arrival_month": Column(int, Check.between(1, 12)),
        "arrival_date": Column(int, Check.between(1, 31)),
        "market_segment_type": Column(str, nullable=True),
        "repeated_guest": Column(int, Check.isin([0, 1])),
        "no_of_previous_cancellations": Column(int, Check.ge(0)),
        "no_of_previous_bookings_not_canceled": Column(int, Check.ge(0)),
        "avg_price_per_room": Column(float, Check.ge(0)),
        "no_of_special_requests": Column(int, Check.ge(0)),
        "booking_status": Column(str, Check.isin(["Canceled", "Not_Canceled"]))
    },
    strict="filter",
    coerce=True
)

def clean_data_on_the_fly(df: pd.DataFrame) -> pd.DataFrame:
    init_count = len(df)
    df = df.drop_duplicates()
    if len(df) < init_count:
        logger.warning(f" Deleted {init_count - len(df)} duplicate rows.")

    threshold = 0.4
    cols_to_drop = [c for c in df.columns if df[c].isna().mean() > threshold]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.warning(f" Dropped corrupted columns: {cols_to_drop}")

    rows_before_dropping = len(df)
    df = df.dropna()
    rows_after_dropping = len(df)
    dropped_count = rows_before_dropping - rows_after_dropping
    
    if dropped_count > 0:
        logger.warning(f" Dropped {dropped_count} rows containing missing values.")
    else:
        logger.info(" No missing values found in rows.")
            
    return df

def run_validation_pipeline(input_path: str, output_path: str):
    try:
        logger.info(f" Loading: {input_path}")
        df = pd.read_csv(input_path)
        df_cleaned = clean_data_on_the_fly(df)
        logger.info(" Validating Schema...")
        validated_df = hotel_schema.validate(df_cleaned)
        validated_df.to_csv(output_path, index=False)
        logger.info(f" Success! Clean data saved to: {output_path}")
        return validated_df
    except pa.errors.SchemaError as e:
        logger.error(f" Validation failed: {e}")
        raise CustomException(f" Validation failed: {e}")
    except Exception as e:
        logger.error(f" Unexpected error: {e}")
        raise CustomException(f" Unexpected error: {e}")
    
if __name__ == "__main__":
    run_validation_pipeline(
        input_path="data/processed/hotel_data.csv", 
        output_path="data/processed/validated_data.csv"
    )
