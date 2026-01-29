import pandas as pd 
import os
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

def load_data(file_path = r"data\raw\Hotel Reservations.csv"):
    try:

        logger.info("start loading data")

        os.makedirs("data/processed",exist_ok=True)
        df = pd.read_csv(file_path)
        df.to_csv("data/processed/hotel_data.csv")

        logger.info(f"loading data successfully and saved in data/processed")

    except Exception as e :
        logger.error("failed loading data")
        raise CustomException("failed loading data", e)
    


if __name__ == "__main__":
    load_data()