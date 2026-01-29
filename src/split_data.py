from src.logger import get_logger
from src.custom_exception import CustomException
from sklearn.preprocessing import LabelEncoder
import pandas as pd


logger = get_logger(__name__)


def features_target_split(file_path: str = r"data/processed/feature_engineered_data.csv"):
    try:

        logger.info("loading final data!")        
        df = pd.read_csv(file_path)

        logger.info("spliting data")
        x = df.drop(columns=["booking_status","Booking_ID"])
        y = df["booking_status"]
        logger.info("spliting data done!")


        logger.info("encoding label")
        le =LabelEncoder()
        y_encoded = le.fit_transform(y)
        logger.info("encoding label done!")

        
        x.to_csv("data/processed/x.csv", index=False)
        pd.DataFrame({"target": y_encoded}).to_csv("data/processed/y.csv", index=False)

        logger.info("X.csv and y.csv saved successfully.")

    except Exception as e:
        logger.error("spliting data failed!")
        raise CustomException(f"spliting data failed!", e)
    


if __name__ == "__main__":
    features_target_split()
