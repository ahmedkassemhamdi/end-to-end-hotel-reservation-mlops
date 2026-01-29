from src.logger import get_logger
from src.custom_exception import CustomException
import pandas as pd


logger = get_logger(__name__)

def feature_engineering(file_path:str = "data/processed/validated_data.csv"):
    try:

        logger.info("loading processed data")
        df = pd.read_csv(file_path)
        logger.info("data loaded successfully")

        logger.info("start making features")

        df["total_guests"] = df["no_of_adults"] + df["no_of_children"]
        df["total_nights"] = df["no_of_weekend_nights"] + df["no_of_week_nights"]
        df.loc[df['total_nights'] == 0, 'total_nights'] = 1
        df['price_per_night'] = df['avg_price_per_room'] / df['total_nights']
        df['is_weekend_only'] = (df['total_nights'] == df['no_of_weekend_nights']).astype(int)

        logger.info("feature engineering done!")





        df.to_csv("data/processed/feature_engineered_data.csv", index=False)
        logger.info("final data saved in data/processed")
        

    except Exception as e :
        logger.error("feature engineering failed!")
        raise CustomException(f"feature engineering failed!", e)





if __name__ == "__main__":

    feature_engineering()