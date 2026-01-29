from src.logger import get_logger
from src.custom_exception import CustomException
from sklearn.preprocessing import OneHotEncoder,StandardScaler,RobustScaler
from sklearn.compose import ColumnTransformer
from src.utils import load_config
from imblearn.pipeline import Pipeline
import joblib
import os


logger = get_logger(__name__)



def preprocessor(config):
    try:

        
        # standard scaler 
        st_pipe = Pipeline(steps=[
                        ("st_scaler",StandardScaler())
                            ])
        logger.info("standard scaler pipeline done!")

        # Robust scaler
        ro_pipe = Pipeline(steps=[
                            ("ro_scaler",RobustScaler()) 
                            ])
        logger.info("Robust scaler pipeline done!")

        
        # ohe pipeline
        ohe_pipe = Pipeline(steps=[
            ("ohe_encoder",OneHotEncoder(drop="first",sparse_output=False))
                            ])
        logger.info("ohe encoder pipeline done!")



        preprocessing_pipeline = ColumnTransformer(transformers=[
            ("st_pipe",st_pipe,config["preprocessing"]["st_cols"]),
            ("ro_pipe",ro_pipe,config["preprocessing"]["ro_cols"]),
            ("ohe_pipe",ohe_pipe,config["preprocessing"]["ohe_cols"]),
            ("ready_pipe", "passthrough", config["preprocessing"]["ready_cols"])
            ])
        
        logger.info("All pipeline done!")

        os.makedirs("models", exist_ok=True)

        joblib.dump(preprocessing_pipeline, "models/prepipeline.pkl")

        return preprocessing_pipeline
    
    except Exception as e :
        logger.error("error while creating pipeline")
        raise CustomException(f"error while creating pipeline", e)


if __name__ == "__main__":
    config = load_config()
    preprocessor(config)




