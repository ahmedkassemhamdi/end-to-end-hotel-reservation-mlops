from xgboost import XGBClassifier
from src.logger import get_logger
from src.custom_exception import CustomException
import joblib
import mlflow
from sklearn.metrics import make_scorer,f1_score,accuracy_score,precision_score,recall_score,confusion_matrix
from sklearn.model_selection import cross_validate,cross_val_predict,StratifiedKFold
from imblearn.over_sampling import SMOTE
from src.utils import load_config
from imblearn.pipeline import Pipeline
import os
import pandas as pd
from src.preprocessing import preprocessor
from src.evaluation import evaluate


logger = get_logger(__name__)


scoring = {
            "F1" : make_scorer(f1_score),
            "Accuracy" : make_scorer(accuracy_score),
            "Recall" : make_scorer(recall_score),
            "Precision" : make_scorer(precision_score)
        }

config = load_config()

xgb_params = config["xgb_params"]

model = XGBClassifier(**xgb_params)


def xgb_model(x, y, pre_pipeline, with_smote:bool = False):
    
    try:
        

        mlflow.set_experiment("hotel_reservation")
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            if with_smote:
                pipe_line = Pipeline([
                                    ("pre_pipeline", pre_pipeline),
                                    ("smote", SMOTE()),
                                    ("model", model)
                                    ])
                
            else:
                pipe_line = Pipeline([
                                    ("pre_pipeline", pre_pipeline),
                                    ("model", model)
                                    ])
            


            result = cross_validate(pipe_line,
                                    x,
                                    y
                                    ,scoring=scoring,
                                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                    n_jobs=-1,
                                    return_train_score=True
                                    )


            


            mlflow.log_params({
                                **xgb_params,
                            "cv" : 5
                            })
            
            pipe_line.fit(x, y)

            os.makedirs("models", exist_ok=True)
            joblib.dump(pipe_line, "models/xgb_model.pkl")

            logger.info("model trained successfully!")
            logger.info("model saved in models/")




            mlflow.sklearn.log_model(pipe_line,
                                    name="xgb_model",
                                    input_example=x.iloc[:5])
            


            model_results = {
                        'Model_name': "xgb_model",
                        'Train_F1': result['train_F1'].mean().round(3),
                        'Test_F1': result['test_F1'].mean().round(3),
                        'Train_Recall': result['train_Recall'].mean().round(3),
                        'Test_Recall': result['test_Recall'].mean().round(3),
                        'Train_Precision': result['train_Precision'].mean().round(3),
                        'Test_Precision': result['test_Precision'].mean().round(3),
                        'Train_Accuracy': result['train_Accuracy'].mean().round(3),
                        'Test_Accuracy': result['test_Accuracy'].mean().round(3),
                        'Fit_Time': result['fit_time'].sum().round(2)
                    }


            for key, value in model_results.items():
                logger.info(f"{key} : {value}")



            mlflow.log_metrics({
                    'train_f1': model_results['Train_F1'],
                    'test_f1': model_results['Test_F1'],
                    'train_recall': model_results['Train_Recall'],
                    'test_recall': model_results['Test_Recall'],
                    'train_precision': model_results['Train_Precision'],
                    'test_precision': model_results['Test_Precision'],
                    'train_accuracy': model_results['Train_Accuracy'],
                    'test_accuracy': model_results['Test_Accuracy'],
                    'total_fit_time': model_results['Fit_Time']
                })
            
            y_pred, y_proba = evaluate(x, y, pipe_line, run_id)

    except Exception as e:
        logger.error("error while training model")
        raise CustomException(f"error while training model", e)
    





if __name__ == "__main__":

    # load splited data
    x = pd.read_csv("data/processed/x.csv")
    y = pd.read_csv("data/processed/y.csv").values.ravel()

    # preprocessing pipeline
    config = load_config()
    pre_pipeline = preprocessor(config)

    # train the model
    xgb_model(x, y, pre_pipeline)
    logger.info("training model done!")
