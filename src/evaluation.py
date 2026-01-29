from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_predict
import pandas as pd 
from src.logger import get_logger
from src.custom_exception import CustomException
import joblib
import plotly.express as px
import os
import mlflow
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)




logger = get_logger(__name__)

def evaluate(x, y, pipe_line, run_id= None):
    
    try:
        y_pred = cross_val_predict(pipe_line,
                                x,
                                y,
                                cv=cv,
                                n_jobs=-1
                                )
        pd.DataFrame({"y_pred": y_pred}).to_csv("data/processed/y_pred.csv", index=False)

        y_proba = cross_val_predict(pipe_line,
                                x,
                                y,
                                cv=cv,
                                n_jobs=-1,
                                method='predict_proba'
                                )
        pd.DataFrame(y_proba, columns=["proba_0", "proba_1"]).to_csv("data/processed/y_proba.csv", index=False)
        

        os.makedirs("artifact", exist_ok=True)

        # confusion_matrix
        cm = confusion_matrix(y, y_pred)
        cm_fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale='Blues',
        labels=dict(x="Predicted", y="Actual", color="Count")
        )

        cm_fig.write_image("artifact/conf_matrix.png")


        # roc_curve
        fpr, tpr, _ = roc_curve(y, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        fig = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC={roc_auc:.2f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate')
                )
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)

        fig.write_image("artifact/roc_auc.png")
        

        if run_id is not None:
            # Assuming run is already active or we just log to the active run
            mlflow.log_artifact("artifact/conf_matrix.png")
            mlflow.log_artifact("artifact/roc_auc.png")
        else:
            logger.warning("MLflow has no active run! Cannot log artifacts!")

        logger.info("evaluate model done!")
        
        
        return y_pred, y_proba
    
    except Exception as e :
        logger.error(f"evaluate model failed! Error: {str(e)}")
        raise CustomException(f"evaluate model failed!", e)
    
if __name__ == "__main__":

    x = pd.read_csv("data/processed/x.csv")

    y = pd.read_csv("data/processed/y.csv").values.ravel()

    model = joblib.load("models/xgb_model.pkl")

    y_pred, y_proba = evaluate(x, y, model)
