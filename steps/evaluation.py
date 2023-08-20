import logging
import pandas as pd 
import mlflow
from typing import Annotated,Tuple
from sklearn.base import RegressorMixin
from src.evaluation import MSE, RMSE, R2


def evaluate_model(model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame) -> Tuple[
        Annotated[float, "r2_score"],
        Annotated[float, "rmse"]
    ]:
    """ 
    Evaluate your trained model on ingested data
    Args:
        df: pd.DataFrame
    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("mse",mse)
        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("r2",r2)
        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("rmse",rmse)
        return r2, rmse
    except Exception as e:
        logging.info(f"Error while evaluating model: {e}")