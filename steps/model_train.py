import logging
import pandas as pd
from sklearn.base import RegressorMixin
from src.model_developement import LinearRegressionModel

def train_model(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame 
        ) -> RegressorMixin:
    """ 
    Trains the model on the ingested data
    Args:
        X_train: pd.DataFrame
        X_test: pd.DataFrame    
        y_train: pd.DataFrame
        y_test: pd.DataFrame 
    """ 
    try:
        model = LinearRegressionModel()
        trained_model = model.train(X_train,y_train)
        return trained_model
    except Exception as e:
        logging.error(f"Error while training model: {e}")
        raise e