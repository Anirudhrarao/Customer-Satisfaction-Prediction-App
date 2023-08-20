import mlflow

from steps.data_ingest import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model

def train_pipeline(data_path: str):
    # Start an MLflow run
    with mlflow.start_run():
        # Log the data ingestion step
        mlflow.log_param("data_path", data_path)
        mlflow.log_artifact(data_path,"dataset")
        # Ingest and clean data
        df = ingest_data(data_path=data_path)
        X_train, X_test, y_train, y_test = clean_data(df)
        
        # Log data shapes as metrics
        mlflow.log_metric("train_data_shape", X_train.shape[0])
        mlflow.log_metric("test_data_shape", X_test.shape[0])
        
        # Train the model
        model = train_model(X_train, X_test, y_train, y_test)
        
        # Evaluate the model
        r2_score, rmse = evaluate_model(model, X_test, y_test)
        
        # Log evaluation metrics
        mlflow.log_metric("r2_score", r2_score)
        mlflow.log_metric("rmse", rmse)
        
        # Log the model artifact
        mlflow.sklearn.log_model(model, "model")
        
