# 🛒 Customer Satisfaction Prediction App

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit Version](https://img.shields.io/badge/Streamlit-1.25.0%2B-blue)
![MLflow Version](https://img.shields.io/badge/MLflow-2.6.0%2B-blue)

## 📖 Description

The Customer Satisfaction Prediction App is a Streamlit-based application that predicts customer satisfaction scores for orders based on various product and payment-related features. The app utilizes a machine learning model trained with MLflow to make predictions.

## 🚀 Features

- Predict customer satisfaction scores for orders.
- Interactive user interface using Streamlit.
- Load and use a pre-trained MLflow model.

## 🛠️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo

2. Create a virtual environment (recommended):
    
    ```bash
    conda create -p venv python==3.9 -y
    conda activate venv

3. Install the required packages:
    
    ```bash
    pip install -r requirements.txt

4. Run training_pipeline.py:

    ```bash
    python run_pipeline.py

5. Register model on mlflow:

    ```bash
    mlflow models register -m your_model_uri -n your_model_name

6. Run the Streamlit app:

    ```bash
    streamlit run streamlit_app.py
7. Dema and screenshot
##### Screenshot 1:
![App Screenshot](https://github.com/Anirudhrarao/Customer-Satisfaction-Prediction-App/blob/main/Demo/stream.png)
##### Screenshot 2:
![App Screenshot](https://github.com/Anirudhrarao/Customer-Satisfaction-Prediction-App/blob/main/Demo/mlflow.png)

## Authors

- [@Anirudhrarao](https://github.com/Anirudhrarao)


### Training Pipeline

Our standard training pipeline consists of several steps:

- `ingest_data`: This step will ingest the data and create a `DataFrame`.
- `clean_data`: This step will clean the data and remove the unwanted columns.
- `train_model`: This step will train the model and save the model using [MLflow autologging](https://www.mlflow.org/docs/latest/tracking.html).
- `evaluation`: This step will evaluate the model and save the metrics -- using MLflow autologging -- into the artifact store.

### Training & Predicting Pipeline
We have pipeline for training model and after training we can do prediction using predicting pipeline.
