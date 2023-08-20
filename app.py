import mlflow
import mlflow.pyfunc
import streamlit as st
import numpy as np
from pipelines.predicting_pipeline import predict

# Load the MLflow model
loaded_model = mlflow.pyfunc.load_model(model_uri="models:/model/Production")

# Stream lit App
st.title("Payment Prediction App")
st.markdown(
    """ 
    #### Problem Statement 
    The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. I will be using MLflow to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.
    """
    )
st.markdown(
    """ 
    We begin by ingesting data into our system. Subsequently, we train our model using this data, with the entire training process tracked and managed by MLflow. Once the model is trained and its activity is recorded by MLflow, we utilize this trained model for our specific tasks and applications.
    """
    )
st.markdown(
        """ 
    #### Description of Features 
    This app is designed to predict the customer satisfaction score for a given customer. You can input the features of the product listed below and get the customer satisfaction score. 
    | Models        | Description   | 
    | ------------- | -     | 
    | Payment Sequential | Customer may pay an order with more than one payment method. If he does so, a sequence will be created to accommodate all payments. | 
    | Payment Installments   | Number of installments chosen by the customer. |  
    | Payment Value |       Total amount paid by the customer. | 
    | Price |       Price of the product. |
    | Freight Value |    Freight value of the product.  | 
    | Product Name length |    Length of the product name. |
    | Product Description length |    Length of the product description. |
    | Product photos Quantity |    Number of product published photos |
    | Product weight measured in grams |    Weight of the product measured in grams. | 
    | Product length (CMs) |    Length of the product measured in centimeters. |
    | Product height (CMs) |    Height of the product measured in centimeters. |
    | Product width (CMs) |    Width of the product measured in centimeters. |
    """
    )

# Input Features
payment_sequential = st.sidebar.slider("Payment Sequential")
payment_installments = st.sidebar.slider("Payment Installments")
payment_value = st.number_input("Payment Value")
price = st.number_input("Price")
freight_value = st.number_input("freight_value")
product_name_length = st.number_input("Product name length")
product_description_length = st.number_input("Product Description length")
product_photos_qty = st.number_input("Product photos Quantity ")
product_weight_g = st.number_input("Product weight measured in grams")
product_length_cm = st.number_input("Product length (CMs)")
product_height_cm = st.number_input("Product height (CMs)")
product_width_cm = st.number_input("Product width (CMs)")

# Make Predictions
if st.button("Predict"):
    input_data = np.array([[
        payment_sequential,
        payment_installments,
        payment_value,
        price,
        freight_value,
        product_name_length,
        product_description_length,
        product_photos_qty,
        product_weight_g,
        product_length_cm,
        product_height_cm,
        product_width_cm
        ]], dtype=np.float32)

    prediction = loaded_model.predict(input_data)
    st.success(
        "Your Customer Satisfactory rate (range between 0 - 5) with given product details is: {}".format(
            prediction
        )
    )