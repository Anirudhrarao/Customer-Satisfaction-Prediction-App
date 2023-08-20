import mlflow
import logging
import numpy as np 
import mlflow.pyfunc



def predict(
    payment_sequential: int,
    payment_installments: int,
    payment_value: float,
    price: float,
    freight_value: float,
    product_name_length: int,
    product_description_length: int,
    product_photos_qty: int,
    product_weight_g: float,
    product_length_cm: float,
    product_height_cm: float,
    product_width_cm: float
) -> float:
    """
    Predict payment using the loaded MLflow model.

    Args:
    - payment_sequential (int): Payment Sequential
    - payment_installments (int): Payment Installments
    - payment_value (float): Payment Value
    - price (float): Price
    - freight_value (float): Freight Value
    - product_name_length (int): Product Name Length
    - product_description_length (int): Product Description Length
    - product_photos_qty (int): Product Photos Quantity
    - product_weight_g (float): Product Weight (g)
    - product_length_cm (float): Product Length (cm)
    - product_height_cm (float): Product Height (cm)
    - product_width_cm (float): Product Width (cm)

    Returns:
    - prediction (float): Predicted Payment
    """
    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri="models:/model/Production")
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
        return prediction
    except Exception as e:
        logging.error(f"Error while predicting: {e}")