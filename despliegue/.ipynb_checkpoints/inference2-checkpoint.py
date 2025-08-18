import os
import joblib
import pandas as pd

def model_fn(model_dir):
    """Carga el modelo de scikit-learn."""
    model = joblib.load(os.path.join(model_dir, "modelRF.joblib"))
    return model

def predict_fn(input_data, model):
    """Realiza la predicción."""
    # Asume que input_data ya es un DataFrame de Pandas
    return model.predict(input_data)