# inference.py
import os
import json
import joblib
import pandas as pd
import numpy as np # Importar numpy

# --------- Columnas esperadas según entrenamiento ----------
EXPECTED_COLUMNS = [
    "valor_transaccion",
    "tipo_transaccion",
    "id_aliado",
    "genero",
    "estrato_social",
    "estado_civil",
    "ocupacion",
    "ciudad",
]

# --------- Carga del pipeline ----------
def model_fn(model_dir: str):
    """Carga el pipeline completo (preprocessor + modelo)."""
    model_path = os.path.join(model_dir, "rf_pipeline.joblib")
    return joblib.load(model_path)

# --------- Parseo de entrada ----------
def input_fn(request_body, request_content_type: str):
    """Convierte el JSON a DataFrame y asegura columnas y tipos."""
    if request_content_type != "application/json":
        raise ValueError(f"Content-Type no soportado: {request_content_type}")

    data = json.loads(request_body)

    # Normalizar la entrada a lista de dicts
    if isinstance(data, dict) and "instances" in data:
        instances = data["instances"]
    elif isinstance(data, dict):
        instances = [data]
    elif isinstance(data, list):
        instances = data
    else:
        raise ValueError("Formato JSON no reconocido.")

    df = pd.DataFrame(instances)

    # Validar columnas
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas en el request: {missing}. Recibidas: {list(df.columns)}")

    # Reordenar y asegurar que todas las columnas esperadas estén presentes, incluso si están vacías en el payload
    # Esto es crucial para que el DataFrame tenga la estructura correcta para el pipeline.
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            # Agrega la columna si falta, con NaN si es numérico o None si es categórico
            if col == "valor_transaccion": # Ejemplo, si tienes más columnas numéricas
                df[col] = np.nan
            else:
                df[col] = None # O np.nan, dependiendo de cómo manejes los nulos en tu preprocesador


    df = df[EXPECTED_COLUMNS] # Asegurarse del orden correcto

    # Forzar tipos con manejo de errores o conversiones más robustas si es necesario.
    # Aquí es donde podría estar la clave del problema.
    try:
        df["valor_transaccion"] = df["valor_transaccion"].astype(float)
    except ValueError as e:
        raise ValueError(f"Error al convertir 'valor_transaccion' a float: {e}")

    categorical_cols = ["tipo_transaccion", "id_aliado", "genero", "estrato_social",
                        "estado_civil", "ocupacion", "ciudad"]
    for cat_col in categorical_cols:
        try:
            # Para id_aliado y estrato_social, si son numéricos en la entrada,
            # asegúrate de convertirlos a string de manera que tu preprocesador los maneje bien.
            # Por ejemplo, si tu preprocesador espera "21" y no 21 para 'id_aliado'
            # o "6" en lugar de 6 para 'estrato_social'.
            df[cat_col] = df[cat_col].astype(str)
        except Exception as e:
            raise ValueError(f"Error al convertir '{cat_col}' a string: {e}")

    # Asegúrate de que no haya NaN's o None's inesperados si tu modelo no los maneja bien.
    # Considera una estrategia para imputar o manejar valores faltantes aquí si es necesario.
    # Por ejemplo: df.fillna(value_to_impute, inplace=True)
    
    return df

# --------- Predicción ----------
def predict_fn(input_data: pd.DataFrame, model):
    """Ejecuta predicción usando el pipeline y devuelve resultados serializables."""
    # Asegúrate de que el input_data que llega aquí tenga los tipos y formato esperados.
    # A veces, el problema es que el preprocesador dentro del pipeline recibe un tipo inesperado
    # incluso después de pasar por input_fn.
    preds = model.predict(input_data)
    
    output = {"predictions": preds.tolist()}

    # Si hay probabilidad (clasificación)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)
        # Asegúrate de que las probabilidades también se conviertan a una lista de Python para JSON
        output["probabilities"] = probabilities.tolist()

    return output

# --------- Serialización ----------
def output_fn(prediction_output, response_content_type: str):
    """Convierte la salida a JSON."""
    if response_content_type != "application/json":
        raise ValueError(f"response_content_type no soportado: {response_content_type}")
    
    # Asegúrate de que prediction_output contiene solo tipos que json.dumps pueda manejar.
    # Los diccionarios y listas de Python son seguros.
    return json.dumps(prediction_output)