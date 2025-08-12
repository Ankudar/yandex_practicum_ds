from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

# Пути к моделям
MODEL_PATH = "../models/heart_pred.pkl"
PREPROCESSOR_PATH = "../models/preprocessor.pkl"

# Загружаем модель и препроцессор отдельно
model_bundle = joblib.load(MODEL_PATH)
model = model_bundle["model"]  # XGBClassifier
threshold = model_bundle["threshold"]
selected_features = model_bundle["selected_features"]

preprocessor = joblib.load(PREPROCESSOR_PATH)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(BytesIO(contents))
    print(df.columns.tolist())

    if "id" not in df.columns:
        return {"error": "CSV должен содержать колонку 'id'"}

    # Оставляем только нужные фичи
    X = df[selected_features]

    # Предобработка отдельно
    X_processed = preprocessor.transform(X)

    # Предсказания с учётом порога
    proba = model.predict_proba(X_processed)[:, 1]
    preds = (proba >= threshold).astype(int)

    result = pd.DataFrame({"id": df["id"], "prediction": preds})

    return result.to_dict(orient="records")
