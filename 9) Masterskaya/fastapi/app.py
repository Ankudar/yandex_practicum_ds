from io import BytesIO

import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile

app = FastAPI()


def load_model_bundle(model_path):
    """Загружаем модельный бандл и достаём pipeline, threshold, features."""
    model_bundle = joblib.load(model_path)

    pipeline = model_bundle["model"]  # Это твой Pipeline
    threshold = model_bundle["threshold"]
    selected_features = model_bundle["selected_features"]

    # Достаём preprocessor из пайплайна по имени шага
    if "preprocessor" in pipeline.named_steps:
        preprocessor = pipeline.named_steps["preprocessor"]
    else:
        raise KeyError(
            f"В pipeline нет шага 'preprocessor'. "
            f"Доступные шаги: {list(pipeline.named_steps.keys())}"
        )

    return preprocessor, pipeline, threshold, selected_features


# Загружаем сразу всё из одной модели
preprocessor, model_pipeline, model_threshold, model_features = load_model_bundle(
    "../models/heart_pred.pkl"
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Читаем CSV
    contents = await file.read()
    df = pd.read_csv(BytesIO(contents))

    if "id" not in df.columns:
        return {"error": "CSV должен содержать колонку 'id'"}

    # Оставляем только нужные фичи
    X = df[model_features]

    # Предобработка через pipeline
    X_processed = preprocessor.transform(X)

    # Предсказания с учётом порога
    proba = model_pipeline.predict_proba(X_processed)[:, 1]
    preds = (proba >= model_threshold).astype(int)

    # Формируем ответ
    result = pd.DataFrame({"id": df["id"], "prediction": preds})

    return result.to_dict(orient="records")
