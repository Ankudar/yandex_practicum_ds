import sys

sys.path.append("../src/modeling/")
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
from datapreprocessor import DataPreProcessor
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Пути и директории
MODEL_PATH = "../models/heart_pred.pkl"
PREPROCESSOR_PATH = "../models/train_preprocessor.pkl"
RESULTS_DIR = Path("../data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Загрузка модели и препроцессора
model_bundle = joblib.load(MODEL_PATH)
model = model_bundle["model"]
threshold = model_bundle["threshold"]
selected_features = model_bundle.get("selected_features", None)

preprocessor = joblib.load(PREPROCESSOR_PATH)

# Монтируем директорию для отдачи сохранённых файлов
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))

        if "id" not in df.columns:
            return JSONResponse(
                status_code=400,
                content={"error": "CSV должен содержать колонку 'id'"},
            )

        # Инициализация препроцессора с параметрами
        drop_cols = ["income", "ck-mb", "troponin"]
        ohe_cols = ["gender"]
        preprocessor = DataPreProcessor(drop_cols=drop_cols, ohe_cols=ohe_cols)

        # Загружаем сохранённый pipeline в препроцессор
        preprocessor.pipeline = joblib.load(PREPROCESSOR_PATH)

        # Полная предобработка
        df_processed = preprocessor.transform(df)

        # Оставляем нужные признаки
        if selected_features:
            missing_feats = [
                f for f in selected_features if f not in df_processed.columns
            ]
            if missing_feats:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"В обработанных данных отсутствуют признаки: {missing_feats}"
                    },
                )
            X = df_processed[selected_features]
        else:
            X = df_processed.drop(columns=["id"], errors="ignore")

        # Предсказания
        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= threshold).astype(int)

        result_df = pd.DataFrame({"id": df["id"], "prediction": preds})

        # Сохраняем результат с суффиксом _pred.csv
        orig_name = Path(file.filename).stem
        save_path = RESULTS_DIR / f"{orig_name}_pred.csv"
        result_df.to_csv(save_path, index=False)

        # Формируем JSON для ответа
        result_json = {
            str(row["id"]): int(row["prediction"]) for _, row in result_df.iterrows()
        }

        return {
            "message": "Файл с предсказаниями сохранён. По ссылке можно скачать CSV.",
            "download_url": f"http://localhost:8000/results/{save_path.name}",
            "predictions": result_json,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
