import os
import sys
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # ../.. от app.py
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))  # добавляем src в PYTHONPATH

from config import DROP_COLS, OHE_COLS, ORD_COLS  # type: ignore
from modeling.datapreprocessor import DataPreProcessor  # type: ignore

app = FastAPI()

PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "train_preprocessor.pkl"

RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = PROJECT_ROOT / "models"
model_path = MODELS_DIR / "heart_pred.pkl"

if not model_path.exists():
    raise FileNotFoundError(f"Не найден файл модели: {model_path}")

MODEL = model_path
model_bundle = joblib.load(MODEL)
model = model_bundle["model"]
threshold = model_bundle["threshold"]
selected_features = model_bundle.get("selected_features", None)

preprocessor_pipeline = joblib.load(PREPROCESSOR_PATH)
preprocessor = DataPreProcessor(
    drop_cols=DROP_COLS, ohe_cols=OHE_COLS, ord_cols=ORD_COLS, name="train"
)
preprocessor.pipeline = preprocessor_pipeline


@app.get("/health")
def health():
    return {"status": "OK"}


app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")


@app.get("/")
async def root():
    return JSONResponse(
        content={"message": "Перейдите на http://localhost:8000/static/index.html"}
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))

        if "id" not in df.columns:
            return JSONResponse(
                status_code=400, content={"error": "CSV должен содержать колонку 'id'"}
            )

        # --- препроцессинг ---
        df_processed = preprocessor.transform(df)
        alerts = []

        if selected_features:
            processed_cols = set(df_processed.columns) - {"id"}
            expected_cols = set(selected_features)

            extra_feats = sorted(processed_cols - expected_cols)
            if extra_feats:
                alerts.append(
                    f"⚠️ В файле есть лишние признаки: {extra_feats}. Они будут проигнорированы."
                )
                df_processed = df_processed.drop(columns=extra_feats, errors="ignore")
                processed_cols = set(df_processed.columns) - {"id"}

            missing_feats = sorted(expected_cols - processed_cols)
            if missing_feats:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "❌ Не хватает признаков после препроцессинга",
                        "missing": missing_feats,
                    },
                )

            X = df_processed.loc[:, selected_features]
        else:
            X = df_processed.drop(columns=["id"], errors="ignore")

        # --- предсказание ---
        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= threshold).astype(int)

        result_df = pd.DataFrame(
            {"id": df["id"], "prediction": preds, "probability": proba}
        )

        # словари для фронта (только если строк ≤1000)
        n_rows = len(result_df)
        if n_rows <= 1000:
            result_json = {
                str(i): int(p) for i, p in zip(result_df["id"], result_df["prediction"])
            }
            prob_json = {
                str(i): float(pr)
                for i, pr in zip(result_df["id"], result_df["probability"])
            }
        else:
            result_json = None
            prob_json = None

        # --- сохраняем файлы ---
        orig_name = Path(file.filename).stem  # type: ignore
        csv_path = RESULTS_DIR / f"{orig_name}_pred_with_proba.csv"
        json_path = RESULTS_DIR / f"{orig_name}_pred_with_proba.json"
        result_df.to_csv(csv_path, index=False)
        result_df.to_json(json_path, orient="records", force_ascii=False)

        # --- ответ ---
        resp = {
            "status": "success",
            "alerts": alerts or None,
            "summary": {
                "total_rows": n_rows,
                "positive": int(preds.sum()),
                "negative": int((preds == 0).sum()),
            },
            "download_csv": f"/results/{csv_path.name}",
            "download_json": f"/results/{json_path.name}",
        }

        # только если строк мало — добавляем таблицу
        if n_rows <= 1000:
            resp["predictions"] = result_json
            resp["probabilities"] = prob_json
        else:
            resp["preview_note"] = (
                "⚠️ Файл содержит более 1000 строк. "
                "Полный результат можно скачать по ссылкам."
            )

        return resp

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"status": "failed", "error": str(e)}
        )
