import glob
import os
import sys
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # fastapi/.. -> 9) Masterskaya
sys.path.append(str(PROJECT_ROOT))  # Добавляем src в путь

from src.config import DROP_COLS, OHE_COLS, ORD_COLS, TARGET_COL  # type: ignore
from src.modeling.datapreprocessor import DataPreProcessor  # type: ignore

app = FastAPI()


PREPROCESSOR_PATH = "../models/train_preprocessor.pkl"
RESULTS_DIR = Path("../data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = Path("../models")
model_files = sorted(
    MODELS_DIR.glob("heart_pred_*.pkl"), key=os.path.getmtime, reverse=True
)
if not model_files:
    raise FileNotFoundError(
        "Не найдено ни одной модели 'heart_pred_*.pkl' в папке models"
    )

MODEL = model_files[0]
model_bundle = joblib.load(MODEL)
model = model_bundle["model"]
threshold = model_bundle["threshold"]
selected_features = model_bundle.get("selected_features", None)

preprocessor_pipeline = joblib.load(PREPROCESSOR_PATH)
preprocessor = DataPreProcessor(
    drop_cols=DROP_COLS, ohe_cols=OHE_COLS, ord_cols=ORD_COLS, name="train"
)
preprocessor.pipeline = preprocessor_pipeline

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

        df_processed = preprocessor.transform(df)

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

        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= threshold).astype(int)

        result_df = pd.DataFrame(
            {"id": df["id"], "prediction": preds, "probability": proba}
        )

        result_json = {
            str(row["id"]): int(row["prediction"]) for _, row in result_df.iterrows()
        }
        prob_json = {
            str(row["id"]): float(row["probability"]) for _, row in result_df.iterrows()
        }

        # Сохраняем CSV и JSON
        orig_name = Path(file.filename).stem  # type: ignore
        csv_path = RESULTS_DIR / f"{orig_name}_pred.csv"
        json_path = RESULTS_DIR / f"{orig_name}_pred.json"

        result_df.to_csv(csv_path, index=False)
        result_df.to_json(json_path, orient="records", force_ascii=False)

        return {
            "status": "success",
            "message": "Файл с предсказаниями сохранён.",
            "download_csv": f"http://localhost:8000/results/{csv_path.name}",
            "download_json": f"http://localhost:8000/results/{json_path.name}",
            "summary": {
                "total_rows": len(result_df),
                "positive": int(preds.sum()),
                "negative": int((preds == 0).sum()),
            },
            "predictions": result_json,
            "probabilities": prob_json,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"status": "failed", "error": str(e)}
        )
