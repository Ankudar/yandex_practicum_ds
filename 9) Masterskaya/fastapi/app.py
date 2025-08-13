import sys

sys.path.append("../src/modeling/")

from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
from datapreprocessor import DataPreProcessor
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

MODEL_PATH = "../models/heart_pred.pkl"
PREPROCESSOR_PATH = "../models/train_preprocessor.pkl"
RESULTS_DIR = Path("../data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

model_bundle = joblib.load(MODEL_PATH)
model = model_bundle["model"]
threshold = model_bundle["threshold"]
selected_features = model_bundle.get("selected_features", None)

preprocessor_pipeline = joblib.load(PREPROCESSOR_PATH)

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

        drop_cols = ["income", "ck-mb", "troponin"]
        ohe_cols = ["gender"]
        preprocessor = DataPreProcessor(drop_cols=drop_cols, ohe_cols=ohe_cols)
        preprocessor.pipeline = preprocessor_pipeline
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
        result_df = pd.DataFrame({"id": df["id"], "prediction": preds})
        result_json = {
            str(row["id"]): int(row["prediction"]) for _, row in result_df.iterrows()
        }

        # Сохраняем CSV и JSON
        orig_name = Path(file.filename).stem
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
        }

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"status": "failed", "error": str(e)}
        )
