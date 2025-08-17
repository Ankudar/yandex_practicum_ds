import sys
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# --- Пути ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # ../.. от app.py
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))  # добавляем src в PYTHONPATH

from config import DROP_COLS, OHE_COLS, ORD_COLS  # type: ignore
from modeling.datapreprocessor import DataPreProcessor  # type: ignore


# --- Класс Predictor для FastAPI ---
class Predictor:
    def __init__(self, model_path, preprocessor_path):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Не найден файл модели: {self.model_path}")
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(f"Не найден препроцессор: {self.preprocessor_path}")

        self.model_bundle = joblib.load(self.model_path)
        self.model = self.model_bundle["model"]
        self.threshold = self.model_bundle["threshold"]
        self.selected_features = self.model_bundle.get("selected_features", None)

        self.preprocessor = DataPreProcessor(
            drop_cols=DROP_COLS, ohe_cols=OHE_COLS, ord_cols=ORD_COLS, name="train"
        )
        self.preprocessor.pipeline = joblib.load(self.preprocessor_path)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.preprocessor.clean_column_names(df)
        df_proc = self.preprocessor.transform(df)
        if self.selected_features:
            missing_feats = [
                f for f in self.selected_features if f not in df_proc.columns
            ]
            if missing_feats:
                raise ValueError(
                    f"Отсутствуют признаки после препроцессинга: {missing_feats}"
                )
            df_proc = df_proc[
                [f for f in self.selected_features if f in df_proc.columns]
            ]
        return df_proc

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df_proc = self.preprocess(df)
        proba = self.model.predict_proba(df_proc)[:, 1]
        preds = (proba >= self.threshold).astype(int)
        return pd.DataFrame({"id": df["id"], "prediction": preds, "probability": proba})


# --- Инициализация FastAPI ---
app = FastAPI()
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = PROJECT_ROOT / "models"
predictor = Predictor(
    model_path=MODELS_DIR / "heart_pred.pkl",
    preprocessor_path=MODELS_DIR / "train_preprocessor.pkl",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")


@app.get("/health")
def health():
    return {"status": "OK"}


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

        result_df = predictor.predict(df)

        n_rows = len(result_df)
        alerts = []
        if predictor.selected_features:
            extra_feats = set(result_df.columns) - {"id", "prediction", "probability"}
            if extra_feats:
                alerts.append(f"⚠️ В файле есть лишние признаки: {sorted(extra_feats)}")

        # --- Сохраняем файлы ---
        orig_name = Path(file.filename).stem  # type: ignore
        csv_path = RESULTS_DIR / f"{orig_name}_pred_with_proba.csv"
        json_path = RESULTS_DIR / f"{orig_name}_pred_with_proba.json"
        result_df.to_csv(csv_path, index=False)
        result_df.to_json(json_path, orient="records", force_ascii=False)

        resp = {
            "status": "success",
            "alerts": alerts or None,
            "summary": {
                "total_rows": n_rows,
                "positive": int(result_df["prediction"].sum()),
                "negative": int((result_df["prediction"] == 0).sum()),
            },
            "download_csv": f"/results/{csv_path.name}",
            "download_json": f"/results/{json_path.name}",
        }

        if n_rows <= 1000:
            resp["predictions"] = {
                str(i): int(p) for i, p in zip(result_df["id"], result_df["prediction"])
            }
            resp["probabilities"] = {
                str(i): float(p)
                for i, p in zip(result_df["id"], result_df["probability"])
            }
        else:
            resp["preview_note"] = (
                "⚠️ Файл содержит более 1000 строк. Полный результат можно скачать по ссылкам."
            )

        return resp

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"status": "failed", "error": str(e)}
        )
