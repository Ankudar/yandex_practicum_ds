import math
import sys
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Annotated, Optional

import joblib
import pandas as pd
from fastapi import Body, FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import DROP_COLS, OHE_COLS, ORD_COLS  # type: ignore
from modeling.datapreprocessor import DataPreProcessor  # type: ignore

app = FastAPI()
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = PROJECT_ROOT / "models"


class GenderEnum(str, Enum):
    male = "Male"
    female = "Female"
    zero = "0.0"  # по хорошему нельзя так делать, надо писать ошибку, чтобы меняли данные на нормальные
    one = "1.0"


# рабочий код сделан под нашу текущую тестовую базу, в комментариях указано как надо делать по правильному (наверное :) )
class PatientData(BaseModel):
    id: Annotated[
        int, Field(ge=0, description="id всегда целое, неотрицательное")
    ]  # id всегда целое, неотрицательное

    Age: Optional[
        Annotated[
            float,
            Field(
                ge=0,
                le=1,
                description="Возраст пациента (нормализованное значение 0-1)",
            ),
        ]
    ] = None
    # Должно быть так: Optional[conint(ge=0, le=120)]

    Cholesterol: Optional[
        Annotated[
            float,
            Field(ge=0, le=1, description="Холестерин (нормализованное значение)"),
        ]
    ] = None
    # Должно быть так: Optional[confloat(ge=50, le=500)]

    Heart_rate: Optional[
        Annotated[
            float,
            Field(
                ge=0,
                le=1,
                description="Частота сердечных сокращений (нормализованное значение)",
            ),
        ]
    ] = None
    # Должно быть так: Optional[conint(ge=30, le=250)]

    Diabetes: Optional[
        Annotated[float, Field(ge=0, le=1, description="Диабет (0/1)")]
    ] = None
    # Должно быть так: Optional[conint(ge=0, le=1)]

    Family_History: Optional[
        Annotated[float, Field(ge=0, le=1, description="Семейная история (0/1)")]
    ] = None
    # Должно быть так: Optional[conint(ge=0, le=1)]

    Smoking: Optional[
        Annotated[float, Field(ge=0, le=1, description="Курение (0/1)")]
    ] = None
    # Должно быть так: Optional[conint(ge=0, le=1)]

    Obesity: Optional[
        Annotated[float, Field(ge=0, le=1, description="Ожирение (0/1)")]
    ] = None
    # Должно быть так: Optional[conint(ge=0, le=1)]

    Alcohol_Consumption: Optional[
        Annotated[float, Field(ge=0, le=1, description="Употребление алкоголя (0/1)")]
    ] = None
    # Должно быть так: Optional[conint(ge=0, le=1)]

    Exercise_Hours_Per_Week: Optional[
        Annotated[
            float,
            Field(ge=0, le=1, description="Физ. нагрузка (нормализованное значение)"),
        ]
    ] = None
    # Должно быть так: Optional[confloat(ge=0, le=50)]

    Diet: Optional[
        Annotated[
            float, Field(ge=0, le=15, description="Диета (нормализованное значение)")
        ]
    ] = None
    # Должно быть так: Optional[conint(ge=0, le=16)]

    Previous_Heart_Problems: Optional[
        Annotated[
            float, Field(ge=0, le=1, description="Предыдущие проблемы с сердцем (0/1)")
        ]
    ] = None
    # Должно быть так: Optional[conint(ge=0, le=1)]

    Medication_Use: Optional[
        Annotated[float, Field(ge=0, le=1, description="Приём лекарств (0/1)")]
    ] = None
    # Должно быть так: Optional[conint(ge=0, le=1)]

    Stress_Level: Optional[
        Annotated[
            float,
            Field(ge=0, le=1, description="Уровень стресса (нормализованное значение)"),
        ]
    ] = None
    # Должно быть так: Optional[confloat(ge=0, le=10)]

    Sedentary_Hours_Per_Day: Optional[
        Annotated[
            float,
            Field(
                ge=0, le=1, description="Сидячие часы в день (нормализованное значение)"
            ),
        ]
    ] = None
    # Должно быть так: Optional[confloat(ge=0, le=24)]

    Income: Optional[
        Annotated[
            float, Field(ge=0, le=1, description="Доход (нормализованное значение)")
        ]
    ] = None
    # Должно быть так: Optional[confloat(ge=0)]

    BMI: Optional[
        Annotated[
            float,
            Field(
                ge=0, le=1, description="Индекс массы тела (нормализованное значение)"
            ),
        ]
    ] = None
    # Должно быть так: Optional[confloat(ge=10, le=70)]

    Triglycerides: Optional[
        Annotated[
            float,
            Field(ge=0, le=1, description="Триглицериды (нормализованное значение)"),
        ]
    ] = None
    # Должно быть так: Optional[confloat(ge=30, le=1000)]

    Physical_Activity_Days_Per_Week: Optional[
        Annotated[
            float,
            Field(
                ge=0,
                le=1,
                description="Физическая активность, дней/нед (нормализованное значение)",
            ),
        ]
    ] = None
    # Должно быть так: Optional[conint(ge=0, le=7)]

    Sleep_Hours_Per_Day: Optional[
        Annotated[
            float,
            Field(ge=0, le=1, description="Сон в часах (нормализованное значение)"),
        ]
    ] = None
    # Должно быть так: Optional[confloat(ge=0, le=24)]

    Blood_sugar: Optional[
        Annotated[
            float,
            Field(ge=0, le=1, description="Уровень сахара (нормализованное значение)"),
        ]
    ] = None
    # Должно быть так: Optional[confloat(ge=50, le=500)]

    CK_MB: Optional[
        Annotated[
            float,
            Field(
                ge=0, le=1, description="Креатинкиназа-МВ (нормализованное значение)"
            ),
        ]
    ] = None
    # Должно быть так: Optional[confloat(ge=0, le=500)]

    Troponin: Optional[
        Annotated[
            float, Field(ge=0, le=1, description="Тропонин (нормализованное значение)")
        ]
    ] = None
    # Должно быть так: Optional[confloat(ge=0, le=100)]

    Gender: Optional[GenderEnum] = Field(
        None, description="Пол пациента (Male/Female/0.0/1.0)"
    )
    # Должно быть так: Enum [Male, Female]

    Systolic_blood_pressure: Optional[
        Annotated[
            float,
            Field(
                ge=0,
                le=1,
                description="Систолическое давление (нормализованное значение)",
            ),
        ]
    ] = None
    # Должно быть так: Optional[confloat(ge=50, le=300)]

    Diastolic_blood_pressure: Optional[
        Annotated[
            float,
            Field(
                ge=0,
                le=1,
                description="Диастолическое давление (нормализованное значение)",
            ),
        ]
    ] = None
    # Должно быть так: Optional[confloat(ge=30, le=200)]

    @field_validator("Diabetes", "Smoking", "Obesity", mode="before")
    def nan_to_none(cls, v):
        """Преобразование NaN → None"""
        if v is None:
            return None
        try:
            if isinstance(v, float) and math.isnan(v):
                return None
        except:
            pass
        return v


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


try:
    predictor = Predictor(
        model_path=MODELS_DIR / "heart_pred.pkl",
        preprocessor_path=MODELS_DIR / "train_preprocessor.pkl",
    )
except FileNotFoundError as e:
    predictor = None
    print(f"\033[91m{e}\033[0m")

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
    if predictor is None:
        return JSONResponse(
            status_code=500,
            content={"status": "failed", "error": "❌ Файл модели не обнаружен"},
        )
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))

        if "id" not in df.columns:
            return JSONResponse(
                status_code=400, content={"error": "CSV должен содержать колонку 'id'"}
            )

        try:
            _ = [PatientData(**row.to_dict()) for _, row in df.iterrows()]
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Ошибка валидации входных данных: {str(e)}"},
            )

        result_df = predictor.predict(df)

        n_rows = len(result_df)
        alerts = []
        if predictor.selected_features:
            extra_feats = set(result_df.columns) - {"id", "prediction", "probability"}
            if extra_feats:
                alerts.append(f"⚠️ В файле есть лишние признаки: {sorted(extra_feats)}")

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
