import os

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


class DataPreProcessor:
    def __init__(
        self,
        drop_cols=None,
        ohe_cols=None,
        processed_dir="../data/processed/",
        models_dir="../models/",
    ):
        self.drop_cols = drop_cols or []
        self.ohe_cols = ohe_cols or []
        self.processed_dir = processed_dir
        self.models_dir = models_dir
        self.encoder = None

        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    @staticmethod
    def process_spaces(s):
        if isinstance(s, str):
            s = s.strip()
            s = " ".join(s.split())
        return s

    @staticmethod
    def replace_spaces(s):
        if isinstance(s, str):
            s = s.strip()
            s = "_".join(s.split())
        return s

    def clean_column_names(self, df):
        df.columns = [
            self.replace_spaces(self.process_spaces(col)).lower() for col in df.columns
        ]
        # Очистка строковых значений во всех ячейках
        df = df.applymap(
            lambda x: self.process_spaces(x).lower() if isinstance(x, str) else x
        )
        return df

    def fit_transform(self, df: pd.DataFrame):
        # Удаляем служебные колонки
        for col in ["Unnamed: 0", "id"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Очистка колонок и строк
        df = self.clean_column_names(df)

        # Заполнение пропусков -1
        df.fillna(-1, inplace=True)

        # Фильтрация пола
        df["gender"] = df["gender"].replace({"1.0": "male", "0.0": "female"})

        # Удаление заданных колонок
        df = df.drop(columns=self.drop_cols, errors="ignore")

        # One-Hot кодирование
        self.encoder = OneHotEncoder(sparse_output=False, drop="first")

        valid_ohe_cols = [col for col in self.ohe_cols if df[col].nunique() > 1]

        if valid_ohe_cols:
            encoded = self.encoder.fit_transform(df[valid_ohe_cols])
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.encoder.get_feature_names_out(valid_ohe_cols),
                index=df.index,
            ).astype("float64")
            df = pd.concat([df.drop(columns=valid_ohe_cols), encoded_df], axis=1)

        # Генерация новых признаков
        required_cols = [
            "systolic_blood_pressure",
            "diastolic_blood_pressure",
            "cholesterol",
            "triglycerides",
            "obesity",
            "exercise_hours_per_week",
            "age",
            "smoking",
            "stress_level",
            "sedentary_hours_per_day",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Отсутствуют признаки для генерации новых фич: {missing_cols}"
            )

        df["pulse_pressure"] = (
            df["systolic_blood_pressure"] - df["diastolic_blood_pressure"]
        )
        df["bp_ratio"] = df["systolic_blood_pressure"] / (
            df["diastolic_blood_pressure"] + 1e-6
        )
        df["chol_trig_ratio"] = df["cholesterol"] / (df["triglycerides"] + 1e-6)
        df["obesity_exercise_interaction"] = (
            df["obesity"] * df["exercise_hours_per_week"]
        )
        df["age_smoking_interaction"] = df["age"] * df["smoking"]
        df["stress_sedentary_ratio"] = df["stress_level"] / (
            df["sedentary_hours_per_day"] + 1
        )
        df["bp_mean"] = (
            df["systolic_blood_pressure"] + df["diastolic_blood_pressure"]
        ) / 2

        # Масштабирование числовых колонок
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        scaler = RobustScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # Удаление дубликатов
        df = df.drop_duplicates()

        # Сохранение результата
        df.to_csv(os.path.join(self.processed_dir, "train_data.csv"), index=False)

        # Сохраняем pipeline (OneHotEncoder + Scaler)
        pipeline_object = Pipeline(
            steps=[("encoder", self.encoder), ("scaler", scaler)]
        )
        joblib.dump(pipeline_object, os.path.join(self.models_dir, "preprocessor.pkl"))

        return df
