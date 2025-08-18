import os
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from config import DROP_COLS, MODELS_DIR, OHE_COLS, ORD_COLS, PROCESSED_DIR, TARGET_COL
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
)


class DataPreProcessor:
    def __init__(
        self,
        drop_cols=None,
        ohe_cols=None,
        ord_cols=None,
        name="default",
        processed_dir=None,
        models_dir=None,
    ):
        norm = self._normalize_name
        self.drop_cols = [norm(c) for c in (drop_cols or [])]
        self.ohe_cols = [norm(c) for c in (ohe_cols or [])]
        self.ord_cols = [norm(c) for c in (ord_cols or [])]
        self.name = name
        self.processed_dir = processed_dir or PROCESSED_DIR
        self.models_dir = models_dir or MODELS_DIR
        self.pipeline = None
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    @staticmethod
    def _normalize_name(s):
        if isinstance(s, str):
            return re.sub(r"\s+", "_", s.strip().lower())
        return s

    def clean_column_names(self, df):
        df.columns = [self._normalize_name(col) for col in df.columns]
        return df.map(
            lambda x: (
                re.sub(r"\s+", " ", str(x)).strip().lower() if isinstance(x, str) else x
            )
        )

    def _preprocess_base(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.drop(
            columns=[c for c in ["Unnamed: 0", "unnamed:_0"] if c in df.columns]
        )
        df = self.clean_column_names(df).fillna(-1)

        if "gender" in df.columns:
            df["gender"] = df["gender"].replace({"1.0": "male", "0.0": "female"})

        df = df.drop(columns=self.drop_cols, errors="ignore")

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
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Отсутствуют признаки: {missing}")

        # Новые фичи
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

        return df

    def _get_feature_lists(self, df):
        ohe_cols = [c for c in self.ohe_cols if c in df.columns]
        ord_cols = [c for c in self.ord_cols if c in df.columns]
        numeric_cols = [
            c
            for c in df.select_dtypes(include=["float64", "int64"]).columns
            if c not in ohe_cols + ord_cols
        ]
        return ohe_cols, ord_cols, numeric_cols

    def fit_transform(self, df: pd.DataFrame):
        df_original = self._preprocess_base(df.copy())

        target_series = None
        if TARGET_COL in df_original.columns:
            target_series = df_original[TARGET_COL]
            df_no_target = df_original.drop(columns=[TARGET_COL])
        else:
            df_no_target = df_original

        ohe_cols, ord_cols, numeric_cols = self._get_feature_lists(df_no_target)

        preprocessor = ColumnTransformer(
            transformers=[
                ("ohe", OneHotEncoder(sparse_output=False, drop="first"), ohe_cols),
                ("ord", OrdinalEncoder(), ord_cols),
                ("num", RobustScaler(), numeric_cols),
            ],
            remainder="drop",
        )

        self.pipeline = Pipeline(
            [("pre", preprocessor), ("minmax", MinMaxScaler(feature_range=(0, 1)))]
        )

        X_scaled = self.pipeline.fit_transform(df_no_target)

        output_cols = (
            list(
                preprocessor.named_transformers_["ohe"].get_feature_names_out(ohe_cols)
            )
            + ord_cols
            + numeric_cols
        )

        df_transformed = pd.DataFrame(X_scaled, columns=output_cols, index=df.index)
        df_transformed[output_cols] = np.clip(df_transformed[output_cols], 0.0, 1.0)

        if target_series is not None:
            df_out = pd.concat([df_transformed, target_series], axis=1)
        else:
            df_out = df_transformed

        df_out.to_csv(os.path.join(self.processed_dir, "train_data.csv"), index=False)

        joblib.dump(
            self.pipeline,
            os.path.join(self.models_dir, f"{self.name}_preprocessor.pkl"),
        )

        return df_out

    def transform(self, df: pd.DataFrame):
        if self.pipeline is None:
            self.pipeline = joblib.load(
                os.path.join(self.models_dir, f"{self.name}_preprocessor.pkl")
            )

        df = self._preprocess_base(df)
        if TARGET_COL in df.columns:
            df = df.drop(columns=[TARGET_COL])

        ohe_cols, ord_cols, numeric_cols = self._get_feature_lists(df)

        X_scaled = self.pipeline.transform(df)

        output_cols = (
            list(
                self.pipeline.named_steps["pre"]
                .named_transformers_["ohe"]
                .get_feature_names_out(ohe_cols)
            )
            + ord_cols
            + numeric_cols
        )
        df_transformed = pd.DataFrame(X_scaled, columns=output_cols, index=df.index)
        df_transformed[output_cols] = np.clip(df_transformed[output_cols], 0.0, 1.0)

        return df_transformed
