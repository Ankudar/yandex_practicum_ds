import logging
import os
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import config
from modeling.datapreprocessor import DataPreProcessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class Predictor:
    def __init__(
        self,
        model_path: str,
        preprocessor_path: str,
        test_data_path: str,
        result_file: str,
    ):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.test_data_path = Path(test_data_path)
        self.result_file = Path(result_file)

        self.model_bundle = None
        self.model = None
        self.threshold = None
        self.selected_features = None
        self.preprocessor = None

        os.makedirs(self.result_file.parent, exist_ok=True)

    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Не найден файл модели: {self.model_path}")

        self.model_bundle = joblib.load(self.model_path)
        self.model = self.model_bundle["model"]
        self.threshold = self.model_bundle["threshold"]
        self.selected_features = self.model_bundle.get("selected_features", None)
        logger.info(f"Модель загружена из {self.model_path}")

    def load_preprocessor(self):
        self.preprocessor = DataPreProcessor(
            drop_cols=config.DROP_COLS,  # type: ignore
            ohe_cols=config.OHE_COLS,
            ord_cols=config.ORD_COLS,
        )
        self.preprocessor.pipeline = joblib.load(self.preprocessor_path)
        logger.info(f"Препроцессор загружен из {self.preprocessor_path}")

    def load_test_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.test_data_path)
        if config.TARGET_COL in df.columns:
            df = df.drop(columns=[config.TARGET_COL])
        df = self.preprocessor.clean_column_names(df)  # type: ignore
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df_proc = self.preprocessor.transform(df)  # type: ignore
        if self.selected_features:
            missing_feats = [
                f for f in self.selected_features if f not in df_proc.columns
            ]
            if missing_feats:
                logger.warning(
                    f"В обработанных данных отсутствуют признаки из selected_features: {missing_feats}"
                )
            df_proc = df_proc[
                [f for f in self.selected_features if f in df_proc.columns]
            ]
        return df_proc

    def predict(self, df_proc: pd.DataFrame) -> pd.DataFrame:
        pred_proba = self.model.predict_proba(df_proc)[:, 1]  # type: ignore
        predictions = (pred_proba >= self.threshold).astype(int)
        return predictions

    def run(self):
        try:
            self.load_model()
            self.load_preprocessor()
            df_test = self.load_test_data()
            df_test_proc = self.preprocess(df_test)

            if "id" not in df_test.columns:
                raise ValueError("В тестовом файле отсутствует колонка 'id'")

            predictions = self.predict(df_test_proc)
            result_df = pd.DataFrame({"id": df_test["id"], "prediction": predictions})
            result_df.to_csv(self.result_file, index=False)
            logger.info(f"Результаты предсказаний сохранены в {self.result_file}")

        except Exception as e:
            logger.error(f"Ошибка в процессе предсказания: {e}")
            raise


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    MODELS_DIR = PROJECT_ROOT / "models"

    predictor = Predictor(
        model_path=MODELS_DIR / "heart_pred.pkl",  # type: ignore
        preprocessor_path=MODELS_DIR / "train_preprocessor.pkl",  # type: ignore
        test_data_path=PROJECT_ROOT / "data" / "raw" / "heart_test.csv",  # type: ignore
        result_file=PROJECT_ROOT / "data" / "results" / "heart_test_pred.csv",  # type: ignore
    )
    predictor.run()
