import sys

sys.path.append("./src/modeling/")

import logging
import os

import pandas as pd
from datapreprocessor import DataPreProcessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE_DIR, "..", "..", "models", "heart_pred.pkl")
PREPROCESSOR = os.path.join(BASE_DIR, "..", "..", "models", "train_preprocessor.pkl")
TEST_DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "raw", "heart_test.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "..", "data", "results")
RESULT_FILE = os.path.join(RESULTS_DIR, "heart_predictions.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    try:
        import joblib

        model_bundle = joblib.load(MODEL)
        model = model_bundle["model"]
        threshold = model_bundle["threshold"]
        selected_features = model_bundle.get("selected_features", None)

        df_test = pd.read_csv(TEST_DATA_PATH)

        drop_cols = ["income", "ck-mb", "troponin"]
        ohe_cols = ["gender"]
        preprocessor = DataPreProcessor(drop_cols=drop_cols, ohe_cols=ohe_cols)

        # Очистка имён колонок до препроцессинга
        df_test = preprocessor.clean_column_names(df_test)

        # Загружаем сохранённый pipeline
        preprocessor.pipeline = joblib.load(
            os.path.join(preprocessor.models_dir, PREPROCESSOR)
        )

        # Применяем transform (генерация новых признаков + кодирование + масштабирование)
        df_test_proc = preprocessor.transform(df_test)

        # После transform оставляем только нужные признаки, если они есть
        if selected_features:
            missing_feats = [
                f for f in selected_features if f not in df_test_proc.columns
            ]
            if missing_feats:
                logger.warning(
                    f"В обработанных данных отсутствуют признаки из selected_features: {missing_feats}"
                )
            df_test_proc = df_test_proc[
                [f for f in selected_features if f in df_test_proc.columns]
            ]

        pred_proba = model.predict_proba(df_test_proc)[:, 1]
        predictions = (pred_proba >= threshold).astype(int)

        if "id" not in df_test.columns:
            raise ValueError("В тестовом файле отсутствует колонка 'id'")

        result_df = pd.DataFrame({"id": df_test["id"], "prediction": predictions})
        result_df.to_csv(RESULT_FILE, index=False)
        logger.info(f"Результаты предсказаний сохранены в {RESULT_FILE}")

    except Exception as e:
        logger.error(f"Ошибка в процессе предсказания: {e}")
        raise


if __name__ == "__main__":
    main()
