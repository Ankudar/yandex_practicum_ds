import glob
import logging
import os
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # ../../.. от predict.py
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))  # добавляем src в PYTHONPATH

import config
from modeling.datapreprocessor import DataPreProcessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent  # папка src/modeling
MODELS_DIR = PROJECT_ROOT / "models"  # ../../models относительно predict.py

model_path = MODELS_DIR / "heart_pred.pkl"

if not model_path.exists():
    raise FileNotFoundError(f"Не найден файл модели: {model_path}")

MODEL = model_path

# Препроцессор без изменений
PREPROCESSOR = os.path.join(MODELS_DIR, "train_preprocessor.pkl")

TEST_DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "raw", "heart_test.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "..", "data", "results")
RESULT_FILE = os.path.join(RESULTS_DIR, "heart_test_pred.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    try:
        # Загружаем модель и метаданные
        model_bundle = joblib.load(MODEL)
        model = model_bundle["model"]
        threshold = model_bundle["threshold"]
        selected_features = model_bundle.get("selected_features", None)

        # Загружаем тестовые данные
        df_test = pd.read_csv(TEST_DATA_PATH)

        # Убираем таргет из данных на всякий случай
        if config.TARGET_COL in df_test.columns:
            df_test = df_test.drop(columns=[config.TARGET_COL])

        # Инициализируем препроцессор с параметрами из конфига
        preprocessor = DataPreProcessor(
            drop_cols=config.DROP_COLS,
            ohe_cols=config.OHE_COLS,
            ord_cols=config.ORD_COLS,
        )

        # Очистка имён колонок
        df_test = preprocessor.clean_column_names(df_test)

        # Загружаем сохранённый pipeline
        preprocessor.pipeline = joblib.load(PREPROCESSOR)

        # Применяем transform
        df_test_proc = preprocessor.transform(df_test)

        # Фильтруем по отобранным признакам, если они есть
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

        # Предсказания
        pred_proba = model.predict_proba(df_test_proc)[:, 1]
        predictions = (pred_proba >= threshold).astype(int)

        # Проверяем наличие id
        if "id" not in df_test.columns:
            raise ValueError("В тестовом файле отсутствует колонка 'id'")

        # Сохраняем результат
        result_df = pd.DataFrame({"id": df_test["id"], "prediction": predictions})
        result_df.to_csv(RESULT_FILE, index=False)
        logger.info(f"Результаты предсказаний сохранены в {RESULT_FILE}")

    except Exception as e:
        logger.error(f"Ошибка в процессе предсказания: {e}")
        raise


if __name__ == "__main__":
    main()
