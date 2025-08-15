import glob
import logging
import os
import sys

import joblib
import pandas as pd

# Добавляем корень проекта в PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import src.config as config
from src.modeling.datapreprocessor import DataPreProcessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "..", "models")

# Поиск самой свежей модели по маске heart_pred_*.pkl
model_files = sorted(
    glob.glob(os.path.join(MODELS_DIR, "heart_pred.pkl")),
    key=os.path.getmtime,
    reverse=True,
)
if not model_files:
    raise FileNotFoundError("Не найдено ни одной модели 'heart_pred.pkl'")
MODEL = model_files[0]

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
