import logging
import os

import joblib
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "models", "heart_pred.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "..", "..", "models", "preprocessor.pkl")
TEST_DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "raw", "heart_test.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "..", "data", "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    try:
        # Загрузка модели
        model_bundle = joblib.load(MODEL_PATH)
        model = model_bundle["model"]
        threshold = model_bundle["threshold"]
        selected_features = model_bundle.get("selected_features", None)

        # Загрузка препроцессора
        preprocessor = joblib.load(PREPROCESSOR_PATH)

        # Загрузка тестовых данных
        df_test = pd.read_csv(TEST_DATA_PATH)

        # Отбор признаков, если есть selected_features
        if selected_features:
            df_test_proc = df_test[selected_features].copy()
        else:
            df_test_proc = df_test.copy()

        # Препроцессинг
        df_test_proc = preprocessor.transform(df_test_proc)

        # Предсказание вероятностей (предполагаем, что model поддерживает predict_proba)
        pred_proba = model.predict_proba(df_test_proc)[:, 1]

        # Применяем threshold для бинарных предсказаний
        predictions = (pred_proba >= threshold).astype(int)

        # Формируем итоговый DataFrame
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
