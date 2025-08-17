# данные для обработки/предобработки
TARGET_COL = "heart_attack_risk_(binary)"
DROP_COLS = ["ck-mb", "troponin", "id"]
OHE_COLS = ["gender"]
ORD_COLS = ["diet", "stress_level", "physical_activity_days_per_week"]

# маршруты для модулей
PROCESSED_DIR = "../data/processed/"
MODELS_DIR = "../models/"
