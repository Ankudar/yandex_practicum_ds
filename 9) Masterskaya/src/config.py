TARGET_COL = "heart_attack_risk_(binary)"

# Столбцы, которые нужно исключить перед обучением/предсказанием
DROP_COLS = ["ck-mb", "troponin", "id"]

# Категориальные признаки для One-Hot Encoding
OHE_COLS = ["gender"]

# Категориальные последовательные для ordinal encode
ORD_COLS = ["diet", "stress_level", "physical_activity_days_per_week"]
