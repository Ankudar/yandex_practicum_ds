import os

import pandas as pd
from sklearn.metrics import classification_report

# Путь к папке data/result из корня проекта
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "results")
)


def check_predictions(student_file, correct_file):
    corr_df = pd.read_csv(correct_file, index_col=0)
    stud_df = pd.read_csv(student_file, index_col=0)

    assert "prediction" in stud_df.columns, "В файле студента нет колонки 'prediction'"
    assert len(stud_df) == len(corr_df), "Количество строк в файлах не совпадает"

    return classification_report(corr_df["prediction"], stud_df["prediction"])


def test_predictions_match():
    student_path = os.path.join(BASE_DIR, "heart_test_pred.csv")
    correct_path = os.path.join(BASE_DIR, "correct_answers.csv")

    assert os.path.exists(student_path), f"Файл не найден: {student_path}"
    assert os.path.exists(correct_path), f"Файл не найден: {correct_path}"

    report = check_predictions(student_path, correct_path)
    print(report)
