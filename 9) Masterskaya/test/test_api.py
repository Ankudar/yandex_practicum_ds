import os

import requests

API_URL = "http://127.0.0.1:8000/predict"

TEST_FILE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "heart_test.csv"
)


def test_api_predict_file():
    with open(TEST_FILE_PATH, "rb") as f:
        files = {"file": (os.path.basename(TEST_FILE_PATH), f, "text/csv")}
        response = requests.post(API_URL, files=files)

    print("Статус:", response.status_code)
    print("Ответ:", response.text)  # выводим весь ответ целиком

    assert response.status_code == 200, f"API вернул код {response.status_code}"

    data = response.json()
    assert "predictions" in data, "В ответе нет ключа 'predictions'"

    predictions = data["predictions"]
    assert isinstance(predictions, dict), "'predictions' должен быть словарём"
    assert all(
        pred in [0, 1] for pred in predictions.values()
    ), "prediction должен быть 0 или 1"
