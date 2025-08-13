<!DOCTYPE html>
<html lang="ru">
<body>

<h1>💓 Предсказание риска сердечных приступов</h1>

<p>Репозиторий содержит pipeline ML и FastAPI-сервис для предсказания риска сердечного приступа по медицинским и поведенческим данным.</p>

<h2>🎯 Цель проекта</p>
<p>Разработать воспроизводимый ML-продукт. Дать готовую модель и библиотеку для инференса. Сервис принимает CSV тестовой выборки и возвращает предсказания в формате JSON "id, prediction"</p>

<h2>📦 Что в репозитории</h2>
<ul>
  <li><code>notebooks/</code> — EDA, экспериментальные исследования (не содержит production-кода приложения).</li>
  <li><code>data/raw/</code>, <code>data/processed/</code>, <code>data/results/</code> — сырые, обработанные данные и результаты.</li>
  <li><code>heart_attack_predict/</code> — приложение и библиотеки (ООП-классы): загрузка, предобработка, модель, инференс, API.</li>
  <li><code>tests/</code> — unit/integration тесты и скрипт проверки финальной метрики.</li>
  <li><code>requirements.txt</code> — зависимости.</li>
  <li><code>README.md</code> — этот файл.</li>
  <li><code>prediction.json</code> — пример выходного файла в формате JSON (id, prediction).</li>
</ul>

<h2>🧩 Стек</h2>
<ul>
  <li>Python 3.10+</li>
  <li>pandas, numpy</li>
  <li>scikit-learn, xgboost</li>
  <li>FastAPI (интерфейс)</li>
  <li>pytest (тесты)</li>
  <li>MLflow</li>
</ul>

<h2>📁 Структура проекта</h2>
<pre>
📦9) Masterskaya
├── README.md               <- Описание проекта, инструкции для разработчиков и пользователей.
├── requirements.txt        <- Зависимости Python.
📦 data                     <- Данные проекта.
 ├── raw                    <- Оригинальные выгрузки данных.
 │   ├── heart_train.csv
 │   └── heart_test.csv
 ├── processed              <- Предобработанные данные для обучения моделей.
 │   └── train_data.csv
 └── results                <- Результаты работы моделей (предсказания, CSV/JSON файлы).
     ├── heart_predictions.csv
     ├── heart_test_pred.csv
     └── heart_test_pred.json
📦 fastapi                  <- Приложение для работы с моделью через веб.
 ├── static                 <- Статические файлы (HTML, JS, CSS).
 │   └── index.html
 └── app.py                 <- Основной FastAPI-приложение.
📦 mlruns                   <- Логи MLflow (эксперименты, метрики, параметры, артефакты).
📦 models                   <- Сохранённые модели и препроцессоры (.pkl, .joblib).
 ├── corr_preprocessor.pkl
 ├── heart_pred.pkl
 └── train_preprocessor.pkl
📦 notebook                 <- Jupyter-ноутбуки для исследования данных и прототипирования.
 └── search.ipynb
📦 reports                  <- Отчёты и визуализации.
 ├── fastapi                <- Отчёты и визуализации для FastAPI.
 └── mlflow                 <- Отчёты MLflow.
📦 src                      <- Основной исходный код проекта.
 └── modeling               <- Модули для обучения и инференса моделей.
     ├── datapreprocessor.py
     ├── predict.py
     └── train.py
📦 test                     <- Тесты проекта.
 └── test.py
</pre>

<h2>⚙️ Архитектура и ООП</h2>
<p>Ключевые классы и их роли:</p>
<ul>
  <li><code>DataPreProcessor</code> — основной класс для предобработки данных:
    <ul>
      <li>Очистка и стандартизация имён колонок и строковых значений.</li>
      <li>Удаление служебных колонок и пропусков.</li>
      <li>Фильтрация и кодирование категориальных признаков (One-Hot Encoding).</li>
      <li>Масштабирование числовых колонок (RobustScaler).</li>
      <li>Генерация новых признаков (например, <code>pulse_pressure</code>, <code>bp_ratio</code>).</li>
      <li>Сохранение и загрузка пайплайна для повторного использования.</li>
    </ul>
  </li>
  <li><code>ModelTrainer</code> — обучение модели:
    <ul>
      <li>Подбор гиперпараметров (Optuna).</li>
      <li>Кросс-валидация и подбор оптимального порога классификации.</li>
      <li>Логирование метрик и артефактов (MLflow).</li>
      <li>Сравнение новой модели с предыдущей по заданной метрике.</li>
    </ul>
  </li>
  <li><code>Predictor</code> — генерация предсказаний на новых данных:
    <ul>
      <li>Загрузка обученной модели и пайплайна предобработки.</li>
      <li>Применение <code>transform</code> для новых CSV-файлов.</li>
      <li>Формирование CSV и JSON с результатами.</li>
    </ul>
  </li>
  <li><code>FastAPIApp</code> — веб-обёртка для <code>Predictor</code>:
    <ul>
      <li>Эндпоинт для загрузки CSV и получения предсказаний.</li>
      <li>Эндпоинт для доступа к статическому фронтенду (HTML/JS).</li>
      <li>Логирование и обработка ошибок запросов.</li>
    </ul>
  </li>
</ul>

<h2>🔬 Ключевые шаги</h2>
<ol>
  <li>EDA(notebooks/).</li>
  <li>Очистка и преобразование типов, обработка пропусков.</li>
  <li>Feature engineering и масштабирование.</li>
  <li>Обучение моделей (XGBCLassifier). Сравнение по выбранной метрике.</li>
  <li>Сохранение пайплайна и модели.</li>
  <li>API для инференса и скрипт генерации <code>prediction.csv</code>.</li>
</ol>

<h2>📐 Метрика</h2>
<p>F2 score с упором на минимизацию FN (Beta > 1)</p>
<p>В репозитории показаны расчёты F2/Ballance accuracy и процедура выбора порога с учётом FN = 0.</p>

<h2>📁 Формат предсказаний</h2>
<p>Тестовый файл результатов будет в виде CSV с двумя столбцами, для приложения ответ в формате JSON:</p>
<pre>
id,prediction
1,0
2,1
...
</pre>
<p><strong>prediction</strong> — {0,1} где 1 = высокий риск.</p>

<h2>🚀 Быстрый старт (локально)</h2>
<ol>
  <li>Клонировать репозиторий:
    <pre>git clone &lt;repo_url&gt;
cd &lt;repo_dir&gt;</pre>
  </li>
  <li>Создать виртуальное окружение и установить зависимости:
    <pre>python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate.bat

pip install -r requirements.txt</pre>
  </li>
  <li>Прогнать Jupyter Notebook для подготовки препроцессоров:
    <pre>jupyter notebook notebook/search.ipynb</pre>
    После выполнения ячеек будут созданы препроцессоры в <code>models/train_preprocessor.pkl</code>
  </li>
  <li>Обучить модель:
    <pre>python src/modeling/train.py</pre>
    В результате будет сохранён файл модели в <code>models/heart_pred.pkl</code>
  </li>
  <li>Сгенерировать предсказания локально:
    <pre>python src/modeling/predict.py</pre>
    Файл с результатами будет сохранён в <code>data/results/heart_predictions.csv</code>
  </li>
  <li>Запустить API:
    <pre>uvicorn fastapi.app:app --reload
# Swagger UI: http://127.0.0.1:8000/docs
# Web UI: http://127.0.0.1:8000/static/index.html</pre>
  </li>
</ol>

<h2>🔧 API</h2>
<p>Основные endpoints:</p>
<ul>
  <li><code>POST /predict/file</code> — отправить CSV, получить JSON с предсказаниями и/или ссылку на <code>pred_{file_name}.csv</code>.</li>
  <li><code>GET /Root</code> — получить ссылку на web ui.</li>
</ul>

<footer>
  <p>Версия README: 1.4</p>
</footer>

</body>
</html>