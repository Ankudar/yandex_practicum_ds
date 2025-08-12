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
📦 Проект: Предсказание риска сердечных приступов
├── Makefile              <- Утилиты командной строки, например: make data, make train.
├── README.md             <- Главный файл описания проекта для разработчиков и пользователей.
├── requirements.txt      <- Зависимости Python (можно собрать через pip freeze > requirements.txt).

📦airflow                 <- Оркестрация пайплайнов через Apache Airflow.
 └── 📂dags
     └── heart_attack_predict.py  <- DAG, реализующий пайплайн подготовки и обучения новой модели при дополнении данных.

📁 fastapi                <- Папка с приложением.
 └── app.py                <- Код приложения для работы с новыми данными.

📁 data                   <- Хранилище данных различных стадий обработки.
├── 📁 raw                <- Оригинальные данные, неизменяемые выгрузки.
├── 📁 processed          <- Финальные подготовленные данные, готовые для моделирования.
└── 📁 results            <- Результаты прогонов, предсказания моделей.

📁 src                    <- Основной код проекта.
└── 📁 modeling           <- Модуль с обучением и инференсом моделей.
    ├── train.py          <- Код для обучения моделей.
    └── predict.py        <- Код для генерации предсказаний.

📁 mlruns                 <- Каталог для логов MLflow (эксперименты, метрики, параметры, артефакты).
└── 📁 models             <- Сохранённые модели и артефакты через MLflow.

📁 models                 <- Дополнительные сериализованные модели, препроцессоры и т.д. (.pkl, .joblib).

📁 notebooks              <- Jupyter-ноутбуки для исследования данных, тестов и прототипирования.

📁 reports                <- Автоматически сгенерированные отчёты, отчётные документы.
└── 📁 figures            <- Графики, визуализации и изображения, используемые в отчётах.
</pre>

<h2>⚙️ Архитектура и ООП</h2>
<p>Ключевые классы (в <code>heart_attack_predict/</code>):</p>
<ul>
  <li><code>DataLoader</code> — чтение CSV, базовая валидация.</li>
  <li><code>Preprocessor</code> — трансформации, кодирование, сохранение пайплайна (fit/transform).</li>
  <li><code>FeatureSelector</code> — удаление утечек, коррелированных и бесполезных признаков.</li>
  <li><code>ModelTrainer</code> — обучение, подбор гиперпараметров (опционально), логирование метрик и артефактов.</li>
  <li><code>Predictor</code> — загрузка пайплайна и модели, генерация предсказаний.</li>
  <li><code>FastAPIApp</code> — обёртка API (endpoints -> использует Predictor).</li>
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
  <li>Клонировать репозиторий: <code>git clone &lt;repo_url&gt;</code></li>
  <li>Виртуальное окружение и зависимости:
    <pre>
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate.bat  # Windows
pip install -r requirements.txt
    </pre>
  </li>
  <li>Обучить модель (пример):
    <pre>
python -m heart_attack_predict.model --config config/train_config.yaml
    </pre>
  </li>
  <li>Сгенерировать предсказания локально:
    <pre>
python -m heart_attack_predict.predict --input data/raw/test.csv --output data/results/prediction.csv
    </pre>
  </li>
  <li>Запустить API:
    <pre>
uvicorn app:app --reload --reload --host 127.0.0.1 --port 8000
web ui -> http://127.0.0.1:8000/docs
    </pre>
  </li>
</ol>

<h2>🔧 API</h2>
<p>Основные endpoints:</p>
<ul>
  <li><code>POST /predict/file</code> — отправить CSV, получить JSON с предсказаниями и/или ссылку на <code>prediction.csv</code>.</li>
  <li><code>POST /predict/rows</code> — отправить CSV с несколькими строками для быстрого предсказания.</li>
  <li><code>GET /health</code> — проверка статуса сервиса.</li>
</ul>

<h2>🧪 Тестирование и воспроизводимость</h2>
<ul>
  <li>Тесты: <code>pytest</code> в <code>tests/</code>.</li>
  <li>Фиксирование версий библиотек в <code>requirements.txt</code>.</li>
  <li>Конфигурируемые параметры через YAML/JSON (пути, seed, гиперпараметры).</li>
</ul>

<h2>📚 Документация</h2>
<p>В репозитории есть:</p>
<ul>
  <li>Описание классов и методов в docstrings.</li>
  <li>Инструкция по запуску и развёртыванию (раздел "Быстрый старт").</li>
  <li>Notebooks с EDA и метриками экспериментов.</li>
</ul>

<h2>📝 Требования к сдаче</h2>
<ul>
  <li>Репозиторий на GitHub/GitLab.</li>
  <li>Jupyter notebook с исследованием и выводами.</li>
  <li>Код приложения отдельно от notebooks.</li>
  <li>prediction.csv в <code>data/results/</code>.</li>
  <li>Инструкция по запуску или демонстрация работы на финальной встрече.</li>
</ul>

<footer>
  <p>Версия README: 1.3</p>
</footer>

</body>
</html>