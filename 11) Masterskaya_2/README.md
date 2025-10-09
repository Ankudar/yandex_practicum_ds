<h1>Техническое задание — Маркетинг (apparel)</h1>

<style>
body {
  font-family: Inter, Segoe UI, Arial, sans-serif;
  line-height: 1.5;
  color: #111;
  padding: 24px;
  max-width: 900px;
}
h1, h2 {
  color: #0b3;
}
pre {
  background: #f6f6f6;
  padding: 12px;
  border-radius: 6px;
  overflow: auto;
}
</style>

<h2>Краткая цель</h2>
<p>Предсказать вероятность покупки клиента в течение 90 дней.</p>

<h2>Задачи</h2>
<ul>
  <li>Изучение данных.</li>
  <li>Разработка признаков.</li>
  <li>Построение модели классификации.</li>
  <li>Улучшение модели и максимизация roc_auc.</li>
  <li>Тестирование и валидация.</li>
</ul>

  <h2>Ожидаемый результат в репозитории</h2>
  <ul>
    <li>Jupyter notebook (<code>project.ipynb</code>) с разведочным анализом, подготовкой признаков, обучением и тестированием.</li>
    <li>Файл <code>README.md</code> (этот файл).</li>
    <li>Список зависимостей <code>requirements.txt</code>.</li>
    <li>Модуль с рабочими утилитами <code>func.py</code>.</li>
  </ul>

  <h2>Стек</h2>
  <ul>
    <li>Python</li>
    <li>pandas</li>
    <li>scikit-learn</li>
  </ul>

  <h2>Датасеты (исходные таблицы)</h2>
  <h3>apparel-purchases</h3>
  <ul>
    <li><code>client_id</code> — идентификатор клиента</li>
    <li><code>quantity</code> — количество единиц товара</li>
    <li><code>price</code> — цена товара</li>
    <li><code>category_ids</code> — список/строка вложенных категорий (пример: ['4','28','44','1594'])</li>
    <li><code>date</code> — дата покупки (yyyy-mm-dd)</li>
    <li><code>message_id</code> — id сообщения из рассылки (опционально)</li>
  </ul>

  <h3>apparel-messages</h3>
  <ul>
    <li><code>bulk_campaign_id</code> — id кампании</li>
    <li><code>client_id</code></li>
    <li><code>message_id</code></li>
    <li><code>event</code> — тип действия (sent, opened, click, purchase ...)</li>
    <li><code>channel</code> — канал рассылки</li>
    <li><code>date</code> — дата события</li>
    <li><code>created_at</code> — полное время создания</li>
  </ul>

  <h3>apparel-target_binary</h3>
  <ul>
    <li><code>client_id</code></li>
    <li><code>target</code> — бинарный признак: совершил покупку в следующих 90 днях</li>
  </ul>

  <h3>Агрегированные таблицы рассылок</h3>
  <p>Для ускорения работы предоставлены агрегаты по дням.</p>
  <ul>
    <li><strong>full_campaign_daily_event</strong>: колонки <code>count_event*</code> и <code>nunique_event*</code> по типам событий.</li>
    <li><strong>full_campaign_daily_event_channel</strong>: аналогично, но с разбивкой по каналам <code>channel</code>.</li>
    <li>Важно: <em>не суммировать</em> колонки <code>nunique_*</code> по дням без учета пересечений клиентов.</li>
  </ul>

  <h2>Файл структуры проекта (рекомендация)</h2>
  <pre><code>├─ data/
│  ├─ raw/                # исходные csv
│  ├─ processed/          # скрипты сохраняют здесь агрегаты и фичи
├─ notebooks/
│  └─ project.ipynb       # EDA, фичи, обучение, анализ важности
├─ src/
│  └─ func.py             # все рабочие функции (preprocess, features, train, eval)
├─ requirements.txt
└─ README.md
</code></pre>

  <h2>Рекомендации по обработке category_ids</h2>
  <ol>
    <li>Парсить строки в списки и нормализовать типы (str → int, где применимо).</li>
    <li>Выделить уровни вложенности: level_1, level_2, level_3....</li>
    <li>Создать бинарные признаки для часто встречающихся категорий (top-N). </li>
    <li>Для вариативных длин применить следующие варианты:
      <ul>
        <li>фиксировать первые K уровней и последний уровень; пропуски заполнять <code>NA</code>;</li>
        <li>агрегировать частоту вхождения каждой категории по клиенту (count, unique_days, last_seen);</li>
        <li>использовать embedding-подход или target-encoding для большого количества категорий.</li>
      </ul>
    </li>
    <li>Для изменений в дереве категорий формировать признак «изменение вложенности» или «новая комбинация».</li>
  </ol>

  <h2>Идеи признаков (быстро)
  </h2>
  <ul>
    <li>Поведение по временным окнам: покупки за 7/30/90/180 дней.</li>
    <li>RPV: revenue per visit / per client за окна.</li>
    <li>Частота покупок, средний чек, median чек.</li>
    <li>Доля категорий в корзине (top categories share).</li>
    <li>Взаимодействие с рассылками: open_rate, click_rate, purchases_after_message(window=30d).</li>
    <li>Lag-признаки: дни с последней покупки, дни с последнего открытия письма.</li>
    <li>Кросс-признаки: канал × category, campaign × event_rate.</li>
  </ul>

  <h2>Метрики и валидация</h2>
  <ul>
    <li>Основная метрика: <code>roc_auc</code>.</li>
    <li>Вторичные: precision@k, recall, pr_auc, calibration (Brier score).</li>
    <li>Стратегия валидации: временная валидация (time-based split). Формировать тренировочные окна и тестовые окна, чтобы не было утечки данных по времени.</li>
    <li>Оценка стабильности: кросс-валидация по временным периодам или rolling window.</li>
  </ul>

  <h2>Тренировка модели</h2>
  <p>Рекомендации:</p>
  <ul>
    <li>Начать с простых бустинговых моделей (LightGBM / CatBoost) и логистической регрессии для baseline.</li>
    <li>Кросс-валидация по времени и поиск гиперпараметров (Grid/Random/Optuna).</li>
    <li>Логирование экспериментов в MLflow (опционально).</li>
  </ul>

  <h2>Тестирование и проверка</h2>
  <ul>
    <li>Проверить влияние утечки данных. Все фичи должны использовать информацию, доступную до момента предсказания.</li>
    <li>Сделать тест на стабильность производительности по когорте клиентов и по времени.</li>
    <li>Построить confusion matrix и PR/ROC кривые для итоговой модели.</li>
  </ul>

  <h2>Файлы проекта</h2>
  <ul>
    <li><code>notebooks/project.ipynb</code> — основная тетрадь исследования и обучения.</li>
    <li><code>src/func.py</code> — утилиты для загрузки, предобработки, генерации фич, обучения и оценки.</li>
    <li><code>requirements.txt</code> — зависимости.</li>
  </ul>

  <h2>Как запустить (пример)</h2>
  <pre><code>git clone &lt;repo&gt;
cd repo
python -m venv .venv
source .venv/bin/activate  # или .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab  # открыть notebooks/project.ipynb
</code></pre>

  <h2>Примечания</h2>
  <ul>
    <li>Модуль <code>func.py</code> содержит все рабочие функции проекта. Используйте его для повторяемости пайплайна.</li>
    <li>В ноутбуке <code>project.ipynb</code> описаны все шаги: EDA, предобработка, обучение, анализ важности признаков.</li>
  </ul>

  <footer>
    <p>Сгенерировано по техническому заданию. Правки или дополнительные требования добавляйте прямо в README или в issues репозитория.</p>
  </footer>
</body>
</html>

