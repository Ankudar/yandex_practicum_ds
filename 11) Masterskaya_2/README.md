<h1>Техническое задание — Маркетинг (apparel)</h1>

<style>
body {
  font-family: Inter, Segoe UI, Arial, sans-serif;
  line-height: 1.5;
  color: #111;
  max-width: 900px;
}
h1, h2, h3 {
  color: #0b3;
}
pre {
  background: #f6f6f6;
  padding: 12px;
  border-radius: 6px;
  overflow: auto;
}
code {
  background: #f0f0f0;
  padding: 2px 4px;
  border-radius: 4px;
}
</style>

<h2>Краткая цель</h2>
<p>Предсказать вероятность покупки клиента в течение 90 дней.</p>

<h2>Задачи</h2>
<ul>
  <li>Изучить данные.</li>
  <li>Разработать полезные признаки.</li>
  <li>Построить модель классификации.</li>
  <li>Улучшить модель и максимизировать метрику <code>roc_auc</code>.</li>
  <li>Провести тестирование и валидацию.</li>
</ul>

<h2>Содержимое проекта</h2>
<ul>
  <li><code>notebook/project.ipynb</code> — EDA, генерация признаков, обучение и анализ важности.</li>
  <li><code>notebook/func.py</code> — рабочие функции проекта (preprocess, features, train, eval).</li>
  <li><code>README.md</code> — описание проекта.</li>
  <li><code>requirements.txt</code> — список зависимостей.</li>
</ul>

<h2>Стек</h2>
<ul>
  <li>Python</li>
  <li>pandas</li>
  <li>scikit-learn</li>
  <li>jupyter</li>
</ul>

<h2>Датасеты</h2>

<h3>apparel-purchases</h3>
<ul>
  <li><code>client_id</code> — идентификатор клиента</li>
  <li><code>quantity</code> — количество единиц товара</li>
  <li><code>price</code> — цена товара</li>
  <li><code>category_ids</code> — вложенные категории (пример: ['4','28','44','1594'])</li>
  <li><code>date</code> — дата покупки</li>
  <li><code>message_id</code> — id сообщения из рассылки</li>
</ul>

<h3>apparel-messages</h3>
<ul>
  <li><code>bulk_campaign_id</code> — идентификатор кампании</li>
  <li><code>client_id</code></li>
  <li><code>message_id</code></li>
  <li><code>event</code> — тип действия (sent, opened, click, purchase ...)</li>
  <li><code>channel</code> — канал рассылки</li>
  <li><code>date</code> — дата события</li>
  <li><code>created_at</code> — дата-время создания</li>
</ul>

<h3>apparel-target_binary</h3>
<ul>
  <li><code>client_id</code></li>
  <li><code>target</code> — 1, если клиент совершил покупку в следующие 90 дней</li>
</ul>

<h2>Дополнительные таблицы</h2>

<h3>full_campaign_daily_event</h3>
<p>Агрегация общей базы рассылок по дням и типам событий.</p>
<ul>
  <li><code>date</code></li>
  <li><code>bulk_campaign_id</code></li>
  <li><code>count_event*</code> — количество событий</li>
  <li><code>nunique_event*</code> — уникальные клиенты по каждому событию</li>
</ul>

<h3>full_campaign_daily_event_channel</h3>
<p>Агрегация по событиям и каналам.</p>
<ul>
  <li><code>date</code></li>
  <li><code>bulk_campaign_id</code></li>
  <li><code>count_event*_channel*</code> — количество событий по каналам</li>
  <li><code>nunique_event*_channel*</code> — уникальные клиенты по каналам</li>
</ul>

<h2>Структура проекта</h2>
<pre><code>📦Masterskaya_2
 ┣ 📂data
 ┃ ┣ 📜apparel-messages.csv
 ┃ ┣ 📜apparel-purchases.csv
 ┃ ┣ 📜apparel-target_binary.csv
 ┃ ┣ 📜full_campaign_daily_event.csv
 ┃ ┗ 📜full_campaign_daily_event_channel.csv
 ┣ 📂docs
 ┃ ┣ 📜Описание данных.pdf
 ┃ ┗ 📜Техническое задание Маркетинг.pdf
 ┣ 📂notebook
 ┃ ┣ 📜func.py
 ┃ ┗ 📜project.ipynb
 ┣ 📜requirements.txt
 ┗ 📜README.md
</code></pre>

<h2>Создание новых признаков</h2>
<ul>
  <li>Окна активности (7/30/90/180 дней).</li>
  <li>RPV: revenue per visit/client за окно.</li>
  <li>Средний и медианный чек, частота покупок.</li>
  <li>Доля топ-категорий в корзине.</li>
  <li>Показатели рассылок: open_rate, click_rate, purchases_after_message(30d).</li>
  <li>Lag-признаки: дни с последней покупки или открытия письма.</li>
  <li>Кросс-признаки: канал × категория, кампания × event_rate.</li>
</ul>

<h2>Метрики и валидация</h2>
<ul>
  <li>Основная метрика: <code>roc_auc</code>.</li>
  <li>Дополнительно: precision@k, recall, pr_auc, Brier score.</li>
  <li>Валидация: time-based split без утечки данных по времени.</li>
  <li>Проверка стабильности: rolling window или скользящее окно.</li>
</ul>

<h2>Как запустить</h2>
<pre><code>git clone &lt;repo_url&gt;
cd Masterskaya_2
python -m venv .venv
source .venv/bin/activate  # или .venv\Scripts\activate для Windows
pip install -r requirements.txt
jupyter lab  # открыть notebook/project.ipynb
</code></pre>

<h2>Примечания</h2>
<ul>
  <li>Модуль <code>func.py</code> содержит все функции пайплайна.</li>
  <li>В ноутбуке <code>project.ipynb</code> описан полный цикл: EDA → фичи → обучение → анализ.</li>
</ul>
