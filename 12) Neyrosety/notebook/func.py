# func.py
import ast
import logging
import re

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import HTML, display
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === Библиотеки ML ===
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)


class EarlyStoppingCallback:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.no_improvement_count = 0

    def __call__(self, study, trial):
        if study.best_value > self.best_score + self.min_delta:
            self.best_score = study.best_value
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count >= self.patience:
            study.stop()
            logger.info(f"Ранняя остановка: нет улучшений {self.patience} trials")


# форматирования текста
def format_display(text):
    return HTML(
        f"<span style='font-size: 1.5em; font-weight: bold; font-style: italic;'>{text}</span>"
    )


# сделаем функцию оценки пропусков в датасетах
def missing_data(data):
    missing_data = data.isna().sum()
    missing_data = missing_data[missing_data > 0]
    display(missing_data)


# функция для обработки пробелов
def process_spaces(s):
    if isinstance(s, str):
        s = s.strip()
        s = " ".join(s.split())
    return s


# замена пробелов на нижнее подчеркинвание в названии столбцов
def replace_spaces(s):
    if isinstance(s, str):
        s = s.strip()
        s = "_".join(s.split())
    return s


def drop_duplicated(data):
    # проверка дубликатов
    display(format_display("Проверим дубликаты и удалим, если есть"))
    num_duplicates = data.duplicated().sum()
    display(num_duplicates)

    if num_duplicates > 0:
        display("Удаляем")
        data = data.drop_duplicates(keep="first").reset_index(
            drop=True
        )  # обновляем DataFrame
    else:
        display("Дубликаты отсутствуют")
    return data


def normalize_columns(columns):
    new_cols = []
    for col in columns:
        # вставляем "_" перед заглавной буквой (латиница или кириллица), кроме первой
        col = re.sub(r"(?<!^)(?=[A-ZА-ЯЁ])", "_", col)
        # приводим к нижнему регистру
        col = col.lower()
        new_cols.append(col)
    return new_cols


def check_data(data):
    # приведем все к нижнему регистру
    data.columns = normalize_columns(data.columns)

    # удалим лишние пробелы в строках
    data = data.map(process_spaces)

    # и в названии столбцов
    data.columns = [replace_spaces(col) for col in data.columns]

    # строки в ячейках строчными буквами
    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].str.lower()

    # общая информация
    display(format_display("Общая информация базы данных"))
    display(data.info())

    # 5 строк
    display(format_display("5 случайных строк"))
    display(data.sample(5))

    # пропуски
    display(format_display("Число пропусков в базе данных"))
    display(missing_data(data))

    # проверка на наличие пропусков
    if data.isnull().sum().sum() > 0:
        display(format_display("Визуализация пропусков"))
        msno.bar(data)
        plt.show()

    # средние характеристики
    display(format_display("Характеристики базы данных"))
    display(data.describe().T)

    # data = drop_duplicated(data)

    return data  # возвращаем измененные данные

def plot_combined(data, col=None, target=None, col_type=None, legend_loc="best"):
    """
    Строит графики для числовых столбцов в DataFrame, автоматически определяя их типы (дискретные или непрерывные).

    :param data: DataFrame, содержащий данные для визуализации.
    :param col: Список столбцов для построения графиков. Если None, будут использованы все числовые столбцы.
    :param target: Столбец, по которому будет производиться разделение (для hue в графиках).
    :param col_type: Словарь, определяющий типы столбцов ('col' для непрерывных и 'dis' для дискретных).
                     Если None, типы будут определены автоматически.
    :param legend_loc: Положение легенды для графиков (по умолчанию 'best').
    :return: None. Графики отображаются с помощью plt.show().
    """

    # Определяем числовые столбцы
    if col is None:
        numerical_columns = data.select_dtypes(
            include=["int", "float"]
        ).columns.tolist()
    else:
        numerical_columns = col

    # Если col_type не указан, определяем типы автоматически
    if col_type is None:
        col_type = {}
        for col in numerical_columns:
            unique_count = data[col].nunique()
            if unique_count > 20:
                col_type[col] = "col"  # Непрерывные данные
            else:
                col_type[col] = "dis"  # Дискретные данные

    total_plots = len(numerical_columns) * 2
    ncols = 2
    nrows = (total_plots + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 5 * nrows))
    axs = axs.flatten()

    index = 0

    for col in numerical_columns:
        # Определяем тип графика
        plot_type = col_type.get(col)
        if plot_type is None:
            raise ValueError(f"Тип для столбца '{col}' не указан в col_type.")

        # Гистограмма или countplot
        if index < len(axs):
            if plot_type == "col":
                if target is not None:
                    sns.histplot(
                        data, x=col, hue=target, bins=20, kde=True, ax=axs[index]
                    )
                    handles, labels = axs[index].get_legend_handles_labels()
                    if handles:
                        axs[index].legend(title=target, loc=legend_loc)
                else:
                    sns.histplot(data[col].dropna(), bins=20, kde=True, ax=axs[index])
                axs[index].set_title(f"Гистограмма: {col}")
            elif plot_type == "dis":
                if target is not None:
                    sns.countplot(data=data, x=col, hue=target, ax=axs[index])
                    handles, labels = axs[index].get_legend_handles_labels()
                    if handles:
                        axs[index].legend(title=target, loc=legend_loc)
                else:
                    sns.countplot(data=data, x=col, ax=axs[index])
                axs[index].set_title(f"Countplot: {col}")
                # поворот подписей X для дискретных
                axs[index].tick_params(axis="x", rotation=90)
            index += 1

        # Боксплот
        if index < len(axs):
            sns.boxplot(x=data[col], ax=axs[index])
            axs[index].set_title(f"Боксплот: {col}")
            # тоже поворачиваем, если дискретные значения
            if plot_type == "dis":
                axs[index].tick_params(axis="x", rotation=90)
            index += 1

    # Отключаем оставшиеся оси
    for j in range(index, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()


def calc_target_correlations(df, target_col: str = None, drop_cols: list = None):  # type: ignore
    """
    Считает корреляции признаков с таргетом, строит heatmap и рассчитывает VIF.
    Результаты выводятся прямо в Jupyter.
    """
    if drop_cols is None:
        drop_cols = []

    df_tmp = df.copy()

    # Преобразуем категориальные в числовые
    cat_cols = df_tmp.select_dtypes(include=["object", "category"]).columns
    for c in cat_cols:
        df_tmp[c] = df_tmp[c].astype("category").cat.codes

    # Числовые колонки
    numeric_cols = df_tmp.select_dtypes(exclude=["object", "category"]).columns.tolist()
    if target_col not in numeric_cols:
        raise ValueError(f"target_col '{target_col}' должен быть числовым")

    # Корреляции с target
    corr_df = (
        df_tmp[numeric_cols]
        .corr()[target_col]
        .drop(target_col)
        .sort_values(key=np.abs, ascending=False)
    )
    display("=== Корреляция с таргетом ===")
    display(corr_df)

    # Heatmap
    heatmap_cols = [
        col for col in numeric_cols if col not in drop_cols or col == target_col
    ]
    corr_matrix = df_tmp[heatmap_cols].corr()

    plt.figure(figsize=(12, 10))
    plt.imshow(corr_matrix, interpolation="nearest", cmap="coolwarm", aspect="auto")
    plt.xticks(
        range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90, fontsize=8
    )
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns, fontsize=8)
    plt.colorbar()
    plt.title("Correlation Heatmap (включая target)")

    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            value = corr_matrix.iloc[i, j]
            plt.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="black",
            )

    plt.tight_layout()
    plt.show()

    # VIF
    vif_cols = [
        col for col in numeric_cols if col != target_col and col not in drop_cols
    ]
    X_vif = df_tmp[vif_cols].copy()
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_vif), columns=vif_cols)

    vif_data = pd.DataFrame()
    vif_data["feature"] = vif_cols
    vif_data["VIF"] = [
        variance_inflation_factor(X_scaled.values, i) for i in range(X_scaled.shape[1])
    ]
    vif_data = vif_data.sort_values("VIF", ascending=False)

    display("=== VIF ===")
    display(vif_data)


def suggest_param(trial, name, spec):
    if isinstance(spec, tuple):
        if len(spec) == 3 and spec[2] == "log":
            return trial.suggest_float(name, spec[0], spec[1], log=True)
        elif len(spec) == 3:  # int с шагом
            return trial.suggest_int(name, spec[0], spec[1], step=spec[2])
        else:  # обычный int
            return trial.suggest_int(name, spec[0], spec[1])
    elif isinstance(spec, list):
        return trial.suggest_categorical(name, spec)
    else:
        raise ValueError(f"Unsupported param spec: {spec}")


def plot_results(
    model, X_test, y_test, train_losses, test_losses, X_test_original=None
):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)

    y_test_np = y_test.cpu().numpy()
    predictions_np = predictions.squeeze().cpu().numpy()

    df_pred = pd.DataFrame(
        {
            "actual_temperature": y_test_np,
            "predicted_temperature": predictions_np,
            "absolute_error": np.abs(y_test_np - predictions_np),
        }
    )

    if X_test_original is not None:
        X_test_original = X_test_original.reset_index(drop=True)
        df_pred = pd.concat([X_test_original, df_pred], axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # График 1: Факт vs Прогноз
    num_stars = len(y_test_np)
    x_pos = np.arange(num_stars)

    bars_pred = ax1.bar(
        x_pos,
        predictions_np,
        width=0.3,
        color="yellow",
        alpha=0.9,
        edgecolor="darkorange",
        linewidth=1.5,
        label="Прогноз",
        zorder=3,
    )

    bars_actual = ax1.bar(
        x_pos,
        y_test_np,
        width=0.9,
        color="lightblue",
        alpha=0.5,
        edgecolor="lightblue",
        linewidth=1,
        label="Факт",
        zorder=2,
    )

    ax1.set_xlabel("Номер звезды в таблице данных")
    ax1.set_ylabel("Температура звезды (K)")
    ax1.set_title("Сравнение фактических и предсказанных температур звезд")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x_pos)

    xticks_step = max(1, num_stars // 10)
    xticks_positions = np.arange(0, num_stars, xticks_step)
    ax1.set_xticks(xticks_positions)
    ax1.set_xticklabels(xticks_positions, rotation=90)

    train_rmse = [loss for loss in train_losses]
    test_rmse = [loss for loss in test_losses]

    # График 2: Потери (RMSE)
    ax2.plot(train_losses, label="Ошибка на обучении (RMSE)", color="blue", linewidth=2)
    ax2.plot(test_losses, label="Ошибка на тесте (RMSE)", color="orange", linewidth=2)
    ax2.set_xlabel("Эпоха обучения")
    ax2.set_ylabel("Ошибка (RMSE)")
    ax2.set_title("Динамика RMSE при обучении модели")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    mae = mean_absolute_error(y_test_np, predictions_np)
    r2 = r2_score(y_test_np, predictions_np)
    rmse = np.sqrt(mean_squared_error(y_test_np, predictions_np))

    logger.info(f"Средняя абсолютная ошибка (MAE): {mae:.2f} K")
    logger.info(f"Среднеквадратичная ошибка (RMSE): {rmse:.2f} K")
    logger.info(f"Коэффициент детерминации (R²): {r2:.4f}")
    logger.info(f"Средняя температура: {y_test_np.mean():.2f} K")
    logger.info(f"Стандартное отклонение: {y_test_np.std():.2f} K")

    logger.info("\nПервые 10 предсказаний:")
    display(df_pred.head(10))

    logger.info("\nСтатистика по ошибкам предсказания:")
    error_stats = df_pred["absolute_error"].describe()
    display(error_stats)

    return mae, rmse, r2, df_pred


def plot_categorical_columns(data, col=None, target=None, top_n=None):
    """
    Визуализация категориальных столбцов: только столбчатые графики (с группировкой по target).
    top_n — показывать только top_n категорий, остальные сворачивать в 'other'.
    """
    categorical_columns = data.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if col is not None:
        if col not in data.columns:
            print(f"Столбец '{col}' не найден в DataFrame.")
            return
        categorical_columns = [col]

    if len(categorical_columns) == 0:
        print("Категориальных столбцов нет.")
        return

    n = len(categorical_columns)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 5 * nrows))
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]

    idx = 0
    for c in categorical_columns:
        # value_counts с NaN
        vc = data[c].fillna("NaN").value_counts()

        if top_n is not None and len(vc) > top_n:
            head = vc.iloc[:top_n].copy()
            rest = vc.iloc[top_n:].sum()
            head["other"] = rest
            vc = head

        labels = vc.index.tolist()
        cmap = plt.colormaps.get_cmap("tab20").resampled(max(1, len(labels)))
        colors = [cmap(i) for i in range(len(labels))]

        if target is not None and target in data.columns:
            grouped = data.groupby([target, c]).size().unstack(fill_value=0)
            cmap2 = plt.cm.get_cmap("tab20", max(1, len(grouped.columns)))
            bar_colors = [cmap2(i) for i in range(len(grouped.columns))]
            grouped.plot(kind="bar", ax=axs[idx], color=bar_colors)
            axs[idx].legend(title=target)
        else:
            vc.plot(kind="bar", ax=axs[idx], color=colors)

        axs[idx].set_title(f"{c} (гистограмма)")
        axs[idx].set_ylabel("Частота")
        axs[idx].tick_params(axis="x", rotation=90)
        idx += 1

    for j in range(idx, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_scatter_with_numerical(data, target_column):
    """
    Функция для построения диаграммы рассеяния с регрессионной линией.

    :param data: DataFrame с данными.
    :param target_column: Название столбца с целевым признаком.
    """
    numerical_columns = data.select_dtypes(
        include=["float64", "int64"]
    ).columns.tolist()

    if target_column in numerical_columns:
        numerical_columns.remove(target_column)

    for column in numerical_columns:
        plt.figure(figsize=(15, 6))

        # Диаграмма рассеяния с регрессионной линией
        plt.subplot(1, 1, 1)
        sns.regplot(
            data=data,
            x=column,
            y=target_column,
            scatter_kws={"alpha": 0.5},
            line_kws={"color": "red"},
        )
        plt.title(f"Зависимость {target_column} от {column}")
        plt.xlabel(column)
        plt.ylabel(target_column)

        plt.tight_layout()
        plt.show()
