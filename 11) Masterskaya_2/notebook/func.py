# func.py
import ast
import logging
import re

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display
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


def parse_category_ids(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x


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
                fontsize=5,
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
