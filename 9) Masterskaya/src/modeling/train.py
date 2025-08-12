import json
import logging
import os
import warnings
from datetime import datetime
from itertools import product

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap  # type: ignore
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# для наглядности лучше или хуже новая модель
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

# Путь к текущему файлу train.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.join(
    BASE_DIR, "..", "..", "data", "processed", "train_data.csv"
)
TRAIN_DATA = pd.read_csv(TRAIN_DATA_PATH, delimiter=",", decimal=",")

MODELS_DIR = os.path.join(BASE_DIR, "..", "..", "models")


TEST_SIZE = 0.25
RANDOM_STATE = 40
N_TRIALS = 2
N_SPLITS = 5
METRIC = "precision"
TARGET_COL = "heart_attack_risk_(binary)"
N_JOBS = -1
THRESHOLDS = np.arange(0.1, 0.9, 0.01)
MIN_PRECISION = (
    0.6  # гугл говорит, что меньше 0.9 табу для медицины, но пока попробуем так
)
MLFLOW_EXPERIMENT = "heat_pred"


def get_confusion_counts(cm):
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = get_confusion_counts(cm)
    return tn, fp, fn, tp


def is_new_model_better(new_metrics, old_metrics, delta=0.001):
    def round_check(x):
        return round(x or 0, 4)  # type: ignore

    new_score = round_check(new_metrics.get(METRIC, 0))
    old_score = round_check(old_metrics.get(METRIC, 0))

    if new_score > old_score + delta:
        return True
    if new_score < old_score - delta:
        return False

    return False


def manual_optuna_progress(study, n_trials, func):
    for _ in tqdm(range(n_trials), desc="Optuna Tuning"):
        study.optimize(func, n_trials=1, catch=(Exception,), n_jobs=N_JOBS)


def objective(trial, X_train, y_train, all_columns):
    """
    Целевая функция Optuna для подбора гиперпараметров модели XGBoost с отбором признаков.

    Выполняет:
    1. Отбор признаков с помощью SelectKBest.
    2. Подбор гиперпараметров для XGBClassifier.
    3. Кросс-валидацию Stratified K-Fold с подбором порога классификации.

    Параметры
    ---------
    trial : optuna.Trial
        Объект эксперимента Optuna.
    X_train : pd.DataFrame
        Признаки обучающей выборки.
    y_train : pd.Series
        Целевая переменная обучающей выборки.
    all_columns : list
        Список всех признаков.

    Возвращает
    ----------
    float
        Лучшее значение метрики в соответствии с выбранным критерием.
    """
    try:
        k_best = trial.suggest_int("k_best", 5, min(40, X_train.shape[1]))
        selector = SelectKBest(score_func=f_classif, k=k_best)
        X_train_sel = selector.fit_transform(X_train, y_train)
        selected_idx = selector.get_support(indices=True)
        selected_features = X_train.columns[selected_idx].tolist()

        trial.set_user_attr("selected_features", selected_features)
        trial.set_user_attr("n_selected_features", len(selected_features))

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 10.0),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }

        skf = StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
        )
        recalls, precisions, f1s, accuracies, roc_auc = [], [], [], [], []
        fn_list, fp_list, tn_list, tp_list = [], [], [], []
        fold_thresholds = []

        for train_idx, valid_idx in skf.split(X_train_sel, y_train):
            model = XGBClassifier(**params)
            X_tr, X_val = X_train_sel[train_idx], X_train_sel[valid_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

            model.fit(X_tr, y_tr)
            y_proba = model.predict_proba(X_val)[:, 1]

            best_fp = float("inf")
            best_fn = float("inf")
            best_score = -np.inf
            best_threshold = 0.5

            for t in THRESHOLDS:
                y_pred_t = (y_proba >= t).astype(int)
                cm_t = confusion_matrix(y_val, y_pred_t, labels=[0, 1])
                tn, fp, fn, tp = get_confusion_counts(cm_t)

                if fn > FN_STOP:
                    continue

                precision_t = precision_score(y_val, y_pred_t, zero_division=0)
                if precision_t < MIN_PRECISION:
                    continue

                if METRIC == "f1":
                    score_t = f1_score(y_val, y_pred_t, zero_division=0)
                elif METRIC == "accuracy":
                    score_t = accuracy_score(y_val, y_pred_t)
                elif METRIC == "recall":
                    score_t = recall_score(y_val, y_pred_t)
                elif METRIC == "precision":
                    score_t = precision_score(y_val, y_pred_t, zero_division=0)
                elif METRIC == "roc_auc":
                    score_t = roc_auc_score(y_val, y_proba)
                else:
                    logger.warning(
                        f"Неизвестная метрика '{METRIC}', используем recall по умолчанию"
                    )
                    score_t = recall_score(y_val, y_pred_t)

                if score_t is None:
                    continue

                if fn <= best_fn + MAX_FN_SOFT and (
                    fn < best_fn
                    or (fn == best_fn and fp < best_fp)
                    or (fn == best_fn and fp == best_fp and score_t > best_score)
                ):
                    best_fp = fp
                    best_fn = fn
                    best_score = score_t
                    best_threshold = t

            fold_thresholds.append(best_threshold)
            y_pred = (y_proba >= best_threshold).astype(int)

            recalls.append(recall_score(y_val, y_pred))
            precisions.append(precision_score(y_val, y_pred, zero_division=0))
            f1s.append(f1_score(y_val, y_pred, zero_division=0))
            accuracies.append(accuracy_score(y_val, y_pred))
            roc_auc.append(roc_auc_score(y_val, y_proba))

            cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
            tn, fp, fn, tp = get_confusion_counts(cm)

            fn_list.append(fn)
            fp_list.append(fp)
            tn_list.append(tn)
            tp_list.append(tp)

        mean_recall = np.mean(recalls)
        mean_precision = np.mean(precisions)
        mean_f1 = np.mean(f1s)
        mean_accuracy = np.mean(accuracies)
        mean_roc_auc = np.mean(roc_auc)

        if METRIC == "f1":
            score = mean_f1
        elif METRIC == "accuracy":
            score = mean_accuracy
        elif METRIC == "recall":
            score = mean_recall
        elif METRIC == "precision":
            score = mean_precision
        elif METRIC == "roc_auc":
            score = mean_roc_auc
        else:
            logger.warning(
                f"Неизвестная метрика '{METRIC}', используется recall по умолчанию."
            )
            score = mean_recall

        logger.info(
            f"\nTrial {trial.number} → k_best: {k_best}\n"
            f"Recall: {mean_recall:.3f}, Precision: {mean_precision:.3f}, F1: {mean_f1:.3f},\n"
            f"Accuracy: {mean_accuracy:.3f}, Score: {score:.3f}\n"
            f"FN: {np.mean(fn_list):.1f}, FP: {np.mean(fp_list):.1f}, TN: {np.mean(tn_list):.1f}, TP: {np.mean(tp_list):.1f}\n"
        )

        final_threshold = np.mean(fold_thresholds)
        trial.set_user_attr("best_threshold", final_threshold)

        if np.isnan(score) or np.isinf(score):
            return -1
        return score

    except Exception as e:
        logger.exception(f"Ошибка в objective: {e}")
        return -1


def run_optuna_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    metric,
    n_trials,
    experiment_name,
    model_output_path,
    current_time,
):
    try:
        logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
        logger.info(f"Experiment: {experiment_name}")
        mlflow.set_experiment(experiment_name)

        def optuna_objective(trial):
            return objective(trial, X_train, y_train, X_train.columns)

        study = optuna.create_study(study_name=experiment_name, direction="maximize")
        manual_optuna_progress(study, n_trials, optuna_objective)

        best_params = study.best_trial.params.copy()

        # Извлекаем признаки
        selected_features = study.best_trial.user_attrs["selected_features"]
        n_selected_features = study.best_trial.user_attrs["n_selected_features"]
        best_params.pop("k_best", None)  # убираем параметр, если есть

        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

        # Извлекаем лучший порог
        best_threshold = study.best_trial.user_attrs["best_threshold"]

        # Обучаем финальную модель XGBoost
        final_model = XGBClassifier(
            **best_params,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            eval_metric="logloss",
        )
        final_model.fit(X_train, y_train)

        # Предсказания
        y_pred_proba = final_model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= best_threshold).astype(int)

        # Метрики на тесте
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = get_confusion_counts(cm)

        final_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "f1": f1_score(y_test, y_pred),
        }

        # Проверка — стоит ли сохранять модель
        save_model = True
        if os.path.exists(model_output_path):
            try:
                old_model_bundle = joblib.load(model_output_path)
                old_metrics = old_model_bundle.get("metrics", {})
                logger.info(
                    "Старая модель: "
                    + ", ".join(
                        f"{k}: {GREEN + str(v) + RESET}" if k == metric else f"{k}: {v}"
                        for k, v in old_metrics.items()
                    )
                )
                logger.info(
                    "Новая модель: "
                    + ", ".join(
                        f"{k}: {GREEN + str(v) + RESET}" if k == metric else f"{k}: {v}"
                        for k, v in final_metrics.items()
                    )
                )

                save_model = is_new_model_better(final_metrics, old_metrics)
                if save_model:
                    logger.info(f"{GREEN}Новая модель лучше — сохраняем.{RESET}")
                else:
                    logger.info(
                        f"{RED}Старая модель лучше — не сохраняем новую.{RESET}"
                    )
            except Exception as e:
                logger.warning(
                    f"Не удалось загрузить старую модель: {e}. Сохраняем новую."
                )
                save_model = True

        # Сохранение модели
        if save_model:
            joblib.dump(
                {
                    "model": final_model,
                    "threshold": best_threshold,
                    "metrics": final_metrics,
                    "selected_features": selected_features,
                },
                model_output_path,
            )
            logger.info(f"Модель сохранена в {model_output_path}")

        # Метрики на трейне
        input_example = pd.DataFrame(X_test[:1], columns=X_test.columns)
        input_example = input_example.astype("float64")
        y_pred_proba_train = final_model.predict_proba(X_train)[:, 1]
        y_pred_train = (y_pred_proba_train >= best_threshold).astype(int)

        recall_train = recall_score(y_train, y_pred_train)
        precision_train = precision_score(y_train, y_pred_train)
        cm_train = confusion_matrix(y_train, y_pred_train, labels=[0, 1])
        tn, fp, fn, tp = get_confusion_counts(cm_train)

        final_metrics_train = {
            "accuracy": accuracy_score(y_train, y_pred_train),
            "precision": precision_train,
            "recall": recall_train,
            "roc_auc": roc_auc_score(y_train, y_pred_proba_train),
            "f1": f1_score(y_train, y_pred_train),
        }

        # Логирование в MLflow
        log_with_mlflow(
            final_model=final_model,
            metric=metric,
            best_params=best_params,
            best_threshold=best_threshold,
            study=study,
            X_test=X_test,
            y_test=y_test,
            final_metrics=final_metrics,
            final_metrics_train=final_metrics_train,
            selected_features=selected_features,
            model_output_path=model_output_path,
            run_name=f"model_{current_time}",
            n_trials=n_trials,
            input_example=input_example,
            y_pred_proba=y_pred_proba,
            y_pred=y_pred,
        )

    except Exception as e:
        logger.info(f"Ошибка: {e}")
        raise


def log_with_mlflow(
    final_model,
    metric,
    best_params,
    best_threshold,
    study,
    X_test,
    y_test,
    final_metrics,
    final_metrics_train,
    selected_features,
    model_output_path,
    run_name,
    n_trials,
    input_example,
    y_pred_proba,
    y_pred,
):
    try:
        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run(run_name=run_name):
            # Лог параметров
            mlflow.log_params(best_params)
            mlflow.log_param("threshold", round(best_threshold, 4))
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("model_type", "XGBClassifier")
            mlflow.log_param("n_selected_features", len(selected_features))

            # Лог метрик
            for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                mlflow.log_metric(f"{key}_train", round(final_metrics_train[key], 4))
                mlflow.log_metric(f"{key}_test", round(final_metrics[key], 4))

            # Создаем input_example из X_test (первой строки), явно задаем колонки
            input_example = pd.DataFrame(X_test[:1], columns=X_test.columns)

            mlflow.sklearn.log_model(  # type: ignore
                final_model,
                name="model",
                input_example=input_example,
            )

            # Логируем сохраненный файл модели
            mlflow.log_artifact(model_output_path)

    except Exception as e:
        logger.error(f"Ошибка в упрощенном log_with_mlflow: {e}")
        raise


def today():
    now = pd.to_datetime(datetime.now())
    return now


if __name__ == "__main__":
    MLRUNS_PATH = os.path.join(BASE_DIR, "..", "..", "mlruns")
    MLRUNS_PATH = os.path.abspath(MLRUNS_PATH)

    mlflow.set_tracking_uri(f"file://{MLRUNS_PATH}")

    # Приведение всех столбцов к числовому типу
    for col in TRAIN_DATA.columns:
        try:
            TRAIN_DATA[col] = pd.to_numeric(TRAIN_DATA[col])
        except ValueError:
            pass

    TRAIN_DATA.set_index("id", inplace=True)

    # Найти integer столбцы с пропусками и привести их к float64
    int_cols_with_na = [
        col
        for col in TRAIN_DATA.select_dtypes(include=["int64", "Int64"]).columns
        if TRAIN_DATA[col].isnull().any()
    ]
    for col in int_cols_with_na:
        TRAIN_DATA[col] = TRAIN_DATA[col].astype("float64")

    # Для остальных числовых столбцов можно сделать общий каст к float64 (опционально)
    num_cols = TRAIN_DATA.select_dtypes(include=["number"]).columns
    TRAIN_DATA[num_cols] = TRAIN_DATA[num_cols].astype("float64")

    X_main = TRAIN_DATA.drop(columns=[TARGET_COL])
    y_main = TRAIN_DATA[TARGET_COL]

    # Сетка параметров
    fn_penalty_grid = range(1, 2)
    fp_penalty_grid = range(1, 2)
    fn_stop_grid = range(1, 2)
    max_fn_soft_grid = range(1, 2)

    for fn_penalty, fp_penalty, fn_stop_val, max_fn_soft_val in product(
        fn_penalty_grid, fp_penalty_grid, fn_stop_grid, max_fn_soft_grid
    ):
        FN_PENALTY_WEIGHT = fn_penalty
        FP_PENALTY_WEIGHT = fp_penalty
        FN_STOP = fn_stop_val
        MAX_FN_SOFT = max_fn_soft_val

        logger.info(
            f"=== Запуск с параметрами: FN_PENALTY_WEIGHT={FN_PENALTY_WEIGHT}, "
            f"FP_PENALTY_WEIGHT={FP_PENALTY_WEIGHT}, FN_STOP={FN_STOP}, MAX_FN_SOFT={MAX_FN_SOFT} ==="
        )

        X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(
            X_main,
            y_main,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_main,
        )

        run_optuna_experiment(
            X_train=X_train_data,
            y_train=y_train_data,
            X_test=X_test_data,
            y_test=y_test_data,
            metric=METRIC,
            n_trials=N_TRIALS,
            experiment_name=f"{MLFLOW_EXPERIMENT}",
            model_output_path=f"{MODELS_DIR}/heart_pred.pkl",
            current_time=today(),
        )
