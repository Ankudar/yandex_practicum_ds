import json
import logging
import os
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
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
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
N_TRIALS = 200  # число итераций для оптуны
N_SPLITS = 10  # cv split
METRIC = "f2"
TARGET_COL = "heart_attack_risk_(binary)"
N_JOBS = -1
THRESHOLDS = np.arange(0.1, 0.9, 0.01)
MIN_PRECISION = 0.85  # гугл говорит, что меньше 0.9 табу для медицины
MLFLOW_EXPERIMENT = "heat_pred"


def get_confusion_counts(cm):
    if cm.shape == (2, 2):
        return cm.ravel()
    else:
        raise ValueError(f"Ожидалась матрица размером 2x2, получена {cm.shape}")


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
        study.optimize(func, n_trials=1, catch=(Exception,))


def objective(trial, X_train, y_train):
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

    Возвращает
    ----------
    float
        Лучшее значение метрики в соответствии с выбранным критерием.
    """
    try:
        k_best = trial.suggest_int("k_best", 7, min(40, X_train.shape[1]))
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
            "n_jobs": 1,
        }

        skf = StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
        )
        recalls, precisions, f1s, f2s, accuracies, roc_auc, bal_acc = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
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
                elif METRIC == "f2":
                    score_t = fbeta_score(y_val, y_pred_t, beta=1.2, zero_division=0)
                elif METRIC == "bal_acc":
                    score_t = balanced_accuracy_score(y_val, y_pred_t)
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

                if fp < best_fp or (fp == best_fp and score_t > best_score):
                    best_fp = fp
                    best_score = score_t
                    best_threshold = t

            fold_thresholds.append(best_threshold)
            y_pred = (y_proba >= best_threshold).astype(int)

            recalls.append(recall_score(y_val, y_pred))
            precisions.append(precision_score(y_val, y_pred, zero_division=0))
            f1s.append(f1_score(y_val, y_pred, zero_division=0))
            f2s.append(fbeta_score(y_val, y_pred, beta=2, zero_division=0))
            bal_acc.append(balanced_accuracy_score(y_val, y_pred_t))
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
        mean_f2 = np.mean(f2s)
        mean_accuracy = np.mean(accuracies)
        mean_roc_auc = np.mean(roc_auc)
        mean_bal_acc = np.mean(bal_acc)

        if METRIC == "f1":
            score = mean_f1
        elif METRIC == "f2":
            score = mean_f2
        elif METRIC == "bal_acc":
            score = mean_bal_acc
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
            f"Recall: {mean_recall:.3f}, Precision: {mean_precision:.3f},\n"
            f"Accuracy: {mean_accuracy:.3f}, {METRIC}_Score: {score:.3f}\n"
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
            return objective(trial, X_train, y_train)

        study = optuna.create_study(study_name=experiment_name, direction="maximize")
        manual_optuna_progress(study, n_trials, optuna_objective)

        best_params = study.best_trial.params.copy()

        # Извлекаем признаки
        selected_features = study.best_trial.user_attrs["selected_features"]
        n_selected_features = study.best_trial.user_attrs["n_selected_features"]
        best_params.pop("k_best", None)

        X_train = X_train[selected_features].astype(float)
        X_test = X_test[selected_features].astype(float)

        # Извлекаем лучший порог
        best_threshold = study.best_trial.user_attrs["best_threshold"]

        # Обучаем финальную модель XGBoost
        final_model = XGBClassifier(
            **best_params,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            eval_metric="logloss",
            tree_method="hist",
        )

        logger.info(f"Гиперпараметры подобраны, начинается финальное обучение модели")
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
            "f2": fbeta_score(y_test, y_pred, beta=2, zero_division=0),
            "bal_acc": balanced_accuracy_score(y_test, y_pred),
        }

        logger.info(
            f"Модель обучена с параметрами: "
            f"accuracy={final_metrics['accuracy']:.4f}, "
            f"precision={final_metrics['precision']:.4f}, "
            f"recall={final_metrics['recall']:.4f}, "
            f"roc_auc={final_metrics['roc_auc']:.4f}, "
            f"f1={final_metrics['f1']:.4f}, "
            f"f2={final_metrics['f2']:.4f}, "
            f"bal_acc={final_metrics['bal_acc']:.4f}, "
        )

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
        input_example = input_example.astype(np.float64)

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
            "f2": fbeta_score(y_train, y_pred_train, beta=2, zero_division=0),
            "bal_acc": balanced_accuracy_score(y_train, y_pred_train),
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
        n_selected_features = len(selected_features)
        cm_test = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = get_confusion_counts(cm_test)

        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(best_params)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("n_splits", N_SPLITS)
            mlflow.log_param("model_type", "XGBClassifier")
            mlflow.log_param("threshold", round(best_threshold, 5))
            mlflow.log_param(
                "cv_best_threshold",
                round(study.best_trial.user_attrs["best_threshold"], 5),
            )

            mlflow.log_param("opt_metric", f"{metric}")
            mlflow.log_metric("fn_test", fn)
            mlflow.log_metric("fp_test", fp)

            mlflow.log_metric("f1_train", round(final_metrics_train["f1"], 5))
            mlflow.log_metric("f1_test", round(final_metrics["f1"], 5))

            mlflow.log_metric("f2_train", round(final_metrics_train["f2"], 5))
            mlflow.log_metric("f2_test", round(final_metrics["f2"], 5))

            mlflow.log_metric("bal_acc_train", round(final_metrics_train["bal_acc"], 5))
            mlflow.log_metric("bal_acc_test", round(final_metrics["bal_acc"], 5))

            mlflow.log_metric(
                "accuracy_train", round(final_metrics_train["accuracy"], 5)
            )
            mlflow.log_metric("accuracy_test", round(final_metrics["accuracy"], 5))

            mlflow.log_metric("recall_train", round(final_metrics_train["recall"], 5))
            mlflow.log_metric("recall_test", round(final_metrics["recall"], 5))

            mlflow.log_metric(
                "precision_train", round(final_metrics_train["precision"], 5)
            )
            mlflow.log_metric("precision_test", round(final_metrics["precision"], 5))

            mlflow.log_metric("roc_auc_train", round(final_metrics_train["roc_auc"], 5))
            mlflow.log_metric("roc_auc_test", round(final_metrics["roc_auc"], 5))

            mlflow.log_artifact(model_output_path)
            mlflow.sklearn.log_model(final_model, name="final_model", input_example=input_example)  # type: ignore

            # Confusion matrix
            fig_cm, ax_cm = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm)
            fig_cm.tight_layout()
            fig_cm.savefig("confusion_matrix.png")
            plt.close(fig_cm)
            mlflow.log_artifact("confusion_matrix.png")
            os.remove("confusion_matrix.png")

            # ROC curve
            fig_roc, ax_roc = plt.subplots()
            RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=ax_roc)
            fig_roc.tight_layout()
            fig_roc.savefig("roc_curve.png")
            plt.close(fig_roc)
            mlflow.log_artifact("roc_curve.png")
            os.remove("roc_curve.png")

            # SHAP
            if hasattr(final_model, "feature_names_in_"):
                X_test = X_test[final_model.feature_names_in_]

            explainer = shap.Explainer(final_model, X_test)
            shap_values = explainer(X_test, check_additivity=False)

            shap_class_1 = shap_values

            # Строим dot plot
            plt.figure()
            shap.summary_plot(
                shap_class_1, X_test, plot_type="dot", show=False, max_display=39
            )
            plt.tight_layout()
            plt.savefig("shap_dot_plot.png")
            plt.close()

            # Логгируем артефакт в MLflow
            mlflow.log_artifact("shap_dot_plot.png")
            os.remove("shap_dot_plot.png")

            # Threshold vs metrics
            f1s, precisions, recalls = [], [], []
            for t in THRESHOLDS:
                y_pred_temp = (y_pred_proba >= t).astype(int)
                p, r, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred_temp, average="binary", zero_division=0
                )
                f1s.append(f1)
                precisions.append(p)
                recalls.append(r)

            plt.figure(figsize=(8, 6))
            plt.plot(THRESHOLDS, f1s, label="F1")
            plt.plot(THRESHOLDS, precisions, label="Precision")
            plt.plot(THRESHOLDS, recalls, label="Recall")
            plt.axvline(
                float(best_threshold),
                color="gray",
                linestyle="--",
                label=f"Threshold = {round(best_threshold, 3)}",
            )
            plt.xlabel("Threshold")
            plt.ylabel("Score")
            plt.title("Threshold vs F1 / Precision / Recall")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("threshold_metrics.png")
            plt.close()
            mlflow.log_artifact("threshold_metrics.png")
            os.remove("threshold_metrics.png")

            # Гистограмма предсказанных вероятностей
            plt.figure(figsize=(8, 6))
            plt.hist(y_pred_proba, bins=50, alpha=0.7)
            plt.axvline(
                float(best_threshold),
                color="red",
                linestyle="--",
                label=f"Threshold = {round(best_threshold, 3)}",
            )
            plt.title("Distribution of predicted probabilities")
            plt.xlabel("Probability")
            plt.ylabel("Count")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("proba_distribution.png")
            plt.close()
            mlflow.log_artifact("proba_distribution.png")
            os.remove("proba_distribution.png")

            # TP/FP/FN/TN vs Threshold plot
            tps, fps, fns, tns = [], [], [], []

            for t in THRESHOLDS:
                y_pred_temp = (y_pred_proba >= t).astype(int)
                cm_temp = confusion_matrix(y_test, y_pred_temp, labels=[0, 1])
                tn, fp, fn, tp = get_confusion_counts(cm_temp)
                tps.append(tp)
                fps.append(fp)
                fns.append(fn)
                tns.append(tn)

            plt.figure(figsize=(8, 6))
            plt.plot(THRESHOLDS, tps, label="TP")
            plt.plot(THRESHOLDS, fps, label="FP")
            plt.plot(THRESHOLDS, fns, label="FN")
            plt.plot(THRESHOLDS, tns, label="TN")
            plt.axvline(
                float(best_threshold),
                color="gray",
                linestyle="--",
                label=f"Threshold = {round(best_threshold, 3)}",
            )
            plt.xlabel("Threshold")
            plt.ylabel("Count")
            plt.title("TP / FP / FN / TN vs Threshold")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("threshold_confusion_counts.png")
            plt.close()
            mlflow.log_artifact("threshold_confusion_counts.png")
            os.remove("threshold_confusion_counts.png")

            selected_features_path = "selected_features.txt"
            with open(selected_features_path, "w") as f:
                for feat in selected_features:
                    f.write(f"{feat}\n")

            mlflow.log_param("n_selected_features", n_selected_features)
            mlflow.log_artifact(selected_features_path)
            os.remove(selected_features_path)

            # матрица кореляции
            corr_matrix = X_test[selected_features].corr(method="pearson")
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                square=True,
                cbar_kws={"shrink": 0.75},
                linewidths=0.5,
                linecolor="gray",
                annot_kws={"size": 6},
            )
            plt.title("Correlation Heatmap (Test Data)")
            plt.tight_layout()
            plt.savefig("correlation_heatmap.png")
            plt.close()
            mlflow.log_artifact("correlation_heatmap.png")
            os.remove("correlation_heatmap.png")

            # покажет высококорелирующиеся пары (|corr| > 0.9)
            high_corr_output = "high_corr_pairs.txt"
            corr_abs = corr_matrix.abs()
            upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))

            with open(high_corr_output, "w") as f:
                for col in upper.columns:
                    for row in upper.index:
                        val = upper.loc[row, col]
                        if pd.notnull(val) and val > 0.9:
                            f.write(f"{row} - {col}: {val:.3f}\n")

            mlflow.log_artifact(high_corr_output)
            os.remove(high_corr_output)

            params_path = "best_params.json"
            with open(params_path, "w") as f:
                json.dump(best_params, f, indent=4)

            mlflow.log_artifact(params_path)
            os.remove(params_path)

            experiment_config = {
                "TEST_SIZE": TEST_SIZE,
                "RANDOM_STATE": RANDOM_STATE,
                "N_TRIALS": N_TRIALS,
                "N_SPLITS": N_SPLITS,
                "METRIC": METRIC,
                "TARGET_COL": TARGET_COL,
                "FN_PENALTY_WEIGHT": FN_PENALTY_WEIGHT,
                "FP_PENALTY_WEIGHT": FP_PENALTY_WEIGHT,
                "MIN_PRECISION": MIN_PRECISION,
                "FN_STOP": FN_STOP,
                "MAX_FN_SOFT": MAX_FN_SOFT,
            }

            with open("experiment_config.json", "w") as f:
                json.dump(experiment_config, f, indent=4)
            mlflow.log_artifact("experiment_config.json")
            os.remove("experiment_config.json")

            # Логирование прогресса оптимизации Optuna (оценка trial на каждой итерации)
            scores = [trial.value for trial in study.trials if trial.value is not None]

            plt.figure(figsize=(10, 6))
            plt.plot(scores, marker="o", linestyle="-", alpha=0.8)
            plt.xlabel("Trial Number")
            plt.ylabel("Score")
            plt.title("Optuna Optimization Progress")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("optuna_progress.png")
            plt.close()
            mlflow.log_artifact("optuna_progress.png")
            os.remove("optuna_progress.png")

    except Exception as e:
        logger.info(f"Ошибка: {e}")
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

    # Найти integer столбцы с пропусками и привести их к float64
    int_cols_with_na = [
        col
        for col in TRAIN_DATA.select_dtypes(include=["int64", "Int64"]).columns
        if TRAIN_DATA[col].isnull().any()
    ]
    for col in int_cols_with_na:
        TRAIN_DATA[col] = TRAIN_DATA[col].astype("float64")

    # Для остальных числовых столбцов можно к float64
    num_cols = TRAIN_DATA.select_dtypes(include=["number"]).columns
    TRAIN_DATA[num_cols] = TRAIN_DATA[num_cols].astype("float64")

    X_main = TRAIN_DATA.drop(columns=[TARGET_COL])
    y_main = TRAIN_DATA[TARGET_COL]

    # Сетка параметров для полбора лучших
    fn_penalty_grid = np.arange(0, 2, 0.5)
    fp_penalty_grid = np.arange(0, 2, 0.5)
    fn_stop_grid = range(0, 2)
    max_fn_soft_grid = range(0, 2)

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

        X_train, X_test, y_train, y_test = train_test_split(
            X_main,
            y_main,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_main,
        )

        run_optuna_experiment(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            metric=METRIC,
            n_trials=N_TRIALS,
            experiment_name=f"{MLFLOW_EXPERIMENT}",
            model_output_path=f"{MODELS_DIR}/heart_pred.pkl",
            current_time=today(),
        )
