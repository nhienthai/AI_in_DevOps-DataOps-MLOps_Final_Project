"""Optuna hyperparameter search, one nested MLflow run per trial.

The search optimises cross-validated macro-F1 rather than a single held-out
score. With ``neutral`` at roughly 4% of UIT-VSFC, a single split rewards trials
that got a favourable fold, and the sweep then confidently reports a
configuration that does not reproduce. Cross-validating inside the objective
costs ``n_splits`` times more compute and is the only way the ranking means
anything.

Each trial becomes a nested MLflow run under one parent, so the sweep reads as a
single experiment with its trials attached rather than as N unrelated runs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import optuna

from sentiment.training.evaluate import cross_validate_baseline

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest(trial: optuna.Trial) -> Dict[str, Any]:
    """Draw one hyperparameter configuration."""
    max_ngram = trial.suggest_int("tfidf_max_ngram", 1, 3)
    return {
        "tfidf_max_ngram": max_ngram,
        "tfidf_max_features": trial.suggest_categorical(
            "tfidf_max_features", [5000, 10000, 20000, 50000]
        ),
        "tfidf_min_df": trial.suggest_int("tfidf_min_df", 1, 5),
        "tfidf_sublinear_tf": trial.suggest_categorical("tfidf_sublinear_tf", [True, False]),
        "clf_C": trial.suggest_float("clf_C", 0.01, 20.0, log=True),
        "clf_class_weight": trial.suggest_categorical("clf_class_weight", ["balanced", None]),
    }


def _split_params(config: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Split a flat trial configuration into vectoriser and classifier kwargs."""
    tfidf_params = {
        "ngram_range": (1, int(config["tfidf_max_ngram"])),
        "max_features": int(config["tfidf_max_features"]),
        "min_df": int(config["tfidf_min_df"]),
        "sublinear_tf": bool(config["tfidf_sublinear_tf"]),
    }
    clf_params = {
        "C": float(config["clf_C"]),
        "class_weight": config["clf_class_weight"],
    }
    return tfidf_params, clf_params


def tune_baseline(
    texts: List[str],
    labels: List[int],
    n_trials: int = 12,
    n_splits: int = 3,
    seed: int = 42,
    log_to_mlflow: bool = True,
) -> Dict[str, Any]:
    """Search baseline hyperparameters, maximising cross-validated macro-F1.

    Args:
        texts: Training texts.
        labels: Integer labels aligned with ``texts``.
        n_trials: Number of Optuna trials. Each is one nested MLflow run.
        n_splits: Folds used inside the objective.
        seed: Seed for the sampler and the folds, so the sweep is reproducible.
        log_to_mlflow: When false, run the search without touching MLflow. Used
            by tests, which must not require a tracking server.

    Returns:
        The best configuration, its cross-validated scores, and every trial.
    """
    trials: List[Dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        config = _suggest(trial)
        tfidf_params, clf_params = _split_params(config)
        scores = cross_validate_baseline(
            texts,
            labels,
            n_splits=n_splits,
            seed=seed,
            tfidf_params=tfidf_params,
            clf_params=clf_params,
        )
        record = {
            "trial": trial.number,
            "params": config,
            "cv_macro_f1_mean": scores["cv_macro_f1_mean"],
            "cv_macro_f1_std": scores["cv_macro_f1_std"],
            "cv_accuracy_mean": scores["cv_accuracy_mean"],
        }
        trials.append(record)

        if log_to_mlflow:
            import mlflow

            with mlflow.start_run(nested=True, run_name=f"trial-{trial.number:03d}"):
                mlflow.log_params({f"tune.{key}": value for key, value in config.items()})
                mlflow.log_metrics(
                    {
                        "cv_macro_f1_mean": scores["cv_macro_f1_mean"],
                        "cv_macro_f1_std": scores["cv_macro_f1_std"],
                        "cv_accuracy_mean": scores["cv_accuracy_mean"],
                    }
                )

        logger.info(
            "trial %d: macro-F1 %.4f (+/- %.4f)",
            trial.number,
            scores["cv_macro_f1_mean"],
            scores["cv_macro_f1_std"],
        )
        return float(scores["cv_macro_f1_mean"])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name="baseline-tfidf-logreg",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_tfidf, best_clf = _split_params(dict(study.best_trial.params))
    result = {
        "best_trial": study.best_trial.number,
        "best_value": float(study.best_value),
        "best_params": dict(study.best_trial.params),
        "best_tfidf_params": best_tfidf,
        "best_clf_params": best_clf,
        "n_trials": len(trials),
        "trials": trials,
    }

    if log_to_mlflow:
        import mlflow

        mlflow.log_params({f"best.{key}": value for key, value in study.best_trial.params.items()})
        mlflow.log_metric("best_cv_macro_f1", float(study.best_value))

    logger.info(
        "best trial %d with cross-validated macro-F1 %.4f",
        study.best_trial.number,
        study.best_value,
    )
    return result
