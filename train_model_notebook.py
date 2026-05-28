from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "Obesity"
ORDINAL_COLUMNS = ["FCVC", "NCP", "CH2O", "FAF", "TUE"]
RAW_NUMERIC_COLUMNS = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
RAW_CATEGORICAL_COLUMNS = [
    "Gender",
    "family_history",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
]
ENGINEERED_NUMERIC_COLUMNS = [f"{column}_rounded" for column in ORDINAL_COLUMNS]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.copy()
        for column in ORDINAL_COLUMNS:
            data[f"{column}_rounded"] = data[column].round()

        return data


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    df.columns = [column.strip() for column in df.columns]

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found in the CSV.")

    return df.drop_duplicates().reset_index(drop=True)


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_columns = RAW_NUMERIC_COLUMNS + ENGINEERED_NUMERIC_COLUMNS

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=numeric_steps), numeric_columns),
            ("categorical", categorical_pipeline, RAW_CATEGORICAL_COLUMNS),
        ]
    )


def build_candidate_pipelines() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("features", FeatureEngineer()),
                ("preprocessor", build_preprocessor(scale_numeric=True)),
                ("model", LogisticRegression(max_iter=3000, random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("features", FeatureEngineer()),
                ("preprocessor", build_preprocessor(scale_numeric=False)),
                ("model", RandomForestClassifier(n_estimators=400, random_state=42)),
            ]
        ),
    }


def evaluate_candidates(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[str, dict[str, dict[str, float]]]:
    candidates = build_candidate_pipelines()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results: dict[str, dict[str, float]] = {}

    for name, pipeline in candidates.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=splitter, scoring="accuracy")
        results[name] = {
            "cv_mean_accuracy": float(scores.mean()),
            "cv_std_accuracy": float(scores.std()),
        }

    best_name = max(results, key=lambda model_name: results[model_name]["cv_mean_accuracy"])
    return best_name, results


def train_and_evaluate(df: pd.DataFrame) -> tuple[Pipeline, dict[str, Any]]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    best_name, candidate_metrics = evaluate_candidates(X_train, y_train)
    best_pipeline = build_candidate_pipelines()[best_name]
    best_pipeline.fit(X_train, y_train)

    predictions = best_pipeline.predict(X_test)
    holdout_accuracy = accuracy_score(y_test, predictions)

    metrics = {
        "dataset_rows_after_deduplication": int(len(df)),
        "target_column": TARGET_COLUMN,
        "best_model": best_name,
        "candidate_metrics": candidate_metrics,
        "holdout_accuracy": float(holdout_accuracy),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=best_pipeline.classes_).tolist(),
        "class_labels": list(best_pipeline.classes_),
    }

    return best_pipeline, metrics


def save_artifacts(model: Pipeline, metrics: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "obesity_pipeline.joblib"
    metrics_path = output_dir / "metrics.json"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def train_from_csv(
    data_path: str | Path,
    output_dir: str | Path = "artifacts",
    save: bool = True,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train the obesity model from a notebook cell."""
    csv_path = Path(data_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = load_dataset(csv_path)
    model, metrics = train_and_evaluate(df)

    if save:
        save_artifacts(model, metrics, output_dir)

    return model, metrics


def metrics_summary(metrics: dict[str, Any]) -> pd.DataFrame:
    """Return the main training metrics as a compact DataFrame."""
    candidate_rows = []
    for model_name, model_metrics in metrics["candidate_metrics"].items():
        candidate_rows.append(
            {
                "model": model_name,
                "cv_mean_accuracy": model_metrics["cv_mean_accuracy"],
                "cv_std_accuracy": model_metrics["cv_std_accuracy"],
                "selected": model_name == metrics["best_model"],
            }
        )

    summary = pd.DataFrame(candidate_rows).sort_values(
        by="cv_mean_accuracy",
        ascending=False,
    )
    summary["holdout_accuracy"] = metrics["holdout_accuracy"]
    summary["dataset_rows_after_deduplication"] = metrics[
        "dataset_rows_after_deduplication"
    ]
    return summary.reset_index(drop=True)


def classification_report_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    """Return sklearn's classification report as a notebook-friendly DataFrame."""
    return pd.DataFrame(metrics["classification_report"]).T


def confusion_matrix_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    """Return the confusion matrix with class labels on rows and columns."""
    labels = metrics["class_labels"]
    return pd.DataFrame(
        metrics["confusion_matrix"],
        index=pd.Index(labels, name="actual"),
        columns=pd.Index(labels, name="predicted"),
    )


def save_metrics_json(metrics: dict[str, Any], path: str | Path) -> None:
    """Save metrics to a custom JSON path from a notebook."""
    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


__all__ = [
    "FeatureEngineer",
    "load_dataset",
    "build_preprocessor",
    "build_candidate_pipelines",
    "evaluate_candidates",
    "train_and_evaluate",
    "save_artifacts",
    "train_from_csv",
    "metrics_summary",
    "classification_report_frame",
    "confusion_matrix_frame",
    "save_metrics_json",
]
