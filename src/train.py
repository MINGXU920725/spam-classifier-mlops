"""Train, evaluate, and save the deployable baseline model."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import normalize_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "spam_sms.csv"
DEFAULT_MODEL = PROJECT_ROOT / "artifacts" / "spam_classifier.joblib"
DEFAULT_METRICS = PROJECT_ROOT / "artifacts" / "metrics.json"


def load_dataset(path: Path) -> tuple[pd.Series, pd.Series]:
    data = pd.read_csv(path)
    required = {"v1", "v2"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    labels = data["v1"].map({"ham": 0, "spam": 1, 0: 0, 1: 1})
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError("Column v1 must contain only ham/spam or 0/1 labels")
    return data["v2"].fillna(""), labels.astype(int)


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train(data_path: Path, model_path: Path, metrics_path: Path) -> dict:
    texts, labels = load_dataset(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    report = classification_report(
        y_test,
        predictions,
        target_names=["ham", "spam"],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "spam_f1": float(f1_score(y_test, predictions, pos_label=1)),
        "spam_precision": float(report["spam"]["precision"]),
        "spam_recall": float(report["spam"]["recall"]),
        "test_samples": int(len(y_test)),
    }
    version = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    bundle = {
        "pipeline": pipeline,
        "model_version": version,
        "metrics": metrics,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    metrics_path.write_text(
        json.dumps({"model_version": version, **metrics}, indent=2),
        encoding="utf-8",
    )
    return {"model_version": version, **metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = train(args.data, args.model_out, args.metrics_out)
    print(json.dumps(result, indent=2))
