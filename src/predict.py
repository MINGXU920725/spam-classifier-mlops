"""Load a trained model and expose one-message prediction."""

from __future__ import annotations

from pathlib import Path

import joblib


class SpamPredictor:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Run `python -m src.train` first."
            )
        bundle = joblib.load(self.model_path)
        self.pipeline = bundle["pipeline"]
        self.model_version = bundle.get("model_version", "unknown")
        self.metrics = bundle.get("metrics", {})

    def predict(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        spam_probability = float(self.pipeline.predict_proba([text])[0][1])
        label = "spam" if spam_probability >= 0.5 else "ham"
        return {
            "label": label,
            "spam_probability": round(spam_probability, 6),
            "model_version": self.model_version,
        }
