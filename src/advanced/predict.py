"""Load saved V2 artifacts and predict without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from src.advanced.features import combine_features, soft_assignment, word_feature_vector
from src.preprocessing import normalize_text

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = ROOT / "artifacts" / "advanced-v2"


class AdvancedSpamPredictor:
    def __init__(self, artifact_dir: str | Path = DEFAULT_ARTIFACTS, cpu: bool = False) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.metadata = json.loads(
            (self.artifact_dir / "metadata.json").read_text(encoding="utf-8")
        )
        bundle = joblib.load(self.artifact_dir / "pipeline_bundle.joblib")
        self.frequent_itemsets = bundle["frequent_itemsets"]
        self.cluster_labels = bundle["cluster_labels"]
        self.cluster_exemplars = bundle["cluster_exemplars"]
        self.scaler = bundle["scaler"]
        self.classifier = bundle["classifier"]
        self.classifier_kind = bundle["classifier_kind"]
        self.model_version = bundle["model_version"]
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.artifact_dir / "tokenizer")
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.artifact_dir / "bert_model"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
        self.model.to(self.device).eval()

    def _embedding(self, text: str) -> np.ndarray:
        encoded = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self.metadata["max_length"],
            return_tensors="pt",
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded, output_hidden_states=True, return_dict=True)
        return output.hidden_states[-1][:, 0, :].cpu().numpy()

    def predict(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        word = word_feature_vector(text, self.frequent_itemsets)
        semantic = soft_assignment(
            self._embedding(text), self.cluster_labels, self.cluster_exemplars
        )
        features = combine_features(semantic, word)
        expected = self.metadata["combined_feature_count"]
        if features.shape[1] != expected:
            raise RuntimeError(
                f"feature dimension mismatch: expected {expected}, got {features.shape[1]}"
            )
        model_input = (
            self.scaler.transform(features)
            if self.classifier_kind == "logistic_regression"
            else features
        )
        probability = float(self.classifier.predict_proba(model_input)[0, 1])
        normalized = normalize_text(text)
        return {
            "label": "spam" if probability >= self.metadata["classification_threshold"] else "ham",
            "spam_probability": round(probability, 6),
            "model_version": self.model_version,
            "detected_patterns": {
                "phone_number": "phone_number" in normalized,
                "currency": "currency" in normalized,
                "url": "url" in normalized,
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    print(json.dumps(AdvancedSpamPredictor(args.artifacts, args.cpu).predict(args.text), indent=2))


if __name__ == "__main__":
    main()
