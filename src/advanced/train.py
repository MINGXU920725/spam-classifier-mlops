"""Train the complete notebook-inspired V2 model and save every inference artifact."""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import hdbscan
import joblib
import numpy as np
import pandas as pd
import torch
from mlxtend.frequent_patterns import apriori
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from src.advanced.features import combine_features, soft_assignment, tokenize, word_feature_vector

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "spam_sms.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "advanced-v2"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EncodedSmsDataset(Dataset):
    def __init__(self, encodings: dict[str, torch.Tensor], labels: np.ndarray) -> None:
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {name: values[index] for name, values in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    labels = frame["v1"].map({"ham": 0, "spam": 1, 0: 0, 1: 1})
    if labels.isna().any():
        raise ValueError("v1 must contain only ham/spam or 0/1")
    return frame["v2"].fillna("").astype(str).to_numpy(), labels.astype(int).to_numpy()


def mine_frequent_itemsets(
    texts: np.ndarray, labels: np.ndarray, support: float
) -> list[list[str]]:
    transactions = [set(tokenize(text)) for text in texts[labels == 1]]
    vocabulary = sorted(set().union(*transactions))
    binary = pd.DataFrame(
        [[word in transaction for word in vocabulary] for transaction in transactions],
        columns=vocabulary,
        dtype=bool,
    )
    itemsets = apriori(binary, min_support=support, use_colnames=True)
    itemsets["lexical"] = itemsets["itemsets"].map(lambda values: "|".join(sorted(values)))
    itemsets = itemsets.sort_values(["support", "lexical"], ascending=[False, True])
    return [sorted(values) for values in itemsets["itemsets"]]


def encode(tokenizer: DistilBertTokenizerFast, texts: np.ndarray, max_length: int) -> dict:
    return tokenizer(
        texts.tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )


def fine_tune(
    model: DistilBertForSequenceClassification,
    dataset: EncodedSmsDataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            total_loss += float(output.loss.detach().cpu())
        print(f"epoch={epoch + 1}/{epochs} mean_loss={total_loss / len(loader):.6f}")


def embeddings(
    model: DistilBertForSequenceClassification,
    encodings: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    results: list[np.ndarray] = []
    count = len(encodings["input_ids"])
    with torch.no_grad():
        for start in range(0, count, batch_size):
            batch = {
                key: value[start : start + batch_size].to(device)
                for key, value in encodings.items()
            }
            output = model(**batch, output_hidden_states=True, return_dict=True)
            results.append(output.hidden_states[-1][:, 0, :].cpu().numpy())
    return np.vstack(results)


def metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    report = classification_report(
        y_true, predictions, target_names=["ham", "spam"], output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "spam_precision": float(report["spam"]["precision"]),
        "spam_recall": float(report["spam"]["recall"]),
        "spam_f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    texts, labels = load_data(args.data)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=args.test_size, random_state=args.seed, stratify=labels
    )

    print("Mining frequent spam itemsets...")
    frequent_itemsets = mine_frequent_itemsets(x_train, y_train, args.min_support)
    train_word = np.vstack([word_feature_vector(text, frequent_itemsets) for text in x_train])
    test_word = np.vstack([word_feature_vector(text, frequent_itemsets) for text in x_test])

    print("Loading and fine-tuning DistilBERT...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.base_model)
    model = DistilBertForSequenceClassification.from_pretrained(args.base_model, num_labels=2)
    train_encoding = encode(tokenizer, x_train, args.max_length)
    test_encoding = encode(tokenizer, x_test, args.max_length)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}")
    fine_tune(
        model,
        EncodedSmsDataset(train_encoding, y_train),
        device,
        args.epochs,
        args.train_batch_size,
        args.learning_rate,
    )

    print("Extracting semantic embeddings...")
    train_embeddings = embeddings(model, train_encoding, device, args.embedding_batch_size)
    test_embeddings = embeddings(model, test_encoding, device, args.embedding_batch_size)

    print("Training HDBSCAN and creating semantic features...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size, min_samples=args.min_samples
    )
    clusterer.fit(train_embeddings)
    cluster_labels = sorted(int(value) for value in np.unique(clusterer.labels_) if value != -1)
    if not cluster_labels:
        raise RuntimeError("HDBSCAN produced no valid clusters; adjust clustering parameters")
    cluster_exemplars = {
        label: np.asarray(clusterer.exemplars_[label], dtype=np.float32) for label in cluster_labels
    }
    train_semantic = soft_assignment(train_embeddings, cluster_labels, cluster_exemplars)
    test_semantic = soft_assignment(test_embeddings, cluster_labels, cluster_exemplars)
    train_features = combine_features(train_semantic, train_word)
    test_features = combine_features(test_semantic, test_word)

    scaler = StandardScaler().fit(train_features)
    logistic = LogisticRegression(random_state=args.seed, max_iter=1000).fit(
        scaler.transform(train_features), y_train
    )
    random_forest = RandomForestClassifier(n_estimators=100, random_state=args.seed).fit(
        train_features, y_train
    )
    candidates = {
        "logistic_regression": (
            logistic,
            logistic.predict(scaler.transform(test_features)),
        ),
        "random_forest": (random_forest, random_forest.predict(test_features)),
    }
    candidate_metrics = {name: metrics(y_test, result[1]) for name, result in candidates.items()}
    selected_name = max(candidate_metrics, key=lambda name: candidate_metrics[name]["spam_f1"])
    selected_model = candidates[selected_name][0]

    version = datetime.now(UTC).strftime("advanced-v2-%Y%m%d-%H%M%S")
    output = args.output
    bert_dir = output / "bert_model"
    tokenizer_dir = output / "tokenizer"
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(bert_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    bundle = {
        "frequent_itemsets": frequent_itemsets,
        "cluster_labels": cluster_labels,
        "cluster_exemplars": cluster_exemplars,
        "scaler": scaler,
        "classifier": selected_model,
        "classifier_kind": selected_name,
        "model_version": version,
    }
    joblib.dump(bundle, output / "pipeline_bundle.joblib")
    metadata = {
        "model_version": version,
        "base_model": args.base_model,
        "max_length": args.max_length,
        "feature_order": ["semantic", "word"],
        "semantic_feature_count": len(cluster_labels) + 1,
        "word_feature_count": len(frequent_itemsets),
        "combined_feature_count": int(train_features.shape[1]),
        "cluster_order": [*cluster_labels, -1],
        "classifier_kind": selected_name,
        "label_mapping": {"0": "ham", "1": "spam"},
        "classification_threshold": 0.5,
        "seed": args.seed,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    evaluation = {
        "model_version": version,
        "selected_classifier": selected_name,
        "candidates": candidate_metrics,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
    }
    (output / "metrics.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(json.dumps({**metadata, "metrics": evaluation}, indent=2))
    return evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--min-support", type=float, default=0.1)
    parser.add_argument("--min-cluster-size", type=int, default=30)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
