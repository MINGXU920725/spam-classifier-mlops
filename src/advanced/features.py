"""Feature functions shared by advanced training and inference."""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import normalize_text

TOKEN_RE = re.compile(r"(?u)\b\w\w+\b|currency|phone_number|url")


def tokenize(text: str) -> list[str]:
    """Tokenize like the notebook while removing English stop words and numbers."""
    normalized = normalize_text(text)
    return [
        token
        for token in TOKEN_RE.findall(normalized)
        if not token.isdigit() and token not in ENGLISH_STOP_WORDS
    ]


def word_feature_vector(text: str, frequent_itemsets: Iterable[Iterable[str]]) -> np.ndarray:
    """Return one binary value per saved frequent itemset, preserving saved order."""
    words = set(tokenize(text))
    return np.asarray(
        [float(set(itemset).issubset(words)) for itemset in frequent_itemsets],
        dtype=np.float32,
    )


def soft_assignment(
    embeddings: np.ndarray,
    cluster_labels: list[int],
    cluster_exemplars: dict[int, np.ndarray],
) -> np.ndarray:
    """Reproduce the notebook's cosine-similarity cluster features plus noise."""
    rows: list[np.ndarray] = []
    for embedding in np.asarray(embeddings):
        sample = embedding.reshape(1, -1)
        similarities = [
            float(cosine_similarity(sample, cluster_exemplars[label]).mean())
            for label in cluster_labels
        ]
        noise_strength = 1.0 - max(similarities, default=0.0)
        strengths = np.asarray([*similarities, noise_strength], dtype=np.float64)
        total = strengths.sum()
        if np.isclose(total, 0.0):
            strengths = np.full(len(strengths), 1.0 / len(strengths))
        else:
            strengths /= total
        rows.append(strengths.astype(np.float32))
    return np.vstack(rows)


def combine_features(semantic: np.ndarray, word: np.ndarray) -> np.ndarray:
    """Keep the notebook's exact order: semantic features first, word features second."""
    semantic = np.asarray(semantic)
    word = np.asarray(word)
    if semantic.ndim == 1:
        semantic = semantic.reshape(1, -1)
    if word.ndim == 1:
        word = word.reshape(1, -1)
    if len(semantic) != len(word):
        raise ValueError("semantic and word feature row counts must match")
    return np.concatenate([semantic, word], axis=1)
