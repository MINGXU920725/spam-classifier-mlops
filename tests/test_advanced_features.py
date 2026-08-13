import numpy as np

from src.advanced.features import combine_features, soft_assignment, word_feature_vector


def test_saved_itemset_order_controls_word_feature_order() -> None:
    itemsets = [["free"], ["claim", "prize"], ["missing"]]
    result = word_feature_vector("Claim your FREE prize", itemsets)
    assert result.tolist() == [1.0, 1.0, 0.0]


def test_soft_assignment_and_combined_dimensions() -> None:
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    exemplars = {
        0: np.array([[1.0, 0.0]], dtype=np.float32),
        1: np.array([[0.0, 1.0]], dtype=np.float32),
    }
    semantic = soft_assignment(embeddings, [0, 1], exemplars)
    combined = combine_features(semantic, np.array([[1.0, 0.0]], dtype=np.float32))
    assert semantic.shape == (1, 3)
    assert np.isclose(semantic.sum(), 1.0)
    assert combined.shape == (1, 5)

