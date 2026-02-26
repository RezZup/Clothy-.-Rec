"""Tests for MMR diversity reranking algorithm."""
import numpy as np
from backend.app.ml.diversity_reranker import mmr_rerank

def test_mmr_selects_diverse_items():
    query = np.array([1.0, 0.0], dtype=np.float32)
    # Cand 0 and Cand 1 are identical duplicates pointing to query
    # Cand 2 is slightly different but still relevant
    candidates = np.array([
        [1.0, 0.0],
        [0.99, 0.01],
        [0.8, 0.6]
    ], dtype=np.float32)
    ids = ["id_0", "id_1", "id_2"]
    
    # Under high diversity penalty (lambda = 0.5), id_2 should be prioritized over duplicate id_1
    reranked = mmr_rerank(query, candidates, ids, lambda_param=0.4, top_k=2)
    assert reranked[0] == "id_0"
    assert reranked[1] == "id_2"
