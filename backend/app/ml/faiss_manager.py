"""
FAISS index management and sub-millisecond vector similarity search.
"""

import numpy as np
from typing import List, Dict, Any, Tuple

class FaissIndexManager:
    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.item_ids = []
        self.vectors = None
        
    def initialize_mock_index(self, num_items: int = 1000):
        """Initialize mock normalized index for unit tests."""
        np.random.seed(42)
        raw_vecs = np.random.randn(num_items, self.dimension).astype(np.float32)
        # L2 Normalize
        norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
        self.vectors = raw_vecs / norms
        self.item_ids = [f"item_{i:05d}" for i in range(num_items)]

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Cosine similarity search using dot product on L2-normalized vectors."""
        if self.vectors is None:
            raise RuntimeError("Index not initialized")
        q_norm = query_vec / np.linalg.norm(query_vec)
        scores = np.dot(self.vectors, q_norm.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.item_ids[idx], float(scores[idx])) for idx in top_indices]
