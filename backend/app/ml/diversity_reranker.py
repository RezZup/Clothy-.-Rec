"""
Maximal Marginal Relevance (MMR) reranking to prevent recommendation homogeneity.
"""

import numpy as np
from typing import List, Tuple

def mmr_rerank(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    candidate_ids: List[str],
    lambda_param: float = 0.65,
    top_k: int = 5
) -> List[str]:
    """
    MMR balances relevance to query against diversity among selected items.
    MR = argmax [ lambda * Sim(Di, Q) - (1 - lambda) * max(Sim(Di, Dj)) ]
    """
    if len(candidate_ids) == 0:
        return []
        
    sim_to_query = np.dot(candidate_vecs, query_vec)
    selected_indices = [int(np.argmax(sim_to_query))]
    remaining_indices = set(range(len(candidate_ids))) - {selected_indices[0]}
    
    while len(selected_indices) < min(top_k, len(candidate_ids)):
        best_idx = -1
        best_mmr_score = -float("inf")
        
        for cand_idx in remaining_indices:
            sim_q = sim_to_query[cand_idx]
            sim_selected = max(np.dot(candidate_vecs[cand_idx], candidate_vecs[sel]) for sel in selected_indices)
            mmr_score = lambda_param * sim_q - (1 - lambda_param) * sim_selected
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = cand_idx
                
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        
    return [candidate_ids[i] for i in selected_indices]
