from __future__ import annotations
import math
from typing import Optional

from src.core.fuzzy_distance import FuzzyDistance


class FuzzyScorer:
    """
    Combines BM25 scores with fuzzy matching penalty.
    
    Formula: score = bm25_score * exp(-λ * edit_distance_norm)
    
    Where:
    - bm25_score: Traditional BM25 ranking score
    - edit_distance_norm: Normalized Levenshtein distance (0-1)
    - λ: Tuning parameter (3=lenient, 5=balanced, 7=strict)
    """
    
    def __init__(self, lambda_param: float = 5.0):
        """
        Initialize fuzzy scorer.
        
        Args:
            lambda_param: Penalty intensity.
                - 3.0: Very lenient (typos heavily tolerated)
                - 5.0: Balanced (default)
                - 7.0: Strict (heavy penalty for typos)
        """
        self.lambda_param = lambda_param
    
    def calculate_fuzzy_penalty(self, edit_distance_norm: float) -> float:
        """
        Calculate fuzzy penalty from normalized distance.
        
        Penalty = exp(-λ * distance_norm)
        
        Args:
            edit_distance_norm: Distance normalized to [0, 1]
        
        Returns:
            Penalty factor (0 < penalty <= 1)
        """
        penalty = math.exp(-self.lambda_param * edit_distance_norm)
        return penalty
    
    def score_with_fuzzy(
        self,
        bm25_score: float,
        query: str,
        best_match_term: Optional[str] = None,
        threshold: float = 0.5,
    ) -> tuple[float, dict]:
        """
        Apply fuzzy penalty to BM25 score.
        
        Args:
            bm25_score: Original BM25 score from ranking
            query: Search query
            best_match_term: Most relevant term from document (filename or key term)
            threshold: Ignore doc if edit_distance_norm > threshold
        
        Returns:
            Tuple of (final_score, metadata_dict)
            
        Metadata includes:
            - bm25_score: Original score
            - edit_distance_norm: Normalized distance
            - fuzzy_penalty: Applied penalty factor
            - final_score: After penalty
            - filtered_out: Whether doc was filtered by threshold
        """
        metadata = {
            "bm25_score": bm25_score,
            "edit_distance_norm": 0.0,
            "fuzzy_penalty": 1.0,
            "final_score": bm25_score,
            "filtered_out": False,
        }
        
        if not best_match_term:
            return bm25_score, metadata
        
        edit_distance_norm = FuzzyDistance.normalized_fuzzy_distance(query, best_match_term)
        metadata["edit_distance_norm"] = edit_distance_norm
        
        if threshold > 0 and edit_distance_norm > threshold:
            metadata["filtered_out"] = True
            return 0.0, metadata
        
        fuzzy_penalty = self.calculate_fuzzy_penalty(edit_distance_norm)
        metadata["fuzzy_penalty"] = fuzzy_penalty
        
        final_score = bm25_score * fuzzy_penalty
        metadata["final_score"] = final_score
        
        return final_score, metadata
    
    def score_batch(
        self,
        scored_results: list[tuple[float, str]],
        query: str,
        threshold: float = 0.5,
    ) -> list[tuple[float, str, dict]]:
        """
        Apply fuzzy penalty to a batch of BM25 scores.
        
        Args:
            scored_results: List of (bm25_score, match_term) tuples
            query: Search query
            threshold: Filter threshold
        
        Returns:
            List of (final_score, match_term, metadata) tuples, sorted by score
        """
        fuzzy_results = []
        
        for bm25_score, match_term in scored_results:
            final_score, metadata = self.score_with_fuzzy(
                bm25_score,
                query,
                best_match_term=match_term,
                threshold=threshold,
            )
            
            if metadata["filtered_out"]:
                continue
            
            fuzzy_results.append((final_score, match_term, metadata))
        
        fuzzy_results.sort(key=lambda x: x[0], reverse=True)
        
        return fuzzy_results