from __future__ import annotations
import math
from typing import Iterable


class TFIDFRanker:
    """Traditional TF-IDF ranking."""

    def __init__(self, index):
        self.index = index

    def rank_documents(
        self, doc_ids: Iterable[int], query_terms: list[str]
    ) -> list[tuple[int, float]]:
        """Rank documents by TF-IDF score."""
        scores: dict[int, float] = {}
        for term in query_terms:
            idf = math.log(
                1 + self.index.doc_count / (1 + self.index.get_document_frequency(term))
            )
            for doc_id in doc_ids:
                tf = self.index.get_term_frequency(term, doc_id)
                if tf > 0:
                    scores[doc_id] = scores.get(doc_id, 0) + tf * idf
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class BM25Ranker:
    """BM25 ranking algorithm (better for variable-length documents)."""

    def __init__(self, index, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            index: InvertedIndex instance
            k1: saturation parameter (1.5 typical)
            b: length normalization (0.75 typical)
        """
        self.index = index
        self.k1 = k1
        self.b = b
        self._avg_doc_length: float | None = None

    @property
    def avg_doc_length(self) -> float:
        """Cache average document length."""
        if self._avg_doc_length is None:
            self._avg_doc_length = self._compute_avg_doc_length()
        return self._avg_doc_length

    def _compute_avg_doc_length(self) -> float:
        """Compute average indexed bytes across all documents."""
        try:
            avg = self.index.db.get_avg_indexed_bytes()
            return avg if avg > 0 else 256_000
        except (AttributeError, Exception):
            return 256_000

    def rank_documents(
        self,
        doc_ids: Iterable[int],
        query_terms: list[str],
        fuzzy_confidence_map: dict[str, float] | None = None,
    ) -> list[tuple[int, float]]:
        """
        Rank documents using BM25 with optional fuzzy confidence boost.
        Formula:
        score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * (|D| / avgdl)))
                    * fuzzy_confidence(qi)
        Args:
            doc_ids: Document IDs to rank
            query_terms: Query terms (may include fuzzy matches)
            fuzzy_confidence_map: Maps term -> confidence score (0-1), 1.0 = exact, <1.0 = fuzzy
        """
        scores: dict[int, float] = {}
        avg_length = self.avg_doc_length
        fuzzy_confidence_map = fuzzy_confidence_map or {}
        for term in query_terms:
            df = self.index.get_document_frequency(term)
            idf = math.log(1 + (self.index.doc_count - df + 0.5) / (df + 0.5))
            fuzzy_confidence = fuzzy_confidence_map.get(term, 1.0)
            for doc_id in doc_ids:
                tf = self.index.get_term_frequency(term, doc_id)
                if tf == 0:
                    continue
                doc_data = self.index.get_document(doc_id)
                doc_length = doc_data.get("content_indexed_bytes", 0) or 1
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * (doc_length / avg_length)
                )
                bm25_component = idf * (numerator / denominator)
                final_score = bm25_component * fuzzy_confidence
                scores[doc_id] = scores.get(doc_id, 0) + final_score
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
