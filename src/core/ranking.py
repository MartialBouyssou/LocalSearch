from __future__ import annotations

import math
from typing import Iterable


class TFIDFRanker:
    """Traditional TF-IDF ranking."""

    def __init__(self, index):
        self.index = index

    def rank_documents(self, doc_ids: Iterable[int], query_terms: list[str]) -> list[tuple[int, float]]:
        """Rank documents by TF-IDF score."""
        scores: dict[int, float] = {}

        for term in query_terms:
            idf = math.log(1 + self.index.doc_count / (1 + self.index.get_document_frequency(term)))

            for doc_id in doc_ids:
                tf = self.index.get_term_frequency(term, doc_id)
                if tf > 0:
                    scores[doc_id] = scores.get(doc_id, 0) + tf * idf

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class BM25Ranker:
    """
    BM25 ranking algorithm with token-count-based length normalisation.

    Key design decisions:
    - Document length is measured in **tokens** (sum of posting frequencies),
      not in bytes.  Raw bytes skew scores heavily for PDFs where the byte
      count grows with embedded images and metadata, not actual text density.
    - All DB access is batched: one query for postings, one for token counts.
      This replaces the previous N×M individual round-trips.
    - ``avg_doc_length`` is computed once and cached for the lifetime of the
      ranker instance.
    """

    def __init__(self, index, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            index: InvertedIndex instance.
            k1:    Term-frequency saturation parameter (1.5 typical).
            b:     Length normalisation strength (0.75 typical).
        """
        self.index = index
        self.k1 = k1
        self.b = b
        self._avg_doc_length: float | None = None

    @property
    def avg_doc_length(self) -> float:
        """Average document length in tokens, computed once and cached."""
        if self._avg_doc_length is None:
            self._avg_doc_length = self.index.get_avg_token_count()
        return self._avg_doc_length

    def rank_documents(self, doc_ids: Iterable[int], query_terms: list[str]) -> list[tuple[int, float]]:
        """
        Rank documents using BM25.

        Formula:
        score(D,Q) = Σ IDF(qi) * tf*(k1+1) / (tf + k1*(1-b+b*|D|/avgdl))

        where |D| is the token count of document D (not byte size).
        All DB access uses two batched queries to minimise round-trips.
        """
        doc_ids_list = list(doc_ids)
        if not doc_ids_list or not query_terms:
            return []

        postings = self.index.get_postings_batch(query_terms, doc_ids_list)
        token_counts = self.index.get_doc_token_counts_batch(doc_ids_list)
        avg_length = self.avg_doc_length
        total_docs = self.index.doc_count

        scores: dict[int, float] = {}

        for term in query_terms:
            df = self.index.get_document_frequency(term)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            term_postings = postings.get(term, {})

            for doc_id, tf in term_postings.items():
                doc_length = token_counts.get(doc_id, 1)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / avg_length))
                scores[doc_id] = scores.get(doc_id, 0) + idf * (numerator / denominator)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
