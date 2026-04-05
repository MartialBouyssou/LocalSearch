from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

from src.core.models import SearchResult, Document
from src.core.tokenizer import Tokenizer
from src.core.ranking import BM25Ranker
from src.core.index import InvertedIndex
from src.core.fuzzy_scorer import FuzzyScorer
from src.core.fuzzy_distance import FuzzyDistance
from src.infrastructure.db_storage import DBStorage
from src.infrastructure.content_extractor import ContentExtractor


class SearchEngine:
    """
    Full-text search engine with BM25 ranking and optional fuzzy matching.
    
    Features:
    - BM25 relevance ranking
    - Fuzzy matching with typo tolerance
    - Lazy content upgrade for partial documents
    - Configurable fuzzy penalty function
    """
    
    def __init__(
        self,
        db_storage: DBStorage,
        extractor: ContentExtractor,
        enable_fuzzy: bool = True,
        fuzzy_lambda: float = 5.0,
    ):
        """
        Initialize search engine.
        
        Args:
            db_storage: Database storage instance
            extractor: Content extractor instance
            enable_fuzzy: Enable fuzzy matching by default
            fuzzy_lambda: Fuzzy penalty parameter (3=lenient, 5=balanced, 7=strict)
        
        Raises:
            RuntimeError: If index is empty
        """
        self.db = db_storage
        self.extractor = extractor
        self.enable_fuzzy = enable_fuzzy
        self.fuzzy_lambda = fuzzy_lambda
        
        self.db.open()
        self.index = InvertedIndex(self.db)
        self.ranker = BM25Ranker(self.index)
        self.fuzzy_scorer = FuzzyScorer(lambda_param=fuzzy_lambda) if enable_fuzzy else None
        
        if self.index.doc_count == 0:
            raise RuntimeError("Index is empty. Please index files first.")

    def close(self) -> None:
        """Close database connection."""
        self.db.close()

    def search(
        self,
        query: str,
        top_k: int = 10,
        lazy_upgrade: bool = True,
        use_fuzzy: bool = True,
        fuzzy_threshold: float = 0.5,
    ) -> list[SearchResult]:
        """
        Search index with optional fuzzy matching.
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return
            lazy_upgrade: Upgrade partial documents when filename matches query
            use_fuzzy: Apply fuzzy matching penalty to results
            fuzzy_threshold: Filter documents with fuzzy distance > threshold
                          (0.0 = exact only, 1.0 = any match)
        
        Returns:
            List of SearchResult sorted by relevance score (fuzzy or BM25)
        """
        terms = Tokenizer.tokenize(query)
        if not terms:
            return []

        if use_fuzzy and self.enable_fuzzy and self.fuzzy_scorer:
            term_expansion = self._expand_terms_with_fuzzy(terms, fuzzy_threshold)
            expanded_terms = list({t for hits in term_expansion.values() for t in hits})
        else:
            expanded_terms = terms

        doc_matches = self.index.search_terms(expanded_terms)
        if not doc_matches:
            return []

        doc_ids = list(doc_matches.keys())
        ranked_bm25 = self.ranker.rank_documents(doc_ids, expanded_terms)

        if use_fuzzy and self.enable_fuzzy and self.fuzzy_scorer:
            results = self._apply_fuzzy_ranking(
                ranked_bm25=ranked_bm25,
                doc_matches=doc_matches,
                terms=terms,
                query=query,
                lazy_upgrade=lazy_upgrade,
                fuzzy_threshold=fuzzy_threshold,
                top_k=top_k,
            )
        else:
            results = self._create_results_from_bm25(
                ranked_bm25=ranked_bm25,
                doc_matches=doc_matches,
                terms=terms,
                lazy_upgrade=lazy_upgrade,
                top_k=top_k,
            )
        
        return results

    def _expand_terms_with_fuzzy(
        self,
        query_terms: list[str],
        fuzzy_threshold: float,
    ) -> dict[str, list[str]]:
        """
        Map each query term to the list of index terms that are close enough.

        For each query term we iterate over all indexed terms and keep those
        whose normalized Levenshtein distance is within *fuzzy_threshold*.
        A query term also always maps to itself so that exact matches are
        included regardless of the threshold.

        Args:
            query_terms: Tokenized terms from the user query.
            fuzzy_threshold: Maximum allowed normalized distance (0-1).

        Returns:
            Mapping  query_term -> [matching_index_terms].
        """
        all_index_terms = self.index.get_all_terms()
        expansion: dict[str, list[str]] = {}

        for query_term in query_terms:
            matched: list[str] = []
            for index_term in all_index_terms:
                dist = FuzzyDistance.normalized_fuzzy_distance(query_term, index_term)
                if dist <= fuzzy_threshold:
                    matched.append(index_term)
            if not matched:
                matched = [query_term]
            expansion[query_term] = matched

        return expansion

    def _apply_fuzzy_ranking(
        self,
        ranked_bm25: list[tuple[int, float]],
        doc_matches: dict,
        terms: list[str],
        query: str,
        lazy_upgrade: bool,
        fuzzy_threshold: float,
        top_k: int,
    ) -> list[SearchResult]:
        """
        Apply fuzzy penalty to BM25 scores.

        The penalty is derived from the best (smallest) normalized edit-distance
        between any query term and any index term that was actually matched for the
        document.  Using the minimum distance ensures that an exact token match
        produces zero penalty, while a typo introduces a proportional penalty.

        Formula: final_score = bm25_score * exp(-λ * min_distance)
        """
        results_with_fuzzy: list[SearchResult] = []

        for doc_id, bm25_score in ranked_bm25:
            doc_data = self.index.get_document(doc_id)
            if not doc_data:
                continue

            if lazy_upgrade and int(doc_data.get("content_partial", 0)) == 1:
                filename = (doc_data.get("filename") or "").lower()
                if any(t in filename for t in terms):
                    self._upgrade_document_content(doc_id, doc_data)
                    doc_data = self.index.get_document(doc_id) or doc_data

            matched_index_terms = list(doc_matches.get(doc_id, []))

            min_distance = 1.0
            for query_term in terms:
                for index_term in matched_index_terms:
                    dist = FuzzyDistance.normalized_fuzzy_distance(query_term, index_term)
                    if dist < min_distance:
                        min_distance = dist

            fuzzy_score, metadata = self.fuzzy_scorer.score_with_fuzzy(
                bm25_score=bm25_score,
                query=query,
                best_match_term=None,
                threshold=0,
            )
            fuzzy_penalty = self.fuzzy_scorer.calculate_fuzzy_penalty(min_distance)
            fuzzy_score = bm25_score * fuzzy_penalty
            metadata["edit_distance_norm"] = round(min_distance, 4)
            metadata["fuzzy_penalty"] = round(fuzzy_penalty, 4)
            metadata["final_score"] = fuzzy_score

            filename = doc_data.get("filename", "")
            document = Document(
                doc_id=doc_id,
                filename=filename,
                path=doc_data.get("path", ""),
                extension=doc_data.get("extension", ""),
                content="",
            )

            result = SearchResult(
                document=document,
                score=fuzzy_score,
                matched_terms=matched_index_terms,
                metadata=metadata,
            )
            results_with_fuzzy.append(result)

        results_with_fuzzy.sort(key=lambda r: r.score, reverse=True)

        return results_with_fuzzy[:top_k]

    def _create_results_from_bm25(
        self,
        ranked_bm25: list[tuple[int, float]],
        doc_matches: dict,
        terms: list[str],
        lazy_upgrade: bool,
        top_k: int,
    ) -> list[SearchResult]:
        """
        Create SearchResult objects from BM25 scores without fuzzy matching.
        """
        results: list[SearchResult] = []
        
        for doc_id, score in ranked_bm25[:top_k]:
            doc_data = self.index.get_document(doc_id)
            if not doc_data:
                continue

            if lazy_upgrade and int(doc_data.get("content_partial", 0)) == 1:
                filename = (doc_data.get("filename") or "").lower()
                if any(t in filename for t in terms):
                    self._upgrade_document_content(doc_id, doc_data)
                    doc_data = self.index.get_document(doc_id) or doc_data

            document = Document(
                doc_id=doc_id,
                filename=doc_data.get("filename", ""),
                path=doc_data.get("path", ""),
                extension=doc_data.get("extension", ""),
                content="",
            )
            
            results.append(
                SearchResult(
                    document=document,
                    score=score,
                    matched_terms=list(doc_matches.get(doc_id, [])),
                )
            )

        return results

    def search_with_details(
        self,
        query: str,
        top_k: int = 10,
        lazy_upgrade: bool = True,
        use_fuzzy: bool = True,
        fuzzy_threshold: float = 0.5,
    ) -> tuple[list[SearchResult], dict]:
        """
        Search and return results with detailed calculation metadata.
        
        Returns:
            Tuple of (results, metadata_dict)
            
            metadata_dict contains:
            - query: Original search query
            - terms: Tokenized terms from query
            - use_fuzzy: Whether fuzzy was used
            - fuzzy_threshold: Threshold used
            - count: Number of results
        """
        results = self.search(
            query=query,
            top_k=top_k,
            lazy_upgrade=lazy_upgrade,
            use_fuzzy=use_fuzzy,
            fuzzy_threshold=fuzzy_threshold,
        )
        
        terms = Tokenizer.tokenize(query)
        metadata = {
            "query": query,
            "terms": terms,
            "use_fuzzy": use_fuzzy,
            "fuzzy_threshold": fuzzy_threshold,
            "count": len(results),
        }
        
        return results, metadata

    def get_fuzzy_info(self, term1: str, term2: str) -> dict:
        """
        Get detailed fuzzy matching information between two terms.
        
        Useful for debugging and understanding fuzzy scores.
        
        Args:
            term1: First term
            term2: Second term
        
        Returns:
            Dictionary with distance and similarity information
        """
        distance = FuzzyDistance.levenshtein_distance(term1, term2)
        normalized_distance = FuzzyDistance.normalized_fuzzy_distance(term1, term2)
        similarity = FuzzyDistance.similarity_score(term1, term2)
        
        if self.fuzzy_scorer:
            penalty = self.fuzzy_scorer.calculate_fuzzy_penalty(normalized_distance)
        else:
            penalty = 1.0
        
        return {
            "term1": term1,
            "term2": term2,
            "levenshtein_distance": distance,
            "normalized_distance": round(normalized_distance, 4),
            "similarity_score": round(similarity, 4),
            "fuzzy_penalty": round(penalty, 4),
            "is_exact_match": distance == 0,
            "lambda": self.fuzzy_lambda,
        }

    def _upgrade_document_content(self, doc_id: int, doc_data: dict) -> None:
        """
        Upgrade partial document content with full extraction.
        
        This is called during lazy upgrade when a partially indexed document
        matches the query.
        
        Args:
            doc_id: Document ID to upgrade
            doc_data: Current document data
        """
        full_path = Path(str(doc_data["path"])) / str(doc_data["filename"])

        extracted = self.extractor.extract_full_for_upgrade(full_path)
        if not extracted.text:
            self.db.update_document_content_flags(doc_id, content_partial=0, content_indexed_bytes=0)
            self.db.commit()
            return

        tokens = []
        tokens.extend(Tokenizer.tokenize_filename(str(doc_data["filename"])))
        tokens.extend(Tokenizer.tokenize(extracted.text))
        freq = Counter(tokens)

        self.db.begin()
        try:
            self.db.delete_postings_for_doc(doc_id)
            if freq:
                self.db.ensure_terms(freq.keys())
                self.db.upsert_postings((t, doc_id, f) for t, f in freq.items())

            self.db.update_document_content_flags(
                doc_id,
                content_partial=0 if not extracted.partial else 1,
                content_indexed_bytes=len(extracted.text.encode("utf-8", errors="ignore")),
            )
            self.db.commit()
        except Exception:
            self.db.commit()
            raise