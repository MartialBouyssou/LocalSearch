from __future__ import annotations
from collections import Counter
from pathlib import Path
from src.core.models import SearchResult, Document
from src.core.pattern_matcher import PatternMatcher
from src.core.tokenizer import Tokenizer
from src.core.ranking import BM25Ranker
from src.core.index import InvertedIndex
from src.core.fuzzy_matcher import FuzzyMatcher
from src.core.wildcard_matcher import WildcardMatcher
from src.infrastructure.db_storage import DBStorage
from src.infrastructure.content_extractor import ContentExtractor
from src.infrastructure.term_cache import TermCache
from src.infrastructure.path_filter import PathFilter


class SearchEngine:
    def __init__(self, db_storage: DBStorage, extractor: ContentExtractor):
        self.db = db_storage
        self.extractor = extractor
        self.term_cache = TermCache()
        self.db.open()
        self.index = InvertedIndex(self.db)
        self.ranker = BM25Ranker(self.index)
        if self.index.doc_count == 0:
            raise RuntimeError("Index is empty. Please index files first.")
        self._init_term_cache()

    def _init_term_cache(self) -> None:
        """Load all terms into memory."""
        if not self.term_cache.is_initialized():
            all_terms = self.db.get_all_terms()
            self.term_cache.initialize(all_terms)

    def close(self) -> None:
        self.db.close()

    def search(
        self, query: str, top_k: int = 10, lazy_upgrade: bool = True
    ) -> list[SearchResult]:
        """
        Multi-level search
        """
        query_terms = Tokenizer.tokenize(query)
        if not query_terms:
            return []
        is_wildcard = "*" in query or "?" in query
        if not is_wildcard:
            filename_results = self._search_exact_filename(query, top_k, lazy_upgrade)
            if filename_results:
                return filename_results
        if is_wildcard:
            wildcard_results = self._search_wildcard_filename(
                query, top_k, lazy_upgrade
            )
            if wildcard_results:
                return wildcard_results
            return []
        stem_terms = self._expand_with_stems(query_terms)
        doc_matches = self.db.search_exact_all_terms(stem_terms, limit=100)
        if doc_matches:
            results = self._build_results(
                doc_matches, stem_terms, top_k, lazy_upgrade, match_type="exact"
            )
            filtered = self._filter_results(results)
            if filtered:
                return filtered
        doc_matches = self.db.search_prefix_terms(stem_terms, limit=500)
        if doc_matches:
            results = self._build_results(
                doc_matches, stem_terms, top_k, lazy_upgrade, match_type="prefix"
            )
            filtered = self._filter_results(results)
            if filtered:
                return filtered
        fuzzy_results = self._fuzzy_search_with_azerty(query_terms)
        if fuzzy_results:
            filtered = self._filter_results(fuzzy_results)
            if filtered:
                return filtered
        return []

    def _search_exact_filename(
        self, query: str, top_k: int, lazy_upgrade: bool
    ) -> list[SearchResult]:
        """Search for exact filename matches."""
        query_lower = query.lower()
        query_clean = "".join(c for c in query_lower if c.isalnum() or c in ".-_")
        if not query_clean or len(query_clean) < 2:
            return []
        doc_ids = self.db.search_documents_by_filename_exact(query_clean)
        if doc_ids:
            results = []
            seen = set()
            for doc_id in doc_ids[: top_k * 2]:
                if doc_id in seen:
                    continue
                doc_data = self.index.get_document(doc_id)
                if not doc_data:
                    continue
                path = doc_data.get("path", "")
                filename = doc_data.get("filename", "")
                extension = doc_data.get("extension", "")
                if not PathFilter.should_include(path, filename, extension):
                    continue
                seen.add(doc_id)
                document = Document(
                    doc_id=doc_id,
                    filename=filename,
                    path=path,
                    extension=extension,
                    content="",
                )
                results.append(
                    SearchResult(
                        document=document,
                        score=1000.0,
                        matched_terms=[query_clean],
                        match_type="exact_filename",
                        fuzzy_confidence=1.0,
                    )
                )
            return results[:top_k]
        return []

    def _search_wildcard_filename(
        self, pattern: str, top_k: int, lazy_upgrade: bool
    ) -> list[SearchResult]:
        """
        Search for filenames matching wildcard pattern.
        STRICT: Only matches actual filenames with pattern.
        """
        pattern_lower = pattern.lower()
        try:
            cur = self.db.conn.cursor()
            cur.execute(
                "SELECT id, filename, path, extension FROM documents LIMIT 1000000"
            )
            results = []
            seen = set()
            for row in cur.fetchall():
                doc_id = int(row[0])
                filename = row[1]
                path = row[2]
                extension = row[3]
                if not PatternMatcher.matches_pattern(filename, pattern_lower):
                    continue
                if not PathFilter.should_include(path, filename, extension):
                    continue
                key = (filename, path)
                if key in seen:
                    continue
                seen.add(key)
                document = Document(
                    doc_id=doc_id,
                    filename=filename,
                    path=path,
                    extension=extension,
                    content="",
                )
                results.append(
                    SearchResult(
                        document=document,
                        score=1000.0,
                        matched_terms=[pattern_lower],
                        match_type="wildcard_filename",
                        fuzzy_confidence=1.0,
                    )
                )
                if len(results) >= top_k:
                    break
            return results[:top_k]
        except Exception as e:
            print(f"Error in wildcard search: {e}")
            return []

    def _expand_with_stems(self, query_terms: list[str]) -> list[str]:
        """Expand query terms with their FR/EN stems (for better recall)."""
        from src.core.stemmer import Stemmer

        expanded = list(query_terms)
        for term in query_terms:
            for stem in Stemmer.expand(term):
                if stem not in expanded:
                    expanded.append(stem)
        return expanded

    def _fuzzy_search_with_azerty(self, query_terms: list[str]) -> list[SearchResult]:
        """
        Fuzzy search combining:
        1. Fast prefix-cache lookup (O(log n))
        2. AZERTY-aware Levenshtein on the term cache (bounded by MAX_DISTANCE=2)
        Stays well under 5 s for typical index sizes.
        """
        if not self.term_cache.is_initialized():
            self._init_term_cache()
        all_terms = self.term_cache.get_all_terms()
        if not all_terms:
            return []
        all_fuzzy_matches: dict[str, float] = {}
        for query_term in query_terms:
            if len(query_term) < 2:
                continue
            prefix_matches = self.term_cache.find_prefix_matches(query_term, limit=20)
            for term in prefix_matches:
                prefix_len = len(query_term)
                total_len = len(term)
                confidence = 0.90 + (0.09 * (prefix_len / max(total_len, 1)))
                all_fuzzy_matches[term] = max(
                    all_fuzzy_matches.get(term, 0), confidence
                )
            fuzzy_matches = FuzzyMatcher.find_fuzzy_matches(
                [query_term], all_terms, max_results=30
            )
            for term, confidence in fuzzy_matches:
                all_fuzzy_matches[term] = max(
                    all_fuzzy_matches.get(term, 0), confidence
                )
        if not all_fuzzy_matches:
            return []
        fuzzy_terms = list(all_fuzzy_matches.keys())
        doc_matches = self.db.search_terms(fuzzy_terms)
        if not doc_matches:
            return []
        return self._build_results(
            doc_matches,
            fuzzy_terms,
            10,
            True,
            match_type="fuzzy",
            confidence_map=all_fuzzy_matches,
        )

    def _fuzzy_search_ultra_fast(self, query_terms: list[str]) -> list[SearchResult]:
        """Legacy alias kept for compatibility."""
        return self._fuzzy_search_with_azerty(query_terms)
        """
        ULTRA-FAST fuzzy search:
        1. Prefix match ONLY (no Levenshtein)
        2. Very strict on length
        3. Timeout after 50ms equivalent
        """
        if not self.term_cache.is_initialized():
            self._init_term_cache()
        all_terms = self.term_cache.get_all_terms()
        if not all_terms:
            return []
        all_fuzzy_matches: dict[str, float] = {}
        for query_term in query_terms:
            if len(query_term) < 2:
                continue
            prefix_matches = self.term_cache.find_prefix_matches(query_term, limit=20)
            for term in prefix_matches:
                prefix_len = len(query_term)
                total_len = len(term)
                confidence = 0.90 + (0.09 * (prefix_len / max(total_len, 1)))
                all_fuzzy_matches[term] = max(
                    all_fuzzy_matches.get(term, 0), confidence
                )
        if not all_fuzzy_matches:
            return []
        fuzzy_terms = list(all_fuzzy_matches.keys())
        doc_matches = self.db.search_terms(fuzzy_terms)
        if not doc_matches:
            return []
        return self._build_results(
            doc_matches,
            fuzzy_terms,
            10,
            True,
            match_type="fuzzy",
            confidence_map=all_fuzzy_matches,
        )

    def _filter_results(self, results: list[SearchResult]) -> list[SearchResult]:
        """Filter, deduplicate, and limit to top 10."""
        filtered = []
        seen = set()
        for result in results:
            if PathFilter.is_suspicious_path(result.document.path):
                continue
            if PathFilter.is_likely_cache_file(
                result.document.filename, result.document.extension
            ):
                continue
            key = (result.document.filename, result.document.path)
            if key in seen:
                continue
            seen.add(key)
            filtered.append(result)
            if len(filtered) >= 10:
                break
        return filtered

    def _build_results(
        self,
        doc_matches: dict[int, list[str]],
        matched_terms: list[str],
        top_k: int,
        lazy_upgrade: bool,
        match_type: str = "exact",
        confidence_map: dict[str, float] | None = None,
    ) -> list[SearchResult]:
        """Build results."""
        if not doc_matches:
            return []
        doc_ids = list(doc_matches.keys())
        confidence_map = confidence_map or {}
        ranked = self.ranker.rank_documents(doc_ids, matched_terms, confidence_map)
        results: list[SearchResult] = []
        for doc_id, score in ranked[: top_k * 2]:
            doc_data = self.index.get_document(doc_id)
            if not doc_data:
                continue
            if lazy_upgrade and int(doc_data.get("content_partial", 0)) == 1:
                filename = (doc_data.get("filename") or "").lower()
                if any(t in filename for t in matched_terms):
                    self._upgrade_document_content(doc_id, doc_data)
                    doc_data = self.index.get_document(doc_id) or doc_data
            document = Document(
                doc_id=doc_id,
                filename=doc_data.get("filename", ""),
                path=doc_data.get("path", ""),
                extension=doc_data.get("extension", ""),
                content="",
            )
            first_match = (
                doc_matches[doc_id][0] if doc_matches[doc_id] else matched_terms[0]
            )
            fuzzy_conf = confidence_map.get(first_match, 1.0)
            results.append(
                SearchResult(
                    document=document,
                    score=score,
                    matched_terms=list(doc_matches[doc_id]),
                    match_type=match_type,
                    fuzzy_confidence=fuzzy_conf,
                )
            )
        return results

    def _upgrade_document_content(self, doc_id: int, doc_data: dict) -> None:
        """Upgrade partial document."""
        full_path = Path(str(doc_data["path"])) / str(doc_data["filename"])
        extracted = self.extractor.extract_full_for_upgrade(full_path)
        if not extracted.text:
            self.db.update_document_content_flags(
                doc_id, content_partial=0, content_indexed_bytes=0
            )
            self.db.commit()
            return
        tokens = []
        tokens.extend(Tokenizer.tokenize_filename_with_stems(str(doc_data["filename"])))
        tokens.extend(Tokenizer.tokenize_with_stems(extracted.text))
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
                content_indexed_bytes=len(
                    extracted.text.encode("utf-8", errors="ignore")
                ),
            )
            self.db.commit()
        except Exception:
            self.db.commit()
            raise
