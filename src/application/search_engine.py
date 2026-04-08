from __future__ import annotations

import threading
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


class SearchCancelled(Exception):
    """Raised when a search is cancelled (timeout or Ctrl+C)."""
    pass


class SearchEngine:
    def __init__(self, db_storage: DBStorage, extractor: ContentExtractor):
        """
        Initialize the search engine with database and content extractor.
        
        Args:
            db_storage: Database storage instance for the inverted index.
            extractor: Content extractor for file processing.
            
        Raises:
            RuntimeError: If the index is empty (no documents indexed).
        """
        self.db = db_storage
        self.extractor = extractor
        self.term_cache = TermCache()

        self.db.open()
        self.index = InvertedIndex(self.db)
        self.ranker = BM25Ranker(self.index)

        if self.index.doc_count == 0:
            raise RuntimeError("Index is empty. Please index files first.")

        self._init_term_cache()

        self._filename_cache: list[tuple[str, int, str]] | None = None
        self._filename_cache_lock = threading.Lock()

        self._cancel_event = threading.Event()

    def _init_term_cache(self) -> None:
        """Initialize and load all indexed terms into memory cache for fast prefix/fuzzy matching."""
        if not self.term_cache.is_initialized():
            all_terms = self.db.get_all_terms()
            self.term_cache.initialize(all_terms)

    def _init_filename_cache(self) -> None:
        """Load all filenames into a sorted list for efficient binary-search fuzzy matching."""
        with self._filename_cache_lock:
            if self._filename_cache is None:
                rows = self.db.get_all_filenames()
                self._filename_cache = sorted(
                    [(fn.lower(), doc_id, fn) for doc_id, fn in rows],
                    key=lambda x: x[0],
                )

    def close(self) -> None:
        """Close the search engine and release database resources."""
        self.db.close()

    def cancel(self) -> None:
        """Signal the running search to abort."""
        self._cancel_event.set()

    def _check_cancel(self) -> None:
        """Check if a cancellation signal has been received; raise SearchCancelled if so."""
        if self._cancel_event.is_set():
            raise SearchCancelled()

    def search(
        self,
        query: str,
        top_k: int = 10,
        lazy_upgrade: bool = True,
        timeout_ms: int = 5000,
    ) -> list[SearchResult]:
        """
        Multi-level search with optional timeout (milliseconds).
        Raises SearchCancelled if cancelled via self.cancel() or timeout.
        """
        self._cancel_event.clear()

        result_holder: list[list[SearchResult]] = [[]]
        exc_holder: list[Exception | None] = [None]

        def _run():
            try:
                result_holder[0] = self._search_impl(query, top_k, lazy_upgrade)
            except Exception as e:
                exc_holder[0] = e

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=timeout_ms / 1000.0)

        if t.is_alive():
            self._cancel_event.set()
            raise SearchCancelled(f"Search timed out after {timeout_ms}ms")

        if exc_holder[0] is not None:
            if isinstance(exc_holder[0], SearchCancelled):
                raise exc_holder[0]
            raise exc_holder[0]

        return result_holder[0]

    def _search_impl(self, query: str, top_k: int, lazy_upgrade: bool) -> list[SearchResult]:
        """
        Execute the main search logic with multiple fallback strategies.
        
        Tries search strategies in order: exact filename, wildcard, fuzzy filename,
        exact terms, prefix terms, and fuzzy content search.
        
        Args:
            query: Search query string.
            top_k: Maximum number of results to return.
            lazy_upgrade: Whether to load full content for partial documents.
            
        Returns:
            List of SearchResult objects ranked by relevance.
        """
        query_terms = Tokenizer.tokenize(query)
        if not query_terms:
            return []

        is_wildcard = WildcardMatcher.is_wildcard_query(query)

        if not is_wildcard:
            filename_results = self._search_exact_filename(query, top_k, lazy_upgrade)
            if filename_results:
                return filename_results

        if is_wildcard:
            wildcard_results = self._search_wildcard(query, top_k, lazy_upgrade)
            if wildcard_results:
                return wildcard_results
            return []

        fuzzy_fn_results = self._search_fuzzy_filename(query, top_k, lazy_upgrade)
        if fuzzy_fn_results:
            return fuzzy_fn_results

        self._check_cancel()

        doc_matches = self.db.search_exact_all_terms(query_terms, limit=100)
        if doc_matches:
            results = self._build_results(doc_matches, query_terms, top_k, lazy_upgrade, match_type="exact")
            filtered = self._filter_results(results)
            if filtered:
                return filtered

        self._check_cancel()

        doc_matches = self.db.search_prefix_terms(query_terms, limit=500)
        if doc_matches:
            results = self._build_results(doc_matches, query_terms, top_k, lazy_upgrade, match_type="prefix")
            filtered = self._filter_results(results)
            if filtered:
                return filtered

        self._check_cancel()

        fuzzy_results = self._fuzzy_search_with_azerty(query_terms)
        if fuzzy_results:
            filtered = self._filter_results(fuzzy_results)
            if filtered:
                return filtered

        return []

    def _search_exact_filename(self, query: str, top_k: int, lazy_upgrade: bool) -> list[SearchResult]:
        """
        Search for exact filename matches.
        
        Prioritizes documents matching the exact query string in their filename.
        Cleans the query to alphanumeric characters plus dots, dashes, and underscores.
        
        Args:
            query: Search query (interpreted as filename).
            top_k: Maximum number of results to return.
            lazy_upgrade: Whether to load full content for partial documents.
            
        Returns:
            List of SearchResult objects with exact filename matches.
        """
        query_lower = query.lower()
        query_clean = ''.join(c for c in query_lower if c.isalnum() or c in '.-_')

        if not query_clean or len(query_clean) < 2:
            return []

        doc_ids = self.db.search_documents_by_filename_exact(query_clean)
        if not doc_ids:
            return []

        results = []
        seen: set[int] = set()

        for doc_id in doc_ids[:top_k * 2]:
            if doc_id in seen:
                continue
            doc_data = self.index.get_document(doc_id)
            if not doc_data:
                continue
            path, filename, extension = doc_data.get("path", ""), doc_data.get("filename", ""), doc_data.get("extension", "")
            if not PathFilter.should_include(path, filename, extension):
                continue
            seen.add(doc_id)
            results.append(SearchResult(
                document=Document(doc_id=doc_id, filename=filename, path=path, extension=extension, content=""),
                score=1000.0,
                matched_terms=[query_clean],
                match_type="exact_filename",
                fuzzy_confidence=1.0,
            ))

        return results[:top_k]

    def _search_fuzzy_filename(self, query: str, top_k: int, lazy_upgrade: bool) -> list[SearchResult]:
        """
        Fuzzy match the raw query string against all indexed filenames.
        Uses AZERTY-aware edit distance, limited to filenames within ±3 chars
        of the query length (early-exit on the sorted cache).
        Only kicks in when the query looks like a filename (has a dot or is short).
        """
        query_lower = query.lower().strip()
        if not query_lower or len(query_lower) < 2:
            return []

        self._init_filename_cache()
        assert self._filename_cache is not None

        max_dist = 2
        query_len = len(query_lower)
        candidates: list[tuple[float, int, str, str]] = []

        seen_docs: set[int] = set()

        for fn_lower, doc_id, fn_orig in self._filename_cache:
            self._check_cancel()
            fn_len = len(fn_lower)
            
            if abs(fn_len - query_len) > max_dist + 2:
                continue

            is_match, confidence = FuzzyMatcher.is_fuzzy_match(query_lower, fn_lower)
            if not is_match or confidence < 0.65:
                continue
            if doc_id in seen_docs:
                continue

            doc_data = self.index.get_document(doc_id)
            if not doc_data:
                continue
            path, ext = doc_data.get("path", ""), doc_data.get("extension", "")
            if not PathFilter.should_include(path, fn_orig, ext):
                continue

            seen_docs.add(doc_id)
            candidates.append((confidence, doc_id, fn_orig, path, ext))

        if not candidates:
            return []

        candidates.sort(key=lambda x: x[0], reverse=True)

        results = []
        for confidence, doc_id, filename, path, ext in candidates[:top_k]:
            results.append(SearchResult(
                document=Document(doc_id=doc_id, filename=filename, path=path, extension=ext, content=""),
                score=round(confidence * 1000, 4),
                matched_terms=[query_lower],
                match_type="fuzzy_filename",
                fuzzy_confidence=confidence,
            ))

        return results

    def _search_wildcard(self, pattern: str, top_k: int, lazy_upgrade: bool) -> list[SearchResult]:
        """
        Wildcard search strategy:
        1. Filename match via PatternMatcher (all docs scanned in memory)
        2. Content-term match: find indexed terms that match the pattern via SQL LIKE,
           then return the documents that contain those terms.
        Results are merged and deduplicated.
        """
        pattern_lower = pattern.lower()
        results: list[SearchResult] = []
        seen: set[tuple[str, str]] = set()

        try:
            cur = self.db.conn.cursor()
            cur.execute("SELECT id, filename, path, extension FROM documents LIMIT 1000000")
            for row in cur.fetchall():
                self._check_cancel()
                doc_id, filename, path, extension = int(row[0]), row[1], row[2], row[3]
                if not PatternMatcher.matches_pattern(filename, pattern_lower):
                    continue
                if not PathFilter.should_include(path, filename, extension):
                    continue
                key = (filename, path)
                if key in seen:
                    continue
                seen.add(key)
                results.append(SearchResult(
                    document=Document(doc_id=doc_id, filename=filename, path=path, extension=extension, content=""),
                    score=1000.0,
                    matched_terms=[pattern_lower],
                    match_type="wildcard_filename",
                    fuzzy_confidence=1.0,
                ))
                if len(results) >= top_k:
                    break
        except Exception as e:
            print(f"\nError in wildcard filename search: {e}")

        if len(results) < top_k:
            self._check_cancel()
            term_doc_matches = self.db.search_terms_by_wildcard(pattern_lower, limit=200)
            if term_doc_matches:
                all_matched_terms = list({t for terms in term_doc_matches.values() for t in terms})
                content_results = self._build_results(
                    term_doc_matches, all_matched_terms,
                    top_k - len(results), lazy_upgrade,
                    match_type="wildcard_content",
                )
                for r in content_results:
                    key = (r.document.filename, r.document.path)
                    if key not in seen:
                        seen.add(key)
                        
                        results.append(SearchResult(
                            document=r.document,
                            score=r.score * 0.7,
                            matched_terms=r.matched_terms,
                            match_type="wildcard_content",
                            fuzzy_confidence=r.fuzzy_confidence,
                        ))

        filtered = self._filter_results(results)
        return filtered[:top_k]

    def _fuzzy_search_with_azerty(self, query_terms: list[str]) -> list[SearchResult]:
        """
        Fuzzy content search:
        1. Prefix matches from the sorted term cache (O(log n), always fast)
        2. AZERTY-aware Levenshtein on the term cache, restricted to terms
           within ±2 chars of the query term length (avoids full O(n) scan)
        Hard cap of 200 candidate terms total.
        """
        if not self.term_cache.is_initialized():
            self._init_term_cache()

        all_terms = self.term_cache.get_all_terms()
        if not all_terms:
            return []

        all_fuzzy_matches: dict[str, float] = {}
        MAX_CANDIDATES = 200

        for query_term in query_terms:
            if len(query_term) < 2:
                continue
            self._check_cancel()

            prefix_matches = self.term_cache.find_prefix_matches(query_term, limit=20)
            for term in prefix_matches:
                confidence = 0.90 + 0.09 * (len(query_term) / max(len(term), 1))
                all_fuzzy_matches[term] = max(all_fuzzy_matches.get(term, 0), confidence)

            if len(all_fuzzy_matches) >= MAX_CANDIDATES:
                continue

            q_len = len(query_term)
            max_dist = FuzzyMatcher.MAX_DISTANCE
            nearby = [t for t in all_terms if abs(len(t) - q_len) <= max_dist]
            self._check_cancel()

            fuzzy_matches = FuzzyMatcher.find_fuzzy_matches(
                [query_term], nearby, max_results=30
            )
            for term, confidence in fuzzy_matches:
                all_fuzzy_matches[term] = max(all_fuzzy_matches.get(term, 0), confidence)
                if len(all_fuzzy_matches) >= MAX_CANDIDATES:
                    break

        if not all_fuzzy_matches:
            return []

        self._check_cancel()
        fuzzy_terms = list(all_fuzzy_matches.keys())
        doc_matches = self.db.search_terms(fuzzy_terms)
        if not doc_matches:
            return []

        return self._build_results(
            doc_matches, fuzzy_terms, 10, True,
            match_type="fuzzy",
            confidence_map=all_fuzzy_matches,
        )

    def _fuzzy_search_ultra_fast(self, query_terms: list[str]) -> list[SearchResult]:
        """
        Perform ultra-fast fuzzy search. Currently delegates to the standard fuzzy search.
        
        Args:
            query_terms: List of query terms to search for fuzzily.
            
        Returns:
            List of fuzzy-matched SearchResult objects.
        """
        return self._fuzzy_search_with_azerty(query_terms)

    def _expand_with_stems(self, query_terms: list[str]) -> list[str]:
        """
        Expand query terms with their French and English stems.
        
        Args:
            query_terms: List of base query terms.
            
        Returns:
            List of query terms plus their stems (if available).
        """
        from src.core.stemmer import Stemmer
        expanded = list(query_terms)
        for term in query_terms:
            for stem in Stemmer.expand(term):
                if stem not in expanded:
                    expanded.append(stem)
        return expanded

    def _filter_results(self, results: list[SearchResult]) -> list[SearchResult]:
        """
        Post-process search results: filter suspicious/cache files, deduplicate, and limit to 10 results.
        
        Args:
            results: Unfiltered search results.
            
        Returns:
            Filtered and deduplicated results (max 10 items).
        """
        filtered = []
        seen: set[tuple[str, str]] = set()

        for result in results:
            if PathFilter.is_suspicious_path(result.document.path):
                continue
            if PathFilter.is_likely_cache_file(result.document.filename, result.document.extension):
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
        """
        Build ranked search results from matching document ids.

        The method scores candidate documents with the ranker, optionally
        upgrades partial documents when a filename match suggests they are
        relevant, and converts the final rows into SearchResult objects.

        Args:
            doc_matches: Mapping of document ids to the terms they matched.
            matched_terms: Search terms used to build the candidate set.
            top_k: Maximum number of results to return.
            lazy_upgrade: Whether to load full content for partial documents.
            match_type: Label describing how the matches were produced.
            confidence_map: Optional per-term confidence values for fuzzy search.

        Returns:
            Ranked SearchResult objects ready to return to the caller.
        """
        if not doc_matches:
            return []

        doc_ids = list(doc_matches.keys())
        confidence_map = confidence_map or {}

        ranked = self.ranker.rank_documents(doc_ids, matched_terms, confidence_map)

        results: list[SearchResult] = []
        for doc_id, score in ranked[:top_k * 2]:
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

            first_match = doc_matches[doc_id][0] if doc_matches[doc_id] else (matched_terms[0] if matched_terms else "")
            fuzzy_conf = confidence_map.get(first_match, 1.0)

            results.append(SearchResult(
                document=document,
                score=score,
                matched_terms=list(doc_matches[doc_id]),
                match_type=match_type,
                fuzzy_confidence=fuzzy_conf,
            ))

        return results

    def _upgrade_document_content(self, doc_id: int, doc_data: dict) -> None:
        """
        Upgrade a partial document by loading and indexing its full content.
        
        Used for lazy-loading when a partial document appears in search results.
        Re-indexes the full content and updates the document flags.
        
        Args:
            doc_id: Document ID to upgrade.
            doc_data: Current document metadata dictionary.
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
