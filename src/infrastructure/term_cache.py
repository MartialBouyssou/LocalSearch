from __future__ import annotations

import threading


class TermCache:
    """
    Lightweight in-memory cache for all indexed terms.
    Supports fast prefix matching via binary search.
    """
    
    _lock = threading.RLock()
    
    def __init__(self):
        """Initialize an empty term cache."""
        self._terms: set[str] = set()
        self._terms_list: list[str] = []
        self._initialized = False
    
    def initialize(self, terms: list[str]) -> None:
        """
        Load all indexed terms into the in-memory cache for fast lookups.
        
        Args:
            terms: List of all indexed terms.
        """
        with TermCache._lock:
            self._terms = set(terms)
            self._terms_list = sorted(terms)
            self._initialized = True
    
    def is_initialized(self) -> bool:
        """Check if the cache has been loaded with terms."""
        with TermCache._lock:
            return self._initialized
    
    def get_all_terms(self) -> list[str]:
        """
        Get a copy of all cached terms in sorted order.
        
        Returns:
            List of all indexed terms.
        Add a single term to the cache and resort if needed.
        
        Args:
            term: Term to add.
        """
        with TermCache._lock:
            return self._terms_list.copy()
    
    def add_term(self, term: str) -> None:
        """Add a single term."""
        with TermCache._lock:
            if term not in self._terms:
                self._terms.add(term)
                self._terms_list = sorted(self._terms)
    
    def add_terms(self, terms: list[str]) -> None:
        """Add multiple terms."""
        with TermCache._lock:
            before = len(self._terms)
            self._terms.update(terms)
            if len(self._terms) > before:
                self._terms_list = sorted(self._terms)
    
    def find_prefix_matches(self, prefix: str, limit: int = 100) -> list[str]:
        """
        Find all terms starting with prefix using binary search.
        Very fast O(log n + k) where k is result count.
        """
        if not self._initialized or not prefix:
            return []
        
        prefix_lower = prefix.lower()
        
        with TermCache._lock:
            left, right = 0, len(self._terms_list)
            while left < right:
                mid = (left + right) // 2
                if self._terms_list[mid].lower() < prefix_lower:
                    left = mid + 1
                else:
                    right = mid
            
            result = []
            for i in range(left, len(self._terms_list)):
                term = self._terms_list[i]
                if term.lower().startswith(prefix_lower):
                    result.append(term)
                    if len(result) >= limit:
                        break
                else:
                    break
            
            return result
    
    def clear(self) -> None:
        """Clear cache."""
        with TermCache._lock:
            self._terms.clear()
            self._terms_list.clear()
            self._initialized = False
    
    def stats(self) -> dict[str, int]:
        """Get cache statistics."""
        with TermCache._lock:
            return {
                "term_count": len(self._terms),
                "initialized": self._initialized,
                "estimated_memory_bytes": len(self._terms_list) * 12 + 1_000_000
            }