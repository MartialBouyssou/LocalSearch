from __future__ import annotations

from typing import Dict, Set
from src.infrastructure.db_storage import DBStorage


class InvertedIndex:
    def __init__(self, db_storage: DBStorage):
        self.db = db_storage

    def search_terms(self, terms: list[str]) -> Dict[int, Set[str]]:
        matches = self.db.search_terms(terms)
        return {doc_id: set(ts) for doc_id, ts in matches.items()}

    def get_document(self, doc_id: int) -> dict:
        return self.db.get_document(doc_id) or {}

    def get_term_frequency(self, term: str, doc_id: int) -> int:
        return self.db.get_term_frequency(term, doc_id)

    def get_document_frequency(self, term: str) -> int:
        return self.db.get_document_frequency(term)

    def get_all_terms(self) -> list[str]:
        """Return every distinct term stored in the index."""
        return self.db.get_all_terms()

    @property
    def doc_count(self) -> int:
        return self.db.get_document_count()