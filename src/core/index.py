from __future__ import annotations

from src.infrastructure.db_storage import DBStorage


class InvertedIndex:
    """Inverted index for efficient term-based document retrieval."""

    def __init__(self, db_storage: DBStorage):
        """
        Initialize the inverted index with a database storage backend.
        
        Args:
            db_storage: Database storage instance for index data.
        """
        self.db = db_storage

    def search_terms(self, terms: list[str]) -> dict[int, set[str]]:
        """
        Search for multiple terms in the index.
        
        Args:
            terms: List of search terms.
            
        Returns:
            Dictionary mapping document IDs to sets of matching terms.
        """
        matches = self.db.search_terms(terms)
        return {doc_id: set(ts) for doc_id, ts in matches.items()}

    def get_document(self, doc_id: int) -> dict:
        """
        Retrieve document metadata by ID.
        
        Args:
            doc_id: Document ID to retrieve.
            
        Returns:
            Dictionary containing document metadata, or empty dict if not found.
        """
        return self.db.get_document(doc_id) or {}

    def get_term_frequency(self, term: str, doc_id: int) -> int:
        """
        Get the frequency of a term in a specific document.
        
        Args:
            term: The search term.
            doc_id: Document ID.
            
        Returns:
            Number of occurrences of the term in the document.
        """
        return self.db.get_term_frequency(term, doc_id)

    def get_document_frequency(self, term: str) -> int:
        """
        Get the number of documents containing a term.
        
        Args:
            term: The search term.
            
        Returns:
            Number of documents containing the term.
        """
        return self.db.get_document_frequency(term)

    @property
    def doc_count(self) -> int:
        """Get the total number of indexed documents."""
        return self.db.get_document_count()