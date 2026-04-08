from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    """Represents an indexed document with metadata."""
    doc_id: int
    filename: str
    path: str
    extension: str
    content: str
    
    @property
    def full_path(self) -> str:
        """Get the complete file path by combining path and filename."""
        return str(Path(self.path) / self.filename)


@dataclass
class SearchResult:
    """Represents a search result with relevance scores and metadata."""
    document: Document
    score: float
    matched_terms: list[str]
    match_type: str ="exact_filename"
    fuzzy_confidence: float = 1.0
    
    def __lt__(self, other):
        """Enable sorting by score in descending order."""
        return self.score > other.score