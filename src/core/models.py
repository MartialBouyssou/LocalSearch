from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    """Represents a file in the index"""

    doc_id: int
    filename: str
    path: str
    extension: str
    content: str

    @property
    def full_path(self) -> str:
        return str(Path(self.path) / self.filename)


@dataclass
class SearchResult:
    """Represents a search result with relevance score"""

    document: Document
    score: float
    matched_terms: list[str]
    match_type: str = "exact_filename"
    fuzzy_confidence: float = 1.0

    def __lt__(self, other):
        """Enable sorting by score (descending)"""
        return self.score > other.score
