from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Document:
    """Represents a file in the index"""
    doc_id: int
    filename: str
    path: Path
    extension: str
    content: str
    
    @property
    def full_path(self) -> str:
        return str(self.path / self.filename)


@dataclass
class SearchResult:
    """Represents a search result with relevance score"""
    document: Document
    score: float
    matched_terms: list[str]
    metadata: Optional[dict] = None
    
    def __lt__(self, other):
        """Enable sorting by score (descending)"""
        return self.score > other.score