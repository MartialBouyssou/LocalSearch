from pydantic import BaseModel

class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str
    top_k: int = 10
    timeout_ms: int | None = None

class ReindexRequest(BaseModel):
    """Request model for re-indexing endpoint."""
    include_soft_skips: bool = False

class DocumentResult(BaseModel):
    """Result model for a single document in search results."""
    filename: str
    path: str
    extension: str
    score: float

class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    results: list[DocumentResult]
    elapsed_time: float
    count: int

class StatusResponse(BaseModel):
    """Response model for status endpoint."""
    db_exists: bool
    indexed: bool
    db_path: str
    search_path: str

class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    version: str = "1.0"