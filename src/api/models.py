from pydantic import BaseModel

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

class ReindexRequest(BaseModel):
    include_soft_skips: bool = False

class DocumentResult(BaseModel):
    filename: str
    path: str
    extension: str
    score: float

class SearchResponse(BaseModel):
    results: list[DocumentResult]
    elapsed_time: float
    count: int

class StatusResponse(BaseModel):
    db_exists: bool
    indexed: bool
    db_path: str
    search_path: str

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0"