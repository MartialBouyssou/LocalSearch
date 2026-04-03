from fastapi import APIRouter, HTTPException
import time
from src.api.models import SearchRequest, SearchResponse, DocumentResult
from src.application.search_engine import SearchEngine
from src.infrastructure.config import Config

router = APIRouter(prefix="/search", tags=["search"])

_engine = None
_config = None

def init_search(engine: SearchEngine, config: Config):
    global _engine, _config
    _engine = engine
    _config = config

@router.post("", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Effectue une recherche dans l'index.
    """
    if _engine is None:
        raise HTTPException(status_code=500, detail="Search engine not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        start = time.time()
        results = _engine.search(
            request.query,
            top_k=request.top_k,
            lazy_upgrade=not _config.no_lazy_upgrade
        )
        elapsed = time.time() - start
        
        doc_results = [
            DocumentResult(
                filename=r.document.filename,
                path=r.document.path,
                extension=r.document.extension,
                score=r.score
            )
            for r in results
        ]
        
        return SearchResponse(
            results=doc_results,
            elapsed_time=elapsed,
            count=len(doc_results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")