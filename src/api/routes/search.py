from fastapi import APIRouter, HTTPException
import asyncio
import time
from src.api.models import SearchRequest, SearchResponse, DocumentResult
from src.application.search_engine import SearchEngine, SearchCancelled
from src.infrastructure.config import Config

router = APIRouter(prefix="/search", tags=["search"])

_engine: SearchEngine | None = None
_config: Config | None = None


def init_search(engine: SearchEngine, config: Config):
    """
    Initialize the search route with engine and configuration.
    
    Args:
        engine: Initialized SearchEngine instance.
        config: Application configuration.
    """
    global _engine, _config
    _engine = engine
    _config = config


@router.post("", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Execute a search query against the index.
    
    Respects search timeout from configuration or request body (request overrides config).
    Tracks elapsed time and returns results ranked by relevance.
    
    Args:
        request: SearchRequest with query, top_k, and optional timeout_ms.
        
    Returns:
        SearchResponse with results, elapsed time, and count.
        
    Raises:
        HTTPException: 400 for empty query, 408 for timeout, 500 for internal errors.
    """
    if _engine is None:
        raise HTTPException(status_code=500, detail="Search engine not initialized")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    timeout_ms = getattr(request, "timeout_ms", None) or (_config.search_timeout_ms if _config else 5000)

    try:
        start = time.time()

        loop = asyncio.get_event_loop()
        results = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: _engine.search(
                    request.query,
                    top_k=request.top_k,
                    lazy_upgrade=not (_config.no_lazy_upgrade if _config else False),
                    timeout_ms=timeout_ms,
                ),
            ),
            timeout=timeout_ms / 1000.0 + 1.0,
        )
        elapsed = time.time() - start

        doc_results = [
            DocumentResult(
                filename=r.document.filename,
                path=r.document.path,
                extension=r.document.extension,
                score=r.score,
            )
            for r in results
        ]

        return SearchResponse(results=doc_results, elapsed_time=elapsed, count=len(doc_results))

    except (asyncio.TimeoutError, SearchCancelled):
        raise HTTPException(status_code=408, detail=f"Search timed out after {timeout_ms}ms")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
