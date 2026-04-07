from fastapi import APIRouter, HTTPException, BackgroundTasks
from pathlib import Path
import time
from src.api.models import ReindexRequest, StatusResponse
from src.application.indexing_service import IndexingService
from src.infrastructure.db_storage import DBStorage
from src.infrastructure.content_extractor import ContentExtractor
from src.infrastructure.config import Config

router = APIRouter(prefix="/indexing", tags=["indexing"])
_db_storage = None
_extractor = None
_config = None
_reindex_status = {"running": False, "progress": "idle"}


def init_indexing(db_storage: DBStorage, extractor: ContentExtractor, config: Config):
    global _db_storage, _extractor, _config
    _db_storage = db_storage
    _extractor = extractor
    _config = config


def _reindex_task(include_soft_skips: bool):
    """Tâche en arrière-plan pour l'indexation."""
    global _reindex_status
    try:
        _reindex_status["running"] = True
        _reindex_status["progress"] = "starting..."
        indexing = IndexingService(db_storage=_db_storage, extractor=_extractor)
        start = time.time()
        indexed = indexing.index_directory(
            Path(_config.path),
            recursive=True,
            commit_every=_config.commit_every,
            include_soft_skips=include_soft_skips,
        )
        elapsed = time.time() - start
        _reindex_status["running"] = False
        _reindex_status["progress"] = f"completed in {elapsed:.2f}s ({indexed} files)"
    except Exception as e:
        _reindex_status["running"] = False
        _reindex_status["progress"] = f"error: {str(e)}"


@router.post("/reindex")
async def reindex(request: ReindexRequest, background_tasks: BackgroundTasks):
    """
    Lance une réindexation complète.
    S'exécute en arrière-plan.
    """
    if _reindex_status["running"]:
        raise HTTPException(status_code=409, detail="Reindex already running")
    background_tasks.add_task(
        _reindex_task, include_soft_skips=request.include_soft_skips
    )
    return {"message": "Reindex started", "status": "processing"}


@router.get("/status", response_model=StatusResponse)
async def status():
    """
    Retourne le statut actuel.
    """
    if _config is None:
        raise HTTPException(status_code=500, detail="Config not initialized")
    db_path = Path(_config.db)
    return StatusResponse(
        db_exists=db_path.exists(),
        indexed=db_path.exists(),
        db_path=str(db_path),
        search_path=str(_config.path),
    )


@router.get("/reindex-status")
async def reindex_status():
    """
    Retourne le statut de la réindexation en cours.
    """
    return {
        "running": _reindex_status["running"],
        "progress": _reindex_status["progress"],
    }
