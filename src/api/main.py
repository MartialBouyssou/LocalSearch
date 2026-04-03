from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.infrastructure.config import Config
from src.infrastructure.db_storage import DBStorage
from src.infrastructure.file_reader import FileReader
from src.infrastructure.content_extractor import ContentExtractor, ExtractorConfig
from src.application.search_engine import SearchEngine
from src.api.routes import search, indexing, health

def create_app(config_path: str = "config.json"):
    """
    Factory function pour créer l'app FastAPI.
    """
    app = FastAPI(
        title="LocalSearch API",
        description="REST API for LocalSearch engine",
        version="1.0.0"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    cfg = Config.load(config_path)
    
    db = DBStorage(cfg.db)
    file_reader = FileReader()
    extractor = ContentExtractor(
        file_reader=file_reader,
        cfg=ExtractorConfig(
            max_text_bytes=cfg.max_text_bytes,
            sample_bytes=cfg.sample_bytes
        ),
    )
    engine = SearchEngine(db_storage=db, extractor=extractor)
    
    search.init_search(engine, cfg)
    indexing.init_indexing(db, extractor, cfg)
    
    app.include_router(health.router)
    app.include_router(search.router)
    app.include_router(indexing.router)
    
    @app.on_event("shutdown")
    async def shutdown():
        """Cleanup on shutdown."""
        engine.close()
    
    return app