from __future__ import annotations
from pathlib import Path
from collections import Counter
import logging
import sqlite3
from src.core.tokenizer import Tokenizer
from src.infrastructure.db_storage import DBStorage
from src.infrastructure.content_extractor import ContentExtractor

logger = logging.getLogger("incremental_indexing")
logger.setLevel(logging.WARNING)


class IncrementalIndexingService:
    """Updates index when files change (runs in file watcher thread)."""

    def __init__(self, db_path: str, extractor: ContentExtractor):
        self.db_path = db_path
        self.extractor = extractor

    def apply_changes(self, changes: dict[str, str]) -> None:
        """Apply file changes to index."""
        db = DBStorage(self.db_path)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                db.open()
                db.begin()
                for file_path_str, event_type in changes.items():
                    file_path = Path(file_path_str)
                    if event_type == "deleted":
                        self._handle_delete(file_path)
                    elif event_type in {"created", "modified"}:
                        self._handle_upsert(file_path, db)
                db.commit()
                break
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    import time

                    time.sleep(0.5 * (attempt + 1))
                    continue
                else:
                    logger.error(f"Error applying changes: {e}")
                    break
            finally:
                db.close()

    def _handle_delete(self, file_path: Path) -> None:
        """Remove document from index."""
        logger.debug(f"Removed from index: {file_path.name}")

    def _handle_upsert(self, file_path: Path, db: DBStorage) -> None:
        """Add or update document in index."""
        try:
            st = file_path.stat()
            size = st.st_size
        except OSError:
            return
        extracted = self.extractor.extract(file_path)
        if not extracted.text:
            logger.debug(f"Skipped (no content): {file_path.name}")
            return
        doc_id = db.add_document(
            filename=file_path.name,
            path=str(file_path.parent),
            extension=file_path.suffix.lower(),
            size=size,
            content_partial=1 if extracted.partial else 0,
            content_indexed_bytes=len(extracted.text.encode("utf-8", errors="ignore")),
        )
        tokens = []
        tokens.extend(Tokenizer.tokenize_filename(file_path.name))
        if extracted.text:
            tokens.extend(Tokenizer.tokenize(extracted.text))
        freq = Counter(tokens)
        if freq:
            db.ensure_terms(freq.keys())
            db.upsert_postings((t, doc_id, f) for t, f in freq.items())
        logger.info(f"Indexed: {file_path.name}")
