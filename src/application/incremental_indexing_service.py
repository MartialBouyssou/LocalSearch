from __future__ import annotations

from pathlib import Path
from collections import Counter
import logging
import sqlite3

from src.core.tokenizer import Tokenizer
from src.infrastructure.db_storage import DBStorage
from src.infrastructure.content_extractor import ContentExtractor

logger = logging.getLogger("incremental_indexing")
logger.setLevel(logging.INFO)


class IncrementalIndexingService:
    """Updates index when files change (runs in file watcher thread)."""

    IGNORED_EXTENSIONS = {
        ".swp", ".swo", ".tmp", ".bak", ".lock", 
        ".part", ".crdownload", "~"
    }
    
    IGNORED_FILES = {
        ".DS_Store", "Thumbs.db", ".localsearch_history"
    }

    def __init__(self, db_path: str, extractor: ContentExtractor):
        self.db_path = db_path
        self.extractor = extractor

    def apply_changes(self, changes: dict[str, str]) -> None:
        """
        Apply file changes to index.
        
        Args:
            changes: Dict mapping file_path -> event_type ("created", "modified", "deleted")
        """
        db = DBStorage(self.db_path)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                db.open()
                db.begin()
                
                for file_path_str, event_type in changes.items():
                    file_path = Path(file_path_str)
                    
                    if self._should_ignore(file_path):
                        logger.debug(f"Ignoring {file_path.name} (temporary/system file)")
                        continue
                    
                    if file_path.is_dir():
                        continue
                    
                    if event_type == "deleted":
                        self._handle_delete(file_path, db)
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

    def _should_ignore(self, file_path: Path) -> bool:
        """Check if file should be ignored."""
        if file_path.name in self.IGNORED_FILES:
            return True
        
        if file_path.suffix in self.IGNORED_EXTENSIONS:
            return True
        
        if file_path.name.endswith("~"):
            return True
        
        return False

    def _handle_delete(self, file_path: Path, db: DBStorage) -> None:
        """
        Remove document from index when file is deleted.
        
        Args:
            file_path: Path to deleted file
            db: Database storage instance
        """
        try:
            query = """
                SELECT id FROM documents 
                WHERE filename = ? AND path = ?
            """
            result = db.conn.execute(query, (file_path.name, str(file_path.parent))).fetchone()
            
            if result:
                doc_id = result[0]
                db.delete_postings_for_doc(doc_id)
                db.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
                logger.info(f"Deleted from index: {file_path.name}")
            else:
                logger.debug(f"File not in index: {file_path.name}")
        
        except Exception as e:
            logger.error(f"Error deleting document {file_path.name}: {e}")

    def _handle_upsert(self, file_path: Path, db: DBStorage) -> None:
        """
        Add or update document in index.
        
        Args:
            file_path: Path to file to index
            db: Database storage instance
        """
        try:
            if not file_path.exists():
                logger.debug(f"File no longer exists: {file_path.name}")
                return
            
            if file_path.is_dir():
                return
            
            st = file_path.stat()
            size = st.st_size
            
            if size == 0:
                logger.debug(f"Skipped (empty file): {file_path.name}")
                return
                
        except OSError as e:
            logger.debug(f"Cannot access file: {file_path.name} - {e}")
            return

        try:
            extracted = self.extractor.extract(file_path)
        except Exception as e:
            logger.error(f"Error extracting content from {file_path.name}: {e}")
            return

        if not extracted.text:
            logger.debug(f"Skipped (no content): {file_path.name}")
            return

        try:
            query = """
                SELECT id FROM documents 
                WHERE filename = ? AND path = ?
            """
            result = db.conn.execute(query, (file_path.name, str(file_path.parent))).fetchone()
            
            if result:
                doc_id = result[0]
                db.delete_postings_for_doc(doc_id)
                db.conn.execute("""
                    UPDATE documents 
                    SET content_partial = ?, content_indexed_bytes = ?, size = ?
                    WHERE id = ?
                """, (
                    1 if extracted.partial else 0,
                    len(extracted.text.encode("utf-8", errors="ignore")),
                    size,
                    doc_id
                ))
                logger.info(f"Updated in index: {file_path.name}")
            else:
                doc_id = db.add_document(
                    filename=file_path.name,
                    path=str(file_path.parent),
                    extension=file_path.suffix.lower(),
                    size=size,
                    content_partial=1 if extracted.partial else 0,
                    content_indexed_bytes=len(extracted.text.encode("utf-8", errors="ignore")),
                )
                logger.info(f"Added to index: {file_path.name}")

            tokens = []
            tokens.extend(Tokenizer.tokenize_filename(file_path.name))
            if extracted.text:
                tokens.extend(Tokenizer.tokenize(extracted.text))

            freq = Counter(tokens)
            if freq:
                db.ensure_terms(freq.keys())
                db.upsert_postings((t, doc_id, f) for t, f in freq.items())

        except Exception as e:
            logger.error(f"Error upserting document {file_path.name}: {e}")