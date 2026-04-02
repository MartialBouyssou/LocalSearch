from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Iterable, Optional


class DBStorage:
    _db_lock = threading.RLock()
    
    PRAGMAS = {
        "cache_size": -32000,
        "synchronous": "NORMAL",
        "journal_mode": "WAL",
        "temp_store": "MEMORY",
        "busy_timeout": 15000,
    }

    def __init__(self, db_path: str = "search_index.db"):
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None

    def open(self) -> None:
        with DBStorage._db_lock:
            if self.conn is not None:
                return

            self.conn = sqlite3.connect(
                str(self.db_path),
                timeout=self.PRAGMAS["busy_timeout"] / 1000.0,
                check_same_thread=False
            )
            self.conn.row_factory = sqlite3.Row

            cur = self.conn.cursor()
            cur.execute(f"PRAGMA cache_size={self.PRAGMAS['cache_size']};")
            cur.execute(f"PRAGMA synchronous={self.PRAGMAS['synchronous']};")
            cur.execute(f"PRAGMA journal_mode={self.PRAGMAS['journal_mode']};")
            cur.execute(f"PRAGMA temp_store={self.PRAGMAS['temp_store']};")
            cur.execute("PRAGMA foreign_keys=ON;")
            cur.execute("PRAGMA page_size=4096;")
            cur.execute("PRAGMA query_only=OFF;")
            
            self.conn.commit()
            self._init_db()

    def close(self) -> None:
        with DBStorage._db_lock:
            if self.conn is None:
                return
            try:
                self.conn.commit()
            except sqlite3.ProgrammingError:
                pass
            self.conn.close()
            self.conn = None

    def _init_db(self) -> None:
        assert self.conn is not None
        cur = self.conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL,
                path TEXT NOT NULL,
                extension TEXT,
                size INTEGER,
                content_partial INTEGER NOT NULL DEFAULT 0,
                content_indexed_bytes INTEGER NOT NULL DEFAULT 0,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename)")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS terms (
                id INTEGER PRIMARY KEY,
                term TEXT NOT NULL UNIQUE
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_terms_term ON terms(term)")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS postings (
                term_id INTEGER NOT NULL,
                doc_id INTEGER NOT NULL,
                frequency INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (term_id, doc_id),
                FOREIGN KEY (term_id) REFERENCES terms(id) ON DELETE CASCADE,
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_postings_term ON postings(term_id)")
        
        self.conn.commit()

    def begin(self) -> None:
        with DBStorage._db_lock:
            assert self.conn is not None
            self.conn.execute("BEGIN;")

    def commit(self) -> None:
        with DBStorage._db_lock:
            assert self.conn is not None
            self.conn.commit()

    def clear_index(self) -> None:
        with DBStorage._db_lock:
            assert self.conn is not None
            cur = self.conn.cursor()
            cur.execute("DELETE FROM postings;")
            cur.execute("DELETE FROM terms;")
            cur.execute("DELETE FROM documents;")
            self.conn.commit()

    def add_document(
        self,
        filename: str,
        path: str,
        extension: str,
        size: int = 0,
        content_partial: int = 0,
        content_indexed_bytes: int = 0,
    ) -> int:
        with DBStorage._db_lock:
            assert self.conn is not None
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO documents (filename, path, extension, size, content_partial, content_indexed_bytes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (filename, path, extension, size, content_partial, content_indexed_bytes),
            )
            return int(cur.lastrowid)

    def update_document_content_flags(self, doc_id: int, *, content_partial: int, content_indexed_bytes: int) -> None:
        with DBStorage._db_lock:
            assert self.conn is not None
            self.conn.execute(
                "UPDATE documents SET content_partial = ?, content_indexed_bytes = ? WHERE id = ?",
                (content_partial, content_indexed_bytes, doc_id),
            )

    def ensure_terms(self, terms: Iterable[str]) -> None:
        with DBStorage._db_lock:
            assert self.conn is not None
            cur = self.conn.cursor()
            cur.executemany(
                "INSERT OR IGNORE INTO terms(term) VALUES (?)",
                ((t,) for t in terms),
            )

    def upsert_postings(self, term_doc_freq: Iterable[tuple[str, int, int]]) -> None:
        with DBStorage._db_lock:
            assert self.conn is not None
            cur = self.conn.cursor()
            cur.executemany(
                """
                INSERT INTO postings(term_id, doc_id, frequency)
                SELECT t.id, ?, ?
                FROM terms t
                WHERE t.term = ?
                ON CONFLICT(term_id, doc_id)
                DO UPDATE SET frequency = frequency + excluded.frequency
                """,
                ((doc_id, freq, term) for (term, doc_id, freq) in term_doc_freq),
            )

    def delete_postings_for_doc(self, doc_id: int) -> None:
        with DBStorage._db_lock:
            assert self.conn is not None
            self.conn.execute("DELETE FROM postings WHERE doc_id = ?", (doc_id,))

    def get_document_count(self) -> int:
        with DBStorage._db_lock:
            assert self.conn is not None
            cur = self.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM documents")
            return int(cur.fetchone()[0])

    def get_document(self, doc_id: int) -> Optional[dict]:
        with DBStorage._db_lock:
            assert self.conn is not None
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT id, filename, path, extension, size, content_partial, content_indexed_bytes
                FROM documents
                WHERE id = ?
                """,
                (doc_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def search_terms(self, terms: list[str]) -> dict[int, list[str]]:
        with DBStorage._db_lock:
            assert self.conn is not None
            if not terms:
                return {}

            placeholders = ",".join("?" for _ in terms)
            sql = f"""
                SELECT p.doc_id, t.term
                FROM postings p
                JOIN terms t ON t.id = p.term_id
                WHERE t.term IN ({placeholders})
            """
            cur = self.conn.cursor()
            cur.execute(sql, terms)

            out: dict[int, list[str]] = {}
            for doc_id, term in cur.fetchall():
                out.setdefault(int(doc_id), []).append(str(term))
            return out

    def get_term_frequency(self, term: str, doc_id: int) -> int:
        with DBStorage._db_lock:
            assert self.conn is not None
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT p.frequency
                FROM postings p
                JOIN terms t ON t.id = p.term_id
                WHERE t.term = ? AND p.doc_id = ?
                """,
                (term, doc_id),
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def get_document_frequency(self, term: str) -> int:
        with DBStorage._db_lock:
            assert self.conn is not None
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT COUNT(DISTINCT p.doc_id)
                FROM postings p
                JOIN terms t ON t.id = p.term_id
                WHERE t.term = ?
                """,
                (term,),
            )
            return int(cur.fetchone()[0])

    def get_avg_indexed_bytes(self) -> float:
        with DBStorage._db_lock:
            assert self.conn is not None
            cur = self.conn.cursor()
            cur.execute("SELECT AVG(content_indexed_bytes) FROM documents WHERE content_indexed_bytes > 0")
            result = cur.fetchone()
            avg = float(result[0]) if result and result[0] else 256_000
            return avg if avg > 0 else 256_000

    def vacuum(self) -> None:
        with DBStorage._db_lock:
            assert self.conn is not None
            self.conn.execute("VACUUM;")
            self.conn.commit()