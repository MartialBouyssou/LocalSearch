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
        """
        Initialize database storage with a SQLite database file.
        
        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None

    def open(self) -> None:
        """Open the database connection and initialize the schema if needed."""
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
        """Close the database connection."""
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
        """Initialize the database schema (documents, terms, postings tables)."""
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
        """Start a new database transaction."""
        with DBStorage._db_lock:
            assert self.conn is not None
            self.conn.execute("BEGIN;")

    def commit(self) -> None:
        """Commit the current transaction."""
        with DBStorage._db_lock:
            assert self.conn is not None
            self.conn.commit()

    def clear_index(self) -> None:
        """Delete all documents, terms, and postings from the index (clears the entire database)."""

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
        """
        Add a new document to the index.
        
        Args:
            filename: Name of the file.
            path: Directory path of the file.
            extension: File extension (e.g., ".txt").
            size: File size in bytes.
            content_partial: 1 if content is partially indexed, 0 otherwise.
            content_indexed_bytes: Number of bytes indexed from the content.
            
        Returns:
            The newly created document ID.
        """
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
        """
        Update the content indexing flags for a document (used for lazy loading).
        
        Args:
            doc_id: Document ID to update.
            content_partial: 1 if content is partial, 0 if complete.
            content_indexed_bytes: Number of bytes indexed.
        """
        with DBStorage._db_lock:
            assert self.conn is not None
            self.conn.execute(
                "UPDATE documents SET content_partial = ?, content_indexed_bytes = ? WHERE id = ?",
                (content_partial, content_indexed_bytes, doc_id),
            )

    def ensure_terms(self, terms: Iterable[str]) -> None:
        """
        Ensure that every term in the iterable exists in the terms table.
        
        Args:
            terms: Iterable of term strings to insert if missing.
        """
        with DBStorage._db_lock:
            assert self.conn is not None
            cur = self.conn.cursor()
            cur.executemany(
                "INSERT OR IGNORE INTO terms(term) VALUES (?)",
                ((t,) for t in terms),
            )

    def upsert_postings(self, term_doc_freq: Iterable[tuple[str, int, int]]) -> None:
        """
        Insert postings for the given term/document/frequency tuples.

        If a posting already exists for the same term and document, its
        frequency is incremented instead of creating a duplicate row.
        
        Args:
            term_doc_freq: Iterable of (term, doc_id, frequency) tuples.
        """
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
        """Delete all postings associated with a document id."""
        with DBStorage._db_lock:
            assert self.conn is not None
            self.conn.execute("DELETE FROM postings WHERE doc_id = ?", (doc_id,))

    def get_document_count(self) -> int:
        """
        Returns:
            Total number of indexed documents.
        """
        with DBStorage._db_lock:
            assert self.conn is not None
            cur = self.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM documents")
            return int(cur.fetchone()[0])

    def get_document(self, doc_id: int) -> Optional[dict]:
        """Return the document row for the given id, or None if missing."""
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
        """Original search (kept for compatibility)."""
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

    def search_exact_all_terms(self, terms: list[str], limit: int = 100) -> dict[int, list[str]]:
        """
        Level 1: EXACT match - documents containing ALL query terms.
        Returns docs that have every single query term.
        """
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
                GROUP BY p.doc_id
                HAVING COUNT(DISTINCT p.term_id) = {len(terms)}
                LIMIT {limit}
            """
            cur = self.conn.cursor()
            cur.execute(sql, terms)

            out: dict[int, list[str]] = {}
            for doc_id, term in cur.fetchall():
                out.setdefault(int(doc_id), []).append(str(term))
            return out

    def search_prefix_terms(self, terms: list[str], limit: int = 500) -> dict[int, list[str]]:
        """
        Level 2: PREFIX match - documents containing ANY query term or prefix.
        Uses LIKE for efficient prefix search on indexed terms.
        """
        with DBStorage._db_lock:
            assert self.conn is not None
            if not terms:
                return {}

            where_parts = ["t.term LIKE ?" for _ in terms]
            where_clause = " OR ".join(where_parts)
            like_terms = [f"{t}%" for t in terms]

            sql = f"""
                SELECT p.doc_id, t.term
                FROM postings p
                JOIN terms t ON t.id = p.term_id
                WHERE {where_clause}
                LIMIT {limit}
            """
            cur = self.conn.cursor()
            cur.execute(sql, like_terms)

            out: dict[int, list[str]] = {}
            for doc_id, term in cur.fetchall():
                out.setdefault(int(doc_id), []).append(str(term))
            return out

    def search_documents_by_filename_exact(self, filename: str) -> list[int]:
        """
        Search for exact or partial filename match (case-insensitive).
        Handles special characters and variations.
        """
        with DBStorage._db_lock:
            assert self.conn is not None
            cur = self.conn.cursor()
            
            filename_clean = ''.join(c for c in filename.lower() if c.isalnum() or c in '.-_')
            
            if not filename_clean:
                return []
            
            cur.execute("""
                SELECT DISTINCT id FROM documents 
                WHERE 
                    LOWER(filename) = ?
                    OR LOWER(filename) LIKE ?
                    OR LOWER(?) LIKE ('%' || LOWER(filename) || '%')
                LIMIT 20
            """, (
                filename_clean,
                f"%{filename_clean}%",
                filename_clean
            ))
            
            results = [int(row[0]) for row in cur.fetchall()]
            return results
        
    def search_documents_by_filename_wildcard(self, pattern: str) -> list[int]:
        """
        Search documents by filename using SQL LIKE pattern.
        Pattern uses standard wildcards: % (any chars), _ (single char).
        """
        with DBStorage._db_lock:
            assert self.conn is not None
            cur = self.conn.cursor()
            
            sql_pattern = pattern.replace('*', '%').replace('?', '_')
            
            cur.execute(
                """
                SELECT DISTINCT id FROM documents 
                WHERE LOWER(filename) LIKE ?
                LIMIT 50
                """,
                (sql_pattern.lower(),)
            )
            
            results = [int(row[0]) for row in cur.fetchall()]
            return results
        
    def get_all_filenames(self) -> list[tuple[int, str]]:
        """Fetch all (doc_id, filename) pairs for fuzzy filename matching."""
        with DBStorage._db_lock:
            assert self.conn is not None
            cur = self.conn.cursor()
            cur.execute("SELECT id, filename FROM documents ORDER BY filename")
            return [(int(row[0]), str(row[1])) for row in cur.fetchall()]

    def search_terms_by_wildcard(self, pattern: str, limit: int = 200) -> dict[int, list[str]]:
        """
        Search content terms matching a glob pattern (wildcard content search).
        Converts glob to SQL LIKE: * -> %, ? -> _
        """
        with DBStorage._db_lock:
            assert self.conn is not None
            sql_pattern = pattern.lower().replace("*", "%").replace("?", "_")
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT p.doc_id, t.term
                FROM postings p
                JOIN terms t ON t.id = p.term_id
                WHERE t.term LIKE ?
                LIMIT ?
                """,
                (sql_pattern, limit),
            )
            out: dict[int, list[str]] = {}
            for doc_id, term in cur.fetchall():
                out.setdefault(int(doc_id), []).append(str(term))
            return out

    def get_all_terms(self) -> list[str]:
        """Fetch all indexed terms for fuzzy matching cache."""
        with DBStorage._db_lock:
            assert self.conn is not None
            cur = self.conn.cursor()
            cur.execute("SELECT term FROM terms ORDER BY term")
            return [row[0] for row in cur.fetchall()]

    def get_term_frequency(self, term: str, doc_id: int) -> int:
        """
        Get the frequency of a term in a specific document.
        
        Args:
            term: The search term.
            doc_id: Document ID.
            
        Returns:
            Number of occurrences of the term in the document.
        """
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
        """
        Get the number of documents containing a term.
        
        Args:
            term: The search term.
            
        Returns:
            Number of documents containing the term.
        """
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