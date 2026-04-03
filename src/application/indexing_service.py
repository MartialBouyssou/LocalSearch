"""Indexing service for building the search index."""
from __future__ import annotations

from pathlib import Path
from collections import Counter

from src.core.tokenizer import Tokenizer
from src.infrastructure.db_storage import DBStorage
from src.infrastructure.content_extractor import ContentExtractor


class IndexingService:
    def __init__(self, db_storage: DBStorage, extractor: ContentExtractor):
        self.db = db_storage
        self.extractor = extractor

    def index_directory(
        self,
        directory: Path,
        recursive: bool = True,
        commit_every: int = 500,
        include_soft_skips: bool = False,
    ) -> int:
        """Index all files in directory."""
        from src.infrastructure.file_reader import FileReader

        reader = FileReader()
        file_list = list(reader.scan_directory(directory, recursive=recursive, include_soft_skips=include_soft_skips))
        
        self.db.open()

        indexed = 0
        batch_docs = 0

        self.db.begin()

        for file_path in file_list:
            try:
                extracted = self.extractor.extract(file_path)
                if not extracted.text and not extracted.partial:
                    continue

                doc_id = self.db.add_document(
                    filename=file_path.name,
                    path=str(file_path.parent),
                    extension=file_path.suffix.lower(),
                    size=file_path.stat().st_size,
                    content_partial=1 if extracted.partial else 0,
                    content_indexed_bytes=len(extracted.text.encode("utf-8", errors="ignore")),
                )

                tokens = []
                tokens.extend(Tokenizer.tokenize_filename(file_path.name))
                if extracted.text:
                    tokens.extend(Tokenizer.tokenize(extracted.text))

                freq = Counter(tokens)
                if freq:
                    self.db.ensure_terms(freq.keys())
                    self.db.upsert_postings((t, doc_id, f) for t, f in freq.items())

                indexed += 1
                batch_docs += 1

                if batch_docs >= commit_every:
                    self.db.commit()
                    self.db.begin()
                    batch_docs = 0

            except (OSError, UnicodeDecodeError, PermissionError):
                continue

        self.db.commit()
        self.db.close()

        return indexed