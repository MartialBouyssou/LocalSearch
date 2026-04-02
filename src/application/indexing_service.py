from __future__ import annotations

from collections import Counter
from pathlib import Path

from core.tokenizer import Tokenizer
from infrastructure.db_storage import DBStorage
from infrastructure.content_extractor import ContentExtractor


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
        """
        Index all files in directory (bulk optimized)
        """
        self.db.open()
        self.db.clear_index()

        indexed = 0
        batch_docs = 0

        self.db.begin()
        try:
            for file_path in self.extractor.file_reader.scan_directory(
                Path(directory),
                recursive=recursive,
                include_soft_skips=include_soft_skips,
            ):
                try:
                    st = file_path.stat()
                    size = st.st_size
                except OSError:
                    continue

                extracted = self.extractor.extract(file_path)

                doc_id = self.db.add_document(
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
                    self.db.ensure_terms(freq.keys())
                    self.db.upsert_postings((t, doc_id, f) for t, f in freq.items())

                indexed += 1
                batch_docs += 1

                if batch_docs >= commit_every:
                    self.db.commit()
                    self.db.begin()
                    batch_docs = 0
                    print(f"  [*] Indexed {indexed} files...")

            self.db.commit()
        finally:
            self.db.close()

        print(f"  [*] Indexing complete! ({indexed} files indexed)")
        return indexed