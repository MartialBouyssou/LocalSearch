from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.core.models import SearchResult, Document
from src.core.tokenizer import Tokenizer
from src.core.ranking import BM25Ranker
from src.core.index import InvertedIndex
from src.infrastructure.db_storage import DBStorage
from src.infrastructure.content_extractor import ContentExtractor


class SearchEngine:
    def __init__(self, db_storage: DBStorage, extractor: ContentExtractor):
        self.db = db_storage
        self.extractor = extractor

        self.db.open()
        self.index = InvertedIndex(self.db)
        self.ranker = BM25Ranker(self.index)

        if self.index.doc_count == 0:
            raise RuntimeError("Index is empty. Please index files first.")

    def close(self) -> None:
        self.db.close()

    def search(self, query: str, top_k: int = 10, lazy_upgrade: bool = True) -> list[SearchResult]:
        terms = Tokenizer.tokenize(query)
        if not terms:
            return []

        doc_matches = self.index.search_terms(terms)
        if not doc_matches:
            return []

        doc_ids = list(doc_matches.keys())
        ranked = self.ranker.rank_documents(doc_ids, terms)

        results: list[SearchResult] = []
        for doc_id, score in ranked[:top_k]:
            doc_data = self.index.get_document(doc_id)
            if not doc_data:
                continue

            if lazy_upgrade and int(doc_data.get("content_partial", 0)) == 1:
                filename = (doc_data.get("filename") or "").lower()
                if any(t in filename for t in terms):
                    self._upgrade_document_content(doc_id, doc_data)
                    doc_data = self.index.get_document(doc_id) or doc_data

            document = Document(
                doc_id=doc_id,
                filename=doc_data.get("filename", ""),
                path=doc_data.get("path", ""),
                extension=doc_data.get("extension", ""),
                content="",
            )
            results.append(
                SearchResult(
                    document=document,
                    score=score,
                    matched_terms=list(doc_matches[doc_id]),
                )
            )

        return results

    def _upgrade_document_content(self, doc_id: int, doc_data: dict) -> None:
        """Option A: wipe + rebuild postings for this doc using fuller extraction."""
        full_path = Path(str(doc_data["path"])) / str(doc_data["filename"])

        extracted = self.extractor.extract_full_for_upgrade(full_path)
        if not extracted.text:
            self.db.update_document_content_flags(doc_id, content_partial=0, content_indexed_bytes=0)
            self.db.commit()
            return

        tokens = []
        tokens.extend(Tokenizer.tokenize_filename(str(doc_data["filename"])))
        tokens.extend(Tokenizer.tokenize(extracted.text))
        freq = Counter(tokens)

        self.db.begin()
        try:
            self.db.delete_postings_for_doc(doc_id)
            if freq:
                self.db.ensure_terms(freq.keys())
                self.db.upsert_postings((t, doc_id, f) for t, f in freq.items())

            self.db.update_document_content_flags(
                doc_id,
                content_partial=0 if not extracted.partial else 1,
                content_indexed_bytes=len(extracted.text.encode("utf-8", errors="ignore")),
            )
            self.db.commit()
        except Exception:
            self.db.commit()
            raise