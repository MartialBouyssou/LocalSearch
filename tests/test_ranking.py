"""Tests for ranking algorithms."""

import math
import sys
import tempfile
import unittest
from pathlib import Path

SRC_PATH = str(Path(__file__).resolve().parents[1] / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from core.index import InvertedIndex
from core.ranking import BM25Ranker, TFIDFRanker
from infrastructure.db_storage import DBStorage


def _build_db(tmp_dir: str) -> DBStorage:
    db = DBStorage(str(Path(tmp_dir) / "test_ranking.db"))
    db.open()
    return db


def _index_document(db: DBStorage, filename: str, tokens: list, size: int = 100) -> int:
    from collections import Counter
    doc_id = db.add_document(filename, ".", filename.rsplit(".", 1)[-1], size=size, content_indexed_bytes=size)
    freq = Counter(tokens)
    db.ensure_terms(freq.keys())
    db.upsert_postings((t, doc_id, f) for t, f in freq.items())
    db.commit()
    return doc_id


class TestTFIDFRanker(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = _build_db(self.tmp.name)
        self.index = InvertedIndex(self.db)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_rank_documents_returns_sorted_list(self):
        doc_high = _index_document(self.db, "a.txt", ["python"] * 10)
        doc_low = _index_document(self.db, "b.txt", ["python"] * 2)

        ranker = TFIDFRanker(self.index)
        results = ranker.rank_documents([doc_high, doc_low], ["python"])

        self.assertEqual(results[0][0], doc_high)
        self.assertEqual(results[1][0], doc_low)

    def test_rank_documents_returns_positive_score(self):
        doc_id = _index_document(self.db, "a.txt", ["python"])
        ranker = TFIDFRanker(self.index)
        results = ranker.rank_documents([doc_id], ["python"])
        self.assertGreater(results[0][1], 0)

    def test_rank_documents_ignores_unknown_terms(self):
        doc_id = _index_document(self.db, "a.txt", ["python"])
        ranker = TFIDFRanker(self.index)
        results = ranker.rank_documents([doc_id], ["nonexistentterm"])
        self.assertEqual(results, [])

    def test_rare_term_scores_higher_than_common_term(self):
        doc1 = _index_document(self.db, "a.txt", ["rare", "common"])
        doc2 = _index_document(self.db, "b.txt", ["common"])
        ranker = TFIDFRanker(self.index)
        results = ranker.rank_documents([doc1], ["rare", "common"])
        self.assertGreater(results[0][1], 0)


class TestBM25Ranker(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = _build_db(self.tmp.name)
        self.index = InvertedIndex(self.db)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_rank_documents_sorted_by_frequency(self):
        doc_high = _index_document(self.db, "a.txt", ["python"] * 8, size=500)
        doc_low = _index_document(self.db, "b.txt", ["python"] * 1, size=500)
        ranker = BM25Ranker(self.index)
        results = ranker.rank_documents([doc_high, doc_low], ["python"])
        self.assertEqual(results[0][0], doc_high)

    def test_rank_documents_positive_score(self):
        doc_id = _index_document(self.db, "a.txt", ["search", "engine"], size=200)
        ranker = BM25Ranker(self.index)
        results = ranker.rank_documents([doc_id], ["search"])
        self.assertGreater(results[0][1], 0)

    def test_rank_documents_empty_for_missing_term(self):
        doc_id = _index_document(self.db, "a.txt", ["python"])
        ranker = BM25Ranker(self.index)
        results = ranker.rank_documents([doc_id], ["java"])
        self.assertEqual(results, [])

    def test_multi_term_query_boosts_matching_more_terms(self):
        doc_both = _index_document(self.db, "a.txt", ["python", "search"], size=300)
        doc_one = _index_document(self.db, "b.txt", ["python"], size=300)
        ranker = BM25Ranker(self.index)
        results = ranker.rank_documents([doc_both, doc_one], ["python", "search"])
        self.assertEqual(results[0][0], doc_both)

    def test_avg_doc_length_is_cached(self):
        _index_document(self.db, "a.txt", ["test"], size=1000)
        ranker = BM25Ranker(self.index)
        self.assertEqual(ranker.avg_doc_length, ranker.avg_doc_length)

    def test_length_normalisation_penalises_long_documents(self):
        """
        A document with many tokens (long) should score lower than a short one
        for the same query term frequency, because BM25 normalises by token count.
        """
        doc_short = _index_document(self.db, "a.txt", ["python"] * 3 + ["other"] * 2, size=100)
        doc_long = _index_document(
            self.db, "b.txt",
            ["python"] * 3 + ["padding"] * 200,
            size=100,
        )
        ranker = BM25Ranker(self.index)
        results = dict(ranker.rank_documents([doc_short, doc_long], ["python"]))
        self.assertGreater(results[doc_short], results[doc_long])


if __name__ == "__main__":
    unittest.main()
