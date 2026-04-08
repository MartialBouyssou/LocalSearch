"""Tests for ranking algorithms"""
import unittest
import tempfile
from pathlib import Path
import sys

SRC_PATH = str(Path(__file__).resolve().parents[1])
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from src.core.index import InvertedIndex
from src.core.ranking import TFIDFRanker, BM25Ranker
from src.infrastructure.db_storage import DBStorage


class TestTFIDFRanker(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "test_ranking.db")
        self.db = DBStorage(self.db_path)
        self.db.open()
        self.index = InvertedIndex(self.db)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_rank_returns_higher_score_for_more_occurrences(self):
        """Document with higher TF should rank higher."""
        doc1 = self.db.add_document("a.txt", ".", ".txt", size=100)
        doc2 = self.db.add_document("b.txt", ".", ".txt", size=100)
        self.db.ensure_terms(["python"])
        self.db.upsert_postings([("python", doc1, 5), ("python", doc2, 1)])
        self.db.commit()

        ranker = TFIDFRanker(self.index)
        ranked = ranker.rank_documents([doc1, doc2], ["python"])
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0][0], doc1)  # doc1 has higher TF

    def test_idf_via_rank_documents(self):
        """rank_documents must produce positive scores."""
        doc1 = self.db.add_document("a.txt", ".", ".txt", size=1)
        self.db.add_document("b.txt", ".", ".txt", size=1)
        self.db.ensure_terms(["python"])
        self.db.upsert_postings([("python", doc1, 1)])
        self.db.commit()

        ranker = TFIDFRanker(self.index)
        ranked = ranker.rank_documents([doc1], ["python"])
        self.assertGreater(ranked[0][1], 0)

    def test_term_frequency_via_index(self):
        doc_id = self.db.add_document("a.txt", ".", ".txt", size=1)
        self.db.ensure_terms(["python"])
        self.db.upsert_postings([("python", doc_id, 5)])
        self.db.commit()

        self.assertEqual(self.index.get_term_frequency("python", doc_id), 5)

    def test_document_frequency_via_index(self):
        doc1 = self.db.add_document("a.txt", ".", ".txt", size=1)
        self.db.add_document("b.txt", ".", ".txt", size=1)
        self.db.ensure_terms(["python"])
        self.db.upsert_postings([("python", doc1, 1)])
        self.db.commit()

        self.assertEqual(self.index.get_document_frequency("python"), 1)


class TestBM25Ranker(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "test_bm25.db")
        self.db = DBStorage(self.db_path)
        self.db.open()
        self.index = InvertedIndex(self.db)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_bm25_rank_positive_score(self):
        doc_id = self.db.add_document("a.txt", ".", ".txt", size=100,
                                      content_indexed_bytes=100)
        self.db.ensure_terms(["python"])
        self.db.upsert_postings([("python", doc_id, 3)])
        self.db.commit()

        ranker = BM25Ranker(self.index)
        ranked = ranker.rank_documents([doc_id], ["python"])
        self.assertGreater(ranked[0][1], 0)

    def test_bm25_fuzzy_confidence_lowers_score(self):
        """A fuzzy confidence < 1.0 should reduce the BM25 score."""
        doc_id = self.db.add_document("a.txt", ".", ".txt", size=100,
                                      content_indexed_bytes=100)
        self.db.ensure_terms(["python"])
        self.db.upsert_postings([("python", doc_id, 3)])
        self.db.commit()

        ranker = BM25Ranker(self.index)
        exact_score = ranker.rank_documents([doc_id], ["python"])[0][1]
        fuzzy_score = ranker.rank_documents(
            [doc_id], ["python"], fuzzy_confidence_map={"python": 0.5}
        )[0][1]
        self.assertLess(fuzzy_score, exact_score)

    def test_bm25_higher_tf_ranks_first(self):
        doc1 = self.db.add_document("a.txt", ".", ".txt", size=100,
                                    content_indexed_bytes=100)
        doc2 = self.db.add_document("b.txt", ".", ".txt", size=100,
                                    content_indexed_bytes=100)
        self.db.ensure_terms(["python"])
        self.db.upsert_postings([("python", doc1, 10), ("python", doc2, 1)])
        self.db.commit()

        ranker = BM25Ranker(self.index)
        ranked = ranker.rank_documents([doc1, doc2], ["python"])
        self.assertEqual(ranked[0][0], doc1)


if __name__ == "__main__":
    unittest.main()
