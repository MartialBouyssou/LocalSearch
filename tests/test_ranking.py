"""Tests for ranking algorithms"""
import unittest
import tempfile
from pathlib import Path
import sys

SRC_PATH = str(Path(__file__).resolve().parents[1] / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from core.index import InvertedIndex
from core.ranking import TFIDFRanker
from infrastructure.db_storage import DBStorage


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

    def test_tf_calculation(self):
        doc_id = self.db.add_document("a.txt", ".", ".txt", size=1)
        self.db.ensure_terms(["python"])
        self.db.upsert_postings([("python", doc_id, 5)])
        self.db.commit()

        ranker = TFIDFRanker(self.index)
        tf = ranker.calculate_tf("python", doc_id)
        self.assertEqual(tf, 5)

    def test_idf_calculation(self):
        doc1 = self.db.add_document("a.txt", ".", ".txt", size=1)
        self.db.add_document("b.txt", ".", ".txt", size=1)
        self.db.ensure_terms(["python"])
        self.db.upsert_postings([("python", doc1, 1)])
        self.db.commit()

        ranker = TFIDFRanker(self.index)
        idf = ranker.calculate_idf("python")
        self.assertGreater(idf, 0)


if __name__ == "__main__":
    unittest.main()