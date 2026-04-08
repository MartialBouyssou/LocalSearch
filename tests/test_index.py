"""Tests for inverted index"""
import unittest
import tempfile
from pathlib import Path
import sys

SRC_PATH = str(Path(__file__).resolve().parents[1] / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from core.index import InvertedIndex
from infrastructure.db_storage import DBStorage


class TestInvertedIndex(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "test_index.db")
        self.db = DBStorage(self.db_path)
        self.db.open()
        self.index = InvertedIndex(self.db)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_doc_count_starts_at_zero(self):
        self.assertEqual(self.index.doc_count, 0)

    def test_get_document_after_insert(self):
        doc_id = self.db.add_document("test.txt", ".", ".txt", size=12)
        self.db.commit()

        doc = self.index.get_document(doc_id)
        self.assertEqual(doc["filename"], "test.txt")
        self.assertEqual(doc["path"], ".")

    def test_search_terms_and_frequencies(self):
        doc_id = self.db.add_document("code.py", ".", ".py", size=100)
        self.db.ensure_terms(["python"])
        self.db.upsert_postings([("python", doc_id, 3)])
        self.db.commit()

        matches = self.index.search_terms(["python"])
        self.assertIn(doc_id, matches)
        self.assertIn("python", matches[doc_id])
        self.assertEqual(self.index.get_term_frequency("python", doc_id), 3)
        self.assertEqual(self.index.get_document_frequency("python"), 1)


if __name__ == "__main__":
    unittest.main()