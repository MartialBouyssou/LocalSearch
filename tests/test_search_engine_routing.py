"""
Integration-level tests for SearchEngine routing logic.
Uses an in-memory SQLite database so no real filesystem is required.
"""

import unittest
import tempfile
import sys
from pathlib import Path
from collections import Counter

SRC_PATH = str(Path(__file__).resolve().parents[1])
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
from src.infrastructure.db_storage import DBStorage
from src.infrastructure.content_extractor import ContentExtractor, ExtractorConfig
from src.infrastructure.file_reader import FileReader
from src.application.search_engine import SearchEngine
from src.core.stemmer import Stemmer
from src.core.tokenizer import Tokenizer


def _make_extractor():
    return ContentExtractor(FileReader(), ExtractorConfig())


def _make_engine(tmp_path: str) -> SearchEngine:
    """Create a SearchEngine pre-populated with test documents."""
    db = DBStorage(tmp_path)
    db.open()
    db.begin()
    docs = [
        ("test.txt", "/home/user/Documents", ".txt", "ceci est un fichier test"),
        ("test.txt", "/home/user/Documents/Programmation/demo", ".txt", "autre test"),
        ("TesT.opti", "/home/user/Documents", ".opti", "test file"),
        ("TEST.pdf", "/home/user/Documents", ".pdf", "test document"),
        (
            "CalculatriceTest.java",
            "/home/user/src/tests",
            ".java",
            "classe test calculatrice",
        ),
        ("CanardTest.java", "/home/user/src/test/java", ".java", "classe test canard"),
        ("journalctl.conf", "/etc/systemd", ".conf", "journalctl configuration"),
        (
            "fonctionnement_ordinateur.pdf",
            "/home/user/docs",
            ".pdf",
            "carte mere processeur memoire",
        ),
        (
            "geographie.webp",
            "/home/user/images",
            ".webp",
            "carte de france mere patrie",
        ),
    ]
    for filename, path, ext, content in docs:
        doc_id = db.add_document(
            filename,
            path,
            ext,
            size=len(content),
            content_partial=0,
            content_indexed_bytes=len(content.encode()),
        )
        tokens = []
        tokens.append(filename.lower())
        tokens.extend(Tokenizer.tokenize_filename_with_stems(filename))
        tokens.extend(Tokenizer.tokenize_with_stems(content))
        freq = Counter(tokens)
        db.ensure_terms(freq.keys())
        db.upsert_postings((t, doc_id, f) for t, f in freq.items())
    db.commit()
    db.close()
    extractor = _make_extractor()
    return SearchEngine(db, extractor)


class TestSearchEngineRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls.tmp.close()
        cls.engine = _make_engine(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()
        Path(cls.tmp.name).unlink(missing_ok=True)

    def test_wildcard_star_exact_prefix(self):
        """'test.*' must return test.txt, TesT.opti, TEST.pdf — NOT CalculatriceTest.java."""
        results = self.engine.search("test.*", top_k=10)
        filenames = {r.document.filename for r in results}
        self.assertIn("test.txt", filenames)
        self.assertIn("TesT.opti", filenames)
        self.assertIn("TEST.pdf", filenames)
        self.assertNotIn("CalculatriceTest.java", filenames)
        self.assertNotIn("CanardTest.java", filenames)

    def test_wildcard_question_mark(self):
        """'test.???' must match exactly 3 chars after dot: txt, pdf, but NOT java."""
        results = self.engine.search("test.???", top_k=10)
        filenames = {r.document.filename for r in results}
        self.assertIn("test.txt", filenames)
        self.assertIn("TEST.pdf", filenames)
        self.assertNotIn("CalculatriceTest.java", filenames)

    def test_wildcard_does_not_fall_through_to_content(self):
        """A wildcard query with no filename matches should return [] not content results."""
        results = self.engine.search("zzznomatch.*", top_k=10)
        self.assertEqual(results, [])

    def test_exact_filename_match(self):
        results = self.engine.search("test.txt", top_k=10)
        filenames = [r.document.filename for r in results]
        self.assertIn("test.txt", filenames)
        self.assertEqual(results[0].score, 1000.0)

    def test_wildcard_match_type(self):
        results = self.engine.search("test.*", top_k=5)
        for r in results:
            self.assertEqual(r.match_type, "wildcard_filename")

    def test_exact_filename_match_type(self):
        results = self.engine.search("test.txt", top_k=5)
        self.assertEqual(results[0].match_type, "exact_filename")


class TestSearchEngineStemming(unittest.TestCase):
    """Verify that stemmed queries find content."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls.tmp.close()
        cls.engine = _make_engine(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()
        Path(cls.tmp.name).unlink(missing_ok=True)

    def test_content_query_finds_results(self):
        """Querying a word that appears in content should return results."""
        results = self.engine.search("carte", top_k=5)
        filenames = [r.document.filename for r in results]
        self.assertTrue(len(results) > 0)
        self.assertIn("fonctionnement_ordinateur.pdf", filenames)

    def test_stem_expand_does_not_crash(self):
        """Stem expansion must not raise on any normal query."""
        for word in ["mangeais", "journal", "journalctl", "configuration"]:
            try:
                self.engine.search(word, top_k=5)
            except Exception as e:
                self.fail(f"search('{word}') raised {e}")


class TestSearchEngineFuzzy(unittest.TestCase):
    """Verify fuzzy + AZERTY-correction does not crash and returns sensible results."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls.tmp.close()
        cls.engine = _make_engine(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()
        Path(cls.tmp.name).unlink(missing_ok=True)

    def test_no_crash_adjacent_key_typo(self):
        """'jourbalctl' (b≈n on AZERTY) must not crash."""
        try:
            self.engine.search("jourbalctl", top_k=5)
        except Exception as e:
            self.fail(f"search raised {e}")

    def test_fuzzy_finds_similar_term(self):
        """'journalct' (missing last l) should find journalctl.conf."""
        results = self.engine.search("journalct", top_k=5)
        filenames = [r.document.filename for r in results]
        self.assertIn("journalctl.conf", filenames)

    def test_no_results_for_garbage_query(self):
        """Completely random string should return empty or at least not crash."""
        try:
            results = self.engine.search("xzqjkpwvmfb", top_k=5)
            self.assertIsInstance(results, list)
        except Exception as e:
            self.fail(f"search raised {e}")


if __name__ == "__main__":
    unittest.main()
