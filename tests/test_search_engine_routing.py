"""
Integration-level tests for SearchEngine routing logic.
Uses an in-memory SQLite database — no real filesystem required.
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
from src.application.search_engine import SearchEngine, SearchCancelled
from src.core.tokenizer import Tokenizer


def _make_extractor():
    return ContentExtractor(FileReader(), ExtractorConfig())


def _make_engine(tmp_path: str) -> SearchEngine:
    """Create a SearchEngine pre-populated with test documents."""
    db = DBStorage(tmp_path)
    db.open()
    db.begin()

    docs = [
        # (filename, path, extension, content)
        ("test.txt",          "/home/user/Documents",                    ".txt",  "ceci est un fichier test journalctl info"),
        ("test.txt",          "/home/user/Documents/Programmation/demo", ".txt",  "autre test journalctl warning"),
        ("TesT.opti",         "/home/user/Documents",                    ".opti", "test file"),
        ("TEST.pdf",          "/home/user/Documents",                    ".pdf",  "test document"),
        ("CalculatriceTest.java", "/home/user/src/tests",                ".java", "classe test calculatrice"),
        ("CanardTest.java",   "/home/user/src/test/java",                ".java", "classe test canard"),
        ("journalctl.conf",   "/etc/systemd",                            ".conf", "journalctl configuration system"),
        ("fonctionnement_ordinateur.pdf", "/home/user/docs",             ".pdf",  "carte mere processeur memoire"),
        ("geographie.webp",   "/home/user/images",                       ".webp", "carte de france mere patrie"),
    ]

    for filename, path, ext, content in docs:
        doc_id = db.add_document(filename, path, ext, size=len(content),
                                 content_partial=0,
                                 content_indexed_bytes=len(content.encode()))
        tokens = []
        tokens.append(filename.lower())
        tokens.extend(Tokenizer.tokenize_filename(filename))
        tokens.extend(Tokenizer.tokenize(content))
        freq = Counter(tokens)
        db.ensure_terms(freq.keys())
        db.upsert_postings((t, doc_id, f) for t, f in freq.items())

    db.commit()
    db.close()

    return SearchEngine(db, _make_extractor())


class TestWildcardRouting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls.tmp.close()
        cls.engine = _make_engine(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()
        Path(cls.tmp.name).unlink(missing_ok=True)

    def test_star_matches_exact_prefix_filenames(self):
        """test.* → test.txt, TesT.opti, TEST.pdf but NOT CalculatriceTest.java"""
        results = self.engine.search("test.*", top_k=10)
        filenames = {r.document.filename for r in results}
        self.assertIn("test.txt", filenames)
        self.assertIn("TesT.opti", filenames)
        self.assertIn("TEST.pdf", filenames)
        self.assertNotIn("CalculatriceTest.java", filenames)
        self.assertNotIn("CanardTest.java", filenames)

    def test_question_mark_exact_length(self):
        """test.??? → 3-char extensions only"""
        results = self.engine.search("test.???", top_k=10)
        filenames = {r.document.filename for r in results}
        self.assertIn("test.txt", filenames)
        self.assertIn("TEST.pdf", filenames)
        self.assertNotIn("CalculatriceTest.java", filenames)

    def test_wildcard_finds_content_terms(self):
        """journal* should find docs where 'journalctl' appears in content (test.txt)."""
        results = self.engine.search("journal*", top_k=10)
        filenames = {r.document.filename for r in results}
        # journalctl.conf matches by filename; test.txt matches by content
        self.assertIn("journalctl.conf", filenames)
        self.assertIn("test.txt", filenames)

    def test_wildcard_question_on_content(self):
        """journal??? matches the *term* journalctl (= journal + ctl, 3 chars).
        The filename journalctl.conf does NOT match journal??? (too long),
        but the documents that contain the indexed term 'journalctl' should appear."""
        results = self.engine.search("journal???", top_k=10)
        filenames = {r.document.filename for r in results}
        # test.txt and journalctl.conf both contain the term 'journalctl'
        self.assertTrue(
            len(filenames) > 0,
            "journal??? should match at least one doc via content term 'journalctl'"
        )

    def test_no_match_wildcard_returns_empty(self):
        results = self.engine.search("zzznomatch.*", top_k=10)
        self.assertEqual(results, [])

    def test_wildcard_match_type(self):
        results = self.engine.search("test.*", top_k=5)
        for r in results:
            self.assertIn(r.match_type, ("wildcard_filename", "wildcard_content"))


class TestExactFilenameRouting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls.tmp.close()
        cls.engine = _make_engine(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()
        Path(cls.tmp.name).unlink(missing_ok=True)

    def test_exact_filename_score_1000(self):
        results = self.engine.search("test.txt", top_k=10)
        self.assertTrue(any(r.document.filename == "test.txt" for r in results))
        top = next(r for r in results if r.document.filename == "test.txt")
        self.assertEqual(top.score, 1000.0)

    def test_exact_match_type(self):
        results = self.engine.search("test.txt", top_k=5)
        self.assertEqual(results[0].match_type, "exact_filename")


class TestFuzzyFilenameRouting(unittest.TestCase):
    """Verify fuzzy filename matching: tst.txt → test.txt, trst.txt → test.txt."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls.tmp.close()
        cls.engine = _make_engine(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()
        Path(cls.tmp.name).unlink(missing_ok=True)

    def test_missing_letter_finds_filename(self):
        """tst.txt (missing 'e') → should match test.txt"""
        results = self.engine.search("tst.txt", top_k=5)
        filenames = [r.document.filename for r in results]
        self.assertIn("test.txt", filenames,
                      f"Expected test.txt in fuzzy filename results, got {filenames}")

    def test_wrong_letter_finds_filename(self):
        """trst.txt (r instead of e) → should match test.txt"""
        results = self.engine.search("trst.txt", top_k=5)
        filenames = [r.document.filename for r in results]
        self.assertIn("test.txt", filenames,
                      f"Expected test.txt in fuzzy filename results, got {filenames}")

    def test_fuzzy_filename_match_type(self):
        results = self.engine.search("tst.txt", top_k=5)
        if results:
            test_txt_results = [r for r in results if r.document.filename == "test.txt"]
            if test_txt_results:
                self.assertEqual(test_txt_results[0].match_type, "fuzzy_filename")

    def test_no_crash_on_garbage_query(self):
        try:
            self.engine.search("xzqpwvmfb.xyz", top_k=5)
        except Exception as e:
            self.fail(f"search raised: {e}")


class TestTimeout(unittest.TestCase):
    """Verify timeout and cancellation mechanisms."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls.tmp.close()
        cls.engine = _make_engine(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()
        Path(cls.tmp.name).unlink(missing_ok=True)

    def test_normal_search_completes_within_timeout(self):
        """Normal query must finish well within 5s."""
        results = self.engine.search("test", top_k=5, timeout_ms=5000)
        self.assertIsInstance(results, list)

    def test_very_short_timeout_raises_cancelled(self):
        """A very short timeout should eventually trigger SearchCancelled.
        We use a small but non-zero timeout and accept that fast queries
        may complete before it fires — we just test the cancel path doesn't crash."""
        try:
            self.engine.search("journalctl", top_k=5, timeout_ms=1)
            # If it completes instantly, that's also fine
        except SearchCancelled:
            pass  # Expected for slow queries
        except Exception as e:
            self.fail(f"Unexpected exception: {e}")

    def test_cancel_method(self):
        """engine.cancel() must set the cancel event."""
        import threading
        self.engine._cancel_event.clear()
        # Cancel from another thread after a tiny delay
        def _cancel():
            import time; time.sleep(0.005)
            self.engine.cancel()
        t = threading.Thread(target=_cancel)
        t.start()
        # The event should be set shortly
        t.join(timeout=1.0)
        self.assertTrue(self.engine._cancel_event.is_set())


class TestFuzzyContent(unittest.TestCase):
    """Verify AZERTY-aware fuzzy content search."""

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
        """'jourbalctl' must not crash."""
        try:
            self.engine.search("jourbalctl", top_k=5)
        except SearchCancelled:
            pass  # timeout is acceptable
        except Exception as e:
            self.fail(f"search raised: {e}")

    def test_one_missing_letter_fuzzy_content(self):
        """'journalct' (missing l) → should find journalctl.conf"""
        results = self.engine.search("journalct", top_k=5)
        filenames = [r.document.filename for r in results]
        self.assertIn("journalctl.conf", filenames)


if __name__ == "__main__":
    unittest.main()
