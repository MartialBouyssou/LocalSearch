"""Tests for wildcard pattern matching (PatternMatcher + SearchEngine routing)."""
import unittest
import sys
from pathlib import Path

SRC_PATH = str(Path(__file__).resolve().parents[1])
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from src.core.pattern_matcher import PatternMatcher


class TestPatternMatcherBasic(unittest.TestCase):

    # ---- * wildcard --------------------------------------------------------

    def test_star_matches_any_extension(self):
        self.assertTrue(PatternMatcher.matches_pattern("test.txt", "test.*"))
        self.assertTrue(PatternMatcher.matches_pattern("test.java", "test.*"))
        self.assertTrue(PatternMatcher.matches_pattern("test.opti", "test.*"))

    def test_star_does_not_match_prefix(self):
        """'CalculatriceTest.java' must NOT match 'test.*'."""
        self.assertFalse(PatternMatcher.matches_pattern("CalculatriceTest.java", "test.*"))
        self.assertFalse(PatternMatcher.matches_pattern("MyTest.py", "test.*"))

    def test_star_case_insensitive(self):
        self.assertTrue(PatternMatcher.matches_pattern("TEST.PDF", "test.*"))
        self.assertTrue(PatternMatcher.matches_pattern("TesT.opti", "test.*"))

    def test_leading_star(self):
        self.assertTrue(PatternMatcher.matches_pattern("MyTest.java", "*Test.java"))
        self.assertTrue(PatternMatcher.matches_pattern("CanardTest.java", "*Test.java"))

    def test_star_no_match_different_name(self):
        self.assertFalse(PatternMatcher.matches_pattern("hello.txt", "test.*"))

    # ---- ? wildcard --------------------------------------------------------

    def test_question_mark_exact_one_char(self):
        self.assertTrue(PatternMatcher.matches_pattern("test.txt", "test.???"))
        self.assertTrue(PatternMatcher.matches_pattern("test.pdf", "test.???"))

    def test_question_mark_wrong_length(self):
        """'test.js' (2 chars after dot) must NOT match 'test.???'."""
        self.assertFalse(PatternMatcher.matches_pattern("test.js", "test.???"))
        self.assertFalse(PatternMatcher.matches_pattern("test.java", "test.???"))

    def test_question_mark_in_name(self):
        self.assertTrue(PatternMatcher.matches_pattern("test1.py", "test?.py"))
        self.assertFalse(PatternMatcher.matches_pattern("testAB.py", "test?.py"))

    # ---- filter_by_pattern -------------------------------------------------

    def test_filter_by_pattern(self):
        files = ["test.txt", "test.java", "CalculatriceTest.java", "hello.py"]
        result = PatternMatcher.filter_by_pattern(files, "test.*")
        self.assertIn("test.txt", result)
        self.assertIn("test.java", result)
        self.assertNotIn("CalculatriceTest.java", result)
        self.assertNotIn("hello.py", result)

    def test_filter_by_pattern_question(self):
        files = ["test.txt", "test.js", "test.pdf", "test.java"]
        result = PatternMatcher.filter_by_pattern(files, "test.???")
        self.assertIn("test.txt", result)
        self.assertIn("test.pdf", result)
        self.assertNotIn("test.js", result)
        self.assertNotIn("test.java", result)

    # ---- is_wildcard_query -------------------------------------------------

    def test_is_wildcard_detection(self):
        from src.core.wildcard_matcher import WildcardMatcher
        self.assertTrue(WildcardMatcher.is_wildcard_query("test.*"))
        self.assertTrue(WildcardMatcher.is_wildcard_query("test.???"))
        self.assertFalse(WildcardMatcher.is_wildcard_query("test.txt"))
        self.assertFalse(WildcardMatcher.is_wildcard_query("journalctl"))


if __name__ == "__main__":
    unittest.main()
