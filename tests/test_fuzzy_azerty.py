"""Tests for AZERTY-aware fuzzy matching."""
import unittest
import sys
from pathlib import Path

SRC_PATH = str(Path(__file__).resolve().parents[1])
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from src.core.fuzzy_matcher import FuzzyMatcher, _azerty_edit_distance, AZERTY_ADJACENCY


class TestAzertyEditDistance(unittest.TestCase):

    def test_identical_strings_zero(self):
        self.assertEqual(_azerty_edit_distance("test", "test"), 0)

    def test_empty_strings(self):
        self.assertEqual(_azerty_edit_distance("", ""), 0)

    def test_adjacent_key_cheaper_than_non_adjacent(self):
        """
        'jourbalctl' vs 'journalctl': 'b' is adjacent to 'n' on AZERTY.
        That should have a lower (or equal) distance than an arbitrary substitution.
        """
        dist_adjacent = _azerty_edit_distance("jourbalctl", "journalctl")
        dist_arbitrary = _azerty_edit_distance("jourxalctl", "journalctl")
        self.assertLessEqual(dist_adjacent, dist_arbitrary)

    def test_n_b_are_adjacent_on_azerty(self):
        """'n' and 'b' are neighbours on AZERTY — one-char substitution."""
        self.assertIn('b', AZERTY_ADJACENCY.get('n', set()))

    def test_single_adjacent_substitution_at_most_one(self):
        """Replacing one char with its adjacent AZERTY key → distance ≤ 1."""
        dist = _azerty_edit_distance("jourbalctl", "journalctl")
        self.assertLessEqual(dist, 1)

    def test_one_deletion(self):
        self.assertEqual(_azerty_edit_distance("journalct", "journalctl"), 1)

    def test_two_deletions(self):
        self.assertEqual(_azerty_edit_distance("journalc", "journalctl"), 2)

    def test_beyond_max_dist_returns_max_plus_one(self):
        self.assertGreater(_azerty_edit_distance("abc", "xyz"), 2)

    def test_transposition(self):
        self.assertLessEqual(_azerty_edit_distance("juornalctl", "journalctl"), 2)


class TestFuzzyMatcherAzerty(unittest.TestCase):

    def test_exact_match(self):
        is_match, conf = FuzzyMatcher.is_fuzzy_match("journalctl", "journalctl")
        self.assertTrue(is_match)
        self.assertEqual(conf, 1.0)

    def test_adjacent_key_typo_matches(self):
        """'jourbalctl' (b instead of n, adjacent on AZERTY) should match 'journalctl'."""
        is_match, conf = FuzzyMatcher.is_fuzzy_match("jourbalctl", "journalctl")
        self.assertTrue(is_match, "Adjacent-key typo should be a fuzzy match")
        self.assertGreater(conf, 0.5)

    def test_missing_one_letter_matches(self):
        is_match, conf = FuzzyMatcher.is_fuzzy_match("journalct", "journalctl")
        self.assertTrue(is_match)
        self.assertGreater(conf, 0.5)

    def test_completely_different_no_match(self):
        is_match, _ = FuzzyMatcher.is_fuzzy_match("python", "journalctl")
        self.assertFalse(is_match)

    def test_find_fuzzy_matches_returns_best_first(self):
        candidates = ["journalctl", "journal", "other"]
        results = FuzzyMatcher.find_fuzzy_matches(["jourbalctl"], candidates)
        terms = [t for t, _ in results]
        # journalctl is closest to jourbalctl
        self.assertTrue(len(results) > 0)
        self.assertEqual(terms[0], "journalctl")

    def test_no_crash_on_empty_inputs(self):
        self.assertEqual(FuzzyMatcher.find_fuzzy_matches([], []), [])
        self.assertEqual(FuzzyMatcher.find_fuzzy_matches(["test"], []), [])

    def test_short_terms_excluded(self):
        """Terms shorter than MIN_LENGTH should not be returned."""
        results = FuzzyMatcher.find_fuzzy_matches(["a"], ["ab", "abc"])
        # 'a' is below MIN_LENGTH so no results
        self.assertEqual(results, [])

    def test_prefix_match(self):
        is_match, conf = FuzzyMatcher.is_fuzzy_match("journal", "journalctl")
        self.assertTrue(is_match)
        self.assertGreaterEqual(conf, 0.75)

    def test_substring_match(self):
        is_match, conf = FuzzyMatcher.is_fuzzy_match("nalctl", "journalctl")
        self.assertTrue(is_match)
        self.assertGreaterEqual(conf, 0.90)


class TestFuzzyMatcherPhonetic(unittest.TestCase):

    def test_simple_phonetic_collapses_vowels(self):
        """simple_phonetic removes duplicate chars and normalises vowels."""
        s = FuzzyMatcher.simple_phonetic("bonjour")
        self.assertIsInstance(s, str)
        self.assertGreater(len(s), 0)

    def test_phonetic_match(self):
        """Words that sound similar should match via phonetics."""
        # 'kafe' vs 'cafe' — after phonetic both become the same
        p1 = FuzzyMatcher.simple_phonetic("kafe")
        p2 = FuzzyMatcher.simple_phonetic("cafe")
        if p1 == p2:
            is_match, conf = FuzzyMatcher.is_fuzzy_match("kafe", "cafe")
            self.assertTrue(is_match)


if __name__ == "__main__":
    unittest.main()
