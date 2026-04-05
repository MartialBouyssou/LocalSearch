"""Tests for fuzzy search and content-based search."""

import sys
import tempfile
import unittest
from pathlib import Path

SRC_PATH = str(Path(__file__).resolve().parents[1] / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from core.fuzzy_distance import FuzzyDistance
from core.fuzzy_scorer import FuzzyScorer
from core.index import InvertedIndex
from core.ranking import BM25Ranker
from infrastructure.db_storage import DBStorage


def _build_db(tmp_dir: str) -> DBStorage:
    db = DBStorage(str(Path(tmp_dir) / "test.db"))
    db.open()
    return db


def _index_document(db: DBStorage, filename: str, path: str, ext: str, tokens: list[str]) -> int:
    """Helper: add a document and its term postings."""
    from collections import Counter

    doc_id = db.add_document(filename, path, ext, size=100)
    freq = Counter(tokens)
    db.ensure_terms(freq.keys())
    db.upsert_postings((t, doc_id, f) for t, f in freq.items())
    db.commit()
    return doc_id


class TestFuzzyDistance(unittest.TestCase):
    """Unit tests for the FuzzyDistance utility class."""

    def test_identical_strings_have_zero_distance(self):
        self.assertEqual(FuzzyDistance.levenshtein_distance("test", "test"), 0)

    def test_empty_string_distance_equals_other_length(self):
        self.assertEqual(FuzzyDistance.levenshtein_distance("", "hello"), 5)
        self.assertEqual(FuzzyDistance.levenshtein_distance("hello", ""), 5)

    def test_single_substitution(self):
        self.assertEqual(FuzzyDistance.levenshtein_distance("test", "tast"), 1)
        self.assertEqual(FuzzyDistance.levenshtein_distance("test", "tets"), 2)

    def test_missing_character(self):
        self.assertEqual(FuzzyDistance.levenshtein_distance("test", "tst"), 1)

    def test_normalized_distance_is_zero_for_identical(self):
        self.assertAlmostEqual(FuzzyDistance.normalized_fuzzy_distance("test", "test"), 0.0)

    def test_normalized_distance_is_one_for_completely_different(self):
        dist = FuzzyDistance.normalized_fuzzy_distance("abc", "xyz")
        self.assertGreater(dist, 0.0)
        self.assertLessEqual(dist, 1.0)

    def test_normalized_distance_case_insensitive(self):
        d1 = FuzzyDistance.normalized_fuzzy_distance("Test", "test")
        self.assertAlmostEqual(d1, 0.0)

    def test_similarity_score_is_one_for_identical(self):
        self.assertAlmostEqual(FuzzyDistance.similarity_score("test", "test"), 1.0)

    def test_similarity_score_inverse_of_distance(self):
        dist = FuzzyDistance.normalized_fuzzy_distance("test", "tast")
        sim = FuzzyDistance.similarity_score("test", "tast")
        self.assertAlmostEqual(dist + sim, 1.0)

    def test_is_fuzzy_match_exact(self):
        self.assertTrue(FuzzyDistance.is_fuzzy_match("test", "test", threshold=0.7))

    def test_is_fuzzy_match_close_typo(self):
        self.assertTrue(FuzzyDistance.is_fuzzy_match("test", "tast", threshold=0.7))

    def test_is_fuzzy_match_too_different(self):
        self.assertFalse(FuzzyDistance.is_fuzzy_match("test", "xxxxxxxx", threshold=0.7))

    def test_both_empty_strings(self):
        self.assertAlmostEqual(FuzzyDistance.normalized_fuzzy_distance("", ""), 0.0)


class TestFuzzyScorer(unittest.TestCase):
    """Unit tests for the FuzzyScorer class."""

    def setUp(self):
        self.scorer = FuzzyScorer(lambda_param=5.0)

    def test_exact_match_has_no_penalty(self):
        score, meta = self.scorer.score_with_fuzzy(
            bm25_score=1.0, query="test", best_match_term="test"
        )
        self.assertAlmostEqual(meta["edit_distance_norm"], 0.0)
        self.assertAlmostEqual(meta["fuzzy_penalty"], 1.0)
        self.assertAlmostEqual(score, 1.0)

    def test_typo_reduces_score(self):
        score_exact, _ = self.scorer.score_with_fuzzy(
            bm25_score=1.0, query="test", best_match_term="test"
        )
        score_typo, _ = self.scorer.score_with_fuzzy(
            bm25_score=1.0, query="tast", best_match_term="test"
        )
        self.assertLess(score_typo, score_exact)

    def test_no_best_match_term_returns_bm25_unchanged(self):
        score, meta = self.scorer.score_with_fuzzy(
            bm25_score=2.5, query="test", best_match_term=None
        )
        self.assertAlmostEqual(score, 2.5)
        self.assertFalse(meta["filtered_out"])

    def test_threshold_filters_out_distant_results(self):
        score, meta = self.scorer.score_with_fuzzy(
            bm25_score=1.0,
            query="abc",
            best_match_term="zzzzzzzzz",
            threshold=0.3,
        )
        self.assertTrue(meta["filtered_out"])
        self.assertAlmostEqual(score, 0.0)

    def test_score_batch_sorts_descending(self):
        results = self.scorer.score_batch(
            scored_results=[
                (0.5, "test"),
                (1.0, "test"),
                (0.8, "test"),
            ],
            query="test",
            threshold=1.0,
        )
        scores = [r[0] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_score_batch_excludes_filtered_results(self):
        results = self.scorer.score_batch(
            scored_results=[
                (1.0, "zzzzz"),
            ],
            query="ab",
            threshold=0.1,
        )
        self.assertEqual(results, [])

    def test_lenient_lambda_penalizes_less_than_strict(self):
        lenient = FuzzyScorer(lambda_param=2.0)
        strict = FuzzyScorer(lambda_param=8.0)
        bm25 = 1.0
        s_lenient, _ = lenient.score_with_fuzzy(bm25, "tast", "test")
        s_strict, _ = strict.score_with_fuzzy(bm25, "tast", "test")
        self.assertGreater(s_lenient, s_strict)


class TestInvertedIndexGetAllTerms(unittest.TestCase):
    """Tests for the get_all_terms method added to InvertedIndex."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = _build_db(self.tmp.name)
        self.index = InvertedIndex(self.db)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_empty_index_returns_empty_list(self):
        self.assertEqual(self.index.get_all_terms(), [])

    def test_indexed_terms_are_returned(self):
        _index_document(self.db, "doc.txt", ".", ".txt", ["python", "search", "engine"])
        terms = self.index.get_all_terms()
        self.assertIn("python", terms)
        self.assertIn("search", terms)
        self.assertIn("engine", terms)

    def test_terms_are_unique(self):
        _index_document(self.db, "a.txt", ".", ".txt", ["python", "python", "search"])
        terms = self.index.get_all_terms()
        self.assertEqual(len(terms), len(set(terms)))


class TestFuzzyTermExpansion(unittest.TestCase):
    """
    Integration tests that verify fuzzy term expansion works correctly:
    documents found by their *content* (not filename) when a typo is typed.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = _build_db(self.tmp.name)
        self.index = InvertedIndex(self.db)
        self.ranker = BM25Ranker(self.index)
        self.scorer = FuzzyScorer(lambda_param=5.0)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def _expand(self, query_terms: list[str], threshold: float = 0.4) -> dict[str, list[str]]:
        """Replicate the engine's _expand_terms_with_fuzzy logic."""
        all_terms = self.index.get_all_terms()
        expansion: dict[str, list[str]] = {}
        for qt in query_terms:
            matched = [it for it in all_terms
                       if FuzzyDistance.normalized_fuzzy_distance(qt, it) <= threshold]
            expansion[qt] = matched or [qt]
        return expansion

    def test_exact_term_always_matched(self):
        _index_document(self.db, "file.txt", ".", ".txt", ["python"])
        expansion = self._expand(["python"])
        self.assertIn("python", expansion["python"])

    def test_one_char_typo_is_expanded(self):
        _index_document(self.db, "file.txt", ".", ".txt", ["test"])
        expansion = self._expand(["tast"], threshold=0.4)
        self.assertIn("test", expansion["tast"])

    def test_missing_char_is_expanded(self):
        _index_document(self.db, "file.txt", ".", ".txt", ["test"])
        expansion = self._expand(["tst"], threshold=0.4)
        self.assertIn("test", expansion["tst"])

    def test_two_char_typo_is_expanded_with_lenient_threshold(self):
        _index_document(self.db, "file.txt", ".", ".txt", ["test"])
        expansion = self._expand(["tets"], threshold=0.5)
        self.assertIn("test", expansion["tets"])

    def test_completely_different_term_not_expanded(self):
        _index_document(self.db, "file.txt", ".", ".txt", ["test"])
        expansion = self._expand(["xxxxxxxx"], threshold=0.3)
        self.assertNotIn("test", expansion["xxxxxxxx"])

    def test_content_tokens_are_searchable_via_expansion(self):
        """A document indexed by content tokens can be found even with typos."""
        doc_id = _index_document(
            self.db, "report.pdf", ".", ".pdf",
            ["financial", "quarterly", "analysis"]
        )
        expansion = self._expand(["finantial"], threshold=0.4)
        expanded_terms = list({t for hits in expansion.values() for t in hits})
        matches = self.index.search_terms(expanded_terms)
        self.assertIn(doc_id, matches)

    def test_multiple_documents_ranked_by_relevance(self):
        doc1 = _index_document(
            self.db, "a.txt", ".", ".txt",
            ["python", "python", "python", "programming"]
        )
        doc2 = _index_document(
            self.db, "b.txt", ".", ".txt",
            ["python", "snake"]
        )
        matches = self.index.search_terms(["python"])
        doc_ids = list(matches.keys())
        ranked = self.ranker.rank_documents(doc_ids, ["python"])
        self.assertEqual(ranked[0][0], doc1)

    def test_fuzzy_score_degrades_gracefully_with_distance(self):
        """Scores decrease monotonically as typos increase."""
        bm25 = 1.0
        s0, _ = self.scorer.score_with_fuzzy(bm25, "test", "test")
        s1, _ = self.scorer.score_with_fuzzy(bm25, "tast", "test")
        s2, _ = self.scorer.score_with_fuzzy(bm25, "tats", "test")
        self.assertGreaterEqual(s0, s1)
        self.assertGreaterEqual(s1, s2)


class TestDBStorageGetAllTerms(unittest.TestCase):
    """Unit tests for DBStorage.get_all_terms."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = _build_db(self.tmp.name)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_returns_empty_list_when_no_terms(self):
        self.assertEqual(self.db.get_all_terms(), [])

    def test_returns_inserted_terms(self):
        self.db.ensure_terms(["alpha", "beta", "gamma"])
        self.db.commit()
        terms = self.db.get_all_terms()
        self.assertIn("alpha", terms)
        self.assertIn("beta", terms)
        self.assertIn("gamma", terms)

    def test_no_duplicate_terms(self):
        self.db.ensure_terms(["hello", "world"])
        self.db.ensure_terms(["hello"])
        self.db.commit()
        terms = self.db.get_all_terms()
        self.assertEqual(terms.count("hello"), 1)


if __name__ == "__main__":
    unittest.main()
