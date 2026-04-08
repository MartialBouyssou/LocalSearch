from __future__ import annotations

import re


AZERTY_ADJACENCY: dict[str, set[str]] = {
    'a': {'z', 'q', 's'},
    'z': {'a', 'e', 'q', 's', 'd'},
    'e': {'z', 'r', 's', 'd', 'f'},
    'r': {'e', 't', 'd', 'f', 'g'},
    't': {'r', 'y', 'f', 'g', 'h'},
    'y': {'t', 'u', 'g', 'h', 'j'},
    'u': {'y', 'i', 'h', 'j', 'k'},
    'i': {'u', 'o', 'j', 'k', 'l'},
    'o': {'i', 'p', 'k', 'l', 'm'},
    'p': {'o', 'l', 'm'},
    'q': {'a', 'z', 's', 'w', 'x'},
    's': {'q', 'z', 'd', 'a', 'e', 'w', 'x', 'c'},
    'd': {'s', 'z', 'f', 'e', 'r', 'x', 'c', 'v'},
    'f': {'d', 'e', 'g', 'r', 't', 'c', 'v', 'b'},
    'g': {'f', 'r', 'h', 't', 'y', 'v', 'b', 'n'},
    'h': {'g', 't', 'j', 'y', 'u', 'b', 'n'},
    'j': {'h', 'y', 'k', 'u', 'i', 'n'},
    'k': {'j', 'u', 'l', 'i', 'o'},
    'l': {'k', 'i', 'm', 'o', 'p'},
    'm': {'l', 'o', 'p'},
    'w': {'q', 's', 'x'},
    'x': {'w', 's', 'd', 'c'},
    'c': {'x', 'd', 'f', 'v'},
    'v': {'c', 'f', 'g', 'b'},
    'b': {'v', 'g', 'h', 'n'},
    'n': {'b', 'h', 'j'},
    '0': {'9', 'o', 'p'},
    '1': {'2', 'a', 'z'},
    '2': {'1', '3', 'z', 'e'},
    '3': {'2', '4', 'e', 'r'},
    '4': {'3', '5', 'r', 't'},
    '5': {'4', '6', 't', 'y'},
    '6': {'5', '7', 'y', 'u'},
    '7': {'6', '8', 'u', 'i'},
    '8': {'7', '9', 'i', 'o'},
    '9': {'8', '0', 'o', 'p'},
}


def _azerty_edit_distance(s1: str, s2: str, max_dist: int = 2) -> int:
    """
    AZERTY-aware Damerau-Levenshtein distance.
    Adjacent-key substitutions on AZERTY cost 0.5 (= 1 half-unit).
    We compute in half-units, divide at end.
    """
    len1, len2 = len(s1), len(s2)
    if abs(len1 - len2) > max_dist:
        return max_dist + 1

    INF = (max_dist + 1) * 2
    d = [[INF] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        d[i][0] = i * 2
    for j in range(len2 + 1):
        d[0][j] = j * 2

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            c1, c2 = s1[i - 1], s2[j - 1]
            if c1 == c2:
                sub_cost = 0
            elif c2 in AZERTY_ADJACENCY.get(c1, set()):
                sub_cost = 1
            else:
                sub_cost = 2

            d[i][j] = min(
                d[i - 1][j] + 2,
                d[i][j - 1] + 2,
                d[i - 1][j - 1] + sub_cost,
            )
            
            if i > 1 and j > 1 and c1 == s2[j - 2] and s1[i - 2] == c2:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + sub_cost)

    return (d[len1][len2] + 1) // 2


class FuzzyMatcher:
    """Optimized fuzzy matching: AZERTY-aware Levenshtein + substring + phonetics."""

    MAX_DISTANCE = 2
    MIN_LENGTH = 2

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """AZERTY-aware edit distance (adjacent-key typos cost ~half)."""
        return _azerty_edit_distance(s1, s2, max_dist=FuzzyMatcher.MAX_DISTANCE)

    @staticmethod
    def simple_phonetic(s: str) -> str:
        """
        Apply simple phonetic encoding by removing vowels and deduplicating consecutive characters.
        
        Args:
            s: Input string to encode.
            
        Returns:
            Phonetically encoded string.
        """
        s = re.sub(r'[aeiou]', 'a', s.lower())
        result = []
        prev = ''
        for c in s:
            if c != prev:
                result.append(c)
                prev = c
        return ''.join(result)

    @staticmethod
    def is_fuzzy_match(query_term: str, candidate_term: str) -> tuple[bool, float]:
        """
        Fuzzy match with multiple strategies.
        Returns (is_match, confidence 0-1).
        """
        if len(query_term) < FuzzyMatcher.MIN_LENGTH or len(candidate_term) < FuzzyMatcher.MIN_LENGTH:
            exact = query_term == candidate_term
            return exact, 1.0 if exact else 0.0

        if query_term == candidate_term:
            return True, 1.0

        if query_term in candidate_term or candidate_term in query_term:
            match_len = min(len(query_term), len(candidate_term))
            total_len = max(len(query_term), len(candidate_term))
            confidence = 0.90 + (0.09 * (match_len / total_len))
            return True, min(0.99, confidence)

        if candidate_term.startswith(query_term) or query_term.startswith(candidate_term):
            diff = abs(len(query_term) - len(candidate_term))
            confidence = 0.85 + (0.14 * (1.0 - min(diff, 5) / 5.0))
            return True, max(0.75, confidence)

        lev_dist = _azerty_edit_distance(query_term, candidate_term, FuzzyMatcher.MAX_DISTANCE)
        if lev_dist <= FuzzyMatcher.MAX_DISTANCE:
            confidence = 1.0 - (lev_dist * 0.25)
            return True, confidence

        if len(FuzzyMatcher.simple_phonetic(query_term)) > 2:
            if FuzzyMatcher.simple_phonetic(query_term) == FuzzyMatcher.simple_phonetic(candidate_term):
                return True, 0.65

        return False, 0.0

    @staticmethod
    def find_fuzzy_matches(
        query_terms: list[str],
        candidate_terms: list[str],
        max_results: int = 50
    ) -> list[tuple[str, float]]:
        """
        Find fuzzy matches for multiple query terms against a list of candidates.
        
        Args:
            query_terms: Terms to search for.
            candidate_terms: Terms to search within.
            max_results: Maximum number of results to return.
            
        Returns:
            List of (term, confidence) tuples sorted by confidence (descending).
        """
        matches: dict[str, float] = {}

        for query_term in query_terms:
            if len(query_term) < FuzzyMatcher.MIN_LENGTH:
                continue

            for candidate_term in candidate_terms:
                if candidate_term in matches:
                    continue

                is_match, confidence = FuzzyMatcher.is_fuzzy_match(query_term, candidate_term)
                if is_match and confidence > 0:
                    matches[candidate_term] = max(matches.get(candidate_term, 0), confidence)

        sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
        return sorted_matches[:max_results]
