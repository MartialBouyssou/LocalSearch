from __future__ import annotations
from typing import List, Set

try:
    from snowballstemmer import stemmer as SnowballStemmer

    _fr = SnowballStemmer("french")
    _en = SnowballStemmer("english")
    _SNOWBALL_AVAILABLE = True
except ImportError:
    _SNOWBALL_AVAILABLE = False


class Stemmer:
    """
    Dual-language stemmer (French + English via Snowball).
    Strategy:
    - A term is always kept as-is (exact match still works).
    - Additionally its stem(s) are added so morphological variants
      collapse to the same canonical form at query time.
    Example:
        expand("mangeais") → {"mangeais", "mang"}   (FR stem)
        expand("journal")  → {"journal", "journal"}  (same → dedup)
        expand("journalctl") → {"journalctl", "journalctl"} (no change → exact still there)
    """

    @staticmethod
    def stem_fr(word: str) -> str:
        if _SNOWBALL_AVAILABLE:
            return _fr.stemWord(word.lower())
        return word.lower()

    @staticmethod
    def stem_en(word: str) -> str:
        if _SNOWBALL_AVAILABLE:
            return _en.stemWord(word.lower())
        return word.lower()

    @staticmethod
    def expand(token: str) -> Set[str]:
        """
        Return the token itself PLUS any stems that differ from it.
        Both FR and EN stems are included (union), so a corpus mixing
        languages is handled gracefully.
        """
        t = token.lower()
        results: Set[str] = {t}
        if _SNOWBALL_AVAILABLE and len(t) > 2:
            stem_fr = _fr.stemWord(t)
            stem_en = _en.stemWord(t)
            if stem_fr:
                results.add(stem_fr)
            if stem_en:
                results.add(stem_en)
        return results

    @staticmethod
    def expand_tokens(tokens: List[str]) -> List[str]:
        """
        Expand a list of tokens with their stems.
        Preserves original tokens; stems are appended.
        Deduplication is NOT applied so Counter still counts correctly —
        caller can deduplicate if needed.
        """
        expanded: List[str] = []
        for token in tokens:
            expanded.append(token)
            for stem in Stemmer.expand(token):
                if stem != token:
                    expanded.append(stem)
        return expanded
