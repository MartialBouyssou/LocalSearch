from __future__ import annotations


try:
    from snowballstemmer import stemmer as SnowballStemmer
    _fr = SnowballStemmer("french")
    _en = SnowballStemmer("english")
    _SNOWBALL_AVAILABLE = True
except ImportError:
    _SNOWBALL_AVAILABLE = False

_stem_cache: dict[str, frozenset[str]] = {}


class Stemmer:
    """
    Dual-language stemmer (French + English via Snowball).

    Design:
    - The original token is ALWAYS included in expand() so exact search still works.
    - Additional stems are added so morphological variants share a common form.
    - Results are cached at module level to avoid redundant computation at
      indexing time (huge win: each unique token is stemmed only once).
    - use_stemming=False (default in Config) disables extra stems entirely so
      indexing speed is unchanged from the pre-stemming baseline.
    """

    @staticmethod
    def stem_fr(word: str) -> str:
        """
        Apply French stemming to a word using Snowball if available.
        
        Args:
            word: Word to stem.
            
        Returns:
            Stemmed word (lowercase).
        """
        if _SNOWBALL_AVAILABLE:
            return _fr.stemWord(word.lower())
        return word.lower()

    @staticmethod
    def stem_en(word: str) -> str:
        """
        Apply English stemming to a word using Snowball if available.
        
        Args:
            word: Word to stem.
            
        Returns:
            Stemmed word (lowercase).
        """
        if _SNOWBALL_AVAILABLE:
            return _en.stemWord(word.lower())
        return word.lower()

    @staticmethod
    def expand(token: str, use_stemming: bool = True) -> set[str]:
        """
        Return the token itself PLUS its FR/EN stems (when use_stemming=True).
        Results are cached so each unique token is only stemmed once.
        """
        t = token.lower()

        if not use_stemming or not _SNOWBALL_AVAILABLE or len(t) <= 2:
            return {t}

        cached = _stem_cache.get(t)
        if cached is not None:
            return set(cached)

        results: set[str] = {t}
        stem_fr = _fr.stemWord(t)
        stem_en = _en.stemWord(t)
        if stem_fr:
            results.add(stem_fr)
        if stem_en:
            results.add(stem_en)

        _stem_cache[t] = frozenset(results)
        return results

    @staticmethod
    def expand_tokens(tokens: list[str], use_stemming: bool = True) -> list[str]:
        """
        Expand a list of tokens with their stems (when use_stemming=True).
        Original tokens are always preserved; stems are appended.
        If use_stemming=False this is a no-op and returns the original list unchanged.
        """
        if not use_stemming:
            return tokens

        expanded: list[str] = []
        for token in tokens:
            expanded.append(token)
            for stem in Stemmer.expand(token, use_stemming=True):
                if stem != token:
                    expanded.append(stem)
        return expanded
