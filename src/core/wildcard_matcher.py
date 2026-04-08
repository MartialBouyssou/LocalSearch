from __future__ import annotations

import re


class WildcardMatcher:
    """Fast wildcard pattern matching with support for *, ?, and character ranges."""
    
    @staticmethod
    def glob_to_regex(pattern: str) -> str:
        """
        Convert a glob pattern to a regex pattern string.
        
        Args:
            pattern: Glob pattern with wildcards (*, ?, [abc]).
            
        Returns:
            Regex pattern string (including anchors).
        """
        result = []
        i = 0
        while i < len(pattern):
            c = pattern[i]
            
            if c == '*':
                result.append('.*')
            elif c == '?':
                result.append('.')
            elif c == '[':
                j = i + 1
                if j < len(pattern) and pattern[j] == '^':
                    j += 1
                if j < len(pattern) and pattern[j] == ']':
                    j += 1
                while j < len(pattern) and pattern[j] != ']':
                    j += 1
                if j < len(pattern):
                    result.append(pattern[i:j+1])
                    i = j
                else:
                    result.append(re.escape(c))
            else:
                result.append(re.escape(c))
            
            i += 1
        
        return '^' + ''.join(result) + '$'
    
    @staticmethod
    def is_wildcard_query(query: str) -> bool:
        """
        Check if a query string contains wildcard characters.
        
        Args:
            query: Query string to check.
            
        Returns:
            True if the query contains *, ?, or [ wildcard characters.
        """
        return any(c in query for c in '*?[]')
    
    @staticmethod
    def find_wildcard_matches(
        pattern: str,
        candidates: list[str],
        max_results: int = 50
    ) -> list[tuple[str, float]]:
        """
        Find matches for wildcard pattern.
        Returns list of (candidate, confidence) tuples.
        Confidence is 1.0 for exact wildcard matches.
        """
        if not pattern or not candidates:
            return []
        
        regex_pattern = WildcardMatcher.glob_to_regex(pattern.lower())
        
        try:
            compiled = re.compile(regex_pattern)
        except re.error:
            return []
        
        matches = []
        for candidate in candidates:
            if compiled.match(candidate.lower()):
                matches.append((candidate, 1.0))
                if len(matches) >= max_results:
                    break
        
        return matches
    
    @staticmethod
    def extract_wildcard_parts(pattern: str) -> tuple[str, str, str]:
        """
        Extract prefix, wildcard, suffix from pattern.
        e.g., "journal*" -> ("journal", "*", "")
        e.g., "test?.txt" -> ("test", "?", ".txt")
        """
        match = re.search(r'[*?\[]', pattern)
        
        if not match:
            return pattern, "", ""
        
        idx = match.start()
        prefix = pattern[:idx]
        
        wildcard_start = idx
        wildcard_end = idx
        
        if pattern[idx] == '[':
            while wildcard_end < len(pattern) and pattern[wildcard_end] != ']':
                wildcard_end += 1
            wildcard_end += 1
        else:
            wildcard_end = idx + 1
        
        wildcard = pattern[wildcard_start:wildcard_end]
        suffix = pattern[wildcard_end:]
        
        return prefix, wildcard, suffix