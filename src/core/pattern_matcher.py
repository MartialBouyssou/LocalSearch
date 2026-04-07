import re
from typing import List


class PatternMatcher:
    """
    Strict glob pattern matching:
    * = zero or more chars
    ? = exactly one char
    Case-insensitive matching
    """

    @staticmethod
    def glob_to_regex(pattern: str) -> re.Pattern:
        """
        Convert glob pattern to regex.
        * -> .* (any chars)
        ? -> . (exactly one char)
        """
        escaped = ""
        for c in pattern:
            if c == "*":
                escaped += ".*"
            elif c == "?":
                escaped += "."
            elif c in r".^$+{}[]|()\:":
                escaped += "\\" + c
            else:
                escaped += c
        return re.compile("^" + escaped + "$", re.IGNORECASE)

    @staticmethod
    def matches_pattern(filename: str, pattern: str) -> bool:
        """
        Check if filename matches glob pattern (case-insensitive).
        Examples:
        - "test.txt" matches "test.*" ✓
        - "test.java" matches "test.*" ✓
        - "Test.java" matches "test.*" ✓ (case-insensitive)
        - "CalculatriceTest.java" matches "test.*" ✗ (doesn't start with test.)
        - "test.txt" matches "test.???" ✓ (exactly 3 chars after .)
        - "test.js" matches "test.???" ✗ (only 2 chars after .)
        - "myTest.java" matches "*Test.java" ✓
        """
        try:
            regex = PatternMatcher.glob_to_regex(pattern)
            return regex.match(filename) is not None
        except Exception:
            return False

    @staticmethod
    def filter_by_pattern(filenames: List[str], pattern: str) -> List[str]:
        """Filter list of filenames by pattern."""
        try:
            regex = PatternMatcher.glob_to_regex(pattern)
            return [f for f in filenames if regex.match(f)]
        except Exception:
            return []
