import re


class PatternMatcher:
    """
    Glob pattern matching with support for wildcards and character ranges.
    Implements case-insensitive matching for:
    * = zero or more characters
    ? = exactly one character
    """
    
    @staticmethod
    def glob_to_regex(pattern: str) -> re.Pattern:
        """
        Convert a glob pattern to a compiled regex pattern.
        
        Args:
            pattern: Glob pattern string (supports *, ?, and regex special chars).
            
        Returns:
            Compiled regex pattern for matching.
        """
        escaped = ""
        for c in pattern:
            if c == '*':
                escaped += ".*"
            elif c == '?':
                escaped += "."
            elif c in r'.^$+{}[]|()\:':
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
    def filter_by_pattern(filenames: list[str], pattern: str) -> list[str]:
        """
        Filter a list of filenames by glob pattern (case-insensitive).
        
        Args:
            filenames: List of filenames to filter.
            pattern: Glob pattern to match against.
            
        Returns:
            List of filenames matching the pattern.
        """
        try:
            regex = PatternMatcher.glob_to_regex(pattern)
            return [f for f in filenames if regex.match(f)]
        except Exception:
            return []