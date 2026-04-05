class FuzzyDistance:
    """Static utility class for fuzzy matching calculations."""

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        Calculate Levenshtein distance between two strings.
        
        This is the edit distance (insertions, deletions, substitutions).
        Optimized for small strings with early exit.
        
        Args:
            s1: First string
            s2: Second string
        
        Returns:
            Minimum number of edits to transform s1 into s2
        """
        if len(s1) < len(s2):
            return FuzzyDistance.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                
                current_row.append(min(insertions, deletions, substitutions))
            
            previous_row = current_row
        
        return previous_row[-1]

    @staticmethod
    def normalized_fuzzy_distance(s1: str, s2: str) -> float:
        """
        Normalized Levenshtein distance between 0 and 1.
        
        0 = identical strings
        1 = completely different
        
        Args:
            s1: First string
            s2: Second string
        
        Returns:
            Normalized distance (0.0 to 1.0)
        """
        distance = FuzzyDistance.levenshtein_distance(s1.lower(), s2.lower())
        max_len = max(len(s1), len(s2))
        
        if max_len == 0:
            return 0.0
        
        return distance / max_len

    @staticmethod
    def similarity_score(s1: str, s2: str) -> float:
        """
        Calculate similarity score between 0 and 1.
        
        1 = identical strings
        0 = completely different
        
        This is the inverse of normalized_fuzzy_distance.
        
        Args:
            s1: First string
            s2: Second string
        
        Returns:
            Similarity score (0.0 to 1.0)
        """
        return 1.0 - FuzzyDistance.normalized_fuzzy_distance(s1, s2)

    @staticmethod
    def is_fuzzy_match(s1: str, s2: str, threshold: float = 0.7) -> bool:
        """
        Check if two strings fuzzy-match above a threshold.
        
        Args:
            s1: First string
            s2: Second string
            threshold: Similarity threshold (0.0 to 1.0)
        
        Returns:
            True if similarity >= threshold
        """
        return FuzzyDistance.similarity_score(s1, s2) >= threshold