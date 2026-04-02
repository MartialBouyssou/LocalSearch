"""Text tokenization module"""
import re
from typing import List


class Tokenizer:
    """Handles text tokenization and normalization"""
    
    PUNCTUATION_PATTERN = re.compile(r'[^\w\s]')
    WHITESPACE_PATTERN = re.compile(r'\s+')
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenize and normalize text
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of normalized tokens
        """
        if not text:
            return []

        text = text.lower()

        text = Tokenizer.PUNCTUATION_PATTERN.sub(' ', text)

        tokens = Tokenizer.WHITESPACE_PATTERN.split(text.strip())

        tokens = [t for t in tokens if t and len(t) > 1]

        return tokens
    
    @staticmethod
    def tokenize_filename(filename: str) -> List[str]:
        """Tokenize filename separately"""
        name = filename.rsplit('.', 1)[0]
        name = re.sub(r'[-_.]', ' ', name)
        return Tokenizer.tokenize(name)