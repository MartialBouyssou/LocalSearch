import re


class Tokenizer:
    """Handles text tokenization and normalization for indexing and search."""
    
    PUNCTUATION_PATTERN = re.compile(r'[^\w\s]')
    WHITESPACE_PATTERN = re.compile(r'\s+')

    @staticmethod
    def tokenize_with_stems(text: str, use_stemming: bool = False) -> list[str]:
        """
        Tokenize text and optionally expand each token with French/English stems.
        
        Args:
            text: Input text to tokenize.
            use_stemming: If True, include word stems; if False, return base tokens only.
            
        Returns:
            List of tokens (may include stems if use_stemming is enabled).
        """
        from src.core.stemmer import Stemmer
        base_tokens = Tokenizer.tokenize(text)
        return Stemmer.expand_tokens(base_tokens, use_stemming=use_stemming)

    @staticmethod
    def tokenize_filename_with_stems(filename: str, use_stemming: bool = False) -> list[str]:
        """
        Tokenize filename and optionally expand tokens with French/English stems.
        
        Args:
            filename: Name of the file to tokenize.
            use_stemming: If True, include word stems; if False, return base tokens only.
            
        Returns:
            List of tokens from the filename.
        """
        from src.core.stemmer import Stemmer
        base_tokens = Tokenizer.tokenize_filename(filename)
        return Stemmer.expand_tokens(base_tokens, use_stemming=use_stemming)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        Tokenize and normalize text
        
        Args:
            text: Input text to tokenize
            
        Returns:
            list of normalized tokens
        """
        if not text:
            return []

        text = text.lower()

        text = Tokenizer.PUNCTUATION_PATTERN.sub(' ', text)

        tokens = Tokenizer.WHITESPACE_PATTERN.split(text.strip())

        tokens = [t for t in tokens if t and len(t) > 1]

        return tokens
    
    @staticmethod
    def tokenize_filename(filename: str) -> list[str]:
        """
        Tokenize a filename by splitting on punctuation, dashes, and underscores.
        
        Args:
            filename: Name of the file to tokenize.
            
        Returns:
            List of normalized tokens from the filename (without extension).
        """
        name = filename.rsplit('.', 1)[0]
        name = re.sub(r'[-_.]', ' ', name)
        return Tokenizer.tokenize(name)