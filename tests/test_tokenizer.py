"""Tests for tokenizer module"""

import unittest
import sys
from pathlib import Path

SRC_PATH = str(Path(__file__).resolve().parents[1] / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
from core.tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):
    def test_basic_tokenization(self):
        text = "Hello World"
        tokens = Tokenizer.tokenize(text)
        self.assertEqual(tokens, ["hello", "world"])

    def test_punctuation_removal(self):
        text = "Hello, World!"
        tokens = Tokenizer.tokenize(text)
        self.assertEqual(tokens, ["hello", "world"])

    def test_lowercase(self):
        text = "HELLO WORLD"
        tokens = Tokenizer.tokenize(text)
        self.assertEqual(tokens, ["hello", "world"])

    def test_empty_string(self):
        tokens = Tokenizer.tokenize("")
        self.assertEqual(tokens, [])

    def test_filename_tokenization(self):
        filename = "test_file.py"
        tokens = Tokenizer.tokenize_filename(filename)
        self.assertIn("test", tokens)
        self.assertIn("file", tokens)


if __name__ == "__main__":
    unittest.main()
