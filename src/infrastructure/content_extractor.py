from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
import sys
import io


@dataclass(frozen=True)
class ExtractedContent:
    text: str
    partial: bool


@dataclass(frozen=True)
class ExtractorConfig:
    max_text_bytes: int = 2_000_000
    sample_bytes: int = 256_000



class ContentExtractor:
    TEXT_EXTS = {
        ".txt", ".md", ".rst", ".log",
        ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp",
        ".html", ".css", ".json", ".yaml", ".yml", ".xml", ".toml",
        ".sql", ".sh", ".bat",
    }
    LARGE_TEXT_EXTS = {".csv", ".tsv"}
    PDF_EXTS = {".pdf"}
    ODT_EXTS = {".odt"}

    BINARY_EXTS = {
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".svg",
        ".mp4", ".mkv", ".avi", ".mov", ".webm",
        ".mp3", ".wav", ".flac", ".ogg", ".m4a",
        ".zip", ".7z", ".rar", ".gz", ".tar",
        ".exe", ".dll", ".so", ".bin", ".iso",
        ".db", ".sqlite", ".sqlite3", ".db-wal", ".db-shm",
        ".cache", ".lock", ".tmp",
    }

    ALWAYS_SKIP_FILES = {
        "search_index.db",
        "search_index.db-wal",
        "search_index.db-shm",
        "places.sqlite",
        "places.sqlite-wal",
        "places.sqlite-shm",
    }

    def __init__(self, file_reader, cfg: ExtractorConfig):
        self.file_reader = file_reader
        self.cfg = cfg

    def extract(self, file_path: Path) -> ExtractedContent:
        ext = file_path.suffix.lower()

        if file_path.name in self.ALWAYS_SKIP_FILES:
            return ExtractedContent(text="", partial=False)

        if ext in {".db", ".sqlite", ".sqlite3", ".db-wal", ".db-shm", ".cache", ".lock", ".tmp"}:
            return ExtractedContent(text="", partial=False)

        if ext in self.BINARY_EXTS:
            return ExtractedContent(text="", partial=False)

        if ext in self.PDF_EXTS:
            return ExtractedContent(text=self._extract_pdf(file_path), partial=False)

        if ext in self.ODT_EXTS:
            return ExtractedContent(text=self._extract_odt_text(file_path), partial=False)

        if ext in self.LARGE_TEXT_EXTS:
            return self._extract_large_text_sample(file_path)

        if ext in self.TEXT_EXTS or ext == "":
            return self._extract_text_capped(file_path)

        return ExtractedContent(text="", partial=False)

    def _extract_text_capped(self, file_path: Path) -> ExtractedContent:
        try:
            with open(file_path, "rb") as f:
                data = f.read(self.cfg.max_text_bytes + 1)
            partial = len(data) > self.cfg.max_text_bytes
            data = data[: self.cfg.max_text_bytes]
            return ExtractedContent(text=data.decode("utf-8", errors="ignore"), partial=partial)
        except OSError:
            return ExtractedContent(text="", partial=False)

    def _extract_large_text_sample(self, file_path: Path) -> ExtractedContent:
        try:
            with open(file_path, "rb") as f:
                data = f.read(self.cfg.sample_bytes + 1)
            data = data[: self.cfg.sample_bytes]
            return ExtractedContent(text=data.decode("utf-8", errors="ignore"), partial=True)
        except OSError:
            return ExtractedContent(text="", partial=False)

    def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from PDF, suppressing pypdf warnings."""
        try:
            from pypdf import PdfReader
        except Exception:
            return ""

        try:
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()

            try:
                reader = PdfReader(str(file_path))
                parts: list[str] = []
                for page in reader.pages:
                    t = page.extract_text() or ""
                    if t:
                        parts.append(t)
                return "\n".join(parts)
            finally:
                sys.stderr = old_stderr

        except Exception:
            return ""

    def _extract_odt_text(self, file_path: Path) -> str:
        try:
            with zipfile.ZipFile(file_path, "r") as z:
                with z.open("content.xml") as f:
                    xml_data = f.read()
        except Exception:
            return ""

        try:
            root = ET.fromstring(xml_data)
        except Exception:
            return ""

        texts: list[str] = []
        for node in root.iter():
            if node.text and node.text.strip():
                texts.append(node.text.strip())
        return "\n".join(texts)

    def extract_full_for_upgrade(self, file_path: Path) -> ExtractedContent:
        """Used for lazy upgrade of partial docs."""
        ext = file_path.suffix.lower()

        if ext in self.LARGE_TEXT_EXTS:
            return self._extract_text_capped(file_path)

        return self.extract(file_path)