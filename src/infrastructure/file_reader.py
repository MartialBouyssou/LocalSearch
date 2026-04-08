from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Optional


class FileReader:
    """
    Handles file I/O operations.

    - Uses os.walk() so we can prune directories (big perf win on large trees).
    - Supports:
        * ALWAYS_SKIP_DIR_NAMES: never scan
        * DEFAULT_SKIP_DIR_NAMES: skipped by default, but can be included
    """

    ALWAYS_SKIP_DIR_NAMES = {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".eggs",
        "dist",
        "build",
    }

    DEFAULT_SKIP_DIR_NAMES = {
        ".git", ".hg", ".svn",
        ".idea", ".vscode",
        "node_modules",
        ".venv", "venv",
    }

    @staticmethod
    def read_file(file_path: Path, encoding: str = "utf-8") -> Optional[str]:
        """
        Safely read file content as text with UTF-8 encoding (default).
        
        Args:
            file_path: Path to the file to read.
            encoding: Text encoding to use (default: UTF-8).
            
        Returns:
            File content as string, or None if read fails.
        """
        try:
            with open(file_path, "r", encoding=encoding, errors="strict") as f:
                return f.read()
        except (UnicodeDecodeError, OSError, IOError):
            return None

    @staticmethod
    def scan_directory(
        directory: Path,
        recursive: bool = True,
        skip_hidden: bool = True,
        include_soft_skips: bool = False,
        extra_always_skip_dir_names: Optional[set[str]] = None,
        extra_soft_skip_dir_names: Optional[set[str]] = None,
    ) -> Generator[Path, None, None]:
        """
        Scan directory and yield file paths efficiently using os.walk().
        
        Supports directory pruning to avoid scanning excluded directories (performance win).
        Filtering decisions are delegated to ContentExtractor.
        
        Args:
            directory: Root directory to scan.
            recursive: If True, scan subdirectories; if False, only top level.
            skip_hidden: Skip hidden files/directories (starting with '.').
            include_soft_skips: If True, include soft-skip dirs like .venv, .git.
            extra_always_skip_dir_names: Additional directories to always skip.
            extra_soft_skip_dir_names: Additional directories to conditionally skip.
            
        Yields:
            Path objects for files in the directory tree.
        """
        if not directory.is_dir():
            return

        always_skip = set(FileReader.ALWAYS_SKIP_DIR_NAMES)
        soft_skip = set(FileReader.DEFAULT_SKIP_DIR_NAMES)

        if extra_always_skip_dir_names:
            always_skip |= set(extra_always_skip_dir_names)
        if extra_soft_skip_dir_names:
            soft_skip |= set(extra_soft_skip_dir_names)

        if not recursive:
            for p in directory.iterdir():
                if p.is_file():
                    if skip_hidden and p.name.startswith("."):
                        continue
                    yield p
            return

        for root, dirs, files in os.walk(directory):
            root_path = Path(root)

            pruned = []
            for d in list(dirs):
                if d in always_skip:
                    continue
                if not include_soft_skips and d in soft_skip:
                    continue
                if skip_hidden and d.startswith("."):
                    continue
                pruned.append(d)
            dirs[:] = pruned

            for f in files:
                if skip_hidden and f.startswith("."):
                    continue
                yield root_path / f

    @staticmethod
    def get_file_info(file_path: Path) -> dict:
        """Extract file metadata (best-effort)."""
        try:
            size = file_path.stat().st_size
        except OSError:
            size = 0

        return {
            "filename": file_path.name,
            "path": str(file_path.parent),
            "extension": file_path.suffix.lower(),
            "size": size,
        }