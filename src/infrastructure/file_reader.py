"""File reading utilities"""
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
        """Safely read file content as text (utf-8)."""
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
        Scan directory for files.

        Filtering/deciding what to index is done by ContentExtractor.
        This method is purely responsible for walking the filesystem efficiently.

        Args:
            skip_hidden: skip hidden files/dirs (starting with '.')
            include_soft_skips: if True, do scan dirs like .venv/node_modules/.git
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