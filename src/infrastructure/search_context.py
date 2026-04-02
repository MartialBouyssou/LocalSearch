from __future__ import annotations

from pathlib import Path


class SearchContext:
    """Track current search directory (for :cd / :pwd commands)."""

    def __init__(self, root_path: str = "."):
        self.root = Path(root_path).resolve()
        self.current = self.root

    def cd(self, path: str) -> tuple[bool, str]:
        """
        Change directory within the index root.
        
        Returns: (success, message)
        """
        if path == "" or path == "/":
            self.current = self.root
            return True, str(self.root)

        target = Path(path)
    
        if not target.is_absolute():
            target = self.current / target
        
        target = target.resolve()

        try:
            target.relative_to(self.root)
        except ValueError:
            return False, f"Path outside root: {path}"
        
        if not target.exists():
            return False, f"Path does not exist: {target}"
        
        if not target.is_dir():
            return False, f"Not a directory: {target}"
        
        self.current = target
        return True, str(self.current)

    def pwd(self) -> str:
        """Get current directory."""
        return str(self.current)

    def get_relative_current(self) -> str:
        """Get current path relative to root."""
        try:
            return str(self.current.relative_to(self.root))
        except ValueError:
            return str(self.current)