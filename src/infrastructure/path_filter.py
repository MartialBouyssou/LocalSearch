class PathFilter:
    """Aggressive filtering for cache/noise directories."""
    
    HARD_SKIPS = {
        '.cache',
        'cache2',
        '__pycache__',
        '.pytest_cache',
        '.mozilla',
        '.config',
        'snap/firefox',
        'snap/chromium',
        '.gradle',
        '.m2',
        'node_modules',
        '.venv',
        'venv',
        '.git',
        'Library/PackageCache',
        'Library/Artifacts',
        '/R/',
        '/r/',
        'x86_64-pc-linux-gnu-library',
        'rbibutils',
        'smali_classes',
        'com_spotify_music',
        '.bundle',
        '.rvm',
        'Pods',
        'History', # sensitiv
    }
    
    BINARY_EXTENSIONS = {
        '.smali', '.dex', '.apk', '.class', '.o', '.so', '.pyc',
        '.pyo', '.pyd', '.jar', '.zip', '.tar', '.gz', '.7z', '.xml'
    }
    
    @staticmethod
    def is_suspicious_path(path: str) -> bool:
        """
        Check if a path contains suspicious or cache-related directories.
        
        Args:
            path: File path string to check.
            
        Returns:
            True if the path contains known cache/noise directories.
        """
        path_lower = path.lower()
        
        for skip in PathFilter.HARD_SKIPS:
            skip_lower = skip.lower()
            if f"/{skip_lower}/" in path_lower or f"{skip_lower}/" in path_lower or path_lower.startswith(skip_lower):
                return True
        
        return False
    
    @staticmethod
    def is_likely_cache_file(filename: str, extension: str) -> bool:
        """
        Detect if a file is likely a cache or binary file based on name and extension.
        
        Args:
            filename: Name of the file.
            extension: File extension (with dot).
            
        Returns:
            True if the file is likely a cache file.
        """
        if not filename:
            return True
        
        filename_lower = filename.lower()
        ext_lower = extension.lower() if extension else ""
        
        if ext_lower in PathFilter.BINARY_EXTENSIONS:
            return True
        
        if not ext_lower:
            if len(filename_lower) > 20 and all(c in '0123456789abcdef' for c in filename_lower):
                return True
            if len(filename_lower) > 30:
                return True
        
        return False
    
    @staticmethod
    def should_include(path: str, filename: str, extension: str) -> bool:
        """Return True if file should be included."""
        return not (
            PathFilter.is_suspicious_path(path) or
            PathFilter.is_likely_cache_file(filename, extension)
        )