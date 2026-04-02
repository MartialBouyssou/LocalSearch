"""Index persistence to disk"""
import json
from pathlib import Path
from typing import Optional
from core.index import InvertedIndex


class DiskStorage:
    """Handles index serialization and deserialization"""
    
    def __init__(self, storage_path: str = "search_index.json"):
        self.storage_path = Path(storage_path)
    
    def save_index(self, index: InvertedIndex) -> bool:
        """
        Save index to disk
        
        Args:
            index: InvertedIndex to save
            
        Returns:
            True if successful
        """
        try:
            data = index.to_dict()
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving index: {e}")
            return False
    
    def load_index(self) -> Optional[InvertedIndex]:
        """
        Load index from disk
        
        Returns:
            InvertedIndex or None if not found
        """
        if not self.storage_path.exists():
            return None
        
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return InvertedIndex.from_dict(data)
        except Exception as e:
            print(f"Error loading index: {e}")
            return None
    
    def index_exists(self) -> bool:
        """Check if index file exists"""
        return self.storage_path.exists()