from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Config:
    """Application configuration."""
    db: str = "search_index.db"
    path: str = "."
    topk: int = 10
    commit_every: int = 500
    max_text_bytes: int = 2_000_000
    sample_bytes: int = 256_000
    debounce: float = 5.0
    no_lazy_upgrade: bool = False
    include_soft_skips: bool = False
    no_watch: bool = False
    use_fuzzy: bool = True
    fuzzy_lambda: float = 5.0
    fuzzy_threshold: float = 0.5

    @classmethod
    def load(cls, config_file: str = "config.json") -> Config:
        """Load configuration from JSON file, or use defaults if not found."""
        config_path = Path(config_file)
        
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    data = json.load(f)
                return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[X] Error reading config: {e}. Using defaults.")
                return cls()
        else:
            print(f"[*] Config file '{config_file}' not found. Using defaults.")
            return cls()

    def save(self, config_file: str = "config.json") -> None:
        """Save configuration to JSON file."""
        config_path = Path(config_file)
        with open(config_path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def merge_args(self, args) -> Config:
        """Merge CLI args into config (args override config file)."""
        updates = {}

        for field in self.__dataclass_fields__:
            if hasattr(args, field):
                arg_value = getattr(args, field)
                if arg_value is not None and arg_value != getattr(self, field, None):
                    updates[field] = arg_value

        if updates:
            return Config(**{**asdict(self), **updates})
        return self