from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time
import os
import readline
import atexit

from application.indexing_service import IndexingService
from application.incremental_indexing_service import IncrementalIndexingService
from application.search_engine import SearchEngine
from infrastructure.db_storage import DBStorage
from infrastructure.file_reader import FileReader
from infrastructure.content_extractor import ContentExtractor, ExtractorConfig
from infrastructure.file_watcher import DebouncedFileWatcher
from infrastructure.config import Config
from infrastructure.search_context import SearchContext


def _init_readline(search_context: SearchContext) -> None:
    """Initialize readline with history file and autocomplete."""
    history_file = Path.home() / ".localsearch_history"
    try:
        readline.read_history_file(str(history_file))
    except FileNotFoundError:
        pass
    
    atexit.register(lambda: readline.write_history_file(str(history_file)))
    
    readline.set_completer_delims(" \t\n")
    
    def completer(text: str, state: int) -> str | None:
        """Autocomplete for :cd command."""
        line = readline.get_line_buffer()
        
        if not line.startswith(":cd"):
            return None
        
        if line.startswith(":cd "):
            partial_path = line[4:]
        else:
            return None
        
        if state == 0:
            completer.matches = _get_path_completions(partial_path, search_context)
        
        if state < len(completer.matches):
            return completer.matches[state]
        return None
    
    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")


def _get_path_completions(partial_path: str, search_context: SearchContext) -> list[str]:
    """Get all path completions for a given partial path."""
    try:
        if "/" in partial_path:
            last_slash = partial_path.rfind("/")
            base_path = partial_path[:last_slash + 1]
            prefix = partial_path[last_slash + 1:]
            
            target_dir = search_context.current / base_path
        else:
            base_path = ""
            prefix = partial_path
            target_dir = search_context.current
        
        target_dir = target_dir.resolve()
        
        try:
            target_dir.relative_to(search_context.root)
        except ValueError:
            return []
        
        if not target_dir.exists() or not target_dir.is_dir():
            return []
        
        matches = []
        try:
            for item in sorted(target_dir.iterdir()):
                if item.is_dir() and item.name.startswith(prefix):
                    full_completion = base_path + item.name + "/"
                    matches.append(full_completion)
        except (PermissionError, OSError):
            pass
        
        return matches
    except Exception:
        return []


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="LocalSearch", description="Local file search engine (SQLite).")
    p.add_argument("--config", default="config.json", help="Configuration file path.")
    p.add_argument("--db", default=None, help="SQLite database path.")
    p.add_argument("--path", default=None, help="Directory to index.")
    p.add_argument("--reindex", action="store_true", help="Rebuild index from scratch.")
    p.add_argument("--search", type=str, default=None, help="Run a one-shot search query.")
    p.add_argument("--topk", type=int, default=None, help="Max results to return.")
    p.add_argument("--commit-every", type=int, default=None, help="Commit every N files during reindex.")
    p.add_argument("--max-text-bytes", type=int, default=None, help="Max bytes to read for text-like files.")
    p.add_argument("--sample-bytes", type=int, default=None, help="Sample bytes for large text files (CSV/TSV).")
    p.add_argument("--no-lazy-upgrade", action="store_true", help="Disable lazy upgrade of partial docs.")
    p.add_argument("--include-soft-skips", action="store_true", help="Include .venv, node_modules, .git, etc.")
    p.add_argument("--no-watch", action="store_true", help="Disable file watcher in interactive mode.")
    p.add_argument("--debounce", type=float, default=None, help="Debounce seconds for file watcher.")
    p.add_argument("--save-config", action="store_true", help="Save current config to file and exit.")
    return p


def interactive_loop(
    engine: SearchEngine,
    cfg: Config,
    context: SearchContext,
    watcher: DebouncedFileWatcher | None = None,
) -> None:
    if watcher:
        print(f"[*] File watcher active (watching: {cfg.path})\n")
    
    print("Interactive mode. Type your query, or 'quit' to exit.\n")
    print("Commands info:\n > ':help'\n")
    
    while True:
        prompt = f"[{context.get_relative_current()}] query> "
        q = input(prompt).strip()
        
        if not q:
            continue

        if q.lower() in {":quit", ":exit", ":q"}:
            break

        if q == ":clear":
            clear()
            continue

        if q == ":pwd":
            print(f"Current: {context.pwd()}\n")
            continue

        if q == ":help":
            print("""
                    LocalSearch - Interactive Mode

                    Available commands:
                    - :help    Show this help message
                    - :pwd     Show the current directory
                    - :cd PATH Change the current directory
                    - :clear   Clear the screen
                    - :quit    Exit the application
                    - :q       Exit the application
                    - :exit    Exit the application

                    Search:
                    - Type any query to run a search.
                    - Results show the file name, path, extension, and score.
                    - Searches are limited to the current directory shown in the prompt.

                    Tip:
                    - Use Tab after :cd to autocomplete directories.
                """)
            continue

        if q.startswith(":cd"):
            path = q[3:].strip()
            success, msg = context.cd(path)
            if success:
                print(f"[*] {msg}\n")
            else:
                print(f"[X] {msg}\n")
            continue

        print("[*] Searching...", end="", flush=True)
        start = time.time()
        results = engine.search(q, top_k=cfg.topk, lazy_upgrade=not cfg.no_lazy_upgrade)
        elapsed = time.time() - start
        
        current_path = context.current
        filtered_results = [
            r for r in results
            if Path(r.document.path).resolve().is_relative_to(current_path)
        ]
        
        if not filtered_results:
            print(f"\r[X] No results found. ({elapsed:.2f}s)\n")
            continue

        print(f"\r -> Found {len(filtered_results)} result(s). ({elapsed:.2f}s)\n")
        for i, r in enumerate(filtered_results, 1):
            print(f"{i}. {r.document.filename}")
            print(f"     [Path] {r.document.path}")
            print(f"[Extension] {r.document.extension}")
            print(f"    [Score] {r.score:.4f}")
        print()
    
    if watcher:
        print("\n[*] Stopping file watcher...")
        watcher.stop_watching()

def clear():
    os.system("clear" if os.name == "posix" else "cls")


def main() -> None:
    args = build_parser().parse_args()
    
    cfg = Config.load(args.config)
    cfg = cfg.merge_args(args)

    if args.save_config:
        cfg.save(args.config)
        return

    context = SearchContext(cfg.path)
    _init_readline(context)

    db = DBStorage(cfg.db)
    file_reader = FileReader()
    extractor = ContentExtractor(
        file_reader=file_reader,
        cfg=ExtractorConfig(max_text_bytes=cfg.max_text_bytes, sample_bytes=cfg.sample_bytes),
    )

    db_exists = Path(cfg.db).exists()
    need_index = args.reindex or (not db_exists and args.search is None)

    if need_index:
        if not db_exists:
            print(" => LocalSearch -- first run, indexing...\n")
        else:
            print(" => LocalSearch -- reindex\n")
        
        indexing = IndexingService(db_storage=db, extractor=extractor)
        start = time.time()
        indexed = indexing.index_directory(
            Path(cfg.path),
            recursive=True,
            commit_every=cfg.commit_every,
            include_soft_skips=cfg.include_soft_skips,
        )
        elapsed = time.time() - start
        print(f"[*] Completed in {elapsed:.2f}s\n")

    if args.search is not None:
        print("[*] Searching...", end="", flush=True)
        start = time.time()
        
        engine = SearchEngine(db_storage=db, extractor=extractor)
        try:
            results = engine.search(args.search, top_k=cfg.topk, lazy_upgrade=not cfg.no_lazy_upgrade)
        finally:
            engine.close()

        elapsed = time.time() - start

        if not results:
            print(f"\r[X] No results found. ({elapsed:.2f}s)")
            return

        print(f"\r -> Found {len(results)} result(s). ({elapsed:.2f}s)\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r.document.filename}")
            print(f"     [Path] {r.document.path}")
            print(f"[Extension] {r.document.extension}")
            print(f"    [Score] {r.score:.4f}")
        return

    if not need_index and args.search is None:
        engine = SearchEngine(db_storage=db, extractor=extractor)
        watcher = None
        
        try:
            if not cfg.no_watch:
                incremental = IncrementalIndexingService(db_path=cfg.db, extractor=extractor)
                watcher = DebouncedFileWatcher(debounce_seconds=cfg.debounce)
                watcher.start_watching(
                    paths=[Path(cfg.path)],
                    on_changes=incremental.apply_changes,
                    recursive=True,
                )
            
            interactive_loop(
                engine,
                cfg=cfg,
                context=context,
                watcher=watcher,
            )
        finally:
            if watcher:
                watcher.stop_watching()
            engine.close()


if __name__ == "__main__":
    clear()
    main()