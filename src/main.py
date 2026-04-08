from __future__ import annotations

import argparse
from pathlib import Path
import time
import os
import readline
import atexit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.application.indexing_service import IndexingService
from src.application.incremental_indexing_service import IncrementalIndexingService
from src.application.search_engine import SearchEngine, SearchCancelled
from src.infrastructure.db_storage import DBStorage
from src.infrastructure.file_reader import FileReader
from src.infrastructure.content_extractor import ContentExtractor, ExtractorConfig
from src.infrastructure.file_watcher import DebouncedFileWatcher
from src.infrastructure.config import Config
from src.infrastructure.search_context import SearchContext


def _get_current_tag() -> str | None:
    """Return the current git tag if available."""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError):
        return None

    if result.returncode != 0:
        return None

    tag = result.stdout.strip()
    return tag or None


def _get_github_repo_slug() -> str | None:
    """Extract owner/repo from git remote origin if hosted on GitHub."""
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError):
        return None

    if result.returncode != 0:
        return None

    remote_url = result.stdout.strip()
    if not remote_url:
        return None

    if remote_url.startswith("git@github.com:"):
        slug = remote_url.split("git@github.com:", 1)[1]
    elif "github.com/" in remote_url:
        slug = remote_url.split("github.com/", 1)[1]
    else:
        return None

    slug = slug.strip().rstrip("/")
    if slug.endswith(".git"):
        slug = slug[:-4]

    if "/" not in slug:
        return None

    return slug


def _parse_semver(version: str) -> tuple[tuple[int, ...], tuple[str, ...]] | None:
    """Parse a semver-like tag such as v3.0.0-beta.1."""
    if not version:
        return None

    cleaned = version.strip().lstrip("vV")
    if not cleaned:
        return None

    cleaned = cleaned.split("+", 1)[0]
    core_part, sep, prerelease_part = cleaned.partition("-")

    core_numbers: list[int] = []
    for segment in core_part.split("."):
        if not segment.isdigit():
            return None
        core_numbers.append(int(segment))

    prerelease_segments: tuple[str, ...] = ()
    if sep:
        prerelease_segments = tuple(s for s in prerelease_part.split(".") if s)

    return tuple(core_numbers), prerelease_segments


def _compare_prerelease(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    """Compare semver prerelease segments."""
    if not left and not right:
        return 0
    if not left:
        return 1
    if not right:
        return -1

    max_len = max(len(left), len(right))
    for i in range(max_len):
        if i >= len(left):
            return -1
        if i >= len(right):
            return 1

        l_seg = left[i]
        r_seg = right[i]
        if l_seg == r_seg:
            continue

        l_is_num = l_seg.isdigit()
        r_is_num = r_seg.isdigit()

        if l_is_num and r_is_num:
            l_num = int(l_seg)
            r_num = int(r_seg)
            if l_num < r_num:
                return -1
            if l_num > r_num:
                return 1
            continue

        if l_is_num and not r_is_num:
            return -1
        if not l_is_num and r_is_num:
            return 1

        if l_seg < r_seg:
            return -1
        return 1

    return 0


def _is_version_newer(candidate: str, current: str) -> bool:
    """Return True when candidate tag is newer than current tag."""
    parsed_candidate = _parse_semver(candidate)
    parsed_current = _parse_semver(current)
    if not parsed_candidate or not parsed_current:
        return False

    candidate_core, candidate_pre = parsed_candidate
    current_core, current_pre = parsed_current

    max_len = max(len(candidate_core), len(current_core))
    for i in range(max_len):
        left = candidate_core[i] if i < len(candidate_core) else 0
        right = current_core[i] if i < len(current_core) else 0
        if left > right:
            return True
        if left < right:
            return False

    return _compare_prerelease(candidate_pre, current_pre) > 0


def _fetch_latest_github_release_tag(repo_slug: str, timeout_seconds: float = 1.5) -> str | None:
    """Fetch latest release tag from GitHub API."""
    url = f"https://api.github.com/repos/{repo_slug}/releases/latest"
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "LocalSearch-release-check",
        },
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return None

    tag_name = payload.get("tag_name")
    if not isinstance(tag_name, str):
        return None

    tag_name = tag_name.strip()
    return tag_name or None


def _notify_if_new_release_available() -> None:
    """Print startup message when a newer GitHub release is available."""
    current_tag = _get_current_tag()
    if not current_tag:
        return

    repo_slug = _get_github_repo_slug()
    if not repo_slug:
        return

    latest_tag = _fetch_latest_github_release_tag(repo_slug)
    if not latest_tag:
        return

    if _is_version_newer(latest_tag, current_tag):
        print(
            f"[!] New GitHub release available: {latest_tag} "
            f"(current: {current_tag})"
        )
        print(f"    https://github.com/{repo_slug}/releases/latest\n")


def _init_readline(search_context: SearchContext) -> None:
    """
    Initialize readline with command history and autocompletion for the interactive shell.
    
    Sets up history persistence and enables tab-completion for :cd paths.
    
    Args:
        search_context: Search context containing current search directory.
    """
    history_file = Path.home() / ".localsearch_history"
    try:
        readline.read_history_file(str(history_file))
    except FileNotFoundError:
        pass
    
    atexit.register(lambda: readline.write_history_file(str(history_file)))
    
    readline.set_completer_delims(" \t\n")
    
    def completer(text: str, state: int) -> str | None:
        """Auto-completion function for readline (tab-completion of :cd paths)."""
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
    """
    Generate path completions for a given partial path input.
    
    Returns relative directory paths that start with the given prefix,
    bounded by the search root directory.
    
    Args:
        partial_path: Partial path string entered by user.
        search_context: Search context with root and current directories.
        
    Returns:
        List of relative directory paths starting with the prefix.
    """
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
    """
    Build the command-line argument parser for the application.

    Returns:
        Configured ArgumentParser instance with all supported CLI options.
    """
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
        try:
            results = engine.search(
                q,
                top_k=cfg.topk,
                lazy_upgrade=not cfg.no_lazy_upgrade,
                timeout_ms=cfg.search_timeout_ms,
            )
        except SearchCancelled:
            elapsed = time.time() - start
            print(f"\r[!] Search cancelled. ({elapsed:.2f}s)\n")
            continue
        except KeyboardInterrupt:
            engine.cancel()
            elapsed = time.time() - start
            print(f"\r[!] Search interrupted. ({elapsed:.2f}s)\n")
            continue
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
    """Clear the terminal screen."""
    os.system("clear" if os.name == "posix" else "cls")


def main() -> None:
    args = build_parser().parse_args()

    _notify_if_new_release_available()
    
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
            use_stemming=cfg.use_stemming,
        )
        elapsed = time.time() - start
        print(f"[*] Completed in {elapsed:.2f}s\n")

    if args.search is not None:
        print("[*] Searching...", end="", flush=True)
        start = time.time()
        
        engine = SearchEngine(db_storage=db, extractor=extractor)
        try:
            results = engine.search(
                args.search,
                top_k=cfg.topk,
                lazy_upgrade=not cfg.no_lazy_upgrade,
                timeout_ms=cfg.search_timeout_ms,
            )
        except SearchCancelled:
            print("\r[!] Search timed out.")
            engine.close()
            return
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
            