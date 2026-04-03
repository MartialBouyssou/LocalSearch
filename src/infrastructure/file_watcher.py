from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent


class DebouncedFileWatcher:
    """
    Watches filesystem for changes and debounces them.
    
    Calls a callback after `debounce_seconds` of inactivity.
    """

    def __init__(self, debounce_seconds: float = 5.0):
        self.debounce_seconds = debounce_seconds
        self.observer: Optional[Observer] = None

        self.pending_changes: dict[str, str] = {}
        self.pending_lock = threading.Lock()

        self.debounce_timer: Optional[threading.Timer] = None
        self.timer_lock = threading.Lock()

        self.on_changes_callback: Optional[Callable[[dict[str, str]], None]] = None

    def start_watching(
        self,
        paths: list[Path],
        on_changes: Callable[[dict[str, str]], None],
        recursive: bool = True,
    ) -> None:
        """
        Start watching paths for changes.
        
        Args:
            paths: directories to watch
            on_changes: callback(changes_dict) where keys=paths, values=event_type
            recursive: watch subdirectories
        """
        self.on_changes_callback = on_changes
        
        handler = _DebounceHandler(self)
        self.observer = Observer()
        
        for path in paths:
            self.observer.schedule(handler, str(path), recursive=recursive)
        
        self.observer.start()

    def stop_watching(self) -> None:
        """Stop watching and flush pending changes."""
        if self.observer:
            self.observer.stop()
            self.observer.join()

        self._flush_pending()

    def _on_file_event(self, event_type: str, path: str) -> None:
        """Called when a file event is detected."""
        with self.pending_lock:
            self.pending_changes[path] = event_type

        self._reset_debounce_timer()

    def _reset_debounce_timer(self) -> None:
        """Reset the debounce timer."""
        with self.timer_lock:
            if self.debounce_timer:
                self.debounce_timer.cancel()
            
            self.debounce_timer = threading.Timer(
                self.debounce_seconds,
                self._flush_pending,
            )
            self.debounce_timer.daemon = True
            self.debounce_timer.start()

    def _flush_pending(self) -> None:
        """Fire callback with all pending changes."""
        with self.pending_lock:
            if not self.pending_changes:
                return
            
            changes = dict(self.pending_changes)
            self.pending_changes.clear()
        
        if self.on_changes_callback:
            self.on_changes_callback(changes)


class _DebounceHandler(FileSystemEventHandler):
    """Internal watchdog handler."""

    def __init__(self, watcher: DebouncedFileWatcher):
        self.watcher = watcher

    def on_created(self, event: FileCreatedEvent) -> None:
        if not event.is_directory:
            self.watcher._on_file_event("created", event.src_path)

    def on_deleted(self, event: FileDeletedEvent) -> None:
        if not event.is_directory:
            self.watcher._on_file_event("deleted", event.src_path)

    def on_modified(self, event: FileModifiedEvent) -> None:
        if not event.is_directory:
            self.watcher._on_file_event("modified", event.src_path)