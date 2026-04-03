# LocalSearch

LocalSearch is a local full-text search engine for files and folders.
It builds an index in SQLite, ranks results with BM25, and supports incremental updates through a file watcher.

## Features

- Full-text indexing for local files
- BM25 ranking for search relevance
- One-shot CLI search mode
- Interactive search mode with scoped navigation (:cd, :pwd)
- Incremental indexing with filesystem watcher and debounce
- SQLite-based persistence
- Lazy upgrade for partially indexed large documents
- REST API (FastAPI) for health, search, and indexing operations

## Project Status

This project is currently in beta.
The core flow is functional and ready for local usage, but some edge cases and advanced behaviors are still being refined.

## Requirements

- Python 3.10+
- Linux/macOS/Windows
- Dependencies listed in requirement.txt

## Installation

```bash
git clone <your-repo-url>
cd LocalSearch
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirement.txt
```

### API dependencies

If you want to use the HTTP API, install API dependencies too:

```bash
pip install -r requirement_api.txt
```

## Quick Start

### 1. Configure

Edit config.json (or keep defaults).

Example:

```json
{
  "db": "search_index.db",
  "path": "/home/your-user",
  "topk": 10,
  "commit_every": 500,
  "max_text_bytes": 2000000,
  "sample_bytes": 256000,
  "debounce": 5.0,
  "no_lazy_upgrade": false,
  "include_soft_skips": false,
  "no_watch": false
}
```

### 2. First run (auto-index + interactive mode)

```bash
python start.py
```

On first run, LocalSearch detects that the DB does not exist and performs initial indexing.

### 3. One-shot search

```bash
python start.py --search "your query" --topk 10
```

### 4. Force full reindex

```bash
python start.py --reindex
```

## API Usage

### Start API server

```bash
python api_server.py --config config.json --host 0.0.0.0 --port 8000
```

Optional development mode:

```bash
python api_server.py --reload
```

### API Endpoints

- GET /health
- POST /search
- POST /indexing/reindex
- GET /indexing/status
- GET /indexing/reindex-status

### Example requests

Health check:

```bash
curl http://localhost:8000/health
```

Search:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"python tokenizer", "top_k":10}'
```

Trigger full reindex in background:

```bash
curl -X POST http://localhost:8000/indexing/reindex \
  -H "Content-Type: application/json" \
  -d '{"include_soft_skips": false}'
```

Check index status:

```bash
curl http://localhost:8000/indexing/status
```

Check reindex progress:

```bash
curl http://localhost:8000/indexing/reindex-status
```

## Interactive Commands

In interactive mode:

- :help -> Show help
- :pwd -> Show current search scope
- :cd PATH -> Change current search scope
- :clear -> Clear terminal
- :quit / :q / :exit -> Exit

Search results are filtered to the current directory scope shown in the prompt.

## CLI Options

```text
--config           Configuration file path (default: config.json)
--db               SQLite database path
--path             Directory to index
--reindex          Rebuild index from scratch
--search           Run one-shot query and exit
--topk             Maximum number of results
--commit-every     Commit every N files during reindex
--max-text-bytes   Max bytes read for text-like files
--sample-bytes     Sample bytes for large text files (CSV/TSV)
--no-lazy-upgrade  Disable lazy upgrade for partial docs
--include-soft-skips
                   Include .venv, node_modules, .git, etc.
--no-watch         Disable file watcher in interactive mode
--debounce         Debounce seconds for watcher events
--save-config      Save merged config and exit
```

## Indexed Content

### Text-like files

Examples:

- .txt, .md, .rst, .log
- .py, .js, .ts, .java, .c, .cpp
- .html, .css, .json, .yaml, .yml, .xml, .toml
- .sql, .sh, .bat

### Special handling

- .pdf (via pypdf)
- .odt (content.xml extraction)
- .csv and .tsv are sampled (partial indexing by design)

### Skipped content

- Binary/media/archive files
- SQLite/database/cache/temp files
- Hidden files and hidden directories by default
- Common heavy directories by default (.git, .venv, node_modules, etc.)

## Architecture

- src/core: tokenizer, index abstraction, ranking, models
- src/application: indexing, incremental indexing, search orchestration
- src/infrastructure: SQLite storage, extraction, file scanning, watcher, config
- src/api: FastAPI app factory, routes, and request/response models
- start.py: CLI launcher from project root
- src/main.py: CLI implementation and interactive shell
- api_server.py: API server launcher (uvicorn)

## Running Tests

```bash
python -m unittest discover -s tests -v
```

Current tests cover tokenizer, index behavior, and ranking.

## Known Limitations (Beta)

- Deletion handling in incremental indexing is not yet fully implemented.
- Search is optimized for local datasets and not distributed workloads.
- Very large files may be partially indexed depending on configuration.

## Roadmap Ideas

- Stronger incremental delete/update consistency
- Better snippet/highlight output in results
- Additional file format extractors
- Packaging as a pip-installable CLI

## License

This project is licensed under the MIT License.
Unless otherwise stated, this license applies to all commits in this repository history.
