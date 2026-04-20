# Project Structure

## Database Schema

Ragger uses a simple SQLite schema optimized for hybrid search.

### `memories` Table

Stores documents with embeddings and metadata.

```sql
CREATE TABLE memories
(
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT NOT NULL,
    embedding  BLOB NOT NULL, -- 384 × float32 = 1536 bytes
    timestamp  TEXT NOT NULL, -- ISO 8601
    collection TEXT NOT NULL DEFAULT 'memory',
    category   TEXT NOT NULL DEFAULT '',
    tags       TEXT NOT NULL DEFAULT '',  -- comma-separated
    metadata   TEXT           -- JSON (remaining fields: source, keep, etc.)
);
```

**Fields:**

- `id` — Auto-incrementing primary key
- `text` — Document text (original content)
- `embedding` — Binary blob containing 384 float32 values (1536 bytes)
- `timestamp` — ISO 8601 timestamp (e.g., `2024-03-20T15:23:45`)
- `collection` — Memory collection (indexed, e.g., `memory`, `reference`, `docs`)
- `category` — Classification within collection (indexed, e.g., `fact`, `decision`)
- `tags` — Comma-separated tag list (e.g., `ragger,architecture`)
- `metadata` — JSON-encoded dictionary for remaining fields (source, keep, section, etc.)

**Metadata structure (JSON column):**

```json
{
  "source": "notes.md",
  "section": "API Reference » Authentication",
  "keep": false
}
```

### `memory_usage` Table

Tracks access patterns for usage-based ranking.

```sql
CREATE TABLE memory_usage
(
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL REFERENCES memories (id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    timestamp TEXT    NOT NULL
);
```

**Fields:**

- `id` — Auto-incrementing primary key
- `memory_id` — Foreign key to `memories.id`
- `timestamp` — ISO 8601 timestamp of access

**Purpose:** Log every time a memory appears in search results. Future
versions can use this for boosting frequently-accessed memories or
cleaning up stale content.

### `bm25_index` Table

BM25 term index for keyword search.

```sql
CREATE TABLE bm25_index
(
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL REFERENCES memories (id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    token     TEXT    NOT NULL,
    term_freq INTEGER NOT NULL
);
```

**Fields:**

- `id` — Auto-incrementing primary key
- `memory_id` — Foreign key to `memories.id`
- `token` — Tokenized term (lowercased, stemmed)
- `term_freq` — Number of times this token appears in the document

**Purpose:** Inverted index for BM25 keyword scoring. Rebuilt automatically
when documents are stored. Use `ragger rebuild-bm25` to rebuild the entire
index after bulk changes.

---

## Module Descriptions

### `backend.py`

Abstract base class for storage backends. Provides:

- NumPy-based brute-force cosine similarity search
- Hybrid BM25 blending
- Collection filtering
- Usage tracking hooks
- Query logging

All backends inherit from `MemoryBackend` and implement four methods:

1. `store_raw(text, embedding, metadata, timestamp) -> str`
2. `load_all_embeddings() -> tuple`
3. `count() -> int`
4. `close()`

See [Python API](python-api.md) for details on writing custom backends.

### `sqlite_backend.py`

SQLite implementation of `MemoryBackend`. Default backend for single-user
deployments. Uses Python's built-in `sqlite3` module — no external dependencies.

Features:

- Single-file database (`~/.ragger/memories.db`)
- Binary blob storage for embeddings (384 × float32 = 1536 bytes)
- JSON metadata encoding
- Automatic BM25 index updates on store

### `bm25.py`

Pure Python implementation of Okapi BM25 keyword ranking. No external
dependencies (uses only stdlib).

Features:

- Tunable parameters: `k1` (term frequency saturation), `b` (document length normalization)
- Tokenization: lowercasing, punctuation removal, stopword filtering (optional)
- Persistent index stored in `bm25_index` table
- Automatic updates on document store

See [Search & RAG](search-and-rag.md) for BM25 tuning details.

### `config.py`

INI config file loader with layered override support.

Features:

- Layered config: system (`/etc/ragger.ini`) → user (`~/.ragger/settings.ini`)
- SERVER_LOCKED keys (system values can't be overridden)
- System ceilings (hard limits on user-configurable values)
- Auto-bootstrap (creates default user config on first run)

See [Configuration](configuration.md) for full reference.

### `embedding.py`

Wrapper for sentence-transformers embeddings.

Features:

- Model caching (download once, reuse)
- Device selection (CPU, CUDA, MPS)
- Batch encoding for bulk imports

Default model: `all-MiniLM-L6-v2` (384 dimensions, ~90MB).

### `memory.py`

`RaggerMemory` facade class. Factory for creating backends and providing
a unified API for storing and searching.

Features:

- Config-driven backend selection
- Context manager support (`with RaggerMemory() as memory`)
- Unified `store()` and `search()` interface

See [Python API](python-api.md) for usage examples.

### `server.py`

HTTP server using Flask.

**Endpoints:**

- `GET /health` — Health check
- `GET /count` — Total memory count
- `POST /store` — Store a memory
- `POST /search` — Search memories

**Features:**

- Bearer token authentication (optional)
- JSON request/response
- CORS support (configurable)

See [HTTP API](http-api.md) for endpoint reference.

### `mcp_server.py`

MCP-compliant server for AI agent integration (protocol version `2024-11-05`).

Implements the standard MCP handshake (`initialize`, `notifications/initialized`)
and tool discovery (`tools/list`, `tools/call`). Also accepts plain text queries
as a search shortcut for interactive CLI use.

**Tools:**

- `store` — Store a memory
- `search` — Search memories

See [HTTP API](http-api.md) for MCP details.

### `cli.py`

Verb-style command-line interface.

**Commands:**

- `ragger serve` — Start HTTP server
- `ragger store <text>` — Store a memory
- `ragger search <query>` — Search memories
- `ragger import <file>` — Import a text file
- `ragger export <collection> <dir>` — Export documents
- `ragger count` — Count memories
- `ragger rebuild-bm25` — Rebuild BM25 index
- `ragger mcp` — Run MCP server
- `ragger update-model` — Download embedding model

See [Getting Started](getting-started.md) for CLI usage.

### `lang/`

Internationalization (i18n) strings.

**Files:**

- `en.py` — English strings

Future languages can be added by creating `es.py`, `fr.py`, etc.

---

## C++ Port Architecture Comparison

A [C++ port](https://github.com/diskerror/ragger.cpp) is also available with
the same HTTP API, database format, and config file.

| Component         | Python                             | C++                                |
|-------------------|------------------------------------|------------------------------------|
| **Language**      | Python 3.10+                       | C++20                              |
| **Embedding**     | sentence-transformers (PyTorch)    | ONNX Runtime (ONNXRuntime C++ API) |
| **HTTP server**   | Flask                              | cpp-httplib                        |
| **SQLite**        | Python stdlib `sqlite3`            | SQLite3 C API                      |
| **BM25**          | Pure Python                        | Pure C++                           |
| **Vector search** | NumPy                              | Eigen (matrix library)             |
| **Config**        | Python `configparser`              | inih (C INI parser)                |
| **JSON**          | Python stdlib `json`               | nlohmann/json                      |
| **Performance**   | ~10-50ms for 50K docs              | ~5-20ms for 50K docs               |
| **Binary size**   | N/A (interpreter + packages)       | ~3MB static binary                 |
| **Dependencies**  | ~500MB (Python + packages + model) | ~150MB (binary + model)            |

**Compatibility:**

- Both versions use the same database file
- Both versions use the same config file format
- Both versions expose the same HTTP API
- You can swap between them with zero data migration

**When to use which:**

- **Python:** Rapid development, easy to modify, familiar ecosystem
- **C++:** Faster search, smaller footprint, deployment to resource-constrained systems

---

## Related

- [Python API](python-api.md) — Using Ragger as a library
- [Search & RAG](search-and-rag.md) — How hybrid search works
- [Configuration](configuration.md) — Config file reference
