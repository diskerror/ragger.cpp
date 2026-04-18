# Ragger Feature Suggestions - Personal Additions

## Context

Original file: `~/CLionProjects/Ragger/FEATURE_SUGGESTIONS.md`  
This document extends the existing feature suggestions with additional ideas and refinements.

---

## 10. Memory Visualization & Exploration (New)

**Problem**: Users can't see what memories exist or understand their relationships. Ragger's value is in connecting thoughts, but there's no way to explore those connections visually.

**Solution**: Interactive memory graph viewer.

### Features:
- **Graph view**: Nodes = memories, edges = semantic similarity
- **Filter by collection/tags/timestamp**
- **Cluster detection**: Auto-group related memories (k-means or HDBSCAN)
- **Path exploration**: Click a node → show similar memories → trace connections
- **Timeline view**: Sort memories chronologically with search filtering

### Technical:
- New endpoint: `GET /api/memory/graph?collection=...&limit=100`
  - Returns nodes (id, text_preview, tags, timestamp) + edges (source, target, similarity_score)
- Use precomputed embeddings for fast similarity calculation
- Optional: Store graph structure in separate table (`memory_connections`)

### UX:
- D3.js or Vis.js for visualization
- Hover tooltip shows full content and metadata

---

## 11. Search History & Recall (New)

**Problem**: Users can't revisit past searches or see what they've already found.

**Solution**: Persistent search history with recall capabilities.

### Features:
- **Search log**: Last 50 queries with timestamps, result counts
- **Saved searches**: bookmark queries with custom names
- **Compare results**: View side-by-side of two searches
- **Export history**: Download CSV/JSON of search activity

### Technical:
- New table: `search_history (id, timestamp, query, settings_hash, result_count, memories_json)`
- Settings hash allows matching by config state (useful if tuning)
- Endpoint: `GET /api/history?limit=50&offset=0`

---

## 12. Memory Export & Backup (New)

**Problem**: Users can't easily export or back up their memories outside the app.

**Solution**: Flexible export system with multiple formats.

### Features:
- **Export all memories**: JSON, CSV, Markdown (one file per memory)
- **Export by collection**: Filter dropdown
- **Export by date range**: Start/end date picker
- **Export with full metadata**: tags, source file, creation timestamp

### Formats:
- **JSON**: Full structure for re-import or external processing
- **CSV**: Simple table (id, content, tags, collection)
- **Markdown**: One `.md` file per memory, filename = ID, frontmatter = metadata
- **SQLite dump**: Raw database copy with schema

### Technical:
- New endpoint: `POST /api/export`
  ```json
  {
    "format": "json|csv|md|sqlite",
    "collection": "optional",
    "date_start": "2025-01-01",
    "date_end": "2026-12-31"
  }
  ```
- Return download URL or direct binary response
- Large exports: stream to temp file, return link when ready

---

## 13. Smart Tags & Auto-Classification (New)

**Problem**: Users forget what they've stored or can't find things without remembering exact tags.

**Solution**: AI-assisted tagging and classification.

### Features:
- **Auto-suggest tags**: On memory creation/modify, suggest relevant tags
- **Tag clustering**: Group similar tags (e.g., "c++", "C++", "cpp" → "C++")
- **Topic inference**: Assign top-level topics (code, notes, decisions, research)
- **Tag popularity**: Show most-used tags in sidebar

### Technical:
- Use embedding similarity to find related memories for tag suggestions
- Simple clustering: cosine distance < threshold on tag embeddings
- Endpoint: `POST /api/tag/suggest` returns `{tags: ["tag1", "tag2"]}`

---

## 14. Collaborative Memory (New - Multi-User Extension)

**Problem**: Current multi-user design assumes full isolation. Teams want to share specific memories.

**Solution**: Shared memory spaces within multi-user setup.

### Features:
- **Shared collections**: Mark collection as "shared" → all users can read
- **Granular permissions**: Read/write/admin per user/collection
- **Comments on memories**: Threaded discussions attached to memories
- **Activity feed**: Show who added/modified what

### Technical:
- New table: `collection_permissions (user_id, collection_name, permission_level)`
- Permission levels: `read`, `write`, `admin`
- Endpoint: `GET /api/collection/shared` returns accessible shared collections

---

## 15. Voice Input & Audio Memory (New)

**Problem**: Some users prefer speaking over typing for quick notes.

**Solution**: Audio input integration.

### Features:
- **Record audio**: Browser-based microphone recording
- **Transcribe to text**: Use Whisper or local ASR
- **Store as memory**: Transcribed text + original audio file
- **Search audio content**: Text search covers transcriptions

### Technical:
- New endpoint: `POST /api/memory/from_audio`
  - Accepts audio file (WAV/MP3/M4A)
  - Returns `{memory_id, text_content}`
- Whisper.cpp for local transcription (no API key needed)

---

## 16. Offline-First Sync (New - Advanced)

**Problem**: Users want to use Ragger on multiple devices without constant cloud sync.

**Solution**: Peer-to-peer sync between trusted devices.

### Features:
- **Device pairing**: QR code or manual token
- **Incremental sync**: Only new/modified memories
- **Conflict resolution**: Last-write-wins or manual merge
- **Offline queue**: Queue changes when offline, sync on reconnect

### Technical:
- New endpoint: `GET /api/sync/delta?since=timestamp`
  - Returns memories modified since timestamp
- Sync protocol uses existing memory format + metadata
- No new database schema needed (use `last_modified` column)

---

## 17. Custom Embedding Models (Enhancement to §2, Model tab)

**Problem**: Fixed embedding model limits adaptability. Some users have domain-specific needs.

**Solution**: Allow switching between pre-installed embedding models.

### Features:
- **Model selector** in settings: dropdown of available models
- **Model info**: Show vector dimension, download size, accuracy notes
- **Re-embed on switch**: Option to rebuild embeddings for existing memories

### Technical:
- Embedding models stored in `models/embeddings/`
- Config: `[embedding] model = all-MiniLM-L6-v2` (current default)
- Endpoint: `GET /api/models/embeddings` lists available models
- Re-embed endpoint: `POST /api/embed/rebuild`

---

## 18. Search Query Builder (Enhancement to §4)

**Problem**: Advanced users want complex queries (boolean, filters) but current syntax is limited.

**Solution**: Visual query builder + raw editor toggle.

### Features:
- **Builder mode**: Drag/drop or dropdown form for conditions
  - Collection: [dropdown]
  - Tags: [multi-select]
  - Date range: [date pickers]
  - Content contains: [text input]
  - Minimum score: [slider]
- **Raw mode**: Show generated query syntax (for learning)
- **Query history**: Recent builder configurations

### Technical:
- Parse user selections → build SQL/JSON query
- Example output: `{"collection": "notes", "tags": ["wip", "draft"], "content": "postgres"}`
- Endpoint: `POST /api/search/build` returns query object

---

## 19. Memory Versioning (Enhancement to §5)

**Problem**: Memories are immutable once stored, but users want to track edits.

**Solution**: Simple version history per memory.

### Features:
- **Edit button**: Opens editor with current content
- **Version tabs**: Show previous versions (timestamp, editor)
- **Diff view**: Highlight changes between versions
- **Restore**: Revert to any previous version

### Technical:
- New table: `memory_versions (id, memory_id, timestamp, content, tags_json)`
- On edit: insert new row with old values before updating
- Endpoint: `GET /api/memory/{id}/versions`

---

## 20. Plugin System (New - Extensibility)

**Problem**: Ragger can't be extended for custom workflows or integrations.

**Solution**: Minimal plugin API for hooks and commands.

### Features:
- **Plugin directory**: `plugins/` with `.py` or `.wasm` files
- **Hook points**: 
  - Before/after memory storage
  - Before/after search
  - On config reload
  - On user login
- **Command API**: Add custom HTTP endpoints from plugins

### Technical:
- Python plugins: `import plugin_name; hook_name(args)`
- WASM plugins: Load via Wizer/wasmer, call exported functions
- Plugin manifest: `plugin.json` with name, version, description

---

## Priority Reassessment (Updated)

### Phase 1 (Still critical, additions)
1. Setup wizard (§1)
2. Settings dashboard (§2)
3. Config validation (§3)
4. Server status (§7)
5. **Memory export** (§12) — Backup is essential for trust
6. **Search history** (§11) — Users need to see what they've done

### Phase 2 (Enhances workflow, now includes new features)
7. Search testing UI (§4)
8. Collections & import UI (§5)
9. Onboarding checklist (§6)
10. Settings profiles (§8)
11. **Smart tags** (§13) — Improves findability significantly
12. **Memory visualization** (§10) — Helps users understand their data

### Phase 3 (Advanced/Optional)
13. Admin panel (§9)
14. Voice input (§15) — Niche but delightful
15. Offline sync (§16) — Complex, depends on multi-user demand
16. Custom embeddings (§17) — For advanced users
17. Query builder (§18) — Power user feature
18. Versioning (§19) — Useful for documents
19. Plugin system (§20) — Enables ecosystem growth

---

## Clarifying Questions for You

1. **Memory visualization**: Should the graph show *all* memories or just a sample? Full graphs scale poorly.

2. **Search history retention**: How long should we keep search logs? 30 days? Forever? Configurable?

3. **Export formats**: Do you prefer direct download or async (email link when ready for large exports)?

4. **Smart tags**: Should users opt-in to AI tagging, or enable by default? Any privacy concerns?

5. **Plugin system**: Python-only first, or include WASM from day one?

6. **Versioning**: Should edits create new memories entirely, or keep same ID with version number?

7. **Offline sync**: Do you expect this to work over Bluetooth/Wi-Fi direct, or only via cloud backup (which defeats "offline")?

---

## Architecture Notes for New Features

### Frontend
- Reuse existing dashboard layout; add tabs for new sections
- Graph visualization: D3.js or Vis.js (both work offline)
- Audio recording: Web Audio API + MediaRecorder
- Export downloads: Blob URLs (no server round-trip needed)

### Backend
- No schema changes for most features (add tables where needed)
- Leverage existing embedding infrastructure
- Use same hot-reload pattern for config changes

### Performance
- Large exports: stream to temp file, return download link
- Graph queries: limit results, use caching
- Search history: purge old entries (configurable retention)

---

## Success Metrics (Expanded)

| Feature | Metric |
|---------|--------|
| Setup wizard | 90% complete >8 min, <5 dropouts |
| Settings dashboard | 70% adjust search without restart |
| Memory export | 25% of users export at least once |
| Search history | Users can find previous search in <30 sec |
| Smart tags | 40% adoption rate for suggested tags |
| Graph visualization | Avg session +2 min, 60% explore graph |
