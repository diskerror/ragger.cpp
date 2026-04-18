# GUI & UX Improvements for Non-Technical Users

## Context

Ragger requires compilation and command-line skill, but should be operable by careful users without
deep technical knowledge. Current gaps in **onboarding** and **runtime configuration** are barriers
to adoption—if users can't easily understand what to configure and see the impact of changes, they
won't use it.

**Scope**: Single-user focus, with multi-user admin features as a separate concern.

---

## 1. Guided Setup Wizard (Critical)

**Problem**: First-time users face a blank INI file with 40+ settings. No guidance on what's
important, what's optional, or why each setting matters.

**Solution**: Modal wizard on first access to `/` that walks through essentials.

### Flow:

1. **Welcome** — Brief explanation of Ragger (memory + search, local-first)
2. **Database location** — Confirm default `~/.ragger/memories.db` or choose custom
3. **Search tuning** — Simple sliders for:
	- Hybrid search blend (BM25 vs vector, with visual explanation)
	- Minimum relevance score (`default_min_score`)
	- Default result count (`default_limit`)
4. **Inference (optional)** — If user wants chat:
	- Model selection (dropdown of pre-configured: Claude, Ollama local, etc.)
	- API endpoint & key (with validation hint)
5. **Collections (optional)** — Create initial collections (e.g., "notes", "docs", "decisions")
6. **Complete** — Summary of choices, "Start using Ragger" button

### Technical:

- New HTTP endpoint: `GET /api/setup/status` — returns
  `{wizard_complete: bool, incomplete_steps: [...]}`
- POST `/api/setup/step/<name>` to save individual steps (allows resumption)
- Store in config under `[ui]` section: `wizard_completed = true`
- Hot-reload: wizard endpoint checks config and only shows if `wizard_completed = false`

---

## 2. Settings Dashboard (Critical)

**Problem**: Config changes require manual INI editing + server restart. Users can't see current
config via UI.

**Solution**: Browser-based settings panel grouped by concern.

### Tabs:

#### **Search** (most important for casual users)

- Visual sliders:
	- BM25 vs Vector blend (30/70 default, with explanation)
	- Minimum relevance score (0.0 → 1.0)
	- Default result count (1 → 50)
- **Test panel** (inline): "Type a query, see results as you adjust sliders"
	- Real-time search preview with current settings
	- Show score breakdown (BM25 + vector components)
	- Before/after comparison toggle

#### **Storage** (reference info)

- Database path (read-only if `SERVER_LOCKED`)
- Database size & memory count (info only)
- Default collection (dropdown or text)
- Import settings: minimum chunk size slider

#### **Model & Inference** (advanced)

- Embedding model (read-only: `all-MiniLM-L6-v2`)
- Inference model dropdown (with ping/availability check)
- API endpoint & key (masked input, validation icon)
- Max tokens slider

#### **Chat** (for chat users)

- Turn storage toggle (on/off/session only)
- Summarization settings:
	- Auto-summarize on pause (toggle)
	- Pause timeout in minutes (slider)
	- Auto-summarize on quit (toggle)
- Cleanup: max age in hours, max turns retained
- Context window: persona char limit, memory result limit
- Persona file size limit with current usage bar

#### **Logging** (optional)

- Query logging (toggle)
- HTTP logging (toggle)
- MCP logging (toggle)
- Log file locations (read-only, clickable to view in file browser if single-user)

### UX Details:

- **Validation badges**: ✓ green (valid), ⚠ yellow (warning), ✗ red (error)
	- "API key looks valid" or "No key set"
	- "Server limits allow this" or "Exceeds system ceiling"
- **Live feedback**: Apply immediately, show toast "Settings saved & reloaded"
- **Revert button**: Undo last change, or revert to saved config
- **Reset to defaults** button (modal confirmation)
- **Download config file** link (backup)
- **Import config file** upload (for migration)

### Technical:

- New HTTP endpoints:
	- `GET /api/config` — Return current config as JSON (omit sensitive keys)
	- `POST /api/config/<section>/<key>` — Update single setting + reload
	- `POST /api/config/validate` — Validate entire config without saving
	- `GET /api/config/schema` — Return config schema with descriptions, types, defaults
- Hot-reload: Return `{status: "applied", changed: 3, restart_required: false}` or
  `{..., restart_required: true, reason: "..."}`
- Persist UI state (open tab, scroll position) in localStorage

---

## 3. Config Validation & Error Feedback

**Problem**: Users don't know if their INI edits are valid until they restart the server and try to
use it.

**Solution**: Real-time validation with clear error messages.

### Implementation:

- New endpoint: `POST /api/config/lint` accepts INI text, returns:
  ```json
  {
    "valid": true/false,
    "errors": [
      {"line": 42, "section": "search", "key": "bm25_weight", "message": "Expected float, got 'three'"},
      ...
    ],
    "warnings": [
      {"key": "default_limit", "message": "Exceeds system ceiling of 50"}
    ]
  }
  ```
- **In dashboard**: Show errors/warnings prominently with "Fix" suggestions
- **In text editor**: Line-by-line squiggles (like code editor) if user chooses to edit INI directly

---

## 4. Search Testing & Tuning UI

**Problem**: Users tuning search parameters (BM25 weights, blend ratios) have no way to A/B test
settings before applying.

**Solution**: Dedicated search tuning page.

### Features:

- **Sample queries**: Pre-populated (or user-provided) test queries
- **Side-by-side comparison**:
	- Column A: Current config
	- Column B: Proposed changes (sliders, toggles)
	- Show results in real-time as user adjusts
- **Metrics**:
	- Result count
	- Score distribution (histogram)
	- Top 3 results with individual scores (BM25 + vector breakdown)
	- "Relevant?" toggle per result (unused, but structure for future ML)
- **Batch testing**:
	- Run 5+ test queries
	- Show aggregate stats (avg score, hit rate, etc.)
	- "Approve & save" or "Cancel"

### Technical:

- New endpoint: `POST /api/search/test` accepts:
  ```json
  {
    "query": "deployment requirements",
    "settings": {
      "bm25_weight": 2.5,
      "vector_weight": 7.5,
      "default_min_score": 0.35,
      "default_limit": 10
    }
  }
  ```
  Returns results with score breakdown (don't persist settings).

---

## 5. Collections & Import UI

**Problem**: Collections are powerful but opaque. Users can't easily see what collections exist,
manage them, or understand import chunking.

**Solution**: Collections panel in dashboard.

### Features:

- **Collections list**:
	- Table: Name, Memory count, Last updated, Actions (view, delete, download)
	- Quick stats (total size, avg doc age)
- **Create collection** modal:
	- Name + optional description
	- Pre-fill with common names (offers: notes, docs, decisions, reference, sessions)
- **Import dialog**:
	- Drag-drop file upload (or browse)
	- Preview chunking: show how document will be split
	- Slider for `minimum_chunk_size`
	- Show estimated memory count before import
	- Select target collection
	- "Import" button with progress bar
- **View collection** page:
	- All memories in collection (searchable table)
	- Inline editing of tags/metadata
	- Batch operations (delete, retag)

### Technical:

- New endpoints:
	- `GET /api/collections` — List with stats
	- `POST /api/collections` — Create
	- `DELETE /api/collections/<name>` — Delete (confirm safety)
	- `POST /api/import/preview` — Chunking preview without storing
	- `POST /api/import` — Store chunked document

---

## 6. Onboarding: First-Use Checklist

**Problem**: New users don't know what to do after installation.

**Solution**: Dashboard with "Getting started" checklist.

### Checklist:

- ✓ Wizard completed
- ☐ Created first collection
- ☐ Imported a document / Stored first memory
- ☐ Tested search
- ☐ Configured inference (for chat users)
- ☐ Read "What is Ragger?" (link to docs)

### UX:

- Show on dashboard sidebar or modal until all checked
- Each item links to relevant UI (e.g., "Tested search" → search tuning page)
- Dismissible but toggleable via settings

---

## 7. Server Status & Diagnostics Panel

**Problem**: Users don't know if the server is healthy, why a request failed, or how to debug.

**Solution**: Status page showing system health.

### Features:

- **Server health**:
	- Uptime
	- Memory usage (RSS)
	- Database file size
	- Embedding model loaded (yes/no)
	- Inference endpoint reachable (yes/no/slow)
- **Logs** (last 50 lines, filter by level):
	- Query log (searches)
	- HTTP log (requests)
	- Error log (issues)
	- Clickable timestamps to expand full entry
- **Inference test**:
	- Quick button: "Test inference model"
	- Send dummy prompt, measure latency
	- Show error if unreachable
- **Database integrity**:
	- "Run check" button
	- Report fragmentation, index status
	- Suggest `VACUUM` if needed

### Technical:

- New endpoint: `GET /api/status` returns:
  ```json
  {
    "uptime_seconds": 3600,
    "memory_mb": 128,
    "db_size_mb": 50,
    "memory_count": 1024,
    "embedding_model_loaded": true,
    "inference_reachable": true,
    "inference_latency_ms": 450,
    "logs": [
      {"timestamp": "...", "level": "info", "source": "http", "message": "..."}
    ]
  }
  ```

---

## 8. Settings Profiles (Nice-to-Have)

**Problem**: Users don't know good starting points for different use cases.

**Solution**: Pre-built config templates.

### Profiles:

- **"Local memory"** (default): Local-only, no inference
	- Embedding on, inference off
	- Search tuned for recall
- **"Local with Ollama"**: Local inference
	- Inference endpoint: `http://localhost:11434/v1`
	- Model dropdown with Ollama models
- **"Claude agent"**: Optimized for agentic use
	- Search tuned for precision
	- Chat turned on
	- Inference: Claude API
- **"Team deployment"** (multi-user): Pre-configured for multi-user
	- Suggestions on permissions, ceilings, etc.

### UX:

- Dropdown selector: "Use a profile" → preview → apply
- Can combine profile + manual tweaks
- "Save as profile" button to create custom profile

---

## 9. Auth & Multi-User Admin Panel (Separate Concern)

**Problem** (multi-user only): Admins can't manage users, tokens, or permissions without direct
database access.

**Solution**: Admin panel (separate from user settings).

### Features:

- **User management**:
	- List users, creation date, last login
	- Create new user (generate token, show once)
	- Delete user (confirm safety)
	- Reset token
	- View user config (read-only)
- **System ceilings**:
	- Sliders for max search limit, max persona chars, max memory results
	- Preview impact (e.g., "Users can request max 50 results")
- **Audit log**:
	- User logins, config changes
	- Last 100 entries, filter by user/type

### Access:

- Protected endpoint: `GET /api/admin/*` requires `admin` token (separate from user tokens)
- Configurable in system config: `[server] admin_token = ...`
- Dashboard checks token, shows "Admin panel unavailable" if not provided

---

## Implementation Priority

### Phase 1 (MVP: Solves onboarding + basic config)

1. Setup wizard (§1)
2. Settings dashboard with search tab (§2, search section)
3. Config validation (§3)
4. Server status (§7, basic version)

**Time estimate**: 2-3 weeks (1 person)  
**Unblocks**: Users can get from "installed" to "using" in 15 minutes

### Phase 2 (Enhances workflow)

5. Search testing UI (§4)
6. Collections & import UI (§5)
7. Onboarding checklist (§6)
8. Settings profiles (§8)

**Time estimate**: 2-3 weeks  
**Unblocks**: Users can tune search, manage data, understand impact

### Phase 3 (Optional)

9. Admin panel (§9)
10. Advanced logging & diagnostics
11. Config import/export, migration tools

**Time estimate**: 1-2 weeks (can use existing patterns)

---

## Questions for You

Before implementing, I'd clarify:

1. **Search testing workflow**: Is the goal for users to:
	- Iterate on search tuning iteratively (recommended)?
	- Or just understand how current settings work?
	- Or both?

2. **Inference setup**: Should the UI help users get Ollama/LM Studio running, or assume they've
   done it externally?

3. **Collections default**: Should the UI auto-create common collections on first run (notes, docs,
   decisions)?

4. **Settings scope**: Should users be able to edit embedding model or only inference settings?

5. **Hot-reload expectations**: If a setting requires restart, should we show that clearly and offer
   to restart, or just disable it in UI?

---

## Architecture Notes

### Frontend (existing `web/` dir can be enhanced)

- Settings dashboard: New tab + sidebar navigation
- Reuse existing login/chat components
- Vue.js or vanilla JS for modal/forms (keep light)

### Backend (new HTTP endpoints)

- Config schema endpoint (metadata for UI generation)
- Config CRUD endpoints (with validation)
- Search test endpoint
- Status/diagnostics endpoints
- Admin endpoints (gated)

### Hot-reload

- Signal handler already in place (`SIGHUP` for config reload)
- Extend `reload_config()` to handle specific settings
- Return detailed info about what changed + restart needs

### Database

- No schema changes needed
- Leverage existing `users` table for admin panel (future)

---

## Success Metrics

- **Setup wizard**: User can go from binary → first search in <10 minutes
- **Settings dashboard**: User can adjust search & understand impact without restarting
- **Collections UI**: User can import a document via UI without CLI
- **Admin panel**: Multi-user admin can provision users without database tools

---

## Open Design Questions

1. Should the wizard be **modal** (can't use app until done) or **sidebar** (can skip)?
2. Should config UI **autosave** changes or require explicit "Save" button?
3. Should search test results **show all memories** or **limit to top N** for performance?
4. For collections, should UI support **nested collections** or keep flat?
5. Should admin panel be **separate domain** (e.g., `admin.localhost:8432`) or **same UI with role
   gating**?

These can be decided during design/implementation. Let me know what resonates and where to dig
deeper.
