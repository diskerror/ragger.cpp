# Documents Normalization + Title-in-Signal Implementation Plan

> **For Hermes:** Use subagent-driven-development to implement task-by-task. No code lands until Reid approves this plan.

**Goal:** Normalize the `documents` table by extracting per-document metadata into a new `document_sources` table (giving documents a stable identity), convert `imported_at` to an integer epoch, and fold each document's title into its chunks' embedding + phon signals — all under the existing (not-yet-pushed) `db_version 0.15`.

**Architecture:** `document_sources` (one row per document) holds title/path/year/tags/imported_at; `documents` (chunks) gains `document_source_id` FK + `modified_on`, drops the four extracted columns. Title is appended to chunk text before `encode()`/`phonize()` (text-then-title), so title participates in vector + phon + bm25; `documents_fts` drops `title` (→ text+tags). Two migration vehicles: the repo `0.14.4→0.15` path, and a separate **uncommitted** local fixup for Reid's already-`0.15` live DB.

**Tech Stack:** C++23, SQLite (FTS5, triggers, views), CMake. Test fixtures: `~/.ragger/X*.tgz` (db_version 0.12 snapshots) + a read-only copy of the live 0.15 DB in `/tmp`.

---

## Locked Decisions (from design discussion 2026-09-03)

- **`document_sources` columns (order):** `document_source_id (PK), title, path, year, tags, imported_at` (int epoch). **No FTS** (15 rows; unneeded).
- **`documents` columns (order):** `document_id (PK), text, tags (DEFAULT ''), document_source_id → FK, chunk_index, modified_on (int), embedding_version, embedding, phon`.
- **`imported_at`:** TEXT → INTEGER epoch. Parse existing `'YYYY-MM-DD HH:MM:SS'` (the actual live form) and any `YYYYMMDD` stragglers as **local time**. This deliberately reverses the old "stays TEXT YYYYMMDD" decision (re-confirmed by Reid 2026-09-03).
- **`chunk_index`:** KEPT — positional handle ("where does this doc start / where am I").
- **Tags — repo migration:** `document_sources.tags` = set-**intersection** of all chunk tags sharing a `(path,title)` group; each `documents.tags` = its own set **minus** that intersection. Effective tags at query = **union** (lossless). NB: on the real corpus every group has one identical tagset, so intersection = whole set and chunk remainder = `''` — the set-math is future-proofing for the (future) chunk-tag edit API, not work done today.
- **Tags — local fixup:** unique key = **path alone**; source gets the tags, chunks get `''`.
- **Title in retrieval:** `encode_input = phon_input = text + "\n" + title` (title LAST — keeps chunk body as lead signal, avoids identical phon prefixes across all chunks of one doc). Applies to `documents` ONLY; comment the special-case at `store_document`.
- **`documents_fts`:** `fts5(text, tags, content='documents', content_rowid='document_id')` — `title` removed.
- **Already-0.15 gotcha:** version gate sees `0.15 == 0.15` and skips. The repo migration must self-detect the OLD shape via `column_exists("documents","path")` (NOT the version number) and normalize in place — idempotent.

---

## Fixtures / Safety

- Never touch `~/.ragger/memories.db`. Copy to `/tmp/ragmig/live15.db` (read from live, write to tmp).
- 0.12 fixtures already extracted at `/tmp/ragmig_test/Xmemories_*/memories.db` (14511 / 5654 doc rows).
- Every migration task: run against a tmp copy, verify row-count parity + integrity_check + FTS rebuild, then diff before/after.

---

### Task 1: Update reference schema `scripts/schema_db0.15.sql`

**Objective:** New canonical DDL: add `document_sources`, reshape `documents`, drop `title` from `documents_fts`, update view + triggers + indexes.

**Files:** Modify `scripts/schema_db0.15.sql`

**Details:**
- Add `document_sources` table (columns per locked order) + `document_sources_view` (renders `datetime(imported_at,'unixepoch','localtime')`).
- Reshape `documents`: drop `path,title,year,imported_at`; add `document_source_id INTEGER REFERENCES document_sources(document_source_id)`, `modified_on INTEGER`. Keep `chunk_index`.
- Indexes: drop `idx_documents_imported_at`; add `idx_documents_document_source_id`; add `idx_document_sources_imported_at`.
- `documents_view`: replace path/title/year/imported_at with `document_source_id`, `modified_on`; (optionally LEFT JOIN `document_sources` to still surface title/path in the view for humans — decide in review).
- `documents_fts`: `fts5(text, tags, ...)`; rewrite the 3 sync triggers (ai/ad/au) to drop `title`.

**Verify:** `sqlite3 :memory: < scripts/schema_db0.15.sql` exits 0; `PRAGMA integrity_check`.

---

### Task 2: Repo migration `0.14.4 → 0.15` (document normalization block)

**Objective:** In-process migration reshapes an old-shape `documents` into `document_sources` + slim `documents`, keyed on `(path,title)`, with tag set-math, epoch conversion, and title-appended re-embed.

**Files:** Modify `src/sqlite_backend.cpp` (the migration section ~lines 486–575), gated on `column_exists("documents","path")` (shape detection, NOT version).

**Migration steps (SQL, inside a transaction):**
1. Create `document_sources` + new indexes/triggers/view (idempotent `IF NOT EXISTS`).
2. Populate `document_sources` from `SELECT DISTINCT path, title, year, MIN(imported_at)` grouped by `(path,title)`; convert `imported_at` text→epoch (local). Tags = intersection across the group's chunks (compute in C++ — SQLite has no set-intersection agg; read distinct tagsets per group, intersect token sets).
3. Add `document_source_id`, `modified_on` columns to `documents` (`ALTER TABLE ADD COLUMN`).
4. `UPDATE documents SET document_source_id = (matching source), modified_on = <that source's epoch>`.
5. Set each chunk's `tags` = own tokens minus source-intersection tokens (C++ per-row rewrite).
6. Clear `embedding, embedding_version, phon` for ALL document rows (forces title-appended re-embed by housekeeping/backfill).
7. Drop old columns: SQLite → rebuild via `CREATE TABLE documents_new (...slim...)` + `INSERT SELECT` + drop/rename, OR `ALTER TABLE DROP COLUMN` (SQLite ≥3.35; vendored is 3.54 — DROP COLUMN available). Prefer DROP COLUMN for path/title/year/imported_at.
8. Rebuild `documents_fts` (drop title) + reindex.
9. `set_db_version("0.15")` (no-op if already 0.15; harmless).

**Verify:** against `/tmp/ragmig_test/*/memories.db` copies: `count(documents)` unchanged; `count(document_sources)` == distinct (path,title); no NULL `document_source_id`; every doc row has NULL embedding (pending backfill); `integrity_check` ok.

---

### Task 3: Standalone LOCAL fixup script (NOT committed)

**Objective:** One-shot normalize for Reid's already-0.15 live DB.

**Files:** Create `/tmp/ragmig/fixup_live15.sql` (or a small bash wrapper) — **do not add to repo** (`.gitignore` or keep outside tree).

**Details:** Same body as Task 2 but key = **path alone**, tags → source (chunks `''`), operate on a backup copy first, atomic swap. Runs independently of the binary's version gate.

**Verify:** on `/tmp/ragmig/live15.db` copy: 15 sources, 5654 chunks, parity, integrity.

---

### Task 4: `store_document` — append title, write FK + modified_on

**Objective:** New-write path matches the normalized schema and title-in-signal rule.

**Files:** Modify `src/sqlite_backend.cpp` `store_document()` (~1416) + `DocumentChunk` struct (find def) + `src/main.cpp` import (~95).

**Details:**
- Resolve/insert `document_sources` row for `(path,title,year,tags,imported_at→epoch)`; get `document_source_id`.
- `embed_input = text + "\n" + title` → `encode()` AND `phonize()`. Comment WHY (document-only special case).
- INSERT into slim `documents` (`text, tags, document_source_id, chunk_index, modified_on=import epoch, embedding_version, embedding, phon`).
- `DocumentChunk`: keep title/path/year/tags/imported_at as *input* fields (import still supplies them); they now route to `document_sources`.

**Verify:** import a small test file into a tmp DB; confirm 1 source row + N chunk rows, embeddings non-NULL, phon ends with title tokens.

---

### Task 5: Read/query sites — join to `document_sources`

**Objective:** Everything that read `documents.path/title/year/imported_at` now joins.

**Files (confirmed sites):**
- `src/sqlite_backend.cpp` load path (~1043 `SELECT ... title, tags, imported_at`), `documents_view` (~880), `doc_keyword_scores` (unaffected — already rowid/bm25).
- `src/export.cpp` — dumps documents; must emit both tables (+ FK).
- `src/main.cpp` — any document display/list.
- Dashboard (`src/server.cpp` document endpoints, if any) + `www/` if it renders title/path.

**Verify:** `ragger export` round-trips; dashboard shows title/path via join; search results still show source title.

---

### Task 6: Version-gate + docs

**Files:** `src/sqlite_backend.cpp` (`kExpectedDbVersion` stays `"0.15"`), CHANGELOG, `docs/` mentioning documents schema.

**Verify:** fresh install → 0.15 normalized; 0.12 fixture → migrates; live-0.15 → Task 3 fixup. Full `ctest`.

---

## Open items for review
1. `documents_view` — keep title/path visible via LEFT JOIN, or slim it? (human convenience vs purity)
2. Drop-column vs table-rebuild for removing the 4 old columns (SQLite 3.54 supports DROP COLUMN; but FTS `content=` table rebuilds are finicky — may force full rebuild anyway).
3. Backfill cost: clearing ~5654 doc embeddings triggers a re-embed pass. Acceptable one-time, but confirm housekeeping handles it (it does: `embedding IS NULL` filter).
