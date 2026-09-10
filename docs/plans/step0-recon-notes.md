# STEP 0 — Recon Notes (read-only, no code changes)

**Date:** 2026-09-08
**Scope:** Confirmed the plan's assumptions against the live tree at `~/CLionProjects/Ragger`.

---

## 1. Metaphone function signatures

### C++ API in c_lib (`CLionProjects/c_lib/DoubleMetaphone.h`, namespace `Diskerror`)
```cpp
// Returns {primary} or {primary, alternate}. Alternate present only when it differs from primary.
// Non-alpha input yields an empty vector. Codes upper-case; max_length caps at 4 chars (default).
std::vector<std::string> double_metaphone(std::string_view word, size_t max_length = 4);

// Phonize arbitrary text into a space-joined stream of DMP codes (both primary and alternate per word).
// Example: phonize("Don't panic") -> "TNT PNK"
std::string phonize(std::string_view text);
```

### Ragger wrapper (`include/double_metaphone.h`)
Pure forwarding header — `using Diskerror::double_metaphone;` and `using Diskerror::phonize;`. No extra API.

**Implication for Step 5:** to use ONLY the primary code, call `ragger::double_metaphone(word)` and take `[0]` (empty if no codes). The current `phonize()` emits BOTH codes — v0.16 must NOT use it.

---

## 2. db_version read/write locations

### Constant
- **`src/sqlite_backend.cpp:635`** — `static constexpr std::string_view kExpectedDbVersion = "0.15";`
  The single source of truth for the binary's expected schema version. Step 7 will bump this to `"0.16"`.

### Read
- **`src/sqlite_backend.cpp:646-650`** — `SqliteBackend::Impl::db_version()` reads from `settings` table with key `'db_version'`; returns `""` on missing.

### Write
- **`src/sqlite_backend.cpp:652-658`** — `SqliteBackend::Impl::set_db_version(const std::string& v)` does `INSERT INTO settings ... ON CONFLICT(key) DO UPDATE`.

### Startup hard-gate (CRITICAL DISCOVERY)
- **`src/sqlite_backend.cpp:585-599`** — after schema creation, the binary compares `db_version()` against `kExpectedDbVersion` and **exits with error** if they don't match. There is NO in-binary versioned migration chain; users must run external shell scripts.
  - Fresh installs get auto-stamped `'0.15'` at line 586 (via `INSERT OR IGNORE INTO settings`).
  - Mismatches print "Run scripts/migrate_to_dbX.Y.sh" and call `std::exit(1)`.

---

## 3. Migration architecture — EXTERNAL SCRIPTS, not in-binary

**Files:**
- `scripts/migrate_to_db0.12.sh` (pre-versioning → 0.12)
- `scripts/migrate_to_db0.15.sh` (0.12 → 0.15)
- `scripts/schema_db0.15.sql` (canonical schema DDL for a fresh build)

**The 0.15 migration script pattern (`migrate_to_db0.15.sh`):**
1. Parse args (`--db`, `--yes`).
2. Stop the ragger daemon if operating on the live DB path.
3. Build new schema at `${NAME}_NEW0.15.db` alongside original (never touches original yet).
4. Copy + translate all data into the new file via `ATTACH DATABASE`.
5. Verify row counts old vs new per table; abort loudly on mismatch.
6. Rebuild FTS5 indexes (`INSERT INTO <t>_fts(<t>_fts) VALUES('rebuild')`).
7. Verify FTS docsize matches base row count.
8. `PRAGMA integrity_check`.
9. Stamp `db_version = '0.15'` in the new file.
10. Two atomic renames: original → backup, new → original name.

**Implication for Step 7:** The plan's "one guarded migration block inside `init()`" does NOT match reality. v0.16 must follow the existing pattern: write a new `scripts/migrate_to_db0.16.sh` + update `schema_db0.15.sql` → `schema_db0.16.sql`, then bump `kExpectedDbVersion`.

---

## 4. Migration insertion point (exact line numbers)

For the in-binary code that runs on every open:
- **`src/sqlite_backend.cpp:355-627`** — `create_schema()` function
- **Line 621:** `create_fts_schema();` call — this is where the FTS5 virtual tables + triggers are created. v0.16 will replace this with calls to create the new `terms` / `<t>_terms` junction tables (Step 1).
- **Lines 986-1012:** text-FTS desync probe (resync via `'rebuild'`). Remove in v0.16 — no FTS5 left.

For the migration itself:
- Write a new `scripts/migrate_to_db0.16.sh` mirroring `migrate_to_db0.15.sh`.
- Update `kExpectedDbVersion = "0.16"` at line 635 of sqlite_backend.cpp.

---

## 5. All call sites of `search_text_only` and `rebuild_phon`

### `StorageBackend::search_text_only`
- **Declared:** `include/storage_backend.h:426` (pure virtual)
- **Implemented in Impl:** `src/sqlite_backend.cpp:3405` — uses FTS5 MATCH on `<t>_fts`; falls back to LIKE if missing.
- **Public wrapper:** `src/sqlite_backend.cpp:4352` — delegates to Impl.

### `StorageBackend::rebuild_phon`
- **Declared:** `include/storage_backend.h:468` (pure virtual)
- **Implemented in Impl:** `src/sqlite_backend.cpp:3832` — iterates records, calls `ragger::phonize(text)` per record, writes to the `phon` column.
- **Public wrapper:** `src/sqlite_backend.cpp:4391` — delegates to Impl.

### Call sites of `search_text_only` (in `src/memory.cpp`)
- **Line 310** — keyword fallback inside `RaggerMemory::search()`.
- **Line 319** — second call in the same block (different query path).

### Call sites of `rebuild_phon` (in `src/memory.cpp`)
- **Lines 156** — startup phon-backfill: `int phoned = backend_->rebuild_phon(/*only_missing=*/true, /*progress=*/false);`
- **Line 669** — `RaggerMemory::rebuild_phon()` public wrapper.

---

## 6. Existing FTS5 tables and triggers (to drop in v0.16)

For each of the 5 text tables (`turns`, `turn_summaries`, `summaries`, `decisions`, `documents`):

| Object | Pattern | Notes |
|--------|---------|-------|
| Text FTS virtual table | `<t>_fts` | Content-synced to base via triggers |
| Phonetic FTS virtual table | `<t>_phon_fts` | Same pattern, indexes the `phon` column |
| Text FTS shadow tables | `<t>_fts_data`, `<t>_fts_idx`, `<t>_fts_content`, etc. | Auto-dropped when the virtual table is dropped |
| Phonetic FTS shadow tables | `<t>_phon_fts_data`, etc. | Same |
| AI trigger (after insert) | `<t>_ai` / `<t>_pai` | Inserts into the FTS table |
| AD trigger (after delete) | `<t>_ad` / `<t>_pad` | Deletes from the FTS table |
| AU trigger (after update) | `<t>_au` / `<t>_pau` | Updates the FTS table |

**Drop order for v0.16 migration:** drop triggers first, then virtual tables. SQLite auto-drops shadow tables when a virtual table is dropped.

---

## 7. Current `phon` column locations

The `phon` TEXT column exists on all 5 base tables:
- `turns.phon` (added in the 0.14 migration)
- `turn_summaries.phon`
- `summaries.phon`
- `decisions.phon`
- `documents.phon`

**v0.16 will drop this column from all 5 tables.** Requires SQLite ≥3.35 (`ALTER TABLE <t> DROP COLUMN phon;`). Confirm the SQLite version at migration time; if older, use a table-rebuild fallback (CREATE new schema, copy rows, DROP old, RENAME).

---

## 8. What Step 1 must add

The `terms` table and 5 junction tables do NOT exist yet. They will be created in:
- **Fresh install:** `create_schema()` at line 355 of sqlite_backend.cpp (replaces the FTS5 creation call at line 621).
- **Existing DBs:** inside `migrate_to_db0.16.sh` before data copy, or as part of a fresh-DB schema file (`schema_db0.16.sql`).

The exact DDL is specified in the plan's STEP 1 section.

---

## 9. What Step 3b must add to c_lib

A Porter/Snowball stemmer with a C ABI entry point, e.g.:
```c
char* stem_en(const char* word);   // returns malloc'd string; caller frees
void free_string(char* s);         // or reuse dmp_free pattern
```
Then expose via `include/stemmer.h` (ragger namespace wrapper), mirroring the shape of `double_metaphone.h`.

---

## 10. Verification checklist for Step 0

- [x] Confirmed metaphone function signatures (`Diskerror::double_metaphone`, `Diskerror::phonize`)
- [x] Located exact db_version read/write locations (lines 635, 646–650, 652–658)
- [x] Discovered the hard-gate pattern at lines 585–599 (no in-binary migration chain)
- [x] Confirmed external script architecture (`migrate_to_db0.15.sh`)
- [x] Identified FTS schema creation point (line 621) and phon-FTS creation point (line 1014)
- [x] Enumerated all call sites of `search_text_only` and `rebuild_phon` in memory.cpp
- [x] Documented existing FTS5 tables/triggers/phon columns to drop
- [x] No build, no DB touched

**Output for next step:** metaphone function signature (`Diskerror::double_metaphone(word) -> vector<string>`) and migration insertion point (line 621 of sqlite_backend.cpp + new `migrate_to_db0.16.sh`).
