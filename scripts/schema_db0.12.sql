-- schema_db0.12.sql
--
-- The Ragger memory schema at db_version '0.12'. Building on v4:
--   1. `turn_summaries` split out of `summaries` — level='turn' rows move to
--      their own table with a real `turn_id` FK (ON DELETE SET NULL), so a
--      pruned raw turn's summary survives with an explicit "no raw backing"
--      marker instead of being matched by a fragile (session_id, created_at)
--      compound key. `summaries` now only ever holds
--      'episode' | 'session' | 'project' rows.
--   2. Every timestamp column across all tables becomes INTEGER (Unix epoch
--      seconds — no milliseconds; nothing in this codebase needs sub-second
--      precision). EXCEPTION: `documents.imported_at` stays TEXT 'YYYYMMDD'
--      — it's a date-only batch-grouping key, not a real timestamp, and
--      `documents` rows exist specifically to keep metadata in its most
--      legible form.
--   3. `users.created`/`users.modified` renamed to `users.created_at`/
--      `users.updated_at` to match every other table's naming convention.
--   4. A `_view` per table exposes epoch columns as human-readable
--      `datetime(...)` strings and collapses `embedding`/`phon` BLOB/TEXT
--      columns to `has_embedding`/`has_phon` (0/1) — these views are meant
--      to be opened directly in a plain SQLite browser by a human; raw
--      embedding bytes and phonetic codes are noise in that context.
--
-- Column order per Reid's preferred pattern (unchanged from v4):
--   1. Primary key (id)
--   2. Unique key fields (if any)
--   3. Supporting data fields
--   4. created/modified timestamps
--   5. embedding and phon columns (at the end, after timestamps)
--
-- Applies to all BASE TABLES only. FTS5 virtual tables and their triggers
-- reference base table rows by rowid and are unaffected by the timestamp
-- type change.
--

-- ---------------------------------------------------------------------------
-- users  -- who else has read access to documents.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE,
    token_hash    TEXT NOT NULL,
    password_hash TEXT,
    created_at    INTEGER NOT NULL DEFAULT (unixepoch()),
    updated_at    INTEGER NOT NULL DEFAULT (unixepoch())
);
CREATE TRIGGER IF NOT EXISTS users_modified
    AFTER UPDATE ON users
    BEGIN
        UPDATE users SET updated_at = unixepoch() WHERE id = NEW.id;
    END;

CREATE VIEW IF NOT EXISTS users_view AS
SELECT id, username, token_hash, password_hash,
       datetime(created_at, 'unixepoch', 'localtime') AS created_at,
       datetime(updated_at, 'unixepoch', 'localtime') AS updated_at
FROM users;

-- ---------------------------------------------------------------------------
-- models  -- lookup; turns + summaries + turn_summaries reference it
-- ---------------------------------------------------------------------------
CREATE TABLE models (
    model_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE,
    created_at INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE VIEW IF NOT EXISTS models_view AS
SELECT model_id, name,
       datetime(created_at, 'unixepoch', 'localtime') AS created_at
FROM models;

-- ---------------------------------------------------------------------------
-- sessions  -- lookup; turns + summaries + turn_summaries reference it.
-- Normalizes the long conversation GUIDs (from the agent's turn hook) to a
-- compact integer id, mirroring `models`. The grouping key for session
-- summaries and recipes.
-- ---------------------------------------------------------------------------
CREATE TABLE sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    guid       TEXT NOT NULL UNIQUE,
    created_at INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE VIEW IF NOT EXISTS sessions_view AS
SELECT session_id, guid,
       datetime(created_at, 'unixepoch', 'localtime') AS created_at
FROM sessions;

-- ---------------------------------------------------------------------------
-- turns (L1)  -- raw verbatim exchanges
-- ---------------------------------------------------------------------------
CREATE TABLE turns (
    turn_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_text      TEXT NOT NULL,
    assistant_text TEXT,
    model_id       INTEGER REFERENCES models(model_id),
    session_id     INTEGER REFERENCES sessions(session_id),
    created_at     INTEGER NOT NULL DEFAULT (unixepoch()),
    embedding      BLOB, -- embed(user_text + assistant_text)
    phon           TEXT  -- phonize(user_text + assistant_text): Double Metaphone "sounds-like"
);
CREATE INDEX idx_turns_created_at ON turns(created_at);
CREATE INDEX idx_turns_session    ON turns(session_id);

CREATE VIEW IF NOT EXISTS turns_view AS
SELECT turn_id, user_text, assistant_text, model_id, session_id,
       datetime(created_at, 'unixepoch', 'localtime') AS created_at,
       CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding,
       CASE WHEN phon      IS NULL THEN 0 ELSE 1 END AS has_phon
FROM turns;

-- ---------------------------------------------------------------------------
-- turn_summaries (L2)  -- NEW at db_version 0.12. One row per summarized
-- turn. `turn_id` is a REAL FK, ON DELETE SET NULL: pruning the raw turn
-- leaves the summary behind with turn_id NULL instead of stranding it via a
-- fragile (session_id, created_at) match. `turn_datetime`/`turn_model_id`
-- are copied in explicitly at insert time (not derived via join) so both
-- survive `turns` row deletion. `summary_model_id` (who/what wrote the
-- summary text) is intentionally separate from `turn_model_id` (who
-- answered originally) — conflating them would lose "who actually
-- answered" the moment `turns` is truncated.
--
-- Sentinel for "not yet summarized": summary_model_id IS NULL — NOT
-- text IS NULL. `text` is populated at capture time with the raw
-- "User: ...\n\nAssistant: ..." placeholder (searchable immediately) and
-- overwritten in place once the summarizer produces a real summary;
-- `summary_model_id` only gets set on that second write. Mirrors the v4
-- `summaries.model_id IS NULL` placeholder convention exactly.
-- ---------------------------------------------------------------------------
CREATE TABLE turn_summaries (
    turn_summary_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    text             TEXT,
    turn_id          INTEGER REFERENCES turns(turn_id) ON DELETE SET NULL,
    session_id       INTEGER REFERENCES sessions(session_id),
    turn_model_id    INTEGER REFERENCES models(model_id),
    summary_model_id INTEGER REFERENCES models(model_id),
    turn_datetime    INTEGER NOT NULL,
    summarized_on    INTEGER NOT NULL DEFAULT (unixepoch()),
    embedding        BLOB,
    phon             TEXT
);
CREATE INDEX idx_turn_summaries_turn_id    ON turn_summaries(turn_id);
CREATE INDEX idx_turn_summaries_session_id ON turn_summaries(session_id);
CREATE INDEX idx_turn_summaries_datetime   ON turn_summaries(turn_datetime);
-- Upsert target for finalize_turn_summary: one row per turn_id. NULL
-- turn_id (a pruned turn) is excepted — multiple such rows are fine, they
-- no longer represent "the" summary for any live turn.
CREATE UNIQUE INDEX idx_turn_summaries_turn_id_uniq
    ON turn_summaries(turn_id) WHERE turn_id IS NOT NULL;

CREATE VIEW IF NOT EXISTS turn_summaries_view AS
SELECT
    turn_summary_id, text, turn_id, session_id, turn_model_id, summary_model_id,
    datetime(turn_datetime, 'unixepoch', 'localtime') AS turn_datetime,
    datetime(summarized_on, 'unixepoch', 'localtime') AS summarized_on,
    CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding,
    CASE WHEN phon      IS NULL THEN 0 ELSE 1 END AS has_phon
FROM turn_summaries;

-- ---------------------------------------------------------------------------
-- summaries (L3/L4)  -- episode/session/project rollups only as of 0.12.
-- Turn-level rows moved to `turn_summaries` above.
-- created_at: (re)write / span-start time. updated_at: running rows
-- (session/project) record their last regenerate time; 'episode' rows
-- carry the span-end; on any first insert updated_at == created_at.
-- model_id: the model that produced the summary.
-- ---------------------------------------------------------------------------
CREATE TABLE summaries (
    summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT            NOT NULL,
    level      TEXT            NOT NULL, -- 'episode' | 'session' | 'project'
    status     TEXT            NOT NULL, -- 'current' | 'complete'
    tags       TEXT DEFAULT '' NOT NULL,
    session_id INTEGER REFERENCES sessions(session_id),
    model_id   INTEGER REFERENCES models(model_id),
    created_at INTEGER         NOT NULL DEFAULT (unixepoch()),
    updated_at INTEGER         NOT NULL DEFAULT (unixepoch()),
    embedding  BLOB,
    phon       TEXT            -- phonize(text): Double Metaphone "sounds-like"
);
CREATE INDEX idx_summaries_level ON summaries(level);
CREATE INDEX idx_summaries_status ON summaries(status);
CREATE INDEX idx_summaries_created_at ON summaries(created_at);
CREATE INDEX idx_summaries_session ON summaries(session_id);

CREATE VIEW IF NOT EXISTS summaries_view AS
SELECT summary_id, text, level, status, tags, session_id, model_id,
       datetime(created_at, 'unixepoch', 'localtime') AS created_at,
       datetime(updated_at, 'unixepoch', 'localtime') AS updated_at,
       CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding,
       CASE WHEN phon      IS NULL THEN 0 ELSE 1 END AS has_phon
FROM summaries;

-- ---------------------------------------------------------------------------
-- decisions (L6)
-- ---------------------------------------------------------------------------
CREATE TABLE decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'current', -- current|superseded|revisit|deprecated
    tags        TEXT NOT NULL DEFAULT '',
    created_at  INTEGER NOT NULL DEFAULT (unixepoch()),
    embedding   BLOB,
    phon        TEXT -- phonize(text): Double Metaphone "sounds-like"
);
CREATE INDEX idx_decisions_status ON decisions(status);

CREATE VIEW IF NOT EXISTS decisions_view AS
SELECT decision_id, text, status, tags,
       datetime(created_at, 'unixepoch', 'localtime') AS created_at,
       CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding,
       CASE WHEN phon      IS NULL THEN 0 ELSE 1 END AS has_phon
FROM decisions;

-- ---------------------------------------------------------------------------
-- documents (L5)  -- user-curated RAG; the only sharable table.
-- imported_at STAYS TEXT 'YYYYMMDD' — a date-only batch-grouping key, not a
-- real timestamp; documents rows exist to capture maximal legible metadata
-- up front, which an opaque epoch int works against. Confirmed decision,
-- 2026-07-25 — do not convert this column in a future pass without
-- re-confirming.
-- ---------------------------------------------------------------------------
CREATE TABLE documents (
    document_id INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,            -- body; chapter/section headings inline
    path        TEXT,
    title       TEXT,                     -- identifies the publication
    tags        TEXT NOT NULL DEFAULT '', -- groups + describes
    year        INTEGER,                  -- publish year
    chunk_index INTEGER,
    imported_at TEXT NOT NULL,            -- 'YYYYMMDD', shared per import batch — TEXT by design, see above
    embedding   BLOB,
    phon        TEXT                      -- phonize(text): Double Metaphone "sounds-like"
);
CREATE INDEX idx_documents_imported_at ON documents(imported_at);

CREATE VIEW IF NOT EXISTS documents_view AS
SELECT document_id, text, path, title, tags, year, chunk_index, imported_at,
       CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding,
       CASE WHEN phon      IS NULL THEN 0 ELSE 1 END AS has_phon
FROM documents;

-- ---------------------------------------------------------------------------
-- settings  -- key/value store (embedding_model, dimensions, recipe,
-- vector_type, db_version, ...). No timestamp/embedding columns; no view.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Stamp db_version immediately so a FRESH install (no migration involved —
-- these CREATE TABLE IF NOT EXISTS statements just ran for the first time)
-- passes the binary's startup version-gate check on its very first run,
-- same as a migrated old DB does after migrate_to_db0.12.sh runs.
INSERT OR IGNORE INTO settings (key, value) VALUES ('db_version', '0.12');

-- ---------------------------------------------------------------------------
-- FTS5  -- external-content virtual tables + sync triggers (text search)
-- ---------------------------------------------------------------------------
CREATE VIRTUAL TABLE turns_fts USING fts5 (
    user_text,
    assistant_text,
    content='turns',
    content_rowid='turn_id'
);
CREATE TRIGGER turns_ai
    AFTER INSERT
    ON turns
BEGIN
    INSERT INTO turns_fts(rowid, user_text, assistant_text)
    VALUES (new.turn_id, new.user_text, new.assistant_text);
END;
CREATE TRIGGER turns_ad
    AFTER DELETE
    ON turns
BEGIN
    INSERT INTO turns_fts(turns_fts, rowid, user_text, assistant_text)
    VALUES ('delete', old.turn_id, old.user_text, old.assistant_text);
END;
CREATE TRIGGER turns_au
    AFTER UPDATE
    ON turns
BEGIN
    INSERT INTO turns_fts(turns_fts, rowid, user_text, assistant_text)
    VALUES ('delete', old.turn_id, old.user_text, old.assistant_text);
    INSERT INTO turns_fts(rowid, user_text, assistant_text)
    VALUES (new.turn_id, new.user_text, new.assistant_text);
END;

CREATE VIRTUAL TABLE summaries_fts USING fts5 (
    text,
    tags,
    content='summaries',
    content_rowid='summary_id'
);
CREATE TRIGGER summaries_ai
    AFTER INSERT
    ON summaries
BEGIN
    INSERT INTO summaries_fts(rowid, text, tags)
    VALUES (new.summary_id, new.text, new.tags);
END;
CREATE TRIGGER summaries_ad
    AFTER DELETE
    ON summaries
BEGIN
    INSERT INTO summaries_fts(summaries_fts, rowid, text, tags)
    VALUES ('delete', old.summary_id, old.text, old.tags);
END;
CREATE TRIGGER summaries_au
    AFTER UPDATE
    ON summaries
BEGIN
    INSERT INTO summaries_fts(summaries_fts, rowid, text, tags)
    VALUES ('delete', old.summary_id, old.text, old.tags);
    INSERT INTO summaries_fts(rowid, text, tags)
    VALUES (new.summary_id, new.text, new.tags);
END;

-- turn_summaries_fts: mirrors turns_fts's shape (no tags column, unlike
-- summaries_fts) since turn_summaries has no tags column.
CREATE VIRTUAL TABLE turn_summaries_fts USING fts5 (
    text,
    content='turn_summaries',
    content_rowid='turn_summary_id'
);
CREATE TRIGGER turn_summaries_ai
    AFTER INSERT
    ON turn_summaries
BEGIN
    INSERT INTO turn_summaries_fts(rowid, text)
    VALUES (new.turn_summary_id, new.text);
END;
CREATE TRIGGER turn_summaries_ad
    AFTER DELETE
    ON turn_summaries
BEGIN
    INSERT INTO turn_summaries_fts(turn_summaries_fts, rowid, text)
    VALUES ('delete', old.turn_summary_id, old.text);
END;
CREATE TRIGGER turn_summaries_au
    AFTER UPDATE
    ON turn_summaries
BEGIN
    INSERT INTO turn_summaries_fts(turn_summaries_fts, rowid, text)
    VALUES ('delete', old.turn_summary_id, old.text);
    INSERT INTO turn_summaries_fts(rowid, text)
    VALUES (new.turn_summary_id, new.text);
END;

CREATE VIRTUAL TABLE decisions_fts USING fts5 (
    text,
    tags,
    content='decisions',
    content_rowid='decision_id'
);
CREATE TRIGGER decisions_ai
    AFTER INSERT
    ON decisions
BEGIN
    INSERT INTO decisions_fts(rowid, text, tags)
    VALUES (new.decision_id, new.text, new.tags);
END;
CREATE TRIGGER decisions_ad
    AFTER DELETE
    ON decisions
BEGIN
    INSERT INTO decisions_fts(decisions_fts, rowid, text, tags)
    VALUES ('delete', old.decision_id, old.text, old.tags);
END;
CREATE TRIGGER decisions_au
    AFTER UPDATE
    ON decisions
BEGIN
    INSERT INTO decisions_fts(decisions_fts, rowid, text, tags)
    VALUES ('delete', old.decision_id, old.text, old.tags);
    INSERT INTO decisions_fts(rowid, text, tags)
    VALUES (new.decision_id, new.text, new.tags);
END;

CREATE VIRTUAL TABLE documents_fts USING fts5 (
    title,
    text,
    tags,
    content='documents',
    content_rowid='document_id'
);
CREATE TRIGGER documents_ai
    AFTER INSERT
    ON documents
BEGIN
    INSERT INTO documents_fts(rowid, title, text, tags)
    VALUES (new.document_id, new.title, new.text, new.tags);
END;
CREATE TRIGGER documents_ad
    AFTER DELETE
    ON documents
BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, text, tags)
    VALUES ('delete', old.document_id, old.title, old.text, old.tags);
END;
CREATE TRIGGER documents_au
    AFTER UPDATE
    ON documents
BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, text, tags)
    VALUES ('delete', old.document_id, old.title, old.text, old.tags);
    INSERT INTO documents_fts(rowid, title, text, tags)
    VALUES (new.document_id, new.title, new.text, new.tags);
END;

-- ---------------------------------------------------------------------------
-- Phonetic "sounds-like" (dolphining) FTS5  -- one external-content index over
-- each context table's `phon` column (space-joined Double Metaphone codes),
-- kept in sync by its own triggers. Separate from the text FTS above so phon
-- produces an INDEPENDENT bm25 score for the three-way search blend
-- (vector + text + phon; see config phon_weight).
-- ---------------------------------------------------------------------------
CREATE VIRTUAL TABLE turns_phon_fts USING fts5 (
    phon,
    content='turns',
    content_rowid='turn_id'
);
CREATE TRIGGER turns_pai
    AFTER INSERT
    ON turns
BEGIN
    INSERT INTO turns_phon_fts(rowid, phon) VALUES (new.turn_id, new.phon);
END;
CREATE TRIGGER turns_pad
    AFTER DELETE
    ON turns
BEGIN
    INSERT INTO turns_phon_fts(turns_phon_fts, rowid, phon)
    VALUES ('delete', old.turn_id, old.phon);
END;
CREATE TRIGGER turns_pau
    AFTER UPDATE
    ON turns
BEGIN
    INSERT INTO turns_phon_fts(turns_phon_fts, rowid, phon)
    VALUES ('delete', old.turn_id, old.phon);
    INSERT INTO turns_phon_fts(rowid, phon) VALUES (new.turn_id, new.phon);
END;

CREATE VIRTUAL TABLE summaries_phon_fts USING fts5 (
    phon,
    content='summaries',
    content_rowid='summary_id'
);
CREATE TRIGGER summaries_pai
    AFTER INSERT
    ON summaries
BEGIN
    INSERT INTO summaries_phon_fts(rowid, phon) VALUES (new.summary_id, new.phon);
END;
CREATE TRIGGER summaries_pad
    AFTER DELETE
    ON summaries
BEGIN
    INSERT INTO summaries_phon_fts(summaries_phon_fts, rowid, phon)
    VALUES ('delete', old.summary_id, old.phon);
END;
CREATE TRIGGER summaries_pau
    AFTER UPDATE
    ON summaries
BEGIN
    INSERT INTO summaries_phon_fts(summaries_phon_fts, rowid, phon)
    VALUES ('delete', old.summary_id, old.phon);
    INSERT INTO summaries_phon_fts(rowid, phon) VALUES (new.summary_id, new.phon);
END;

CREATE VIRTUAL TABLE turn_summaries_phon_fts USING fts5 (
    phon,
    content='turn_summaries',
    content_rowid='turn_summary_id'
);
CREATE TRIGGER turn_summaries_pai
    AFTER INSERT
    ON turn_summaries
BEGIN
    INSERT INTO turn_summaries_phon_fts(rowid, phon) VALUES (new.turn_summary_id, new.phon);
END;
CREATE TRIGGER turn_summaries_pad
    AFTER DELETE
    ON turn_summaries
BEGIN
    INSERT INTO turn_summaries_phon_fts(turn_summaries_phon_fts, rowid, phon)
    VALUES ('delete', old.turn_summary_id, old.phon);
END;
CREATE TRIGGER turn_summaries_pau
    AFTER UPDATE
    ON turn_summaries
BEGIN
    INSERT INTO turn_summaries_phon_fts(turn_summaries_phon_fts, rowid, phon)
    VALUES ('delete', old.turn_summary_id, old.phon);
    INSERT INTO turn_summaries_phon_fts(rowid, phon) VALUES (new.turn_summary_id, new.phon);
END;

CREATE VIRTUAL TABLE decisions_phon_fts USING fts5 (
    phon,
    content='decisions',
    content_rowid='decision_id'
);
CREATE TRIGGER decisions_pai
    AFTER INSERT
    ON decisions
BEGIN
    INSERT INTO decisions_phon_fts(rowid, phon) VALUES (new.decision_id, new.phon);
END;
CREATE TRIGGER decisions_pad
    AFTER DELETE
    ON decisions
BEGIN
    INSERT INTO decisions_phon_fts(decisions_phon_fts, rowid, phon)
    VALUES ('delete', old.decision_id, old.phon);
END;
CREATE TRIGGER decisions_pau
    AFTER UPDATE
    ON decisions
BEGIN
    INSERT INTO decisions_phon_fts(decisions_phon_fts, rowid, phon)
    VALUES ('delete', old.decision_id, old.phon);
    INSERT INTO decisions_phon_fts(rowid, phon) VALUES (new.decision_id, new.phon);
END;

CREATE VIRTUAL TABLE documents_phon_fts USING fts5 (
    phon,
    content='documents',
    content_rowid='document_id'
);
CREATE TRIGGER documents_pai
    AFTER INSERT
    ON documents
BEGIN
    INSERT INTO documents_phon_fts(rowid, phon) VALUES (new.document_id, new.phon);
END;
CREATE TRIGGER documents_pad
    AFTER DELETE
    ON documents
BEGIN
    INSERT INTO documents_phon_fts(documents_phon_fts, rowid, phon)
    VALUES ('delete', old.document_id, old.phon);
END;
CREATE TRIGGER documents_pau
    AFTER UPDATE
    ON documents
BEGIN
    INSERT INTO documents_phon_fts(documents_phon_fts, rowid, phon)
    VALUES ('delete', old.document_id, old.phon);
    INSERT INTO documents_phon_fts(rowid, phon) VALUES (new.document_id, new.phon);
END;
