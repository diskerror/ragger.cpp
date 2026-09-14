-- schema_db0.16.sql
--
-- The Ragger memory schema at db_version '0.16'. Building on 0.15, this
-- release REPLACES the FTS5-based text search with a hand-rolled inverted
-- index ("the custom terms index"). See:
--   ~/ownCloud/Ragger/custom-search-schema.md   (design rationale)
--   ~/ownCloud/Ragger/schema_0.16.dot           (ER sketch)
--   docs/plans/migration-plan-fts-index.md      (step-by-step build plan)
--
-- 0.16 changes (vs 0.15):
--   1. BOTH FTS5 layers are GONE. Every `<t>_fts` (text) and `<t>_phon_fts`
--      (phonetic) external-content virtual table -- and its three sync
--      triggers -- is removed for all five text tables (turns, turn_summaries,
--      summaries, decisions, documents). No more FTS5 shadow tables cluttering
--      the schema.
--   2. The `phon` TEXT column is DROPPED from all five text tables. Phonetic
--      matching now lives INSIDE the custom index: Double Metaphone codes are
--      stored as ordinary tokens in `terms` alongside the stemmed literals
--      (they self-filter; no type/flag column). Only the PRIMARY DMP code is
--      emitted (the old phon layer emitted both primary and secondary).
--   3. A single flat `terms` list (unigrams AND bigrams, stemmed literals AND
--      metaphone tokens, no distinction) plus one `<t>_terms` junction table
--      per text table (many-to-many, ON DELETE CASCADE both ways). Deleting a
--      record cascades away its junction rows; the shared `terms` entries stay
--      (reusable, cost-free). NO doc_frequency column: df is computed live at
--      query time as COUNT(*) over the per-table junction (Decision B).
--   4. Per-record `unigram_count` / `bigram_count` INTEGER columns on all five
--      text tables, placed immediately before `embedding_version`. They hold
--      the record's stemmed-literal unigram/bigram token totals, used as the
--      TF length-normalization denominators. Because every stemmed token
--      yields exactly one primary DMP token, the metaphone count equals the40
--      literal count per type -- so these two columns serve both (Decision A).
--
-- 0.15 changes (for reference, carried forward unchanged):
--   1. embedding_version INTEGER on every embedded table (NULL iff embedding
--      IS NULL). The embedding BLOB is payload-only (no leading version byte).
--   2. documents normalized into document_sources (per-document metadata) +
--      slim documents chunks referencing document_source_id.
--
-- Column order per Reid's preferred pattern:
--   1. Primary key (id)
--   2. Unique key fields (if any)
--   3. Supporting data fields
--   4. created/modified timestamps
--   5. unigram_count, bigram_count (TF denominators; NEW at 0.16)
--   6. embedding_version, embedding (embedding_version immediately precedes
--      embedding, since it describes that column). The `phon` column that
--      used to trail here is GONE.
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

CREATE VIEW users_view AS
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

CREATE VIEW models_view AS
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
    name       TEXT,
    name_source TEXT,
    created_at INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE VIEW sessions_view AS
SELECT session_id, guid, name, name_source,
       datetime(created_at, 'unixepoch', 'localtime') AS created_at
FROM sessions;
CREATE INDEX IF NOT EXISTS idx_sessions_name ON sessions(name);

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
    unigram_count  INTEGER NOT NULL DEFAULT 0, -- stemmed-literal unigram tokens (TF denominator)
    bigram_count   INTEGER NOT NULL DEFAULT 0, -- stemmed-literal bigram tokens  (TF denominator)
    embedding_version INTEGER, -- NULL iff embedding IS NULL; tags which embedding_version produced `embedding`
    embedding      BLOB  -- embed(user_text + assistant_text); payload only, no version byte (see embedding_version)
);
CREATE INDEX idx_turns_created_at ON turns(created_at);
CREATE INDEX idx_turns_session    ON turns(session_id);
CREATE INDEX idx_turns_embedding_version ON turns(embedding_version);

CREATE VIEW turns_view AS
SELECT turn_id, user_text, assistant_text, model_id, session_id,
       datetime(created_at, 'unixepoch', 'localtime') AS created_at,
       unigram_count, bigram_count,
       embedding_version,
       CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
FROM turns;

-- ---------------------------------------------------------------------------
-- turn_summaries (L2)  -- NEW at db_version 0.12. One row per summarized
-- turn. `turn_id` is a REAL FK, ON DELETE SET NULL: pruning the raw turn
-- leaves the summary behind with turn_id NULL instead of stranding it via a
-- fragile (session_id, created_at) match. `turn_datetime`/`turn_model_id`
-- are copied in explicitly at insert time (not derived via join) so both
-- survive `turns` row deletion. `summary_model_id` (who/what wrote the
-- summary text) is intentionally separate from `turn_model_id` (who
-- answered originally) -- conflating them would lose "who actually
-- answered" the moment `turns` is truncated.
--
-- Sentinel for "not yet summarized": summary_model_id IS NULL -- NOT
-- text IS NULL. `text` is populated at capture time with the raw
-- "User: ...\n\nAssistant: ..." placeholder (searchable immediately) and
-- overwritten in place once the summarizer produces a real summary;
-- `summary_model_id` only gets set on that second write.
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
    unigram_count    INTEGER NOT NULL DEFAULT 0,
    bigram_count     INTEGER NOT NULL DEFAULT 0,
    embedding_version INTEGER, -- NULL iff embedding IS NULL
    embedding        BLOB      -- payload only, no version byte (see embedding_version)
);
CREATE INDEX idx_turn_summaries_turn_id    ON turn_summaries(turn_id);
CREATE INDEX idx_turn_summaries_session_id ON turn_summaries(session_id);
CREATE INDEX idx_turn_summaries_datetime   ON turn_summaries(turn_datetime);
CREATE INDEX idx_turn_summaries_embedding_version ON turn_summaries(embedding_version);
-- Upsert target for finalize_turn_summary: one row per turn_id. NULL
-- turn_id (a pruned turn) is excepted -- multiple such rows are fine.
CREATE UNIQUE INDEX idx_turn_summaries_turn_id_uniq
    ON turn_summaries(turn_id) WHERE turn_id IS NOT NULL;

CREATE VIEW turn_summaries_view AS
SELECT
    turn_summary_id, text, turn_id, session_id, turn_model_id, summary_model_id,
    datetime(turn_datetime, 'unixepoch', 'localtime') AS turn_datetime,
    datetime(summarized_on, 'unixepoch', 'localtime') AS summarized_on,
    unigram_count, bigram_count,
    embedding_version,
    CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
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
    tags       TEXT DEFAULT '' NOT NULL,
    session_id INTEGER REFERENCES sessions(session_id),
    model_id   INTEGER REFERENCES models(model_id),
    created_at INTEGER         NOT NULL DEFAULT (unixepoch()),
    updated_at INTEGER         NOT NULL DEFAULT (unixepoch()),
    unigram_count INTEGER      NOT NULL DEFAULT 0,
    bigram_count  INTEGER      NOT NULL DEFAULT 0,
    embedding_version INTEGER, -- NULL iff embedding IS NULL
    embedding  BLOB            -- payload only, no version byte (see embedding_version)
);
CREATE INDEX idx_summaries_level ON summaries(level);
CREATE INDEX idx_summaries_created_at ON summaries(created_at);
CREATE INDEX idx_summaries_session ON summaries(session_id);
CREATE INDEX idx_summaries_embedding_version ON summaries(embedding_version);

CREATE VIEW summaries_view AS
SELECT summary_id, text, level, tags, session_id, model_id,
       datetime(created_at, 'unixepoch', 'localtime') AS created_at,
       datetime(updated_at, 'unixepoch', 'localtime') AS updated_at,
       unigram_count, bigram_count,
       embedding_version,
       CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
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
    unigram_count INTEGER NOT NULL DEFAULT 0,
    bigram_count  INTEGER NOT NULL DEFAULT 0,
    embedding_version INTEGER, -- NULL iff embedding IS NULL
    embedding   BLOB           -- payload only, no version byte (see embedding_version)
);
CREATE INDEX idx_decisions_status ON decisions(status);
CREATE INDEX idx_decisions_embedding_version ON decisions(embedding_version);

CREATE VIEW decisions_view AS
SELECT decision_id, text, status, tags,
       datetime(created_at, 'unixepoch', 'localtime') AS created_at,
       unigram_count, bigram_count,
       embedding_version,
       CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
FROM decisions;

-- ---------------------------------------------------------------------------
-- document_sources  -- one row per curated document (db_version 0.15). Holds
-- the per-document metadata that used to be repeated on every chunk of
-- `documents`. Gives a document a stable identity (document_source_id) so it
-- can be addressed / re-curated / replaced as a unit. NO FTS coverage: this
-- table is tiny (one row per document) and its title now rides the chunk
-- embed/terms signals (see store_document), so keyword search needs nothing
-- here. imported_at is an INTEGER unix-epoch (user's local import time).
-- ---------------------------------------------------------------------------
CREATE TABLE document_sources (
    document_source_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT,                     -- identifies the publication
    path        TEXT,                     -- origin file
    year        INTEGER,                  -- publish year
    tags        TEXT NOT NULL DEFAULT '', -- general tags for the whole document
    imported_at INTEGER NOT NULL          -- unix epoch, user's local import time
);
CREATE INDEX idx_document_sources_imported_at ON document_sources(imported_at);

CREATE VIEW document_sources_view AS
SELECT document_source_id, title, path, year, tags,
       datetime(imported_at, 'unixepoch', 'localtime') AS imported_at
FROM document_sources;

-- ---------------------------------------------------------------------------
-- documents (L5)  -- user-curated RAG chunks; the only sharable table.
-- Per-document metadata (title/path/year/imported_at) lives in
-- document_sources, referenced by document_source_id. Each chunk keeps only
-- its distinctive `tags`. chunk_index is a positional handle. modified_on:
-- unix epoch, = the source's imported_at at import; diverges only when the
-- chunk text is later edited.
-- NOTE: title is appended to `text` before encode()/tokenize() at write time
-- (see store_document), so it participates in vector + terms search.
-- ---------------------------------------------------------------------------
CREATE TABLE documents (
    document_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    text               TEXT NOT NULL,            -- body; chapter/section headings inline
    tags               TEXT NOT NULL DEFAULT '', -- chunk-specific tags only (see above)
    chunk_index        INTEGER,
    document_source_id INTEGER REFERENCES document_sources(document_source_id),
    modified_on        INTEGER,                  -- unix epoch; = source.imported_at until edited
    unigram_count      INTEGER NOT NULL DEFAULT 0,
    bigram_count       INTEGER NOT NULL DEFAULT 0,
    embedding_version  INTEGER, -- NULL iff embedding IS NULL
    embedding          BLOB                      -- payload only, no version byte (see embedding_version)
);
CREATE INDEX idx_documents_document_source_id ON documents(document_source_id);
CREATE INDEX idx_documents_embedding_version ON documents(embedding_version);

CREATE VIEW documents_view AS
SELECT document_id, text, tags, chunk_index, document_source_id,
       datetime(modified_on, 'unixepoch', 'localtime') AS modified_on,
       unigram_count, bigram_count,
       embedding_version,
       CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
FROM documents;

-- ---------------------------------------------------------------------------
-- settings  -- key/value store (embedding_model, dimensions, recipe,
-- vector_type, db_version, fts_w_unigram, fts_w_bigram, fts_w_metaphone, ...).
-- No timestamp/embedding columns; no view.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Stamp db_version immediately so a FRESH install (no migration involved --
-- these CREATE TABLE statements just ran for the first time) passes the
-- binary's startup version-gate check on its very first run, same as a
-- migrated old DB does after the in-binary migration runs.
INSERT OR IGNORE INTO settings (key, value) VALUES ('db_version', '0.16');

-- ---------------------------------------------------------------------------
-- Custom terms index (v0.16)  -- a single hand-rolled inverted index that
-- REPLACES both FTS5 layers. One shared `terms` list holds every distinct
-- token: stemmed literal unigrams AND bigrams, plus their primary Double
-- Metaphone codes, all mixed flat (tokens self-filter; no type/flag column).
-- Each text table has its own `<t>_terms` junction (many-to-many) carrying a
-- per-record `count`. df is computed live at query time as COUNT(*) over the
-- junction (no doc_frequency column). See custom-search-schema.md.
-- ---------------------------------------------------------------------------
CREATE TABLE terms (
    term_id INTEGER PRIMARY KEY AUTOINCREMENT,
    term    TEXT NOT NULL UNIQUE
);

-- turns
CREATE TABLE turns_terms (
    turn_id INTEGER NOT NULL REFERENCES turns(turn_id) ON DELETE CASCADE,
    term_id INTEGER NOT NULL REFERENCES terms(term_id) ON DELETE CASCADE,
    count   INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (turn_id, term_id)
);
CREATE INDEX idx_turns_terms_term ON turns_terms(term_id);

-- turn_summaries
CREATE TABLE turn_summaries_terms (
    turn_summary_id INTEGER NOT NULL REFERENCES turn_summaries(turn_summary_id) ON DELETE CASCADE,
    term_id INTEGER NOT NULL REFERENCES terms(term_id) ON DELETE CASCADE,
    count   INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (turn_summary_id, term_id)
);
CREATE INDEX idx_turn_summaries_terms_term ON turn_summaries_terms(term_id);

-- summaries
CREATE TABLE summaries_terms (
    summary_id INTEGER NOT NULL REFERENCES summaries(summary_id) ON DELETE CASCADE,
    term_id INTEGER NOT NULL REFERENCES terms(term_id) ON DELETE CASCADE,
    count   INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (summary_id, term_id)
);
CREATE INDEX idx_summaries_terms_term ON summaries_terms(term_id);

-- documents
CREATE TABLE documents_terms (
    document_id INTEGER NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    term_id INTEGER NOT NULL REFERENCES terms(term_id) ON DELETE CASCADE,
    count   INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (document_id, term_id)
);
CREATE INDEX idx_documents_terms_term ON documents_terms(term_id);

-- decisions
CREATE TABLE decisions_terms (
    decision_id INTEGER NOT NULL REFERENCES decisions(decision_id) ON DELETE CASCADE,
    term_id INTEGER NOT NULL REFERENCES terms(term_id) ON DELETE CASCADE,
    count   INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (decision_id, term_id)
);
CREATE INDEX idx_decisions_terms_term ON decisions_terms(term_id);

-- ---------------------------------------------------------------------------
-- Junction views — resolve term_id to the readable term string.
--
-- The raw <t>_terms tables store term_id, which is meaningless on inspection.
-- These views join through `terms` so you can actually read why a record did
-- or did not match. Rows are ordered most-frequent-first within each record.
--
-- Both literal (stemmed) and DoubleMetaphone tokens live in the same flat
-- `terms` table with no type flag, so a single record's view output mixes
-- them: e.g. 'ragg' and 'memori' (stemmed literals) alongside 'RK' and 'MMR'
-- (primary metaphone codes). That is by design — they self-filter at query
-- time because a stem and its DMP code are different strings.
--
-- INNER JOIN is deliberate: <t>_terms.term_id is NOT NULL with an FK into
-- terms, so an unmatched row would be corruption, and showing nothing for it
-- is the correct signal.
-- ---------------------------------------------------------------------------

CREATE VIEW turns_terms_view AS
SELECT j.turn_id AS turn_id, t.term, j.count
FROM turns_terms j JOIN terms t ON t.term_id = j.term_id
ORDER BY j.turn_id, j.count DESC, t.term;

CREATE VIEW turn_summaries_terms_view AS
SELECT j.turn_summary_id AS turn_summary_id, t.term, j.count
FROM turn_summaries_terms j JOIN terms t ON t.term_id = j.term_id
ORDER BY j.turn_summary_id, j.count DESC, t.term;

CREATE VIEW summaries_terms_view AS
SELECT j.summary_id AS summary_id, t.term, j.count
FROM summaries_terms j JOIN terms t ON t.term_id = j.term_id
ORDER BY j.summary_id, j.count DESC, t.term;

CREATE VIEW documents_terms_view AS
SELECT j.document_id AS document_id, t.term, j.count
FROM documents_terms j JOIN terms t ON t.term_id = j.term_id
ORDER BY j.document_id, j.count DESC, t.term;

CREATE VIEW decisions_terms_view AS
SELECT j.decision_id AS decision_id, t.term, j.count
FROM decisions_terms j JOIN terms t ON t.term_id = j.term_id
ORDER BY j.decision_id, j.count DESC, t.term;
