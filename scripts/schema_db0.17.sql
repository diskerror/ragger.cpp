-- schema_db0.17.sql
--
-- The Ragger memory schema at db_version '0.17'. Building on 0.16, this
-- release adds the AGENT API data model: structured decisions with evidence,
-- projects, and insights. See:
--   ~/ownCloud/Ragger/schema_0.17_full.dot      (ER diagram)
--   ~/ownCloud/Ragger/agent-api/                (design, phase docs, reports)
--   scripts/schema_db0.16.sql                   (previous version)
--
-- This file was checked against a fresh database created by the 0.17 binary
-- (tables, columns, FKs, indexes, views, trigger all identical). The binary's
-- create_schema() remains the source of truth; keep this file in lockstep.
--
-- 0.17 changes (vs 0.16):
--   1. decisions are structured. New columns: trigger ("Use when ..." -- the
--      future-query vocabulary, embedded/indexed FIRST), rationale, scope,
--      supersedes (self-FK; the old row is kept, status 'superseded'),
--      session_id, project_id. status has no DEFAULT and no CHECK;
--      vocabulary: current | provisional | pending | superseded | retired
--      (legacy: roadmap, deprecated). Stray category words that old callers
--      put in status were moved into tags by the migration.
--   2. decision_evidence: many-to-many link from a decision to the
--      turn_summaries it rests on. Real FKs: CASCADE on the decision side,
--      RESTRICT on the turn_summary side (evidence cannot vanish from under
--      a claim).
--   3. protect_orphan_turn_summaries trigger: a turn summary whose turn was
--      culled (turn_id NULL) is the sole record of that exchange; the schema
--      refuses to delete it. Empty-text "done" marker rows stay deletable.
--   4. projects: what a body of work is (name, status, description, path).
--      project_id FKs (ON DELETE SET NULL) on sessions, summaries,
--      decisions and insights. Documents belong to no project (shared
--      reference material).
--   5. document_sources.path is UNIQUE (documents are real, imported files;
--      re-import means deleting the old document first).
--   6. insights + insight_evidence: synthesized knowledge ("these things
--      relate"), same shape as decisions (trigger optional, evidence
--      required by the API), with its own terms junction and view.
--   7. The new tables join the custom terms index (insights_terms) and get
--      views (projects_view, insights_view, insights_terms_view). The other
--      views gained the new columns.
--
-- 0.16 and 0.15 changes carried forward unchanged: custom terms index (no
-- FTS5, no phon column, per-record unigram_count/bigram_count), embedding_version
-- on every embedded table, documents normalized into document_sources.
--
-- Column order per Reid's preferred pattern:
--   1. Primary key (id)
--   2. Unique key fields (if any)
--   3. Supporting data fields (FKs, status, tags, ...)
--   4. created/modified timestamps
--   5. unigram_count, bigram_count (TF denominators)
--   6. embedding_version, embedding (embedding_version immediately precedes
--      embedding, since it describes that column)
--
-- Embedded/indexed text per table: turns = user_text+assistant_text;
-- decisions and insights = trigger || char(10) || text (bare text when
-- trigger is NULL/empty); documents = text + char(10) + source title.
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
-- projects  -- NEW at 0.17. What a body of work is. `path` is the absolute
-- project root (stored with the home directory normalized); the server uses
-- longest-prefix match of a caller's working directory against it to infer
-- the project. status: 'active' | (free-form, e.g. 'archived').
-- ---------------------------------------------------------------------------
CREATE TABLE projects (
    project_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    status      TEXT NOT NULL DEFAULT 'active',
    description TEXT,
    path        TEXT,
    created_at  INTEGER NOT NULL DEFAULT (unixepoch()),
    updated_at  INTEGER NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX idx_projects_status ON projects(status);

CREATE VIEW projects_view AS
SELECT project_id, name, status, description, path,
       datetime(created_at, 'unixepoch', 'localtime') AS created_at,
       datetime(updated_at, 'unixepoch', 'localtime') AS updated_at
FROM projects;

-- ---------------------------------------------------------------------------
-- sessions  -- lookup; turns + summaries + turn_summaries reference it.
-- Normalizes the long conversation GUIDs (from the agent's turn hook) to a
-- compact integer id, mirroring `models`. The grouping key for session
-- summaries and recipes. project_id (0.17) files a session under a project.
-- ---------------------------------------------------------------------------
CREATE TABLE sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    guid       TEXT NOT NULL UNIQUE,
    name       TEXT,
    name_source TEXT,
    project_id INTEGER REFERENCES projects(project_id) ON DELETE SET NULL,
    created_at INTEGER NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX idx_sessions_name    ON sessions(name);
CREATE INDEX idx_sessions_project ON sessions(project_id);

CREATE VIEW sessions_view AS
SELECT session_id, guid, name, name_source, project_id,
       datetime(created_at, 'unixepoch', 'localtime') AS created_at
FROM sessions;

-- ---------------------------------------------------------------------------
-- turns (L1)  -- raw verbatim exchanges
-- ---------------------------------------------------------------------------
CREATE TABLE turns (
    turn_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_text      TEXT NOT NULL,
    assistant_text TEXT,
    model_id       INTEGER REFERENCES models(model_id) ON DELETE SET NULL,
    session_id     INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL,
    created_at     INTEGER NOT NULL,
    unigram_count  INTEGER NOT NULL DEFAULT 0, -- stemmed-literal unigram tokens (TF denominator)
    bigram_count   INTEGER NOT NULL DEFAULT 0, -- stemmed-literal bigram tokens  (TF denominator)
    embedding_version INTEGER, -- NULL iff embedding IS NULL; tags which embedding_version produced `embedding`
    embedding      BLOB  -- embed(user_text + assistant_text); payload only, no version byte
);
CREATE INDEX idx_turns_created_at ON turns(created_at);
CREATE INDEX idx_turns_session    ON turns(session_id);

CREATE VIEW turns_view AS
SELECT turn_id, user_text, assistant_text, model_id, session_id,
       datetime(created_at, 'unixepoch', 'localtime') AS created_at,
       unigram_count, bigram_count,
       embedding_version,
       CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
FROM turns;

-- ---------------------------------------------------------------------------
-- turn_summaries (L2)  -- one row per summarized turn. `turn_id` is a REAL
-- FK, ON DELETE SET NULL: pruning the raw turn leaves the summary behind
-- with turn_id NULL instead of stranding it via a fragile
-- (session_id, created_at) match. `turn_datetime`/`turn_model_id` are copied
-- in explicitly at insert time (not derived via join) so both survive
-- `turns` row deletion. `summary_model_id` (who/what wrote the summary text)
-- is intentionally separate from `turn_model_id` (who answered originally).
--
-- Sentinel for "not yet summarized": summary_model_id IS NULL -- NOT
-- text IS NULL. `text` is populated at capture time with the raw
-- "User: ...\n\nAssistant: ..." placeholder (searchable immediately) and
-- overwritten in place once the summarizer produces a real summary. An
-- EMPTY-string text is a deliberate "done" marker (trivial turn); every query
-- that reads summary text/embedding must filter text != ''.
--
-- This is the evidence layer: decision_evidence and insight_evidence point
-- here.
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
    unigram_count  INTEGER NOT NULL DEFAULT 0,
    bigram_count   INTEGER NOT NULL DEFAULT 0,
    embedding_version INTEGER, -- NULL iff embedding IS NULL
    embedding        BLOB      -- payload only, no version byte
);
CREATE INDEX idx_turn_summaries_turn_id    ON turn_summaries(turn_id);
CREATE INDEX idx_turn_summaries_session_id ON turn_summaries(session_id);
CREATE INDEX idx_turn_summaries_datetime   ON turn_summaries(turn_datetime);
-- Upsert target for finalize_turn_summary: one row per turn_id. NULL
-- turn_id (a pruned turn) is excepted -- multiple such rows are fine.
CREATE UNIQUE INDEX idx_turn_summaries_turn_id_uniq
    ON turn_summaries(turn_id) WHERE turn_id IS NOT NULL;

-- An orphaned turn summary (its turn culled) is the only surviving record of
-- that exchange. Code never deletes one; this makes the schema refuse too.
-- Empty-text marker rows stay deletable (reset_abandoned_turn_summaries).
CREATE TRIGGER IF NOT EXISTS protect_orphan_turn_summaries
BEFORE DELETE ON turn_summaries
WHEN OLD.turn_id IS NULL AND OLD.text <> ''
BEGIN SELECT RAISE(ABORT, 'orphaned turn summary is sole evidence'); END;

CREATE VIEW turn_summaries_view AS
SELECT
    turn_summary_id, text, turn_id, session_id, turn_model_id, summary_model_id,
    datetime(turn_datetime, 'unixepoch', 'localtime') AS turn_datetime,
    datetime(summarized_on, 'unixepoch', 'localtime') AS summarized_on,
    unigram_count, bigram_count,
    embedding_version,
    CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
FROM turn_summaries
WHERE text != '';   -- hides the empty-string "done" marker rows

-- ---------------------------------------------------------------------------
-- summaries (L3/L4)  -- episode/session/project rollups, plus agent notes
-- (model_id NULL). Turn-level rows live in `turn_summaries`.
-- created_at: (re)write / span-start time. updated_at: running rows
-- (session/project) record their last regenerate time; 'episode' rows
-- carry the span-end; on any first insert updated_at == created_at.
-- model_id: the model that produced the summary (NULL for agent notes).
-- project_id (0.17): the project a note/rollup belongs to.
-- ---------------------------------------------------------------------------
CREATE TABLE summaries (
    summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT NOT NULL,
    level      TEXT NOT NULL, -- 'episode' | 'session' | 'project'
    tags       TEXT NOT NULL DEFAULT '',
    session_id INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL,
    model_id   INTEGER REFERENCES models(model_id),
    project_id INTEGER REFERENCES projects(project_id) ON DELETE SET NULL,
    created_at INTEGER NOT NULL DEFAULT (unixepoch()),
    updated_at INTEGER NOT NULL DEFAULT (unixepoch()),
    unigram_count  INTEGER NOT NULL DEFAULT 0,
    bigram_count   INTEGER NOT NULL DEFAULT 0,
    embedding_version INTEGER, -- NULL iff embedding IS NULL
    embedding  BLOB            -- payload only, no version byte
);
CREATE INDEX idx_summaries_level      ON summaries(level);
CREATE INDEX idx_summaries_created_at ON summaries(created_at);
CREATE INDEX idx_summaries_session    ON summaries(session_id);
CREATE INDEX idx_summaries_project    ON summaries(project_id);

CREATE VIEW summaries_view AS
SELECT summary_id, text, level, tags, session_id, model_id, project_id,
       datetime(created_at, 'unixepoch', 'localtime') AS created_at,
       datetime(updated_at, 'unixepoch', 'localtime') AS updated_at,
       unigram_count, bigram_count,
       embedding_version,
       CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
FROM summaries;

-- ---------------------------------------------------------------------------
-- decisions (L6)  -- standing rulings ("always do Y"). Structured at 0.17.
-- `trigger` is the "Use when ..." situation text; it is embedded and indexed
-- first (trigger || char(10) || text). `supersedes` points at the row this
-- one replaces; the old row is kept (status 'superseded'), never rewritten.
-- Text is never edited in place: a change is a new row with supersedes.
-- status has no DEFAULT/CHECK by design; the API validates the vocabulary:
--   current | provisional | pending | superseded | retired
--   (legacy, still readable: roadmap, deprecated)
-- Search shows only current/provisional/roadmap rows; the rest are history
-- or unreviewed drafts, kept in the table.
-- ---------------------------------------------------------------------------
CREATE TABLE decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,
    status      TEXT NOT NULL,
    tags        TEXT NOT NULL DEFAULT '',
    trigger     TEXT,
    rationale   TEXT,
    scope       TEXT,
    supersedes  INTEGER REFERENCES decisions(decision_id),
    session_id INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL,
    project_id  INTEGER REFERENCES projects(project_id) ON DELETE SET NULL,
    created_at  INTEGER NOT NULL DEFAULT (unixepoch()),
    unigram_count  INTEGER NOT NULL DEFAULT 0,
    bigram_count   INTEGER NOT NULL DEFAULT 0,
    embedding_version INTEGER, -- NULL iff embedding IS NULL
    embedding   BLOB           -- payload only, no version byte
);
CREATE INDEX idx_decisions_status     ON decisions(status);
CREATE INDEX idx_decisions_supersedes ON decisions(supersedes);
CREATE INDEX idx_decisions_project    ON decisions(project_id);

CREATE VIEW decisions_view AS
SELECT decision_id, text, status, tags,
       trigger, rationale, scope, supersedes, session_id, project_id,
       datetime(created_at, 'unixepoch', 'localtime') AS created_at,
       unigram_count, bigram_count,
       embedding_version,
       CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
FROM decisions;

-- decision_evidence: which turn_summaries a decision rests on (many-to-many).
-- CASCADE on the decision side (an operator prune of a decision takes its
-- links with it); RESTRICT on the turn_summary side so evidence cannot vanish
-- from under a claim.
CREATE TABLE decision_evidence (
    decision_id     INTEGER NOT NULL
        REFERENCES decisions(decision_id) ON DELETE CASCADE,
    turn_summary_id INTEGER NOT NULL
        REFERENCES turn_summaries(turn_summary_id) ON DELETE RESTRICT,
    PRIMARY KEY (decision_id, turn_summary_id)
);
CREATE INDEX idx_decision_evidence_ts ON decision_evidence(turn_summary_id);

-- ---------------------------------------------------------------------------
-- insights  -- NEW at 0.17. Synthesized knowledge ("these things relate"),
-- distinct from decisions ("always do Y"). Same shape as decisions; `trigger`
-- is OPTIONAL here, and the API requires at least one evidence link (an
-- insight with no sources is just a note). Same status vocabulary, same
-- supersede/history rules, same embed/index text expression as decisions.
-- ---------------------------------------------------------------------------
CREATE TABLE insights (
    insight_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'current',
    tags        TEXT NOT NULL DEFAULT '',
    trigger     TEXT,
    rationale   TEXT,
    scope       TEXT,
    supersedes  INTEGER REFERENCES insights(insight_id),
    session_id INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL,
    project_id  INTEGER REFERENCES projects(project_id) ON DELETE SET NULL,
    created_at  INTEGER NOT NULL DEFAULT (unixepoch()),
    unigram_count  INTEGER NOT NULL DEFAULT 0,
    bigram_count   INTEGER NOT NULL DEFAULT 0,
    embedding_version INTEGER, -- NULL iff embedding IS NULL
    embedding   BLOB           -- payload only, no version byte
);
CREATE INDEX idx_insights_status     ON insights(status);
CREATE INDEX idx_insights_supersedes ON insights(supersedes);
CREATE INDEX idx_insights_project    ON insights(project_id);

CREATE VIEW insights_view AS
SELECT insight_id, text, status, tags,
       trigger, rationale, scope, supersedes, session_id, project_id,
       datetime(created_at, 'unixepoch', 'localtime') AS created_at,
       unigram_count, bigram_count,
       embedding_version,
       CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
FROM insights;

-- insight_evidence: same shape and delete semantics as decision_evidence.
CREATE TABLE insight_evidence (
    insight_id      INTEGER NOT NULL
        REFERENCES insights(insight_id) ON DELETE CASCADE,
    turn_summary_id INTEGER NOT NULL
        REFERENCES turn_summaries(turn_summary_id) ON DELETE RESTRICT,
    PRIMARY KEY (insight_id, turn_summary_id)
);
CREATE INDEX idx_insight_evidence_ts ON insight_evidence(turn_summary_id);

-- ---------------------------------------------------------------------------
-- document_sources  -- one row per curated document (db_version 0.15). Holds
-- the per-document metadata that used to be repeated on every chunk of
-- `documents`. Gives a document a stable identity (document_source_id) so it
-- can be addressed / re-curated / replaced as a unit. NO terms-index
-- coverage: the table is tiny and its title rides the chunk embed/terms
-- signals (see store_document). imported_at is an INTEGER unix-epoch.
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
-- A document is identified by its path: the same file cannot be imported twice.
-- (Old pathless sources were given their document_source_id as a placeholder
-- path by the 0.17 migration.)
CREATE UNIQUE INDEX idx_document_sources_path ON document_sources(path);

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
    embedding          BLOB     -- payload only, no version byte
);
CREATE INDEX idx_documents_document_source_id ON documents(document_source_id);

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
INSERT OR IGNORE INTO settings (key, value) VALUES ('db_version', '0.17');

-- ---------------------------------------------------------------------------
-- Custom terms index (v0.16)  -- a single hand-rolled inverted index. One
-- shared `terms` list holds every distinct token: stemmed literal unigrams
-- AND bigrams, plus their primary Double Metaphone codes, all mixed flat
-- (tokens self-filter; no type/flag column). Each text table has its own
-- `<t>_terms` junction (many-to-many) carrying a per-record `count`. df is
-- computed live at query time as COUNT(*) over the junction (no
-- doc_frequency column). See custom-search-schema.md. 0.17 adds
-- insights_terms.
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

-- insights (0.17)
CREATE TABLE insights_terms (
    insight_id INTEGER NOT NULL REFERENCES insights(insight_id) ON DELETE CASCADE,
    term_id INTEGER NOT NULL REFERENCES terms(term_id) ON DELETE CASCADE,
    count   INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (insight_id, term_id)
);
CREATE INDEX idx_insights_terms_term ON insights_terms(term_id);

-- ---------------------------------------------------------------------------
-- Junction views -- resolve term_id to the readable term string.
--
-- The raw <t>_terms tables store term_id, which is meaningless on inspection.
-- These views join through `terms` so you can actually read why a record did
-- or did not match. Rows are ordered most-frequent-first within each record.
--
-- Both literal (stemmed) and DoubleMetaphone tokens live in the same flat
-- `terms` table with no type flag, so a single record's view output mixes
-- them: e.g. 'ragg' and 'memori' (stemmed literals) alongside 'RK' and 'MMR'
-- (primary metaphone codes). That is by design -- they self-filter at query
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

CREATE VIEW insights_terms_view AS
SELECT j.insight_id AS insight_id, t.term, j.count
FROM insights_terms j JOIN terms t ON t.term_id = j.term_id
ORDER BY j.insight_id, j.count DESC, t.term;
