-- schema_v4.sql
--
-- The v4 Ragger memory schema: reorders columns to follow Reid's preferred pattern:
--   1. Primary key (id)
--   2. Unique key fields (if any)
--   3. Supporting data fields
--   4. created/modified timestamps
--   5. embedding and phon columns (at the end, after timestamps)
--
-- Applies to all BASE TABLES only. FTS5 virtual tables and their triggers are excluded
-- (they remain as-is since they reference base table rows by rowid).
--

-- ---------------------------------------------------------------------------
-- users  -- who else has read access to documents.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE,
    token_hash    TEXT NOT NULL,
    password_hash TEXT,
    created       TEXT NOT NULL DEFAULT (datetime('now')),
    modified      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ---------------------------------------------------------------------------
-- models  -- lookup; turns + summaries reference it
-- ---------------------------------------------------------------------------
CREATE TABLE models (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name     TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ---------------------------------------------------------------------------
-- sessions  -- lookup; turns + summaries reference it. Normalizes the long
-- conversation GUIDs (from the agent's turn hook) to a compact integer id,
-- mirroring `models`. The grouping key for session summaries and recipes.
-- ---------------------------------------------------------------------------
CREATE TABLE sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    guid       TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ---------------------------------------------------------------------------
-- turns (L1)  -- raw verbatim exchanges
-- ---------------------------------------------------------------------------
CREATE TABLE turns (
    turn_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_text      TEXT NOT NULL,
    assistant_text TEXT,
    model_id       INTEGER REFERENCES models(model_id),
    session_id     INTEGER REFERENCES sessions(session_id),
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    embedding      BLOB, -- embed(user_text + assistant_text)
    phon           TEXT  -- phonize(user_text + assistant_text): Double Metaphone "sounds-like"
);
CREATE INDEX idx_turns_created_at ON turns(created_at);
CREATE INDEX idx_turns_session    ON turns(session_id);

-- ---------------------------------------------------------------------------
-- summaries (L2/L3/L4)
-- created_at: 'turn' copies its source turn's ts; 'session'/'project'/'episode'
-- use (re)write / span-start time. updated_at: running rows (session/project)
-- record their last regenerate time; 'episode' rows carry the span-end; on any
-- first insert updated_at == created_at. model_id: 'turn' copies the turn's
-- model; 'session'/'project'/'episode' get the model that produced the summary.
-- ---------------------------------------------------------------------------
CREATE TABLE summaries (
    summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT            NOT NULL,
    level      TEXT            NOT NULL, -- 'turn' | 'episode' | 'session' | 'project'
    status     TEXT            NOT NULL, -- 'current' | 'complete'
    tags       TEXT DEFAULT '' NOT NULL,
    session_id INTEGER REFERENCES sessions(session_id),
    model_id   INTEGER REFERENCES models(model_id),
    created_at TEXT            NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT            NOT NULL DEFAULT (datetime('now')),
    embedding  BLOB,
    phon       TEXT            -- phonize(text): Double Metaphone "sounds-like"
);
CREATE INDEX idx_summaries_level ON summaries(level);
CREATE INDEX idx_summaries_status ON summaries(status);
CREATE INDEX idx_summaries_created_at ON summaries(created_at);
CREATE INDEX idx_summaries_session ON summaries(session_id);

-- ---------------------------------------------------------------------------
-- decisions (L6)
-- ---------------------------------------------------------------------------
CREATE TABLE decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'current', -- current|superseded|revisit|deprecated
    tags        TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    embedding   BLOB,
    phon        TEXT -- phonize(text): Double Metaphone "sounds-like"
);
CREATE INDEX idx_decisions_status ON decisions(status);

-- ---------------------------------------------------------------------------
-- documents (L5)  -- user-curated RAG; the only sharable table
-- ---------------------------------------------------------------------------
CREATE TABLE documents (
    document_id INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,            -- body; chapter/section headings inline
    path        TEXT,
    title       TEXT,                     -- identifies the publication
    tags        TEXT NOT NULL DEFAULT '', -- groups + describes
    year        INTEGER,                  -- publish year
    chunk_index INTEGER,
    imported_at TEXT NOT NULL,            -- 'YYYYMMDD', shared per import batch
    embedding   BLOB,
    phon        TEXT                      -- phonize(text): Double Metaphone "sounds-like"
);
CREATE INDEX idx_documents_imported_at ON documents(imported_at);

-- ---------------------------------------------------------------------------
-- settings  -- key/value store (embedding_model, dimensions, recipe, vector_type, ...)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

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
