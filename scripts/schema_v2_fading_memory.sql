-- schema_v2_fading_memory.sql
--
-- The v2 fading-memory storage schema: turns / summaries / decisions /
-- documents, plus the models + sessions lookup tables, plus FTS5 (issue #49)
-- replacing the hand-rolled bm25_* sidecars.
--
-- This is the reference DDL — it becomes the body of the C++ create_schema()
-- in sqlite_backend.cpp. Pre-v2 data is preserved separately by export
-- (see scripts/, backup-*.sql / memories-content-*.json) and re-imported
-- later; there is no in-place transform.
--
-- Out of scope here (unchanged, separate concerns): settings.

-- ---------------------------------------------------------------------------
-- users  -- who else has read access to documents.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    username   TEXT NOT NULL UNIQUE,
    token_hash TEXT NOT NULL,
    created    TEXT NOT NULL,
    modified   TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- models  -- lookup; turns + summaries reference it
-- ---------------------------------------------------------------------------
CREATE TABLE models (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name     TEXT NOT NULL UNIQUE
);

-- ---------------------------------------------------------------------------
-- sessions  -- lookup; turns + summaries reference it. Normalizes the long
-- conversation GUIDs (from the agent's turn hook) to a compact integer id,
-- mirroring `models`. The grouping key for session summaries and recipes.
-- ---------------------------------------------------------------------------
CREATE TABLE sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    guid       TEXT NOT NULL UNIQUE
);

-- ---------------------------------------------------------------------------
-- turns (L1)  -- raw verbatim exchanges
-- ---------------------------------------------------------------------------
CREATE TABLE turns (
    turn_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id       INTEGER REFERENCES models (model_id) ON DELETE SET NULL,
    session_id     INTEGER REFERENCES sessions (session_id) ON DELETE SET NULL,
    user_text      TEXT NOT NULL,
    assistant_text TEXT,
    embedding      BLOB, -- embed(user_text + assistant_text)
    timestamp      TEXT NOT NULL
);
CREATE INDEX idx_turns_timestamp ON turns (timestamp);
CREATE INDEX idx_turns_session   ON turns (session_id);

-- ---------------------------------------------------------------------------
-- summaries (L2/L3/L4)
-- timestamp: 'turn' copies its source turn's ts; 'session'/'project' use
-- (re)write time. model_id: 'turn' copies the turn's model; 'session'/
-- 'project' get the model that produced the summary.
-- ---------------------------------------------------------------------------
CREATE TABLE summaries (
    summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id   INTEGER REFERENCES models (model_id) ON DELETE SET NULL,
    session_id INTEGER REFERENCES sessions (session_id) ON DELETE SET NULL,
    text       TEXT NOT NULL,
    embedding  BLOB,
    level      TEXT NOT NULL, -- 'turn' | 'session' | 'project'
    status     TEXT NOT NULL, -- 'current' | 'complete'
    tags       TEXT NOT NULL DEFAULT '',
    timestamp  TEXT NOT NULL
);
CREATE INDEX idx_summaries_level ON summaries (level);
CREATE INDEX idx_summaries_status ON summaries (status);
CREATE INDEX idx_summaries_timestamp ON summaries (timestamp);
CREATE INDEX idx_summaries_session ON summaries (session_id);

-- ---------------------------------------------------------------------------
-- decisions (L6)
-- ---------------------------------------------------------------------------
CREATE TABLE decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,
    embedding   BLOB,
    status      TEXT NOT NULL, -- current|superseded|revisit|deprecated
    tags        TEXT NOT NULL DEFAULT '',
    timestamp   TEXT NOT NULL
);
CREATE INDEX idx_decisions_status ON decisions (status);

-- ---------------------------------------------------------------------------
-- documents (L5)  -- user-curated RAG; the only sharable table
-- ---------------------------------------------------------------------------
CREATE TABLE documents (
    document_id INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,            -- body; chapter/section headings inline
    embedding   BLOB,
    path        TEXT,
    title       TEXT,                     -- identifies the publication
    tags        TEXT NOT NULL DEFAULT '', -- groups + describes
    year        INTEGER,                  -- publish year
    chunk_index INTEGER,
    imported_at TEXT NOT NULL             -- 'YYYYMMDD', shared per import batch
);
CREATE INDEX idx_documents_imported_at ON documents (imported_at);

-- ---------------------------------------------------------------------------
-- FTS5  -- external-content virtual tables + sync triggers
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
