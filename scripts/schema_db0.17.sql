-- schema_db0.17.sql
--
-- The Ragger memory schema at db_version '0.17'. This release delivers the
-- AGENT API data model in its final shape: episodes, unified knowledge, and
-- the compact terms index. See:
--   ~/ownCloud/Ragger/v0.17-rework/           (phases, reports, canonical DDL)
--   ~/ownCloud/Ragger/schema-0.17.ddl.sql     (the delta DDL for the leg)
--   scripts/schema_db0.16.sql                 (previous version)
--
-- This file was generated from a fresh database created by the 0.17 binary
-- and is verified against it by scripts/verify-schema-sql.sh (run from
-- scripts/test.sh). The binary's create_schema() remains the source of
-- truth; keep this file in lockstep.
--
-- 0.17 changes (vs 0.16):
--   1. sessions: project_id is GONE; started_on/ended_on span columns are
--      derived from the session's turns (MIN/MAX created_at).
--   2. episodes: narrative segments MOVE out of summaries (level='episode');
--      ids preserved, span_start/span_end from the old created_at/updated_at.
--      turn_summaries gains episode_id (recorded membership, backfilled by
--      span containment within the turn summary's own session).
--   3. knowledge: ONE table for kind decision | insight | note, replacing
--      decisions, insights, AND summaries (session/project rows became notes;
--      the generic store API writes notes). status: current | provisional |
--      superseded | retired | pending (no CHECK; vocabulary may grow).
--      decision requires trigger; insight requires evidence; note neither.
--      supersedes is self-referential and may cross kinds (promotion).
--      knowledge_turn_evidence / knowledge_document_evidence replace the old
--      evidence tables; the document side keeps a breadcrumb path so evidence
--      survives document deletion.
--   4. summaries, projects, decisions, insights and every junction/view of
--      those are GONE (the 0.16->0.17 leg migrates the data; see
--      ~/ownCloud/Ragger/v0.17-rework/migrate-0.16-to-0.17.sql).
--   5. Terms junctions are WITHOUT ROWID tables clustered on
--      (term_id, <pk>): one copy per posting instead of the 0.16 triple,
--      with the secondary idx_<table>_terms_row for the write path.
--   6. documents gains UNIQUE (document_source_id, chunk_index) — one chunk
--      per position per document. document_sources.path stays UNIQUE.
--   7. protect_orphan_turn_summaries trigger: a turn summary whose turn was
--      culled is the sole record of that exchange; the schema refuses to
--      delete it. Empty-text marker rows stay deletable.

CREATE TABLE users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT NOT NULL UNIQUE,
                token_hash    TEXT NOT NULL,
                password_hash TEXT,
                created_at    INTEGER NOT NULL DEFAULT (unixepoch()),
                updated_at    INTEGER NOT NULL DEFAULT (unixepoch())
            );

CREATE TRIGGER users_modified
            AFTER UPDATE ON users
            BEGIN
                UPDATE users SET updated_at = unixepoch()
                WHERE id = NEW.id;
            END;

CREATE TABLE settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

CREATE TABLE models (
                model_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT NOT NULL UNIQUE,
                created_at INTEGER NOT NULL DEFAULT (unixepoch())
            );

CREATE TABLE sessions (
                session_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                guid        TEXT NOT NULL UNIQUE,
                name        TEXT,
                name_source TEXT,
                started_on  INTEGER,
                ended_on    INTEGER,
                created_at  INTEGER NOT NULL DEFAULT (unixepoch())
            );

CREATE TABLE turns (
                turn_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_text      TEXT NOT NULL,
                assistant_text TEXT,
                model_id       INTEGER REFERENCES models(model_id) ON DELETE SET NULL,
                session_id     INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL,
                created_at     INTEGER NOT NULL,
                unigram_count  INTEGER NOT NULL DEFAULT 0,
                bigram_count   INTEGER NOT NULL DEFAULT 0,
                embedding_version INTEGER,
                embedding      BLOB
            );

CREATE TABLE turn_summaries (
                turn_summary_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                text             TEXT,
                turn_id          INTEGER REFERENCES turns(turn_id) ON DELETE SET NULL,
                session_id       INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL,
                episode_id       INTEGER REFERENCES episodes(episode_id) ON DELETE SET NULL,
                turn_model_id    INTEGER REFERENCES models(model_id),
                summary_model_id INTEGER REFERENCES models(model_id),
                turn_datetime    INTEGER NOT NULL,
                summarized_on    INTEGER NOT NULL DEFAULT (unixepoch()),
                unigram_count    INTEGER NOT NULL DEFAULT 0,
                bigram_count     INTEGER NOT NULL DEFAULT 0,
                embedding_version INTEGER,
                embedding        BLOB
            );

CREATE TABLE document_sources (
                document_source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT,
                path        TEXT,
                year        INTEGER,
                tags        TEXT NOT NULL DEFAULT '',
                imported_at INTEGER NOT NULL
            );

CREATE TABLE documents (
                document_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                text               TEXT NOT NULL,
                tags               TEXT NOT NULL DEFAULT '',
                chunk_index        INTEGER,
                document_source_id INTEGER REFERENCES document_sources(document_source_id),
                modified_on        INTEGER,
                unigram_count      INTEGER NOT NULL DEFAULT 0,
                bigram_count       INTEGER NOT NULL DEFAULT 0,
                embedding_version  INTEGER,
                embedding          BLOB
            );

CREATE TABLE terms (
            term_id INTEGER PRIMARY KEY AUTOINCREMENT,
            term    TEXT NOT NULL UNIQUE
        );

CREATE TABLE turns_terms (
    term_id INTEGER NOT NULL REFERENCES terms(term_id) ON DELETE CASCADE,
    turn_id INTEGER NOT NULL REFERENCES turns(turn_id) ON DELETE CASCADE,
    count   INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (term_id, turn_id)
) WITHOUT ROWID;

CREATE INDEX idx_turns_terms_row ON turns_terms(turn_id);

CREATE TABLE knowledge_terms (
    term_id INTEGER NOT NULL REFERENCES terms(term_id) ON DELETE CASCADE,
    knowledge_id INTEGER NOT NULL REFERENCES knowledge(knowledge_id) ON DELETE CASCADE,
    count   INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (term_id, knowledge_id)
) WITHOUT ROWID;

CREATE INDEX idx_knowledge_terms_row ON knowledge_terms(knowledge_id);

CREATE TABLE documents_terms (
    term_id INTEGER NOT NULL REFERENCES terms(term_id) ON DELETE CASCADE,
    document_id INTEGER NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    count   INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (term_id, document_id)
) WITHOUT ROWID;

CREATE INDEX idx_documents_terms_row ON documents_terms(document_id);

CREATE TABLE turn_summaries_terms (
    term_id INTEGER NOT NULL REFERENCES terms(term_id) ON DELETE CASCADE,
    turn_summary_id INTEGER NOT NULL REFERENCES turn_summaries(turn_summary_id) ON DELETE CASCADE,
    count   INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (term_id, turn_summary_id)
) WITHOUT ROWID;

CREATE INDEX idx_turn_summaries_terms_row ON turn_summaries_terms(turn_summary_id);

CREATE TABLE episodes_terms (
    term_id INTEGER NOT NULL REFERENCES terms(term_id) ON DELETE CASCADE,
    episode_id INTEGER NOT NULL REFERENCES episodes(episode_id) ON DELETE CASCADE,
    count   INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (term_id, episode_id)
) WITHOUT ROWID;

CREATE INDEX idx_episodes_terms_row ON episodes_terms(episode_id);

CREATE INDEX idx_turns_created_at      ON turns(created_at);

CREATE INDEX idx_turns_session         ON turns(session_id);

CREATE UNIQUE INDEX idx_document_sources_path ON document_sources(path);

CREATE INDEX idx_sessions_name          ON sessions(name);

CREATE TABLE knowledge (
                knowledge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind        TEXT NOT NULL,            -- decision | insight | note (no CHECK)
                text        TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'current',
                tags        TEXT NOT NULL DEFAULT '',
                trigger     TEXT,
                rationale   TEXT,
                scope       TEXT,
                supersedes  INTEGER REFERENCES knowledge(knowledge_id),
                session_id  INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL,
                model_id    INTEGER REFERENCES models(model_id),
                created_at  INTEGER NOT NULL DEFAULT (unixepoch()),
                unigram_count INTEGER NOT NULL DEFAULT 0,
                bigram_count  INTEGER NOT NULL DEFAULT 0,
                embedding_version INTEGER,
                embedding   BLOB
            );

CREATE INDEX idx_knowledge_kind       ON knowledge(kind);

CREATE INDEX idx_knowledge_status     ON knowledge(status);

CREATE INDEX idx_knowledge_supersedes ON knowledge(supersedes);

CREATE INDEX idx_knowledge_session    ON knowledge(session_id);

CREATE INDEX idx_knowledge_created_at ON knowledge(created_at);

CREATE INDEX idx_knowledge_embedding_version ON knowledge(embedding_version);

CREATE TABLE knowledge_turn_evidence (
                knowledge_id    INTEGER NOT NULL REFERENCES knowledge(knowledge_id) ON DELETE CASCADE,
                turn_summary_id INTEGER NOT NULL REFERENCES turn_summaries(turn_summary_id) ON DELETE RESTRICT,
                PRIMARY KEY (knowledge_id, turn_summary_id)
            );

CREATE INDEX idx_knowledge_turn_evidence_ts ON knowledge_turn_evidence(turn_summary_id);

CREATE TABLE knowledge_document_evidence (
                evidence_id          INTEGER PRIMARY KEY,
                knowledge_id         INTEGER NOT NULL REFERENCES knowledge(knowledge_id) ON DELETE CASCADE,
                document_id          INTEGER REFERENCES documents(document_id) ON DELETE SET NULL,
                document_source_path TEXT NOT NULL,   -- breadcrumb: survives delete-doc
                chunk_index          INTEGER,         -- breadcrumb hint for relink
                UNIQUE (knowledge_id, document_id)
            );

CREATE INDEX idx_knowledge_document_evidence_doc ON knowledge_document_evidence(document_id);

CREATE TABLE episodes (
                episode_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                text        TEXT NOT NULL,
                tags        TEXT NOT NULL DEFAULT '',
                session_id  INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL,
                model_id    INTEGER REFERENCES models(model_id),
                supersedes  INTEGER REFERENCES episodes(episode_id),
                span_start  INTEGER NOT NULL,
                span_end    INTEGER NOT NULL,
                created_at  INTEGER NOT NULL DEFAULT (unixepoch()),
                unigram_count INTEGER NOT NULL DEFAULT 0,
                bigram_count  INTEGER NOT NULL DEFAULT 0,
                embedding_version INTEGER,
                embedding   BLOB
            );

CREATE INDEX idx_episodes_session    ON episodes(session_id);

CREATE INDEX idx_episodes_span_start ON episodes(span_start);

CREATE INDEX idx_episodes_supersedes ON episodes(supersedes);

CREATE INDEX idx_episodes_embedding_version ON episodes(embedding_version);

CREATE UNIQUE INDEX idx_episodes_session_span_uniq
            ON episodes(session_id, span_start) WHERE supersedes IS NULL;

CREATE INDEX idx_turn_summaries_turn_id    ON turn_summaries(turn_id);

CREATE INDEX idx_turn_summaries_session_id ON turn_summaries(session_id);

CREATE INDEX idx_turn_summaries_episode_id ON turn_summaries(episode_id);

CREATE INDEX idx_turn_summaries_datetime   ON turn_summaries(turn_datetime);

CREATE INDEX idx_turn_summaries_embedding_version ON turn_summaries(embedding_version);

CREATE UNIQUE INDEX idx_turn_summaries_turn_id_uniq
            ON turn_summaries(turn_id) WHERE turn_id IS NOT NULL;

CREATE TRIGGER protect_orphan_turn_summaries
        BEFORE DELETE ON turn_summaries
        WHEN OLD.turn_id IS NULL AND OLD.text <> ''
        BEGIN SELECT RAISE(ABORT, 'orphaned turn summary is sole evidence'); END;

CREATE INDEX idx_document_sources_imported_at ON document_sources(imported_at);

CREATE INDEX idx_documents_document_source_id ON documents(document_source_id);

CREATE UNIQUE INDEX idx_documents_source_chunk_uniq ON documents(document_source_id, chunk_index);

CREATE VIEW users_view AS 
            SELECT id, username,
                   datetime(created_at, 'unixepoch', 'localtime') AS created_at,
                   datetime(updated_at, 'unixepoch', 'localtime') AS updated_at
            FROM users;

CREATE VIEW models_view AS 
            SELECT model_id, name,
                   datetime(created_at, 'unixepoch', 'localtime') AS created_at
            FROM models;

CREATE VIEW sessions_view AS 
            SELECT session_id, guid, name, name_source,
                   datetime(started_on, 'unixepoch', 'localtime') AS started_on,
                   datetime(ended_on, 'unixepoch', 'localtime') AS ended_on,
                   datetime(created_at, 'unixepoch', 'localtime') AS created_at
            FROM sessions;

CREATE VIEW turns_view AS 
            SELECT turn_id, user_text, assistant_text, model_id, session_id,
                   datetime(created_at, 'unixepoch', 'localtime') AS created_at,
                   unigram_count, bigram_count,
                   embedding_version,
                   CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
            FROM turns;

CREATE VIEW turn_summaries_view AS 
            SELECT
                turn_summary_id, text, turn_id, session_id, episode_id,
                turn_model_id, summary_model_id,
                datetime(turn_datetime, 'unixepoch', 'localtime') AS turn_datetime,
                datetime(summarized_on, 'unixepoch', 'localtime') AS summarized_on,
                unigram_count, bigram_count,
                embedding_version,
                CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
            FROM turn_summaries;

CREATE VIEW episodes_view AS 
            SELECT episode_id, text, tags, session_id, model_id, supersedes,
                   datetime(span_start, 'unixepoch', 'localtime') AS span_start,
                   datetime(span_end, 'unixepoch', 'localtime') AS span_end,
                   datetime(created_at, 'unixepoch', 'localtime') AS created_at,
                   unigram_count, bigram_count,
                   embedding_version,
                   CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
            FROM episodes;

CREATE VIEW knowledge_view AS 
            SELECT knowledge_id, kind, text, status, tags,
                   trigger, rationale, scope, supersedes, session_id, model_id,
                   datetime(created_at, 'unixepoch', 'localtime') AS created_at,
                   unigram_count, bigram_count,
                   embedding_version,
                   CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
            FROM knowledge;

CREATE VIEW document_sources_view AS 
            SELECT document_source_id, title, path, year, tags,
                   datetime(imported_at, 'unixepoch', 'localtime') AS imported_at
            FROM document_sources;

CREATE VIEW documents_view AS 
            SELECT document_id, text, tags, chunk_index, document_source_id,
                   datetime(modified_on, 'unixepoch', 'localtime') AS modified_on,
                   unigram_count, bigram_count,
                   embedding_version,
                   CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_embedding
            FROM documents;

CREATE VIEW knowledge_turn_evidence_view AS 
            SELECT knowledge_id, turn_summary_id FROM knowledge_turn_evidence;

CREATE VIEW knowledge_document_evidence_view AS 
            SELECT evidence_id, knowledge_id, document_id, document_source_path, chunk_index
            FROM knowledge_document_evidence;

CREATE VIEW turns_terms_view AS SELECT j.turn_id AS turn_id, t.term, j.count FROM turns_terms j JOIN terms t ON t.term_id = j.term_id ORDER BY j.turn_id, j.count DESC, t.term;

CREATE VIEW knowledge_terms_view AS SELECT j.knowledge_id AS knowledge_id, t.term, j.count FROM knowledge_terms j JOIN terms t ON t.term_id = j.term_id ORDER BY j.knowledge_id, j.count DESC, t.term;

CREATE VIEW documents_terms_view AS SELECT j.document_id AS document_id, t.term, j.count FROM documents_terms j JOIN terms t ON t.term_id = j.term_id ORDER BY j.document_id, j.count DESC, t.term;

CREATE VIEW turn_summaries_terms_view AS SELECT j.turn_summary_id AS turn_summary_id, t.term, j.count FROM turn_summaries_terms j JOIN terms t ON t.term_id = j.term_id ORDER BY j.turn_summary_id, j.count DESC, t.term;

CREATE VIEW episodes_terms_view AS SELECT j.episode_id AS episode_id, t.term, j.count FROM episodes_terms j JOIN terms t ON t.term_id = j.term_id ORDER BY j.episode_id, j.count DESC, t.term;

-- seed row: the version stamp a fresh install carries.
INSERT INTO settings (key, value) VALUES ('db_version', '0.17');
