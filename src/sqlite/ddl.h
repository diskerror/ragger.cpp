/**
 * sqlite/ddl.h — canonical CREATE TABLE statements shared by create_schema()
 * and the 0.16 -> 0.17 migration leg (PHASE4 Step 4). One definition per
 * table, no retyped copies: the leg passes the *_rebuild name, the fresh
 * schema passes the real name. Text follows
 * ownCloud/Ragger/v0.17-rework/schema-0.17.ddl.sql (the target DDL for every
 * table new or rebuilt in 0.17); the migration SQL sections are transcribed
 * in src/sqlite/schema.cpp with a comment until Phase 9 reconciles them.
 */
#pragma once

#include <format>
#include <string>

namespace ragger::sqlite::ddl {

// Turn "CREATE TABLE x" / "CREATE INDEX x" / "CREATE UNIQUE INDEX x" into
// the IF NOT EXISTS form (create_schema() side; the leg uses the plain
// canonical text because it creates rebuilt tables exactly once).
inline std::string if_not_exists(std::string ddl) {
    auto replace_all = [&](const std::string& from, const std::string& to) {
        for (size_t pos = 0; (pos = ddl.find(from, pos)) != std::string::npos; pos += to.size())
            ddl.replace(pos, from.size(), to);
    };
    replace_all("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS ");
    replace_all("CREATE UNIQUE INDEX ", "CREATE UNIQUE INDEX IF NOT EXISTS ");
    replace_all("CREATE INDEX ", "CREATE INDEX IF NOT EXISTS ");
    replace_all("CREATE TRIGGER ", "CREATE TRIGGER IF NOT EXISTS ");
    return ddl;
}

// sessions — grouping key only (guid + title + derived span).
// started_on/ended_on are INTEGER epochs: MIN/MAX of the session's turns,
// maintained on write (Impl::touch_session_span). Dropped: project_id.
inline std::string sessions(const std::string& name) {
    return std::format(R"(CREATE TABLE {} (
                session_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                guid        TEXT NOT NULL UNIQUE,
                name        TEXT,
                name_source TEXT,
                started_on  INTEGER,
                ended_on    INTEGER,
                created_at  INTEGER NOT NULL DEFAULT (unixepoch())
            ))", name);
}

// knowledge (PHASE7) — ONE table for decision | insight | note, replacing
// decisions + insights. kind has no CHECK (the vocabulary may grow).
// supersedes is self-referential and MAY cross kinds (insight -> decision
// promotion). Column order is the canonical §4 order (migration test pins
// it). Text matches schema-0.17.ddl.sql / migrate-0.16-to-0.17.sql §4.
inline std::string knowledge(const std::string& name = "knowledge") {
    return std::format(R"(CREATE TABLE {} (
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
            ))", name);
}

inline std::string knowledge_indexes(const std::string& name = "knowledge") {
    return std::format(R"(
        CREATE INDEX idx_{}_kind       ON {}(kind);
        CREATE INDEX idx_{}_status     ON {}(status);
        CREATE INDEX idx_{}_supersedes ON {}(supersedes);
        CREATE INDEX idx_{}_session    ON {}(session_id);
        CREATE INDEX idx_{}_created_at ON {}(created_at);
        CREATE INDEX idx_{}_embedding_version ON {}(embedding_version);
    )", name, name, name, name, name, name, name, name, name, name, name, name);
}

// Evidence at turn-summary level: CASCADE on the knowledge side, RESTRICT on
// the turn_summary side (evidence cannot vanish from under a claim).
inline std::string knowledge_turn_evidence() {
    return R"(CREATE TABLE knowledge_turn_evidence (
                knowledge_id    INTEGER NOT NULL REFERENCES knowledge(knowledge_id) ON DELETE CASCADE,
                turn_summary_id INTEGER NOT NULL REFERENCES turn_summaries(turn_summary_id) ON DELETE RESTRICT,
                PRIMARY KEY (knowledge_id, turn_summary_id)
            ))";
}

// Evidence at document-chunk level (written from Phase 11; exists now so the
// FK target and the SET NULL semantics are testable). document_id SET NULL
// keeps the breadcrumb (document_source_path) alive after delete-doc.
inline std::string knowledge_document_evidence() {
    return R"(CREATE TABLE knowledge_document_evidence (
                evidence_id          INTEGER PRIMARY KEY,
                knowledge_id         INTEGER NOT NULL REFERENCES knowledge(knowledge_id) ON DELETE CASCADE,
                document_id          INTEGER REFERENCES documents(document_id) ON DELETE SET NULL,
                document_source_path TEXT NOT NULL,   -- breadcrumb: survives delete-doc
                chunk_index          INTEGER,         -- breadcrumb hint for relink
                UNIQUE (knowledge_id, document_id)
            ))";
}

inline std::string knowledge_document_evidence_indexes() {
    return R"(CREATE INDEX idx_knowledge_document_evidence_doc ON knowledge_document_evidence(document_id))";
}

// episodes (PHASE5) — narrative segments over consecutive turn summaries.
// New at 0.17 (never rebuilt), so no name parameter; the supersedes FK is
// self-referential. span_start/span_end are epoch seconds of first/last
// member. The unique index allows only ONE open chain head per
// (session, span_start); superseded copies are exempt.
inline std::string episodes() {
    return R"(CREATE TABLE episodes (
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
            ))";
}

inline std::string episodes_indexes() {
    return R"(
        CREATE INDEX idx_episodes_session    ON episodes(session_id);
        CREATE INDEX idx_episodes_span_start ON episodes(span_start);
        CREATE INDEX idx_episodes_supersedes ON episodes(supersedes);
        CREATE INDEX idx_episodes_embedding_version ON episodes(embedding_version);
        CREATE UNIQUE INDEX idx_episodes_session_span_uniq
            ON episodes(session_id, span_start) WHERE supersedes IS NULL;
    )";
}

// turn_summaries (L2) — one row per summarized turn. Rebuilt at 0.17 to add
// episode_id (recorded membership; replaces the BETWEEN heuristic) and to
// place it in canonical column order. session_id gains ON DELETE SET NULL.
inline std::string turn_summaries(const std::string& name) {
    return std::format(R"(CREATE TABLE {} (
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
            ))", name);
}

// Indexes + the orphan-protection trigger. The trigger names the FINAL table
// (no parameter): call it after the rebuild's RENAME (dropping a table drops
// its triggers). ''-text marker rows stay deletable so
// reset_abandoned_turn_summaries() keeps working.
inline std::string turn_summaries_indexes_and_trigger() {
    return R"(
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
    )";
}

} // namespace ragger::sqlite::ddl
