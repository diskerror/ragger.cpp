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

} // namespace ragger::sqlite::ddl
