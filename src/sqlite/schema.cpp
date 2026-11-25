#include "sqlite/backend_impl.h"
#include "ddl.h"
#include "config.h"
#include "lang.h"
#include "util/fs.h"
#include "util/time.h"
#include "Logger.h"


#include <set>

namespace ragger::sqlite {

namespace {
// True iff `v` parses as "<major>.<minor>" and is older than 0.16. Anything
// that does not parse is NOT "older": the version gate further down rejects it.
bool older_than_0_16(const std::string& v) {
    std::istringstream in(v);
    int major = 0, minor = 0;
    char dot = 0;
    if (!(in >> major >> dot >> minor) || dot != '.') return false;
    return major == 0 && minor < 16;
}
} // namespace


using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

// Internal schema-migration helpers (review 4A): file-local free functions
// taking the Impl by reference. These are implementation details of
// schema.cpp's open/migrate path -- no other Impl translation unit should
// call them, and the anonymous namespace makes that a compile error rather
// than a convention. (Impl itself is pimpl, so its members were already
// invisible outside the sqlite module; this tightens the boundary one level
// further.)
//
// Templated on the Impl type ONLY because Backend::Impl is a private nested
// class: a namespace-scope function cannot NAME the type, but template
// argument deduction from *this inside a member function can. Every call
// site passes Backend::Impl; nothing else can reach these.

/// Archive the raw DB file(s) before the first structural change of a
/// migration run (see the definition for the checkpoint/close/fallback
/// strategy).
template <class Impl>
void maybe_backup_before_migration(Impl& self,
                                   const std::string& current_version);

/// Reopen `self.db` with the exact same open flags/pragmas used in the
/// constructor, and re-point the non-owning text_index_ at the new handle.
template <class Impl>
void reopen_db(Impl& self);

/// Integrity guard: refuse to run if FK enforcement is not actually on.
template <class Impl>
void require_foreign_keys(Impl& self);

/// Read a boolean PRAGMA's current value (e.g. "foreign_keys").
template <class Impl>
bool pragma_bool(Impl& self, const char* name);

/// X16 clause 9: upsert one `settings` row as its own statement (autocommit
/// here — these run before any leg transaction opens). Rule 9: Stmt::exec()
/// returns false on a failed write instead of throwing, so the return is
/// checked and a failure throws.
template <class Impl>
void record_setting(Impl& self, const char* key, const std::string& value);

/// X16 clause 10: read a `settings` row, `fallback` when absent.
template <class Impl>
std::string setting_or(Impl& self, const char* key, const char* fallback);

} // anonymous namespace


    Backend::Impl::Impl(Embedder& emb, const std::string& path)
        : embedder(&emb)
    {
        const auto& cfg = config();
        db_path = path.empty() ? cfg.resolved_db_path() : expand_path(path);
        vtype_ = vector_codec::parse(cfg.embedding_vector_type)
                     .value_or(vector_codec::VectorType::F16);

        // Create parent dirs
        fs::create_directories(fs::path(db_path).parent_path());

        int rc = sqlite3_open(db_path.c_str(), &db);
        if (rc != SQLITE_OK) {
            std::string err = sqlite3_errmsg(db);
            sqlite3_close(db);
            db = nullptr;
            throw std::runtime_error(std::format(lang::ERR_SQLITE_OPEN, err));
        }

        exec("PRAGMA journal_mode=WAL");
        exec("PRAGMA foreign_keys = ON");
        require_foreign_keys(*this);
        // Wait up to 10s on a locked DB instead of failing instantly —
        // WAL allows concurrent readers, but two writers (daemon +
        // import CLI) still serialize; without this any collision is an
        // immediate "database is locked" error.
        sqlite3_busy_timeout(db, 10000);
        text_index_.emplace(db);
        create_schema();

        // X16 clause 4: startup check. A `reindexing` row that is not
        // 'stopped' means a reindex was interrupted (crash, cancel, or a
        // failed post-migration reindex) and the text index may be incomplete.
        // Warn only — never auto-run (rule 11: Reid runs every data
        // operation). A CLI reindex running while a daemon starts gives a
        // harmless spurious warning; accepted.
        {
            Stmt s(db, "SELECT value FROM settings WHERE key = 'reindexing'");
            if (s.step()) {
                const std::string flag = s.column_text(0);
                if (flag != "stopped")
                    Diskerror::Logger::warn(std::format(
                        "text index may be incomplete: reindex {} did not finish; "
                        "run `ragger reindex {}`", flag, flag));
            }
        }

        // Load the current embedding version from the settings table.
        // If absent (fresh DB or pre-version DB), default to 0 and stamp it.
        {
            Stmt s(db, "SELECT value FROM settings WHERE key = 'embedding_version'");
            if (s.step()) {
                embedding_version_ = static_cast<uint8_t>(
                    std::stoi(s.column_text(0)) & 0xff);
            } else {
                embedding_version_ = 1;
                Stmt ins(db,
                    "INSERT OR IGNORE INTO settings (key, value) "
                    "VALUES ('embedding_version', '1')");
                // PHASE9B step 4c: a failed stamp means every later row gets
                // an embedding version that is not persisted; fail loudly.
                if (!ins.exec())
                    throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
            }
        }
    }


    /// DB-only constructor — no embedder.
    /// readonly=true: opens SQLITE_OPEN_READONLY, skips schema creation (for export).
    /// readonly=false: opens read-write and creates users/settings tables.
    Backend::Impl::Impl(const std::string& path, bool readonly)
        : embedder(nullptr), readonly_(readonly)
    {
        db_path = expand_path(path);
        if (!readonly)
            fs::create_directories(fs::path(db_path).parent_path());

        int rc;
        if (readonly) {
            rc = sqlite3_open_v2(db_path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr);
        } else {
            rc = sqlite3_open(db_path.c_str(), &db);
        }
        if (rc != SQLITE_OK) {
            std::string err = sqlite3_errmsg(db);
            sqlite3_close(db);
            db = nullptr;
            throw std::runtime_error(std::format(lang::ERR_SQLITE_OPEN, err));
        }

        if (!readonly) {
            exec("PRAGMA journal_mode=WAL");
            exec("PRAGMA foreign_keys = ON");
            require_foreign_keys(*this);
        }
        sqlite3_busy_timeout(db, 10000);
        text_index_.emplace(db);
        // Only ensure users + settings tables exist (skip memory tables/FTS).
        // Skip entirely for readonly connections (export path — no side-effects).
        if (!readonly) {
            create_user_schema();
        }
    }


    Backend::Impl::~Impl() { close(); }


    // ---- helpers -------------------------------------------------------
    void Backend::Impl::exec(const char* sql) {
        char* errmsg = nullptr;
        int rc = sqlite3_exec(db, sql, nullptr, nullptr, &errmsg);
        if (rc != SQLITE_OK) {
            std::string err = errmsg ? errmsg : "unknown error";
            sqlite3_free(errmsg);
            throw std::runtime_error(std::format(lang::ERR_SQL, err));
        }
    }

    void Backend::Impl::exec(const std::string& sql) { exec(sql.c_str()); }


    /// True if `table` has a column named `col`. Used to guard one-time
    /// ADD COLUMN migrations (SQLite has no ADD COLUMN IF NOT EXISTS).
    /// The table name is inlined (PRAGMA table_info doesn't take a bound
    /// parameter reliably); callers pass internal constants, never user input.
    bool Backend::Impl::column_exists(const std::string& table, const std::string& col) const {
        Stmt s(db, "PRAGMA table_info(" + table + ")");
        while (s.step()) {
            if (s.column_text(1) == col) return true;  // col 1 = column name
        }
        return false;
    }


    bool Backend::Impl::table_exists(const std::string& table) const {
        Stmt s(db, "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?");
        s.bind(1, table);
        return s.step();
    }


    void Backend::Impl::create_schema() {
        // Capture "was this DB already populated?" BEFORE any CREATE TABLE
        // IF NOT EXISTS runs below -- this is the only reliable way to tell
        // a genuinely fresh install (no memory tables at all yet) from an
        // existing pre-0.16 DB that simply never had its db_version row
        // stamped. Only a fresh install is auto-stamped at the bottom of
        // this function; an existing un-migrated DB is refused by the gate
        // above instead of silently being treated as current.
        bool db_preexisted = table_exists("turns");

        // Capture the on-disk schema version BEFORE any CREATE/ALTER below
        // mutates the file, then snapshot the DB if an in-binary migration is
        // pending -- so the backup reflects the pristine pre-migration state.
        // db_version() is only meaningful for an existing DB (settings table
        // already present); a fresh install has no version yet and needs no
        // backup. One snapshot covers a whole migration chain (0.16 -> 0.17):
        // it is taken here, once, before the first structural change.
        const std::string pre_migration_version =
            db_preexisted ? db_version() : std::string();

        // PHASE9A: upgrades are supported from 0.16 only. Refuse anything older
        // HERE -- before the backup and before the first CREATE/ALTER below --
        // so a rejected file is left byte-for-byte untouched. An existing DB
        // with no version row at all is older than 0.12 and is refused too.
        if (db_preexisted &&
            (pre_migration_version.empty() || older_than_0_16(pre_migration_version))) {
            throw std::runtime_error(std::format(
                "Ragger: this database is schema '{}', older than 0.16, and this build "
                "cannot upgrade it directly.\n"
                "Open it once with a Ragger v0.16 build to upgrade it to 0.16, then use this "
                "build. The file has not been modified.",
                pre_migration_version.empty() ? "<none>" : pre_migration_version));
        }

        maybe_backup_before_migration(*this, pre_migration_version);

        // Users, settings, and session tables — credentials (for read-only
        // document access) plus web/chat session persistence. These are a
        // separate concern from the v2 memory tables; one declarative
        // definition, shared with the DB-only constructor.
        create_user_schema();

        // ---- memory schema (issue #33): turns / turn_summaries / episodes /
        //      knowledge / documents / models, with the custom terms index
        //      (v0.16; the FTS5 layers of issue #49 are gone).
        //      Pre-v2 data is exported out-of-band, not migrated in place.

        // models — lookup table; turns, turn_summaries, episodes and
        // knowledge reference it. created_at
        // was added in a later schema revision; the column sits at the end so
        // it maps cleanly to the canonical (id, keys, data, timestamps) order.
        exec(R"(
            CREATE TABLE IF NOT EXISTS models (
                model_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT NOT NULL UNIQUE,
                created_at INTEGER NOT NULL DEFAULT (unixepoch())
            )
        )");

        // sessions — lookup table; turns + summaries reference it. Normalizes
        // the long conversation GUID (from the agent turn hook) to a compact
        // integer id, mirroring `models`. Grouping key only (PHASE4): guid +
        // title + derived started_on/ended_on span; no project_id.
        {
            exec(ddl::if_not_exists(ddl::sessions("sessions")));
        }

        // turns — raw verbatim exchanges. embedding nullable for the
        // deferred-embedding path (partial row written, backfilled later).
        // created_at has NO DEFAULT (matches scripts/schema_db0.17.sql
        // exactly) -- the app always supplies it explicitly on INSERT.
        // unigram_count/bigram_count are TF denominators for the custom terms
        // index (v0.16); placed before embedding_version to keep text-index
        // columns grouped together (cosmetic — SQLite appends physically).
        exec(R"(
            CREATE TABLE IF NOT EXISTS turns (
                turn_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_text      TEXT NOT NULL,
                assistant_text TEXT,
                model_id       INTEGER REFERENCES models(model_id) ON DELETE SET NULL,
                session_id     INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL,
                created_at     INTEGER NOT NULL,
                unigram_count  INTEGER NOT NULL DEFAULT 0,
                bigram_count   INTEGER NOT NULL DEFAULT 0,
                embedding_version INTEGER,
                embedding      BLOB,
                -- X32: prompt-arrival time (NULL = unknown). LAST column on
                -- purpose: the live upgrade can only ALTER TABLE ADD COLUMN,
                -- which appends, so the fresh DDL must match that order.
                started_on     INTEGER
            )
        )");

        // summaries is GONE in 0.17 (PHASE8): its rows moved to episodes
        // (§2) and knowledge kind='note' (§4b) by the 0.16 -> 0.17 leg; a
        // fresh DB never creates it. The legacy ALTERs below stay guarded on
        // table_exists() for pre-0.16 DBs that still have it at open time.

        // turn_summaries -- NEW at db_version 0.12. One row per
        // summarized turn, split out of `summaries`. turn_id is a real FK,
        // ON DELETE SET NULL: pruning the raw turn leaves the summary behind
        // with turn_id NULL rather than stranding it via a fragile
        // (session_id, created_at) match. 0.17 adds episode_id (recorded
        // membership) — the ddl:: definition is the single source of truth
        // for fresh DBs and the 0.16->0.17 rebuild alike.
        exec(ddl::if_not_exists(ddl::turn_summaries("turn_summaries")));

        // In-place migration for pre-sessions databases: add the session_id
        // FK column to turns/summaries if an existing DB predates it. SQLite
        // has no ADD COLUMN IF NOT EXISTS, so guard on pragma_table_info.
        // (Fresh DBs already have the column from the CREATE above.)
        if (!column_exists("turns", "session_id"))
            exec("ALTER TABLE turns ADD COLUMN session_id "
                 "INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL");
        if (table_exists("summaries") && !column_exists("summaries", "session_id"))
            exec("ALTER TABLE summaries ADD COLUMN session_id "
                 "INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL");
        // PHASE9: no index creation here — every index is created once, in
        // the post-migration section below, so fresh / migrated /
        // already-current DBs converge from a single source of truth.

        // document_sources — one row per curated document.
        // Holds title/path/year/tags/imported_at once, instead of repeating
        // them on every chunk. document_source_id gives a document a stable
        // identity. imported_at is a unix epoch (user's local import time).
        // No FTS: tiny table; its title rides the chunk embed/phon signals.
        exec(R"(
            CREATE TABLE IF NOT EXISTS document_sources (
                document_source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT,
                path        TEXT,
                year        INTEGER,
                tags        TEXT NOT NULL DEFAULT '',
                imported_at INTEGER NOT NULL
            )
        )");

        // documents — user-curated RAG chunks; the only sharable table.
        // Per-document metadata lives in document_sources (document_source_id
        // FK). Each chunk keeps only its distinctive tags; chunk_index is a
        // positional handle; modified_on (epoch) = source.imported_at until the
        // chunk text is edited. `text + '\n' + title` is embedded/phonized (see
        // store_document), so title participates in vector + phon search.
        exec(R"(
            CREATE TABLE IF NOT EXISTS documents (
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
            )
        )");

        // In-place migration for pre-0.15 databases: add embedding_version
        // to each embedded table if an existing DB predates it. Fresh DBs
        // already have the column from the CREATE above. New rows get NULL
        // (mirrors a NULL embedding) until backfilled/re-embedded.
        if (!column_exists("turns", "embedding_version"))
            exec("ALTER TABLE turns ADD COLUMN embedding_version INTEGER");
        if (table_exists("summaries") && !column_exists("summaries", "embedding_version"))
            exec("ALTER TABLE summaries ADD COLUMN embedding_version INTEGER");
        if (!column_exists("turn_summaries", "embedding_version"))
            exec("ALTER TABLE turn_summaries ADD COLUMN embedding_version INTEGER");
        if (table_exists("decisions") && !column_exists("decisions", "embedding_version"))
            exec("ALTER TABLE decisions ADD COLUMN embedding_version INTEGER");
        if (!column_exists("documents", "embedding_version"))
            exec("ALTER TABLE documents ADD COLUMN embedding_version INTEGER");

        // In-place migration for pre-0.16 databases: add unigram_count and
        // bigram_count to each text table (TF denominators for the custom
        // terms index). Fresh DBs already have them from the CREATE above;
        // existing rows get 0 until reindexed by the v0.16 migration leg
        // (Step 7 of docs/plans/migration-plan-fts-index.md). Idempotent: once
        // present, these ALTERs never fire again.
        if (!column_exists("turns", "unigram_count"))
            exec("ALTER TABLE turns ADD COLUMN unigram_count INTEGER NOT NULL DEFAULT 0");
        if (!column_exists("turns", "bigram_count"))
            exec("ALTER TABLE turns ADD COLUMN bigram_count INTEGER NOT NULL DEFAULT 0");
        // X32: turns.started_on (prompt-arrival time, NULL = unknown). Same
        // append-only pattern; the column is nullable with no default, so
        // existing rows stay unknown. The leg does not rebuild turns.
        if (!column_exists("turns", "started_on"))
            exec("ALTER TABLE turns ADD COLUMN started_on INTEGER");
        if (table_exists("summaries") && !column_exists("summaries", "unigram_count"))
            exec("ALTER TABLE summaries ADD COLUMN unigram_count INTEGER NOT NULL DEFAULT 0");
        if (table_exists("summaries") && !column_exists("summaries", "bigram_count"))
            exec("ALTER TABLE summaries ADD COLUMN bigram_count INTEGER NOT NULL DEFAULT 0");
        if (!column_exists("turn_summaries", "unigram_count"))
            exec("ALTER TABLE turn_summaries ADD COLUMN unigram_count INTEGER NOT NULL DEFAULT 0");
        if (!column_exists("turn_summaries", "bigram_count"))
            exec("ALTER TABLE turn_summaries ADD COLUMN bigram_count INTEGER NOT NULL DEFAULT 0");
        if (table_exists("decisions") && !column_exists("decisions", "unigram_count"))
            exec("ALTER TABLE decisions ADD COLUMN unigram_count INTEGER NOT NULL DEFAULT 0");
        if (table_exists("decisions") && !column_exists("decisions", "bigram_count"))
            exec("ALTER TABLE decisions ADD COLUMN bigram_count INTEGER NOT NULL DEFAULT 0");
        if (!column_exists("documents", "unigram_count"))
            exec("ALTER TABLE documents ADD COLUMN unigram_count INTEGER NOT NULL DEFAULT 0");
        if (!column_exists("documents", "bigram_count"))
            exec("ALTER TABLE documents ADD COLUMN bigram_count INTEGER NOT NULL DEFAULT 0");


        // Custom terms index (v0.16) — replaces both FTS5 layers with a single
        // flat term table + per-content-type junction tables. Created BEFORE
        // migrations run (the 0.16 -> 0.17 leg repopulates through
        // reindex_table()). See ownCloud/Ragger/custom-search-schema.md for
        // the design rationale and docs/plans/migration-plan-fts-index.md for
        // the step-by-step build plan.
        create_terms_schema();

        // Run any pending in-binary version migration for an existing DB
        // (since PHASE9A: the 0.16 -> 0.17 leg only). Advances
        // db_version toward kExpectedDbVersion so the hard gate below passes.
        // No-op on a fresh install or an already-current DB; an unknown/too-old
        // version is left untouched and rejected by the gate.
        if (db_preexisted)
            run_pending_migrations(pre_migration_version);

        // Schema-version hard gate. In-binary migration for the one supported
        // origin (0.16) ran just above; anything still not matching
        // kExpectedDbVersion here is an unknown/too-old version and is rejected.
        // First, stamp db_version for a genuinely fresh install ONLY --
        // db_preexisted (captured before any CREATE TABLE ran, at the top
        // of this function) distinguishes "no memory tables existed yet"
        // from an existing DB that simply lacks the version row. An
        // existing un-migrated DB never reaches here: the older_than_0_16
        // refusal above rejects it (upgrade via a v0.16 build first)
        // instead of the file being silently treated as current.
        if (!db_preexisted)
            exec(std::format("INSERT OR IGNORE INTO settings (key, value) VALUES ('db_version', '{}')", kExpectedDbVersion));
        {
            std::string actual = db_version();
            if (actual != kExpectedDbVersion) {
                // Throw, never exit: a library must not kill its host
                // process (B §1.11). Callers (main, server, mcp) already
                // report std::exception and exit non-zero.
                throw std::runtime_error(std::format(
                    "Ragger: database schema version mismatch (found '{}', need '{}').\n"
                    "This binary requires a database migrated to schema {}.\n"
                    "Restore the pre-migration backup next to the database, "
                    "or run the build that created this database.",
                    actual.empty() ? "<none>" : actual,
                    kExpectedDbVersion, kExpectedDbVersion));
            }
        }

        // Post-migration reclaim moved BELOW, after the indexes and views
        // are (re)built — "reclaim after everything" (review §3.6).

        // Indexes — (re)created here, after migration, so a table rebuild
        // (which drops the table and with it every index) can't leave them
        // missing. All IF NOT EXISTS, so fresh / migrated / already-current
        // DBs all converge to the same set.
        exec("CREATE INDEX IF NOT EXISTS idx_turns_created_at      ON turns(created_at)");
        exec("CREATE INDEX IF NOT EXISTS idx_turns_session         ON turns(session_id)");
        // A document is identified by its path (0.17): the same file cannot
        // be imported twice. NULL/empty paths from old DBs are given a unique
        // value by migrate_0_16_to_0_17() before this index is built.
        exec("CREATE UNIQUE INDEX IF NOT EXISTS idx_document_sources_path ON document_sources(path)");
        exec("CREATE INDEX IF NOT EXISTS idx_sessions_name          ON sessions(name)");

        // knowledge (PHASE7) — ONE table for decision | insight | note,
        // replacing decisions + insights + their evidence tables (dropped by
        // the 0.16 -> 0.17 leg; a fresh DB never creates the old ones).
        // create_terms_schema() above already ran, before this table existed:
        // SQLite accepts a REFERENCES clause naming a table created later, so
        // the order is fine. Created AFTER migrations so knowledge never takes
        // part in a rebuild dance.
        exec(ddl::if_not_exists(ddl::knowledge()));
        exec(ddl::if_not_exists(ddl::knowledge_indexes()));
        exec(ddl::if_not_exists(ddl::knowledge_episode_evidence()));
        exec(ddl::if_not_exists(ddl::knowledge_episode_evidence_indexes()));
        exec(ddl::if_not_exists(ddl::knowledge_document_evidence()));
        exec(ddl::if_not_exists(ddl::knowledge_document_evidence_indexes()));

        // episodes (PHASE5) — narrative segments. Created AFTER migrations
        // (like the evidence tables) so it never takes part in a rebuild
        // dance; the 0.16->0.17 leg creates and populates it itself (§2).
        exec(ddl::if_not_exists(ddl::episodes()));
        exec(ddl::if_not_exists(ddl::episodes_indexes()));

        // turn_summaries indexes + the orphan-protection trigger. An orphaned
        // Turn summary (its turn culled) is the only surviving record of that
        // exchange; code never deletes one, and this makes the schema refuse
        // too. Empty-text marker rows stay deletable so
        // reset_abandoned_turn_summaries() keeps working. Recreated here after
        // any rebuild (dropping a table drops its indexes and triggers).
        exec(ddl::if_not_exists(ddl::turn_summaries_indexes_and_trigger()));

        exec("CREATE INDEX IF NOT EXISTS idx_document_sources_imported_at ON document_sources(imported_at)");
        exec("CREATE INDEX IF NOT EXISTS idx_documents_document_source_id ON documents(document_source_id)");
        // PHASE9 (migrate-0.16-to-0.17.sql §5): (source, chunk_index) is the
        // final chunk key — one row per position per document. NULL
        // chunk_index rows are unconstrained (SQLite UNIQUE treats NULLs as
        // distinct); the leg's pre-check excludes them for the same reason.
        exec("CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_source_chunk_uniq "
             "ON documents(document_source_id, chunk_index)");

        // Human-readable views (datetime()-rendered timestamps,
        // has_embedding booleans) mirroring scripts/schema_db0.17.sql
        // exactly. Always (re)created so fresh/migrated DBs converge.
        create_views();

        // Post-migration reclaim (PHASE9): runs AFTER the indexes and views
        // above — the migration is committed and the schema complete, so
        // VACUUM reclaims everything the leg dropped. Gated exactly like the
        // pre-migration backup: skipped on a fresh install or a DB that was
        // already current. A failure (disk space: VACUUM needs ~2x the DB
        // size) is a warning, never an error — the migration itself is done.
        if (migration_pending(pre_migration_version)) {
            try {
                Diskerror::Logger::info(
                    "Post-migration VACUUM starting (reclaiming space from dropped "
                    "schema) -- this may take a while on a large DB.");
                exec("VACUUM");
                Diskerror::Logger::info("Post-migration VACUUM complete.");
            } catch (const std::exception& e) {
                Diskerror::Logger::warn(std::format(
                    "Post-migration VACUUM failed ({}); the migration itself "
                    "committed cleanly. Reclaim space later with a manual VACUUM.",
                    e.what()));
            }
        }
    }


    std::string Backend::Impl::db_version() const {
        Stmt s(db, "SELECT value FROM settings WHERE key = 'db_version'");
        if (s.step()) return s.column_text(0);
        return "";  // absent
    }


    void Backend::Impl::set_db_version(const std::string& v) {
        Stmt s(db,
            "INSERT INTO settings (key, value) VALUES ('db_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value");
        s.bind(1, v);
        // PHASE9B step 4c: the version stamp is the migration's commit flag.
        if (!s.exec())
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
    }


    // ---- in-binary version migrations ------------------------------------
    // Historically, DB schema upgrades were external shell scripts
    // (scripts/migrate_to_db*.sh) and the binary merely hard-gated on the
    // version. To make GitHub-cloned upgrades painless, the known upgrade
    // legs now run in-process on open, chained until the DB reaches
    // kExpectedDbVersion. Each leg has the same shape: one
    // BEGIN/try/COMMIT/ROLLBACK transaction, Logger progress,
    // and post-migration verification that aborts (rolls back) on any
    // inconsistency rather than silently shipping a corrupt index.

    /// PHASE9: true iff a pre-migration backup/reopen (and the
    /// post-migration VACUUM) should happen for `current_version` — i.e. it
    /// is a kLegs origin and not already current. Replaces the old
    /// per-version if-chain; maybe_backup_before_migration() and the
    /// VACUUM decision share this one predicate so they stay in lockstep.
    bool Backend::Impl::migration_pending(const std::string& current_version) {
        if (current_version.empty()) return false;                     // fresh/pre-version DB
        if (current_version == kExpectedDbVersion) return false;       // already current
        for (const auto& leg : kLegs)
            if (current_version == leg.from) return true;
        return false;  // unknown version: rejected by the gate without mutation
    }


    /// Archive the raw DB file(s) as "<db>_BACKUP_<timestamp>.tar.gz" (or
    /// .zip / .db, see fallback chain below) before the first structural
    /// change of a migration run. Only fires when an in-binary migration is
    /// actually pending (see migration_pending()). One snapshot covers a
    /// whole chain (since PHASE9A there is one leg: 0.16 -> 0.17).
    ///
    /// Preferred path: WAL-checkpoint in TRUNCATE mode WITHOUT closing the
    /// connection, then archive the now-fully-checkpointed main .db file
    /// while the connection stays open. A TRUNCATE checkpoint that reports
    /// zero "busy" frames means every WAL frame was folded back into the
    /// main file and the -wal file was truncated to empty -- at that point
    /// the main .db file alone is a complete, consistent point-in-time
    /// snapshot, and it's safe to read/archive it while `db` stays open,
    /// PROVIDED no other writer is concurrently active (true here: this is
    /// single-threaded startup code, before any other work on this
    /// connection begins). This avoids ever closing/reopening `db`, so
    /// text_index_'s non-owning pointer never goes stale.
    ///
    /// Falls back to a full close -> archive -> reopen (re-pointing
    /// text_index_ at the new handle) only if the checkpoint could NOT
    /// fully complete without closing (checkpoint reports busy frames,
    /// meaning some other connection/transaction is holding the WAL open --
    /// in that case the main .db file alone would not be a safe snapshot).
    ///
    /// Unlike the old VACUUM INTO approach (which snapshotted the live WAL
    /// database via a separate materialized copy query), this archives the
    /// raw file(s) directly using the existing timestamped naming
    /// convention (tar.gz -> zip -> plain copy fallback chain).
    namespace {
    template <class Impl>
    void maybe_backup_before_migration(Impl& self, const std::string& current_version) {
        // Static member via the instance: the template parameter is deduced
        // from *this, so this never has to NAME the private nested type.
        if (!self.migration_pending(current_version)) return;

        Diskerror::Logger::info(std::format(
            "DB schema {} predates {} -- taking pre-migration backup",
            current_version, std::string(self.kExpectedDbVersion)));

        // ---- 1. WAL-checkpoint (TRUNCATE) -- try without closing first ---
        // "PRAGMA wal_checkpoint(TRUNCATE);" returns one row: (busy, log,
        // checkpointed). busy != 0 means the checkpoint could not complete
        // fully (some frames left un-checkpointed) -- typically because
        // another connection is mid-transaction. busy == 0 means the main
        // .db file now fully reflects the database and the -wal file was
        // truncated to empty: safe to archive while staying open.
        bool checkpoint_complete = false;
        {
            Stmt s(self.db, "PRAGMA wal_checkpoint(TRUNCATE)");
            if (s.step()) {
                int busy = s.column_int(0);
                checkpoint_complete = (busy == 0);
            }
            // Stmt goes out of scope here -- no live prepared statement
            // survives past this block, whichever path we take next.
        }

        bool closed_for_backup = false;
        if (!checkpoint_complete) {
            // Fallback: something prevented a full checkpoint while open
            // (e.g. another connection holding a read/write transaction).
            // Fully close so the archived file(s) are guaranteed quiescent.
            Diskerror::Logger::info(
                "wal_checkpoint(TRUNCATE) could not fully complete while open "
                "-- falling back to close+archive+reopen for the pre-migration "
                "backup.");
            int rc = sqlite3_close(self.db);
            if (rc != SQLITE_OK) {
                throw std::runtime_error(std::format(
                    "pre-migration backup: sqlite3_close failed ({}); refusing to "
                    "archive a DB with abandoned open state", sqlite3_errstr(rc)));
            }
            self.db = nullptr;
            closed_for_backup = true;
        }

        // ---- 2. Archive the raw file(s): tar.gz -> zip -> plain copy -----
        // Shared with the CLI's re-embed/reindex backups (util/fs.h) so both
        // paths use one implementation and naming convention. stats.db is
        // opt-in/discardable telemetry (RAGGER_STATS) written by a separate
        // connection this class doesn't own, so it's bundled best-effort
        // (whatever's on disk right now, no checkpoint) rather than gated
        // behind its own migration logic -- add_db_and_siblings() silently
        // skips it if the file was never created (stats disabled).
        std::string archive_path;
        try {
            archive_path = archive_db_files(
                self.db_path, {config().resolved_stats_db_path()});
        } catch (const std::exception& e) {
            // If we closed for the backup, reopen before throwing so we
            // don't leave the backend in a permanently-closed state, then
            // surface the failure.
            if (closed_for_backup) reopen_db(self);
            throw std::runtime_error(
                std::string("pre-migration backup failed: ") + e.what());
        }

        Diskerror::Logger::info(std::format(
            "Pre-migration backup complete: {}", archive_path));

        // ---- 3. Reopen the connection (only if we closed it) --------------
        if (closed_for_backup) {
            reopen_db(self);
            Diskerror::Logger::info(
                "Reconnected to DB after close+archive pre-migration backup.");
        }

        // ---- 4. X16 clause 9: record the archive in `settings` ------------
        // Written AFTER the archive is complete (and after the reopen), but
        // BEFORE the leg runs, so they survive a rolled-back migration and
        // clause 10 can name the file from the failure message. Each is its
        // own committed statement with the return checked. The basename only:
        // the archive always sits beside the DB, and the DB stays free of
        // absolute paths. backup_timestamp is local time with its UTC offset
        // in RFC 3339 form with a space separator ("2026-11-13 00:30:12-08:00")
        // — SQLite's unixepoch()/datetime() parse it back, so no custom
        // reader is needed anywhere.
        record_setting(self, "backup_file_name",
                       fs::path(archive_path).filename().string());
        record_setting(self, "backup_timestamp", local_time_string());
    }
    } // anonymous namespace


    /// Reopen `db` (after maybe_backup_before_migration's checkpoint+close)
    /// with the exact same open flags/pragmas used in the constructor, and
    /// re-point the non-owning text_index_ at the new handle -- sqlite3_open
    /// is not guaranteed to reuse the same pointer value.
    namespace {
    template <class Impl>
    void reopen_db(Impl& self) {
        int rc = sqlite3_open(self.db_path.c_str(), &self.db);
        if (rc != SQLITE_OK) {
            std::string err = self.db ? sqlite3_errmsg(self.db) : sqlite3_errstr(rc);
            if (self.db) { sqlite3_close(self.db); self.db = nullptr; }
            throw std::runtime_error(std::format(lang::ERR_SQLITE_OPEN, err));
        }
        self.exec("PRAGMA journal_mode=WAL");
        self.exec("PRAGMA foreign_keys = ON");
        require_foreign_keys(self);
        sqlite3_busy_timeout(self.db, 10000);

        // text_index_ holds a NON-OWNING sqlite3* captured at construction;
        // re-emplace it against the new handle. Cheap (no allocation, just a
        // pointer member) and safe (no other state in TextIndex).
        self.text_index_.reset();
        self.text_index_.emplace(self.db);
    }
    } // anonymous namespace


    /// Chain the in-binary migration legs until the DB reaches
    /// kExpectedDbVersion (or a leg we don't know about is hit -> leave it for
    /// the hard gate to reject). `from_version` is the on-disk version captured
    /// before create_schema() mutated anything. Each leg advances db_version;
    /// the loop re-reads it under the single backup taken above (since
    /// PHASE9A there is one leg: 0.16 -> 0.17).
    void Backend::Impl::run_pending_migrations(const std::string& from_version) {
        std::string v = from_version;
        // Guard against an unbounded loop if a leg ever fails to advance the
        // version. Legs are few; a handful of iterations is plenty.
        for (int guard = 0; guard < 8; ++guard) {
            if (v.empty() || v == kExpectedDbVersion) return;
            const Leg* leg = nullptr;
            for (const auto& l : kLegs)
                if (v == l.from) { leg = &l; break; }
            // No known leg from this version -> stop; the hard gate reports it.
            if (!leg) return;
            (this->*leg->fn)();
            v = db_version();
        }
    }


    /// Integrity guard: the schema's FK rules (ON DELETE SET NULL/RESTRICT)
    /// only hold if enforcement is actually on. c_lib's SQLite defaults it
    /// ON and we also set it per connection; this catches a build linked
    /// against a library that silently ignores the PRAGMA (e.g. compiled
    /// with SQLITE_OMIT_FOREIGN_KEY) instead of running unprotected.
    namespace {
    template <class Impl>
    void require_foreign_keys(Impl& self) {
        if (!pragma_bool(self, "foreign_keys"))
            throw std::runtime_error(lang::ERR_FOREIGN_KEYS_OFF);
    }


    template <class Impl>
    bool pragma_bool(Impl& self, const char* name) {
        Stmt s(self.db, std::format("PRAGMA {}", name));
        return s.step() && s.column_int(0) != 0;
    }


    // X16 clause 9: same shape as set_db_version() (the migration's own
    // settings write), as an upsert with the return value checked.
    template <class Impl>
    void record_setting(Impl& self, const char* key, const std::string& value) {
        Stmt s(self.db,
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value");
        s.bind(1, key);
        s.bind(2, value);
        if (!s.exec())
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED,
                                                 sqlite3_errmsg(self.db)));
    }


    // X16 clause 10: read a settings row for the failure message; `fallback`
    // when the row is absent (e.g. a DB whose backup step never ran).
    template <class Impl>
    std::string setting_or(Impl& self, const char* key, const char* fallback) {
        Stmt s(self.db, "SELECT value FROM settings WHERE key = ?");
        s.bind(1, key);
        return s.step() ? s.column_text(0) : std::string(fallback);
    }
    } // anonymous namespace


    /// 0.16 -> 0.17 migration (the whole agent API schema in one step):
    ///   - knowledge (§4/§4a, PHASE7): the table + evidence tables are created,
    ///     decisions rows move in with kind='decision' and ids PRESERVED;
    ///     the old decisions/insights objects are dropped.
    ///   - notes (§4b, PHASE8): the remaining summaries rows move in with
    ///     kind='note'; `summaries` + `summaries_terms` are dropped.
    ///   - sessions rebuilt with started_on/ended_on (PHASE4; "project" is
    ///     a tag now — no projects table, no project_id anywhere).
    /// (Developed as three internal steps; released as one schema version.)
    /// Pragma dance: foreign_keys OFF (decisions_terms has ON DELETE CASCADE into decisions;
    /// DROP TABLE would wipe the index) and legacy_alter_table ON (keeps the
    /// RENAME from repointing other tables' REFERENCES at "decisions_rebuild").
    /// Both must be set OUTSIDE the transaction.
    void Backend::Impl::migrate_0_16_to_0_17() {
        Diskerror::Logger::info("Migrating DB 0.16 -> 0.17 (sessions span, episodes, knowledge)...");

        const bool fk_was_on = pragma_bool(*this, "foreign_keys");
        exec("PRAGMA foreign_keys = OFF");
        exec("PRAGMA legacy_alter_table = ON");

        try {
            // PHASE9B step 4c: unchecked BEGIN/COMMIT — a BUSY COMMIT logged
            // "migration complete" with the transaction still open. exec()
            // throws; the catch below rolls back and rethrows.
            exec("BEGIN");

            // ---- PHASE9: pre-checks BEFORE any change (review: the old
            // placement ran the duplicate-path check after the first
            // rebuild, so its "nothing was changed" message was only true
            // thanks to rollback). The placeholder-path UPDATE is idempotent
            // and belongs to this group; a later failure rolls it back.

            // document_sources.path becomes UNIQUE (0.17: a document is
            // identified by its path). Old DBs may hold sources with no
            // path; give each its own document_source_id as text so the
            // unique index can be built (the value is a placeholder, not a
            // file).
            exec("UPDATE document_sources SET path = CAST(document_source_id AS TEXT) "
                 "WHERE path IS NULL OR path = ''");
            if (int n = sqlite3_changes(db); n > 0)
                Diskerror::Logger::info(std::format(
                    "  document_sources: gave {} pathless source(s) a placeholder path "
                    "(their document_source_id)", n));
            {
                // Two sources with the same path cannot be merged safely
                // here (different titles/tags); refuse with a clear message
                // so the operator decides which to keep.
                Stmt d(db, "SELECT path, COUNT(*) FROM document_sources "
                           "GROUP BY path HAVING COUNT(*) > 1 LIMIT 1");
                if (d.step())
                    throw std::runtime_error(std::format(
                        "0.16 -> 0.17 migration: {} document_sources rows share the "
                        "path '{}' (paths must now be unique). Delete or fix the "
                        "duplicates, then restart; nothing was changed.",
                        d.column_int(1), d.column_text(0)));
            }
            {
                // PHASE9 §5: (source, chunk_index) becomes the chunk key.
                // NULL chunk_index rows stay unconstrained (UNIQUE treats
                // NULLs distinct), so they are excluded here for exactly the
                // semantics the index will have.
                Stmt d(db, "SELECT document_source_id, chunk_index, COUNT(*) FROM documents "
                           "WHERE chunk_index IS NOT NULL "
                           "GROUP BY 1, 2 HAVING COUNT(*) > 1 LIMIT 1");
                if (d.step())
                    throw std::runtime_error(std::format(
                        "0.16 -> 0.17 migration: {} chunks share (source {}, chunk_index {}). "
                        "Delete or fix the duplicates, then restart; nothing was changed.",
                        d.column_int(2), d.column_int64(0), d.column_int64(1)));
            }
            {
                // episodes gain UNIQUE (session_id, span_start), and
                // span_start = old created_at — two episode rows sharing
                // (session, created_at) would collide.
                Stmt d(db, "SELECT session_id, created_at, COUNT(*) FROM summaries "
                           "WHERE level = 'episode' AND session_id IS NOT NULL "
                           "GROUP BY 1, 2 HAVING COUNT(*) > 1 LIMIT 1");
                if (d.step())
                    throw std::runtime_error(std::format(
                        "0.16 -> 0.17 migration: {} episodes share (session {}, span_start {}). "
                        "Episode keys must be unique per (session, span_start). Delete or "
                        "fix the duplicates, then restart; nothing was changed.",
                        d.column_int(2), d.column_int64(0), d.column_int64(1)));
            }

            auto fk_violations = [this]() {
                std::multiset<std::string> out;
                Stmt s(db, "PRAGMA foreign_key_check");
                while (s.step()) {
                    // Full violation identity (table, rowid, parent, fkid):
                    // two FK constraints on one table may point at the SAME
                    // parent table (e.g. turn_summaries.turn_model_id and
                    // summary_model_id -> models), so the 3-part form could
                    // collide and misattribute a NEW violation as one of the
                    // identical pre-existing ones in the multiset diff.
                    out.insert(std::format("{}|{}|{}|{}", s.column_text(0),
                                           s.column_int64(1), s.column_text(2),
                                           s.column_int64(3)));
                }
                return out;
            };
            // Baseline BEFORE any change: pre-existing violations (e.g. a
            // summary whose session was deleted while FKs were off) must not
            // be blamed on this leg.
            const std::multiset<std::string> fk_before = fk_violations();

            // PHASE9: drop EVERY view up front (the rebuild dance and the
            // §6 drops would strand them anyway; create_views() recreates
            // the full set after the leg). Replaces the old explicit
            // DROP VIEWs, which missed views added outside this function
            // (e.g. summaries_terms_view).
            {
                std::vector<std::string> views;
                Stmt v(db, "SELECT name FROM sqlite_master WHERE type = 'view' ORDER BY name");
                while (v.step()) views.push_back(v.column_text(0));
                for (const auto& name : views)
                    exec("DROP VIEW IF EXISTS " + name);  // names from sqlite_master
            }

            // ---- sessions rebuild (PHASE4; migrate-0.16-to-0.17.sql §1,
            // transcribed here until Phase 9 executes the file) ------------
            // §0: save the AUTOINCREMENT counters (PHASE5: turn_summaries
            // joins the mechanism with its own rebuild below).
            exec("CREATE TEMP TABLE IF NOT EXISTS seq_save ("
                 "name TEXT PRIMARY KEY, seq INTEGER)");
            exec("INSERT OR REPLACE INTO seq_save "
                 "SELECT name, seq FROM sqlite_sequence "
                 "WHERE name IN ('sessions','turn_summaries')");

            // §1: rebuild with started_on/ended_on derived from the turns
            // table (NULL for sessions without turns). No project_id.
            exec(ddl::sessions("sessions_rebuild"));
            exec(R"(
                INSERT INTO sessions_rebuild
                    (session_id, guid, name, name_source, started_on, ended_on, created_at)
                SELECT s.session_id, s.guid, s.name, s.name_source,
                       (SELECT MIN(created_at) FROM turns WHERE session_id = s.session_id),
                       (SELECT MAX(created_at) FROM turns WHERE session_id = s.session_id),
                       s.created_at
                FROM sessions s ORDER BY s.session_id
            )");
            exec("DROP TABLE sessions");
            exec("ALTER TABLE sessions_rebuild RENAME TO sessions");
            exec("CREATE INDEX IF NOT EXISTS idx_sessions_name ON sessions(name)");

            // ---- episodes (PHASE5; migrate-0.16-to-0.17.sql §2) ----------
            // level='episode' rows MOVE out of summaries; ids are preserved
            // (episode_id = old summary_id), span = old created_at/updated_at.
            exec(ddl::episodes());
            exec(R"(
                INSERT INTO episodes (episode_id, text, tags, session_id, model_id,
                                      span_start, span_end, modified_on,
                                      unigram_count, bigram_count, embedding_version, embedding)
                SELECT summary_id, text, tags, session_id, model_id,
                       created_at, updated_at, created_at,
                       unigram_count, bigram_count, embedding_version, embedding
                FROM summaries WHERE level = 'episode' ORDER BY summary_id
            )");
            exec(ddl::episodes_indexes());
            // PHASE8: no surgical delete here — `summaries` itself is dropped
            // below (§6). The rows MOVED to episodes; FK actions do not fire
            // inside the leg (foreign_keys OFF), and the junction is dropped
            // with the table, so nothing is left to orphan.

            // ---- turn_summaries rebuild (PHASE5; §3) ---------------------
            // Adds episode_id (recorded membership) in canonical column
            // order and gives session_id ON DELETE SET NULL. Backfill: a
            // content row joins the FIRST episode (by span_start, id) of its
            // OWN session whose span contains turn_datetime — same-second
            // collisions across sessions cannot cross-assign because the
            // subquery is session-scoped. Marker rows ('' text) and NULL-text
            // rows stay unassigned. Ids preserved, so turn_summaries_terms
            // rows keep pointing at the same rows (no reindex needed for
            // them; the all-table reindex below refreshes everything anyway).
            exec(ddl::turn_summaries("turn_summaries_rebuild"));
            exec(R"(
                INSERT INTO turn_summaries_rebuild
                    (turn_summary_id, text, turn_id, session_id, episode_id,
                     turn_model_id, summary_model_id, turn_datetime, summarized_on,
                     unigram_count, bigram_count, embedding_version, embedding)
                SELECT t.turn_summary_id, t.text, t.turn_id, t.session_id,
                       CASE WHEN t.text IS NOT NULL AND t.text <> '' THEN
                            (SELECT e.episode_id FROM episodes e
                              WHERE e.session_id = t.session_id
                                AND t.turn_datetime BETWEEN e.span_start AND e.span_end
                              ORDER BY e.span_start, e.episode_id LIMIT 1)
                       END,
                       t.turn_model_id, t.summary_model_id, t.turn_datetime, t.summarized_on,
                       t.unigram_count, t.bigram_count, t.embedding_version, t.embedding
                FROM turn_summaries t ORDER BY t.turn_summary_id
            )");
            exec("DROP TABLE turn_summaries");
            exec("ALTER TABLE turn_summaries_rebuild RENAME TO turn_summaries");
            // The rebuild dropped the indexes and the orphan-protection
            // trigger (SQL file's ts_rest): recreate both.
            exec(ddl::turn_summaries_indexes_and_trigger());

            // ---- knowledge (PHASE7; migrate-0.16-to-0.17.sql §4 + §4a) ----
            // The table + both evidence tables are created EMPTY (a real
            // 0.16 DB has no evidence rows; §4b notes arrive in Phase 8).
            exec(ddl::knowledge());
            exec(ddl::knowledge_indexes());
            exec(ddl::knowledge_episode_evidence());
            exec(ddl::knowledge_episode_evidence_indexes());
            exec(ddl::knowledge_document_evidence());
            exec(ddl::knowledge_document_evidence_indexes());

            // §4a: decisions -> kind='decision', ids PRESERVED. A status
            // outside the vocabulary (old callers put a category word there)
            // moves to tags and status becomes 'current'. roadmap/deprecated
            // stay as stored. trigger/rationale/scope/supersedes/session_id/
            // model_id start NULL (0.16 had none of those columns).
            // FROM decisions — 0.16 source table.
            {
                const std::string known = status::in_list_sql(status::kMigrationKnown);
                exec("INSERT INTO knowledge (knowledge_id, kind, text, status, tags, created_at,"
                     " unigram_count, bigram_count, embedding_version, embedding)"
                     " SELECT decision_id, 'decision', text,"
                     "        CASE WHEN status IN (" + known + ")"
                     "             THEN status ELSE 'current' END,"
                     "        CASE WHEN status IN (" + known + ")"
                     "             THEN tags"
                     "             WHEN tags = '' THEN status ELSE tags || ',' || status END,"
                     "        created_at, unigram_count, bigram_count, embedding_version, embedding"
                     " FROM decisions ORDER BY decision_id");
            }
            if (int n = sqlite3_changes(db); n > 0)
                Diskerror::Logger::info(std::format(
                    "  knowledge: moved {} decision(s) (ids preserved)", n));

            // §4b: remaining summaries rows (level session|project) ->
            // kind='note'. NOTHING is erased. Classification is by
            // session_id + the `imported` tag, NOT by model_id (the rollup
            // pipeline always sets session_id on its level='session' rows;
            // rows with NULL session_id are hand-written or imported notes,
            // whether or not they carry an importer model name). knowledge_id
            // is omitted -> AUTOINCREMENT continues above the highest moved
            // decision id, in old summary_id order.
            // FROM summaries — 0.16 source table.
            exec(R"(
                INSERT INTO knowledge (kind, text, status, tags, session_id, model_id,
                                       created_at, unigram_count, bigram_count,
                                       embedding_version, embedding)
                SELECT 'note', text,
                       CASE WHEN level = 'session' AND session_id IS NULL THEN 'current'
                            WHEN level = 'project' AND tags LIKE '%imported%' THEN 'current'
                            ELSE 'retired' END,
                       CASE WHEN tags = '' THEN '' ELSE tags || ',' END ||
                       CASE WHEN level = 'session' AND session_id IS NULL THEN 'legacy_note'
                            WHEN level = 'session' THEN 'legacy_session'
                            ELSE 'legacy_project' END,
                       session_id, model_id, created_at,
                       unigram_count, bigram_count, embedding_version, embedding
                FROM summaries WHERE level IN ('session', 'project') ORDER BY summary_id
            )");
            if (int n = sqlite3_changes(db); n > 0)
                Diskerror::Logger::info(std::format(
                    "  knowledge: moved {} note(s) from old summaries", n));

            // Drop the old objects. decisions/decisions_terms exist in every
            // migrated DB; the insights/evidence/junction objects exist only
            // in dev DBs that ran an earlier 0.17 leg — IF EXISTS for safety.
            // (0.16 source tables — removal code, gone-list exception.)
            exec("DROP TABLE decisions");
            exec("DROP TABLE summaries");
            exec("DROP TABLE IF EXISTS decisions_terms");
            exec("DROP TABLE IF EXISTS summaries_terms");
            exec("DROP TABLE IF EXISTS insights");
            exec("DROP TABLE IF EXISTS insights_terms");
            exec("DROP TABLE IF EXISTS decision_evidence");
            exec("DROP TABLE IF EXISTS insight_evidence");
            // PHASE9: idx_turns_embedding_version (0.15/0.16 era) is not in
            // the 0.17 design; turns is not rebuilt by this leg, so it would
            // otherwise survive on real DBs while fresh DBs never have it.
            // No 0.17 query filters turns by embedding_version.
            exec("DROP INDEX IF EXISTS idx_turns_embedding_version");

            // PHASE5 / X16 clause 5: the terms wipe stays the LAST step INSIDE
            // the leg transaction (it is drop/recreate DDL, and a failed leg
            // must leave the old index untouched), but the all-table reindex
            // itself moved BELOW, after the COMMIT: inside one transaction
            // SQLite cannot checkpoint, which is exactly what turned Reid's
            // live upgrade into an 85 GB -wal. The flag is stamped in the same
            // transaction as db_version, so a committed 0.17 always means "a
            // reindex is owed".
            reset_terms_table();

            // ---- §7 (migrate-0.16-to-0.17.sql): settings cleanup, seq
            // restore, version stamp — before the final FK diff (canonical
            // order; PHASE9). The reindex that used to sit here now runs after
            // the COMMIT below (X16 clause 5).

            // PHASE5: the boundary-watermark machinery is deleted with the
            // session/project rollups; drop their settings rows, plus the two
            // dead keys no code reads anymore (S40).
            exec("DELETE FROM settings WHERE key IN "
                 "('session_boundary_watermark_turn_id', "
                 " 'project_boundary_watermark_turn_id', "
                 " 'phon_weight', 'ini_migrated')");

            // Restore the saved AUTOINCREMENT counters (§0 restore; the
            // rebuilds reset sqlite_sequence to MAX(id); the saved watermark
            // is never lower). UPDATE, not INSERT OR REPLACE: sqlite_sequence
            // has NO unique constraint on name, so an INSERT would append a
            // duplicate row instead of replacing (the canonical §7 form).
            // A table that was empty pre-migration has no row in seq_save
            // and none after its rebuild — consistent, nothing to restore.
            exec("UPDATE sqlite_sequence SET seq = "
                 "(SELECT s.seq FROM seq_save s WHERE s.name = sqlite_sequence.name) "
                 "WHERE name IN ('sessions','turn_summaries') "
                 "AND (SELECT s.seq FROM seq_save s WHERE s.name = sqlite_sequence.name) > seq");

            set_db_version("0.17");
            // X16 clause 5: stamped in the SAME transaction as the version, so
            // "committed 0.17" and "reindex owed" can never be observed apart.
            set_reindexing("all");

            // Only NEW violations abort; pre-existing ones are carried
            // through and warned about (fk_before baseline above).
            {
                const std::multiset<std::string> fk_after = fk_violations();
                std::vector<std::string> introduced;
                std::set_difference(fk_after.begin(), fk_after.end(),
                                    fk_before.begin(), fk_before.end(),
                                    std::back_inserter(introduced));
                if (!introduced.empty()) {
                    throw std::runtime_error(std::format(
                        "0.16 -> 0.17 migration introduced {} orphaned foreign key "
                        "row(s) (first: {}); rolling back",
                        introduced.size(), introduced.front()));
                }
                if (!fk_before.empty()) {
                    Diskerror::Logger::warn(std::format(
                        "{} pre-existing foreign-key violation(s) carried through "
                        "the 0.17 migration unchanged (first: {}). Not caused by "
                        "the migration; worth investigating separately.",
                        fk_before.size(), *fk_before.begin()));
                }
            }

            exec("COMMIT");
            exec("PRAGMA legacy_alter_table = OFF");
            if (fk_was_on) exec("PRAGMA foreign_keys = ON");
            Diskerror::Logger::info("DB migration 0.16 -> 0.17 complete.");
        } catch (...) {
            try { Stmt(db, "ROLLBACK").exec(); } catch (...) {}
            try {
                exec("PRAGMA legacy_alter_table = OFF");
                if (fk_was_on) exec("PRAGMA foreign_keys = ON");
            } catch (...) {}
            // X16 clause 10: a failed leg rolls the whole schema transaction
            // back — the file stays a valid, untouched 0.16 and the next start
            // reruns the leg (rule 8: there is nothing half-migrated to flag).
            // The operator must learn WHICH backup to restore straight from
            // the log line, so the recorded name and time are part of the
            // error; they were written before the leg and survived the
            // rollback. Surfacing this through the HTTP/agent API is not in
            // this step; the daemon exits on it (it never serves a half-open
            // DB).
            std::string cause = "unknown error";
            try { throw; } catch (const std::exception& e) { cause = e.what(); }
            throw std::runtime_error(std::format(
                "migration 0.16 -> 0.17 failed and was rolled back; the database "
                "is unchanged. Pre-migration backup: {} ({})\n{}",
                setting_or(*this, "backup_file_name", "<none recorded>"),
                setting_or(*this, "backup_timestamp", "<none recorded>"),
                cause));
        }

        // X16 clause 5: the all-table reindex runs on the COMMITTED 0.17
        // schema, outside the leg transaction, in per-batch commits (256
        // records, TextIndex::kBatch) — the same loop reindex_all() uses.
        // Deliberately outside the try/catch above: once the schema
        // transaction committed, the migration counts as complete, so a
        // failure here is NOT "a rolled-back migration" — it is an incomplete
        // index, and the `reindexing` flag (still 'all') is what reports it.
        // Consequence to state plainly: if the process dies during this
        // reindex the DB is 0.17 with a partial index and the flag set; the
        // user either runs `ragger reindex all` or swaps back the
        // pre-migration backup (the maybe_backup_before_migration archive beside
        // the DB). Never data surgery.
        Diskerror::Logger::info("  reindexing all text tables...");
        reindex_tables({});   // silent: library code owns no stdout
        set_reindexing("stopped");
        Diskerror::Logger::info("  reindex after 0.16 -> 0.17 complete.");
    }


    /// Custom terms index (v0.16) — replaces both FTS5 layers with a single
    /// flat term table + per-content-type junction tables. See
    /// ownCloud/Ragger/custom-search-schema.md for the design rationale and
    /// docs/plans/migration-plan-fts-index.md for the step-by-step build plan.
    ///
    /// The `terms` table holds literal (stemmed) AND metaphone tokens in ONE
    /// flat list with no type/flag column — they self-filter at query time
    /// because a stemmed word and its DMP code are different strings that both
    /// resolve to the same term_id. No doc_frequency column: df is computed
    /// live via COUNT(*) on the per-table *_terms join (Decision C).
    void Backend::Impl::create_terms_schema() {
        text_index_->create_schema();
    }


} // namespace ragger::sqlite
