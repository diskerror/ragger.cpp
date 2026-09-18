#include "sqlite/backend_impl.h"
#include "embedder.h"
#include "config.h"
#include "lang.h"
#include "util/fs.h"
#include "util/time.h"
#include "util/sqlite.h"
#include "double_metaphone.h"
#include "fts_tokenize.h"
#include "sqlite/text_index.h"
#include "vector_codec.h"
#include "Logger.h"
#include <format>
#include "nlohmann_json.hpp"

#include <sqlite3.h>
#include <Eigen/Dense>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unordered_map>
#include <unordered_set>
#include <iomanip>
#include <iterator>
#include <filesystem>
#include <numeric>
#include <optional>
#include <regex>
#include <set>
#include <mutex>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace ragger::sqlite {

using json = nlohmann::json;
namespace fs = std::filesystem;


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
        // Wait up to 10s on a locked DB instead of failing instantly —
        // WAL allows concurrent readers, but two writers (daemon +
        // import CLI) still serialize; without this any collision is an
        // immediate "database is locked" error.
        sqlite3_busy_timeout(db, 10000);
        text_index_.emplace(db);
        create_schema();

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
                ins.exec();
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
    bool Backend::Impl::column_exists(const std::string& table, const std::string& col) {
        Stmt s(db, "PRAGMA table_info(" + table + ")");
        while (s.step()) {
            if (s.column_text(1) == col) return true;  // col 1 = column name
        }
        return false;
    }


    bool Backend::Impl::table_exists(const std::string& table) {
        Stmt s(db, "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?");
        s.bind(1, table);
        return s.step();
    }


    void Backend::Impl::create_schema() {
        // Capture "was this DB already populated?" BEFORE any CREATE TABLE
        // IF NOT EXISTS runs below -- this is the only reliable way to tell
        // a genuinely fresh install (no memory tables at all yet) from an
        // existing pre-0.12 DB that simply never had its db_version row
        // stamped. Only a fresh install should be auto-stamped '0.12' at
        // the bottom of this function; an existing un-migrated DB must
        // fail the startup version-gate check so the user runs
        // scripts/migrate_to_db0.12.sh instead of silently being treated
        // as current.
        bool db_preexisted = table_exists("turns");

        // Capture the on-disk schema version BEFORE any CREATE/ALTER below
        // mutates the file, then snapshot the DB if an in-binary migration is
        // pending -- so the backup reflects the pristine pre-migration state.
        // db_version() is only meaningful for an existing DB (settings table
        // already present); a fresh install has no version yet and needs no
        // backup. One snapshot covers a whole migration chain (0.12 -> 0.15 ->
        // ...): it is taken here, once, before the first structural change.
        const std::string pre_migration_version =
            db_preexisted ? db_version() : std::string();
        maybe_backup_before_migration(pre_migration_version);

        // Users, settings, and session tables — credentials (for read-only
        // document access) plus web/chat session persistence. These are a
        // separate concern from the v2 memory tables; one declarative
        // definition, shared with the DB-only constructor.
        create_user_schema();

        // ---- v2 fading-memory schema (issue #33): turns / summaries /
        //      decisions / documents / models, with FTS5 (issue #49).
        //      Pre-v2 data is exported out-of-band, not migrated in place.

        // models — lookup table; turns + summaries reference it. created_at
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
        // integer id, mirroring `models`. Grouping key for session summaries.
        exec(R"(
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                guid       TEXT NOT NULL UNIQUE,
                name       TEXT,
                name_source TEXT,
                created_at INTEGER NOT NULL DEFAULT (unixepoch())
            )
        )");

        // turns (L1) — raw verbatim exchanges. embedding nullable for the
        // deferred-embedding path (partial row written, backfilled later).
        // created_at has NO DEFAULT (matches scripts/schema_db0.12.sql
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
                embedding      BLOB
            )
        )");

        // summaries (L2/L3/L4) — as of db_version 0.12, level is only ever
        // 'episode' | 'session' | 'project'; turn-level rows moved to the
        // dedicated turn_summaries table below.
        // created_at = insert/create time; updated_at = last regenerate time
        // for running rows (session/project) and span-end for episode rows.
        // Both NOT NULL (updated_at = created_at on first insert).
        exec(R"(
            CREATE TABLE IF NOT EXISTS summaries (
                summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                text       TEXT NOT NULL,
                level      TEXT NOT NULL,
                tags       TEXT NOT NULL DEFAULT '',
                session_id INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL,
                model_id   INTEGER REFERENCES models(model_id),
                created_at INTEGER NOT NULL DEFAULT (unixepoch()),
                updated_at INTEGER NOT NULL DEFAULT (unixepoch()),
                unigram_count  INTEGER NOT NULL DEFAULT 0,
                bigram_count   INTEGER NOT NULL DEFAULT 0,
                embedding_version INTEGER,
                embedding  BLOB
            )
        )");
        exec("CREATE INDEX IF NOT EXISTS idx_summaries_level      ON summaries(level)");
        // idx_summaries_created_at created after migration (see note above).

        // turn_summaries (L2) -- NEW at db_version 0.12. One row per
        // summarized turn, split out of `summaries`. turn_id is a real FK,
        // ON DELETE SET NULL: pruning the raw turn leaves the summary behind
        // with turn_id NULL rather than stranding it via a fragile
        // (session_id, created_at) match. See scripts/schema_db0.12.sql.
        exec(R"(
            CREATE TABLE IF NOT EXISTS turn_summaries (
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
                embedding_version INTEGER,
                embedding        BLOB
            )
        )");
        exec("CREATE INDEX IF NOT EXISTS idx_turn_summaries_turn_id    ON turn_summaries(turn_id)");
        exec("CREATE INDEX IF NOT EXISTS idx_turn_summaries_session_id ON turn_summaries(session_id)");
        exec("CREATE INDEX IF NOT EXISTS idx_turn_summaries_datetime   ON turn_summaries(turn_datetime)");
        // Upsert target for finalize_turn_summary: one row per turn_id. NULL
        // turn_id (a pruned turn) is excepted -- multiple such rows are fine.
        exec("CREATE UNIQUE INDEX IF NOT EXISTS idx_turn_summaries_turn_id_uniq "
             "ON turn_summaries(turn_id) WHERE turn_id IS NOT NULL");

        // In-place migration for pre-sessions databases: add the session_id
        // FK column to turns/summaries if an existing DB predates it. SQLite
        // has no ADD COLUMN IF NOT EXISTS, so guard on pragma_table_info.
        // (Fresh DBs already have the column from the CREATE above.)
        if (!column_exists("turns", "session_id"))
            exec("ALTER TABLE turns ADD COLUMN session_id "
                 "INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL");
        if (!column_exists("summaries", "session_id"))
            exec("ALTER TABLE summaries ADD COLUMN session_id "
                 "INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL");
        exec("CREATE INDEX IF NOT EXISTS idx_turns_session     ON turns(session_id)");
        exec("CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id)");

        // decisions (L6). v4 column order: keys/data, created_at, embedding, phon.
        exec(R"(
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                text        TEXT NOT NULL,
                status      TEXT NOT NULL,
                tags        TEXT NOT NULL DEFAULT '',
                created_at  INTEGER NOT NULL DEFAULT (unixepoch()),
                unigram_count  INTEGER NOT NULL DEFAULT 0,
                bigram_count   INTEGER NOT NULL DEFAULT 0,
                embedding_version INTEGER,
                embedding   BLOB
            )
        )");
        exec("CREATE INDEX IF NOT EXISTS idx_decisions_status ON decisions(status)");

        // document_sources (L5 metadata) — one row per curated document.
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

        // documents (L5) — user-curated RAG chunks; the only sharable table.
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
        if (!column_exists("summaries", "embedding_version"))
            exec("ALTER TABLE summaries ADD COLUMN embedding_version INTEGER");
        if (!column_exists("turn_summaries", "embedding_version"))
            exec("ALTER TABLE turn_summaries ADD COLUMN embedding_version INTEGER");
        if (!column_exists("decisions", "embedding_version"))
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
        if (!column_exists("summaries", "unigram_count"))
            exec("ALTER TABLE summaries ADD COLUMN unigram_count INTEGER NOT NULL DEFAULT 0");
        if (!column_exists("summaries", "bigram_count"))
            exec("ALTER TABLE summaries ADD COLUMN bigram_count INTEGER NOT NULL DEFAULT 0");
        if (!column_exists("turn_summaries", "unigram_count"))
            exec("ALTER TABLE turn_summaries ADD COLUMN unigram_count INTEGER NOT NULL DEFAULT 0");
        if (!column_exists("turn_summaries", "bigram_count"))
            exec("ALTER TABLE turn_summaries ADD COLUMN bigram_count INTEGER NOT NULL DEFAULT 0");
        if (!column_exists("decisions", "unigram_count"))
            exec("ALTER TABLE decisions ADD COLUMN unigram_count INTEGER NOT NULL DEFAULT 0");
        if (!column_exists("decisions", "bigram_count"))
            exec("ALTER TABLE decisions ADD COLUMN bigram_count INTEGER NOT NULL DEFAULT 0");
        if (!column_exists("documents", "unigram_count"))
            exec("ALTER TABLE documents ADD COLUMN unigram_count INTEGER NOT NULL DEFAULT 0");
        if (!column_exists("documents", "bigram_count"))
            exec("ALTER TABLE documents ADD COLUMN bigram_count INTEGER NOT NULL DEFAULT 0");

        // In-place migration to the normalized documents schema (0.15):
        // extract per-document metadata (path/title/year/imported_at) into
        // document_sources and slim `documents` to chunks + a document_source_id
        // FK. Gated on the OLD shape (documents.path exists), NOT the db_version
        // string -- Reid's live DB is already stamped 0.15 with the OLD shape,
        // so a version check would wrongly skip it. Idempotent: once path is
        // gone, this never runs again.
        if (column_exists("documents", "path"))
            migrate_documents_normalize();

        // Custom terms index (v0.16) — replaces both FTS5 layers with a single
        // flat term table + per-content-type junction tables. Created BEFORE
        // migrations run, so migrate_0_15_to_0_16() can populate them via
        // reindex_table(). See ownCloud/Ragger/custom-search-schema.md for
        // the design rationale and docs/plans/migration-plan-fts-index.md for
        // the step-by-step build plan.
        create_terms_schema();

        // Run any pending in-binary version migration for an existing DB
        // (e.g. the 0.12 -> 0.15 embedding version-byte split). Advances
        // db_version toward kExpectedDbVersion so the hard gate below passes.
        // No-op on a fresh install or an already-current DB; an unknown/too-old
        // version is left untouched and rejected by the gate.
        if (db_preexisted)
            run_pending_migrations(pre_migration_version);

        // Schema-version hard gate. In-binary migration for known upgrade
        // origins (0.12, 0.15) ran just above; anything still not matching
        // kExpectedDbVersion here is an unknown/too-old version and is rejected.
        // First, stamp db_version for a genuinely fresh install ONLY --
        // db_preexisted (captured before any CREATE TABLE ran, at the top
        // of this function) distinguishes "no memory tables existed yet"
        // from an existing pre-0.12 DB that simply lacks the version row.
        // An existing un-migrated DB must NOT get auto-stamped here; it
        // needs to fail the check below so the user runs
        // scripts/migrate_to_db0.12.sh instead of silently being treated
        // as current.
        if (!db_preexisted)
            exec(std::format("INSERT OR IGNORE INTO settings (key, value) VALUES ('db_version', '{}')", kExpectedDbVersion));
        {
            std::string actual = db_version();
            if (actual != kExpectedDbVersion) {
                std::cerr << std::format(
                    "Ragger: database schema version mismatch (found '{}', need '{}').\n"
                    "This binary requires a database migrated to schema {}.\n"
                    "Run scripts/migrate_to_db{}.sh against your ~/.ragger/memories.db, "
                    "then restart.\n",
                    actual.empty() ? "<none>" : actual,
                    kExpectedDbVersion, kExpectedDbVersion, kExpectedDbVersion);
                std::exit(1);
            }
        }

        // Post-migration reclaim: if a migration actually ran this session
        // (same gating condition maybe_backup_before_migration used), the
        // chain just dropped a lot of data (old FTS5 tables/triggers, phon
        // columns, etc). Reclaim that space with a plain in-place VACUUM
        // now that the DB is confirmed at kExpectedDbVersion. Skipped on a
        // fresh install or a DB that was already current (no migration).
        if (migration_pending(pre_migration_version)) {
            Diskerror::Logger::info(
                "Post-migration VACUUM starting (reclaiming space from dropped "
                "schema) -- this may take a while on a large DB.");
            exec("VACUUM");
            Diskerror::Logger::info("Post-migration VACUUM complete.");
        }

        // Indexes — (re)created here, after migration, so a table rebuild
        // (which drops the table and with it every index) can't leave them
        // missing. All IF NOT EXISTS, so fresh / migrated / already-current
        // DBs all converge to the same set.
        exec("CREATE INDEX IF NOT EXISTS idx_turns_created_at      ON turns(created_at)");
        exec("CREATE INDEX IF NOT EXISTS idx_turns_session         ON turns(session_id)");
        exec("CREATE INDEX IF NOT EXISTS idx_summaries_level       ON summaries(level)");
        exec("CREATE INDEX IF NOT EXISTS idx_summaries_created_at  ON summaries(created_at)");
        exec("CREATE INDEX IF NOT EXISTS idx_summaries_session     ON summaries(session_id)");
        exec("CREATE INDEX IF NOT EXISTS idx_decisions_status      ON decisions(status)");
        exec("CREATE INDEX IF NOT EXISTS idx_document_sources_imported_at ON document_sources(imported_at)");
        exec("CREATE INDEX IF NOT EXISTS idx_documents_document_source_id ON documents(document_source_id)");
        exec("CREATE INDEX IF NOT EXISTS idx_turn_summaries_turn_id    ON turn_summaries(turn_id)");
        exec("CREATE INDEX IF NOT EXISTS idx_turn_summaries_session_id ON turn_summaries(session_id)");
        exec("CREATE INDEX IF NOT EXISTS idx_turn_summaries_datetime   ON turn_summaries(turn_datetime)");
        exec("CREATE UNIQUE INDEX IF NOT EXISTS idx_turn_summaries_turn_id_uniq "
             "ON turn_summaries(turn_id) WHERE turn_id IS NOT NULL");

        // FTS5 — external-content virtual tables + sync triggers replace
        // the old hand-rolled bm25_* sidecars (issue #49).
        // (v0.16: FTS5 schema removed; custom terms index created below via
        // create_terms_schema().)

        // Human-readable views (datetime()-rendered timestamps,
        // has_embedding booleans) mirroring scripts/schema_db0.12.sql
        // exactly. Always (re)created so fresh/migrated DBs converge.
        create_views();
    }


    std::string Backend::Impl::db_version() {
        Stmt s(db, "SELECT value FROM settings WHERE key = 'db_version'");
        if (s.step()) return s.column_text(0);
        return "";  // absent
    }


    void Backend::Impl::set_db_version(const std::string& v) {
        Stmt s(db,
            "INSERT INTO settings (key, value) VALUES ('db_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value");
        s.bind(1, v);
        s.exec();
    }


    // ---- in-binary version migrations ------------------------------------
    // Historically, DB schema upgrades were external shell scripts
    // (scripts/migrate_to_db*.sh) and the binary merely hard-gated on the
    // version. To make GitHub-cloned upgrades painless, the known upgrade
    // legs now run in-process on open, chained until the DB reaches
    // kExpectedDbVersion. Each leg mirrors migrate_documents_normalize()'s
    // shape: one BEGIN/try/COMMIT/ROLLBACK transaction, Logger progress,
    // and post-migration verification that aborts (rolls back) on any
    // inconsistency rather than silently shipping a corrupt index.

    /// True iff a pre-migration backup/reopen (and, later, a post-migration
    /// VACUUM) should happen for `current_version`. Mirrors the guard
    /// conditions maybe_backup_before_migration() has always used: shared
    /// by that function and by the post-migration VACUUM decision so both
    /// stay in lockstep.
    bool Backend::Impl::migration_pending(const std::string& current_version) {
        if (current_version.empty()) return false;               // fresh/pre-version DB
        if (current_version == kExpectedDbVersion) return false;  // already current
        // Only for versions we actually migrate FROM in-binary. An unknown
        // version is rejected by the gate without mutation.
        if (current_version != "0.12" && current_version != "0.15") return false;
        return true;
    }


    /// Archive the raw DB file(s) as "<db>_BACKUP_<timestamp>.tar.gz" (or
    /// .zip / .db, see fallback chain below) before the first structural
    /// change of a migration run. Only fires when an in-binary migration is
    /// actually pending (see migration_pending()). One snapshot covers a
    /// whole chain (0.12 -> 0.15 -> ...).
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
    void Backend::Impl::maybe_backup_before_migration(const std::string& current_version) {
        if (!migration_pending(current_version)) return;

        Diskerror::Logger::info(std::format(
            "DB schema {} predates {} -- taking pre-migration backup",
            current_version, std::string(kExpectedDbVersion)));

        // ---- 1. WAL-checkpoint (TRUNCATE) -- try without closing first ---
        // "PRAGMA wal_checkpoint(TRUNCATE);" returns one row: (busy, log,
        // checkpointed). busy != 0 means the checkpoint could not complete
        // fully (some frames left un-checkpointed) -- typically because
        // another connection is mid-transaction. busy == 0 means the main
        // .db file now fully reflects the database and the -wal file was
        // truncated to empty: safe to archive while staying open.
        bool checkpoint_complete = false;
        {
            Stmt s(db, "PRAGMA wal_checkpoint(TRUNCATE)");
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
            int rc = sqlite3_close(db);
            if (rc != SQLITE_OK) {
                throw std::runtime_error(std::format(
                    "pre-migration backup: sqlite3_close failed ({}); refusing to "
                    "archive a DB with abandoned open state", sqlite3_errstr(rc)));
            }
            db = nullptr;
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
                db_path, {config().resolved_stats_db_path()});
        } catch (const std::exception& e) {
            // If we closed for the backup, reopen before throwing so we
            // don't leave the backend in a permanently-closed state, then
            // surface the failure.
            if (closed_for_backup) reopen_db();
            throw std::runtime_error(
                std::string("pre-migration backup failed: ") + e.what());
        }

        Diskerror::Logger::info(std::format(
            "Pre-migration backup complete: {}", archive_path));

        // ---- 3. Reopen the connection (only if we closed it) --------------
        if (closed_for_backup) {
            reopen_db();
            Diskerror::Logger::info(
                "Reconnected to DB after close+archive pre-migration backup.");
        }
    }


    /// Reopen `db` (after maybe_backup_before_migration's checkpoint+close)
    /// with the exact same open flags/pragmas used in the constructor, and
    /// re-point the non-owning text_index_ at the new handle -- sqlite3_open
    /// is not guaranteed to reuse the same pointer value.
    void Backend::Impl::reopen_db() {
        int rc = sqlite3_open(db_path.c_str(), &db);
        if (rc != SQLITE_OK) {
            std::string err = db ? sqlite3_errmsg(db) : sqlite3_errstr(rc);
            if (db) { sqlite3_close(db); db = nullptr; }
            throw std::runtime_error(std::format(lang::ERR_SQLITE_OPEN, err));
        }
        exec("PRAGMA journal_mode=WAL");
        exec("PRAGMA foreign_keys = ON");
        sqlite3_busy_timeout(db, 10000);

        // text_index_ holds a NON-OWNING sqlite3* captured at construction;
        // re-emplace it against the new handle. Cheap (no allocation, just a
        // pointer member) and safe (no other state in TextIndex).
        text_index_.reset();
        text_index_.emplace(db);
    }


    /// Chain the in-binary migration legs until the DB reaches
    /// kExpectedDbVersion (or a leg we don't know about is hit -> leave it for
    /// the hard gate to reject). `from_version` is the on-disk version captured
    /// before create_schema() mutated anything. Each leg advances db_version;
    /// the loop re-reads it so 0.12 flows 0.12 -> 0.15 -> (0.16 once its leg
    /// lands) under the single backup taken above.
    void Backend::Impl::run_pending_migrations(const std::string& from_version) {
        std::string v = from_version;
        // Guard against an unbounded loop if a leg ever fails to advance the
        // version. Legs are few; a handful of iterations is plenty.
        for (int guard = 0; guard < 8; ++guard) {
            if (v.empty() || v == kExpectedDbVersion) return;
            if (v == "0.12") {
                migrate_0_12_to_0_15();
                v = db_version();
                continue;
            }
            if (v == "0.15") {
                migrate_0_15_to_0_16();
                v = db_version();
                continue;
            }
            // No known leg from this version -> stop; the hard gate reports it.
            return;
        }
    }


    /// 0.12 -> 0.15 migration (in-binary port of scripts/migrate_to_db0.15.sh).
    ///
    /// By the time this runs, create_schema() has ALREADY performed the
    /// structural half of the 0.15 upgrade on this existing DB:
    ///   - added embedding_version INTEGER to all five embedded tables
    ///     (ALTER ADD COLUMN guards, above), and
    ///   - normalized documents into document_sources via
    ///     migrate_documents_normalize(), which also CLEARS documents'
    ///     embedding/embedding_version/phon (they get re-embedded with the
    ///     title appended).
    ///
    /// The remaining data step is the embedding version-byte SPLIT for the
    /// four tables migrate_documents_normalize did NOT wipe: turns,
    /// turn_summaries, summaries, decisions. Pre-0.15 blobs stored a 1-byte
    /// version tag at byte 0 (EmbeddingCodec offset=1); 0.15 blobs are payload
    /// only (offset=0) with the tag living in the embedding_version column. So
    /// for every row with a non-NULL embedding: embedding_version = blob[0],
    /// embedding = blob[1:]. NULL embeddings stay NULL in both columns.
    ///
    /// Whole thing runs in one transaction. On any verification failure the
    /// transaction is rolled back (original rows untouched) and the throw
    /// propagates -- the pre-migration backup made above is the safety net.
    void Backend::Impl::migrate_0_12_to_0_15() {
        Diskerror::Logger::info(
            "Migrating DB 0.12 -> 0.15 (embedding version-byte split)...");

        // (table, primary-key column) for the four tables whose embeddings
        // survive into 0.15 and therefore need the byte split. documents is
        // intentionally excluded: migrate_documents_normalize() already
        // cleared its embeddings.
        struct T { const char* table; const char* pk; };
        const T tables[] = {
            {"turns",          "turn_id"},
            {"turn_summaries", "turn_summary_id"},
            {"summaries",      "summary_id"},
            {"decisions",      "decision_id"},
        };

        Stmt(db, "BEGIN").exec();
        try {
            for (const auto& t : tables) {
                // Read each row's blob, split off byte 0, write both columns
                // back. Collect first, then write, so we never mutate a table
                // mid-scan.
                struct Row { int64_t id; int version; std::vector<uint8_t> payload; };
                std::vector<Row> rows;
                {
                    Stmt s(db, std::format(
                        "SELECT {}, embedding FROM {} WHERE embedding IS NOT NULL",
                        t.pk, t.table));
                    while (s.step_checked()) {
                        const auto* blob =
                            static_cast<const uint8_t*>(s.column_blob(1));
                        int n = s.column_bytes(1);
                        if (blob == nullptr || n < 1)
                            throw std::runtime_error(std::format(
                                "{}: non-NULL embedding with <1 byte (id={})",
                                t.table, s.column_int64(0)));
                        Row r;
                        r.id = s.column_int64(0);
                        r.version = blob[0];
                        r.payload.assign(blob + 1, blob + n);   // bytes [1, n)
                        rows.push_back(std::move(r));
                    }
                }
                for (const auto& r : rows) {
                    Stmt u(db, std::format(
                        "UPDATE {} SET embedding_version = ?, embedding = ? "
                        "WHERE {} = ?", t.table, t.pk));
                    u.bind(1, r.version);
                    u.bind_blob(2, r.payload.data(),
                                static_cast<int>(r.payload.size()));
                    u.bind(3, r.id);
                    if (!u.exec())
                        throw std::runtime_error(std::format(
                            "{}: embedding split UPDATE failed (id={}): {}",
                            t.table, r.id, sqlite3_errmsg(db)));
                }
                Diskerror::Logger::info(std::format(
                    "  {}: split {} embedding(s)", t.table, rows.size()));
            }

            // ---- verification (abort/rollback on any mismatch) ------------
            // For each migrated table: embedding NULL-ness and
            // embedding_version NULL-ness must move together. (The blob is now
            // the pure payload; we already stripped exactly one byte in C++,
            // so a per-row "1 byte shorter" DB check isn't reconstructable
            // post-split -- the correctness guarantee is the byte-exact
            // std::vector assign above plus this NULL-parity check.)
            for (const auto& t : tables) {
                Stmt s(db, std::format(
                    "SELECT COUNT(*) FROM {} "
                    "WHERE (embedding IS NULL) != (embedding_version IS NULL)",
                    t.table));
                s.step_checked();
                int64_t bad = s.column_int64(0);
                if (bad != 0)
                    throw std::runtime_error(std::format(
                        "{}: {} row(s) with embedding/embedding_version NULL "
                        "mismatch after split", t.table, bad));
            }

            // Structural integrity of the whole file.
            {
                Stmt s(db, "PRAGMA integrity_check");
                s.step_checked();
                std::string res = s.column_text(0);
                if (res != "ok")
                    throw std::runtime_error(
                        "PRAGMA integrity_check failed after 0.12->0.15: " + res);
            }

            set_db_version("0.15");
            Stmt(db, "COMMIT").exec();
            Diskerror::Logger::info("DB migration 0.12 -> 0.15 complete.");
        } catch (...) {
            Stmt(db, "ROLLBACK").exec();
            throw;
        }
    }



    /// One-time in-place migration to the normalized documents schema (0.15).
    /// Extracts per-document metadata (path/title/year/imported_at) from the
    /// old flat `documents` rows into `document_sources`, converts imported_at
    /// text -> unix epoch (stored text is a LOCAL wall-clock timestamp), points
    /// each chunk at its source via document_source_id, redistributes tags
    /// (source = intersection of a (path,title) group's chunk tags; chunk keeps
    /// its own set minus that intersection), clears embedding/embedding_version/
    /// phon on every doc row (so housekeeping re-embeds with title appended),
    /// then drops the four extracted columns. Whole thing runs in one
    /// transaction. Caller gates on column_exists("documents","path").
    void Backend::Impl::migrate_documents_normalize() {
        Diskerror::Logger::info("Migrating documents to normalized schema (document_sources)...");
        Stmt(db, "BEGIN").exec();
        try {
            // document_sources may not exist yet on an old DB (the CREATE TABLE
            // IF NOT EXISTS above makes it, but be defensive/idempotent).
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

            // The old documents_fts / triggers reference new.title/old.title.
            // They MUST go before we UPDATE documents or DROP COLUMN title,
            // else the AFTER UPDATE trigger fires against a missing column.
            exec("DROP TRIGGER IF EXISTS documents_ai");
            exec("DROP TRIGGER IF EXISTS documents_ad");
            exec("DROP TRIGGER IF EXISTS documents_au");
            exec("DROP TABLE IF EXISTS documents_fts");

            // Same for the phon sidecar FTS + its triggers. Critically, the
            // external-content documents_phon_fts must be dropped too: after we
            // rebuild `documents` below, its shadow tables would still hold the
            // OLD rows. The row COUNT happens to match (same 8627), so
            // create_phon_fts_schema()'s docsize-vs-base desync probe would NOT
            // fire a rebuild -- leaving stale rowid->content pointers that make
            // every later phon UPDATE fail "database disk image is malformed".
            // Dropping it here forces a fresh, empty index (docsize 0 != base)
            // so the probe rebuilds and resyncs it.
            exec("DROP TRIGGER IF EXISTS documents_pai");
            exec("DROP TRIGGER IF EXISTS documents_pad");
            exec("DROP TRIGGER IF EXISTS documents_pau");
            exec("DROP TABLE IF EXISTS documents_phon_fts");

            // The old documents_view SELECTs path/title/year/imported_at, so
            // SQLite won't let us DROP those columns while it exists. Drop it
            // here; create_views() rebuilds the new (joined) view afterward.
            exec("DROP VIEW IF EXISTS documents_view");

            // The old idx_documents_imported_at indexes a column we're about to
            // drop -- SQLite refuses DROP COLUMN while an index references it.
            exec("DROP INDEX IF EXISTS idx_documents_imported_at");

            // Helper: split a comma-separated tag string into a token set
            // (trimmed, empties dropped).
            auto split_tags = [](const std::string& s) {
                std::set<std::string> out;
                std::string tok;
                std::stringstream ss(s);
                while (std::getline(ss, tok, ',')) {
                    size_t a = tok.find_first_not_of(" \t");
                    size_t b = tok.find_last_not_of(" \t");
                    if (a != std::string::npos)
                        out.insert(tok.substr(a, b - a + 1));
                }
                return out;
            };
            auto join_tags = [](const std::set<std::string>& s) {
                std::string out;
                for (const auto& t : s) { if (!out.empty()) out += ','; out += t; }
                return out;
            };

            // Group existing chunks by (path, title). Collect each group's
            // year, earliest imported_at (as epoch), the intersection of its
            // chunks' tag sets, and the lowest document_id it appears in (so
            // sources can be inserted in document_id order -- tidy + stable).
            struct Group {
                std::string path, title;
                std::optional<long long> year;
                long long imported_epoch = 0;
                bool have_epoch = false;
                std::set<std::string> tag_intersection;
                bool first_tag = true;
                long long first_doc_id = 0;
            };
            // key = path + '\x1f' + title  (unit separator can't appear in text)
            std::unordered_map<std::string, Group> groups;

            {
                // strftime('%s', ts, 'utc') interprets the stored text as LOCAL
                // wall-clock and returns the corresponding unix epoch. (SQLite's
                // bare strftime treats input as UTC; the 'utc' modifier flips a
                // localtime reading back to UTC epoch.) Handles both the live
                // 'YYYY-MM-DD HH:MM:SS' form and 'YYYYMMDD' stragglers (the
                // latter parses as midnight local).
                Stmt s(db,
                    "SELECT document_id, path, title, year, tags, "
                    "  CAST(strftime('%s', imported_at, 'utc') AS INTEGER) AS ep "
                    "FROM documents ORDER BY document_id");
                while (s.step()) {
                    long long doc_id = s.column_int64(0);
                    std::string path  = s.is_null(1) ? "" : s.column_text(1);
                    std::string title = s.is_null(2) ? "" : s.column_text(2);
                    std::string key = path + "\x1f" + title;
                    auto it = groups.find(key);
                    bool is_new = (it == groups.end());
                    Group& g = groups[key];
                    if (is_new) { g.first_doc_id = doc_id; g.path = path; g.title = title; }
                    if (!s.is_null(3)) g.year = s.column_int64(3);
                    if (!s.is_null(5)) {
                        long long ep = s.column_int64(5);
                        if (!g.have_epoch || ep < g.imported_epoch) {
                            g.imported_epoch = ep; g.have_epoch = true;
                        }
                    }
                    std::set<std::string> ts = split_tags(s.is_null(4) ? "" : s.column_text(4));
                    if (g.first_tag) { g.tag_intersection = ts; g.first_tag = false; }
                    else {
                        std::set<std::string> inter;
                        std::set_intersection(g.tag_intersection.begin(), g.tag_intersection.end(),
                                              ts.begin(), ts.end(),
                                              std::inserter(inter, inter.begin()));
                        g.tag_intersection = std::move(inter);
                    }
                }
            }

            // Insert sources in document_id order (tidy IDs), remember each id.
            std::vector<std::pair<std::string, Group*>> ordered;
            ordered.reserve(groups.size());
            for (auto& [key, g] : groups) ordered.emplace_back(key, &g);
            std::sort(ordered.begin(), ordered.end(),
                      [](const auto& a, const auto& b) {
                          return a.second->first_doc_id < b.second->first_doc_id;
                      });

            std::unordered_map<std::string, long long> group_source_id;
            for (auto& [key, gp] : ordered) {
                Group& g = *gp;
                Stmt ins(db,
                    "INSERT INTO document_sources(title, path, year, tags, imported_at) "
                    "VALUES (?,?,?,?,?)");
                if (g.title.empty()) ins.bind_null(1); else ins.bind(1, g.title);
                if (g.path.empty())  ins.bind_null(2); else ins.bind(2, g.path);
                if (g.year) ins.bind(3, static_cast<int>(*g.year)); else ins.bind_null(3);
                ins.bind(4, join_tags(g.tag_intersection));
                // No usable timestamp anywhere in the group -> use "now".
                ins.bind(5, g.have_epoch ? g.imported_epoch
                                         : static_cast<long long>(std::time(nullptr)));
                ins.step();
                group_source_id[key] = sqlite3_last_insert_rowid(db);
            }

            // Rebuild `documents` with the FINAL column order (ALTER ADD COLUMN
            // would append, diverging from a fresh install's layout). Populate
            // it from the old table, computing per-chunk tags (own set MINUS the
            // source intersection) in C++, pointing at the source, stamping
            // modified_on = source epoch, and CLEARING embedding (it gets
            // rebuilt with the title appended -- see store_document). The four
            // extracted columns are simply not carried over. `phon` is retired
            // (v0.16) so it is never (re)created here, even for very old
            // pre-normalize DBs going through this one-time rebuild.
            exec(R"(
                CREATE TABLE documents_new (
                    document_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    text               TEXT NOT NULL,
                    tags               TEXT NOT NULL DEFAULT '',
                    chunk_index        INTEGER,
                    document_source_id INTEGER REFERENCES document_sources(document_source_id),
                    modified_on        INTEGER,
                    embedding_version  INTEGER,
                    embedding          BLOB
                )
            )");
            {
                Stmt s(db, "SELECT document_id, text, tags, path, title, chunk_index "
                           "FROM documents ORDER BY document_id");
                struct Row { long long id; std::string text, tags; bool tags_null;
                             long long src_id, modified; std::optional<long long> chunk_index; };
                std::vector<Row> rows;
                while (s.step()) {
                    long long id = s.column_int64(0);
                    std::string text = s.column_text(1);
                    std::string path  = s.is_null(3) ? "" : s.column_text(3);
                    std::string title = s.is_null(4) ? "" : s.column_text(4);
                    std::string key = path + "\x1f" + title;
                    const Group& g = groups[key];
                    std::set<std::string> own = split_tags(s.is_null(2) ? "" : s.column_text(2));
                    std::set<std::string> remainder;
                    std::set_difference(own.begin(), own.end(),
                                        g.tag_intersection.begin(), g.tag_intersection.end(),
                                        std::inserter(remainder, remainder.begin()));
                    Row r;
                    r.id = id; r.text = std::move(text);
                    r.tags = join_tags(remainder); r.tags_null = false;
                    r.src_id = group_source_id[key];
                    r.modified = g.have_epoch ? g.imported_epoch
                                              : static_cast<long long>(std::time(nullptr));
                    if (!s.is_null(5)) r.chunk_index = s.column_int64(5);
                    rows.push_back(std::move(r));
                }
                for (const auto& r : rows) {
                    Stmt ins(db,
                        "INSERT INTO documents_new "
                        "(document_id, text, tags, chunk_index, document_source_id, "
                        " modified_on, embedding_version, embedding) "
                        "VALUES (?,?,?,?,?,?,NULL,NULL)");
                    ins.bind(1, r.id).bind(2, r.text).bind(3, r.tags);
                    if (r.chunk_index) ins.bind(4, static_cast<int>(*r.chunk_index));
                    else ins.bind_null(4);
                    ins.bind(5, r.src_id).bind(6, r.modified);
                    ins.step();
                }
            }
            exec("DROP TABLE documents");
            exec("ALTER TABLE documents_new RENAME TO documents");

            Stmt(db, "COMMIT").exec();
            Diskerror::Logger::info(std::format(
                "documents normalized: {} source(s) extracted", groups.size()));
        } catch (...) {
            Stmt(db, "ROLLBACK").exec();
            throw;
        }
        // documents_fts is rebuilt (new shape) by create_fts_schema() below.
    }


    /// Read a boolean PRAGMA's current value (e.g. "foreign_keys").
    bool Backend::Impl::pragma_bool(const char* name) {
        Stmt s(db, std::format("PRAGMA {}", name));
        return s.step() && s.column_int(0) != 0;
    }


    /// Rebuild one table so its PHYSICAL column order matches the fresh-install
    /// CREATE TABLE in create_schema(). `ALTER TABLE ... ADD COLUMN` can only
    /// APPEND, so a DB that reached its current shape via migration ends up with
    /// (e.g.) unigram_count/bigram_count sitting after the embedding BLOB, while
    /// a fresh DB has them before it. That divergence is not just cosmetic: rows
    /// are wide, and a trailing multi-KB embedding pushes the small, human-
    /// meaningful columns off-screen in any ad-hoc `SELECT *` / `.dump` /
    /// sqlite3 CLI inspection. Keeping the narrow, readable columns to the LEFT
    /// of the blobs is the whole point of the fixed order.
    ///
    /// Standard SQLite table-rebuild dance (see "Making Other Kinds Of Table
    /// Schema Changes" in the SQLite docs):
    ///   1. snapshot this table's index DDL (DROP TABLE takes its indexes with it)
    ///   2. CREATE <t>_rebuild with the canonical column order
    ///   3. INSERT ... SELECT the named columns (order-independent, by name)
    ///   4. DROP old, RENAME new into place
    ///   5. replay the saved index DDL
    ///
    /// CALLER CONTRACT: foreign_keys MUST already be OFF and any dependent views
    /// already dropped. `PRAGMA foreign_keys` is a NO-OP inside a transaction, so
    /// it has to be set before BEGIN -- see migrate_0_15_to_0_16().
    ///
    /// `cols` is the shared column list present in BOTH old and new tables. The
    /// count columns are deliberately NOT carried over (they are 0 on a
    /// just-ALTERed DB anyway); reindex_table() populates them right after.
    void Backend::Impl::rebuild_table_column_order(const std::string& table,
                                    const std::string& new_ddl,
                                    const std::string& cols) {
        // 1. Snapshot index DDL. Skip sql IS NULL rows: those are the implicit
        //    indexes SQLite auto-creates for UNIQUE/PRIMARY KEY constraints,
        //    which the new CREATE TABLE already re-declares for itself.
        std::vector<std::string> index_ddl;
        {
            Stmt s(db, "SELECT sql FROM sqlite_master WHERE type='index' "
                       "AND tbl_name=? AND sql IS NOT NULL");
            s.bind(1, table);
            while (s.step()) index_ddl.push_back(s.column_text(0));
        }

        const std::string tmp = table + "_rebuild";
        exec(std::format("DROP TABLE IF EXISTS {}", tmp));
        exec(new_ddl);
        exec(std::format("INSERT INTO {} ({}) SELECT {} FROM {}",
                         tmp, cols, cols, table));
        exec(std::format("DROP TABLE {}", table));
        exec(std::format("ALTER TABLE {} RENAME TO {}", tmp, table));

        for (const auto& ddl : index_ddl) exec(ddl);

        Diskerror::Logger::info(
            std::format("  {}: rebuilt with canonical column order", table));
    }


    /// 0.15 -> 0.16 migration (FTS5 teardown + custom index backfill).
    ///
    /// This is the big cutover:
    /// 1. Drop all FTS5 triggers (both text and phonetic)
    /// 2. Drop both FTS5 virtual tables
    /// 3. Drop the `phon` column from all 5 text tables
    /// 4. Call reindex_table() for each of the 5 tables to populate the custom index
    /// 5. Update db_version to 0.16
    ///
    /// The schema (terms, *_terms, count columns) is created by create_schema()
    /// BEFORE this runs, and reindex_table() (Step 6) populates them.
    void Backend::Impl::migrate_0_15_to_0_16() {
        Diskerror::Logger::info("Migrating DB 0.15 -> 0.16 (custom FTS index)...");

        // Both pragmas MUST be set OUTSIDE the transaction:
        //
        // foreign_keys: a no-op inside a transaction, and it has to be OFF for
        // the table rebuilds below. The *_terms junction tables hold FKs into
        // these five tables with ON DELETE CASCADE -- with enforcement on, the
        // `DROP TABLE <t>` step would cascade away every index row we are about
        // to rebuild (and the parent-key checks would fire mid-swap).
        //
        // legacy_alter_table: with the modern (OFF) behavior, `ALTER TABLE
        // <t>_rebuild RENAME TO <t>` helpfully rewrites REFERENCES clauses in
        // OTHER tables that point at the renamed table -- which would silently
        // repoint every junction table's FK at the transient "<t>_rebuild" name.
        // ON keeps RENAME purely local, which is what the rebuild dance needs.
        const bool fk_was_on = pragma_bool("foreign_keys");
        exec("PRAGMA foreign_keys = OFF");
        exec("PRAGMA legacy_alter_table = ON");

        try {
            Stmt(db, "BEGIN").exec();

            // Repair dangling session references BEFORE baselining FK state.
            // These columns are declared ON DELETE SET NULL, so a NULL here is
            // precisely what the schema intends when a session goes away -- the
            // clause simply never fired (the session was deleted while FK
            // enforcement was off). The summary/turn is the valuable record; the
            // session pointer is not. Null the stale pointer and KEEP the row,
            // rather than deleting content to satisfy a constraint.
            for (const char* t : {"summaries", "turns", "turn_summaries"}) {
                const std::string sql = std::format(
                    "UPDATE {} SET session_id = NULL WHERE session_id IS NOT NULL "
                    "AND session_id NOT IN (SELECT session_id FROM sessions)", t);
                exec(sql);
                if (int n = sqlite3_changes(db); n > 0) {
                    Diskerror::Logger::info(std::format(
                        "  {}: cleared {} dangling session_id reference(s) "
                        "(ON DELETE SET NULL never fired; rows preserved)", t, n));
                }
            }

            // Baseline the FK violations that ALREADY exist in this DB, so the
            // post-rebuild check can flag only NEW orphans. Anything still here
            // after the repair above is a pre-existing data issue unrelated to
            // this migration and must not abort an otherwise-correct upgrade.
            auto fk_violations = [this]() {
                std::multiset<std::string> out;
                Stmt s(db, "PRAGMA foreign_key_check");
                while (s.step()) {
                    out.insert(std::format("{}|{}|{}", s.column_text(0),
                                           s.column_int64(1), s.column_text(2)));
                }
                return out;
            };
            const std::multiset<std::string> fk_before = fk_violations();

            // Drop all FTS5 triggers for the 5 text tables
            // Triggers: <table>_ai/_ad/_au (text) and <table>_pai/_pad/_pau (phon)
            const char* tables[] = {"turns", "turn_summaries", "summaries", "decisions", "documents"};
            const char* triggers_per_table[] = {"ai", "ad", "au", "pai", "pad", "pau"};

            for (const auto* table : tables) {
                for (const auto* tri : triggers_per_table) {
                    std::string trigger_name = std::string(table) + "_" + tri;
                    try {
                        exec(std::format("DROP TRIGGER IF EXISTS {}", trigger_name));
                    } catch (...) {
                        // Trigger may not exist; ignore
                    }
                }
            }

            // Drop FTS5 virtual tables (auto-drops shadow tables)
            for (const auto* table : tables) {
                try {
                    exec(std::format("DROP TABLE IF EXISTS {}_fts", table));
                } catch (...) {}
                try {
                    exec(std::format("DROP TABLE IF EXISTS {}_phon_fts", table));
                } catch (...) {}
            }

            // Drop the `phon` column from all 5 tables (SQLite >= 3.35 required)
            // Safe to check and skip if not supported. The *_view views (created
            // by an earlier create_views() call, possibly from a pre-fix binary)
            // may still SELECT the `phon` column -- SQLite refuses DROP COLUMN
            // while a view references it. Drop them first; create_views() (called
            // after migrations complete, CREATE VIEW IF NOT EXISTS) recreates the
            // now-phon-free versions.
            for (const auto* table : tables) {
                exec(std::format("DROP VIEW IF EXISTS {}_view", table));
            }

            // Full table rebuild rather than `ALTER TABLE ... DROP COLUMN phon`.
            // DROP COLUMN would retire `phon` but leave unigram_count/bigram_count
            // stranded AFTER the embedding BLOB (create_schema()'s ADD COLUMN
            // guards can only append), so a migrated DB's physical layout would
            // permanently differ from a fresh install's. Rebuilding does both jobs
            // at once: `phon` is dropped simply by not being carried over, and the
            // remaining columns land in the canonical fresh-install order with the
            // narrow, readable ones ahead of the wide embedding blob.
            //
            // The DDL below MUST stay in lockstep with create_schema()'s CREATE
            // TABLE statements for these five tables.
            Diskerror::Logger::info("Rebuilding text tables in canonical column order...");
            rebuild_table_column_order("turns", R"(
                CREATE TABLE turns_rebuild (
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
                ))",
                "turn_id, user_text, assistant_text, model_id, session_id, "
                "created_at, embedding_version, embedding");

            rebuild_table_column_order("summaries", R"(
                CREATE TABLE summaries_rebuild (
                    summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text       TEXT NOT NULL,
                    level      TEXT NOT NULL,
                    tags       TEXT NOT NULL DEFAULT '',
                    session_id INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL,
                    model_id   INTEGER REFERENCES models(model_id),
                    created_at INTEGER NOT NULL DEFAULT (unixepoch()),
                    updated_at INTEGER NOT NULL DEFAULT (unixepoch()),
                    unigram_count  INTEGER NOT NULL DEFAULT 0,
                    bigram_count   INTEGER NOT NULL DEFAULT 0,
                    embedding_version INTEGER,
                    embedding  BLOB
                ))",
                "summary_id, text, level, tags, session_id, model_id, "
                "created_at, updated_at, embedding_version, embedding");

            rebuild_table_column_order("turn_summaries", R"(
                CREATE TABLE turn_summaries_rebuild (
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
                    embedding_version INTEGER,
                    embedding        BLOB
                ))",
                "turn_summary_id, text, turn_id, session_id, turn_model_id, "
                "summary_model_id, turn_datetime, summarized_on, "
                "embedding_version, embedding");

            rebuild_table_column_order("decisions", R"(
                CREATE TABLE decisions_rebuild (
                    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text        TEXT NOT NULL,
                    status      TEXT NOT NULL,
                    tags        TEXT NOT NULL DEFAULT '',
                    created_at  INTEGER NOT NULL DEFAULT (unixepoch()),
                    unigram_count  INTEGER NOT NULL DEFAULT 0,
                    bigram_count   INTEGER NOT NULL DEFAULT 0,
                    embedding_version INTEGER,
                    embedding   BLOB
                ))",
                "decision_id, text, status, tags, created_at, "
                "embedding_version, embedding");

            rebuild_table_column_order("documents", R"(
                CREATE TABLE documents_rebuild (
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
                ))",
                "document_id, text, tags, chunk_index, document_source_id, "
                "modified_on, embedding_version, embedding");

            // Now backfill the custom FTS index for each table
            Diskerror::Logger::info("Populating custom FTS index...");
            for (const auto* table : tables) {
                Diskerror::Logger::info(std::format("  Indexing {}...", table));
                // Call reindex_table directly (we're in the Impl class)
                reindex_table(table, /*progress=*/false);
            }

            // Update db_version to 0.16
            set_db_version("0.16");

            // With FK enforcement suspended for the rebuilds, verify we did not
            // orphan anything NEW before making it permanent. Pre-existing
            // violations (captured in fk_before) are logged and carried through
            // unchanged -- they are a data problem, not a migration failure.
            {
                const std::multiset<std::string> fk_after = fk_violations();
                std::vector<std::string> introduced;
                std::set_difference(fk_after.begin(), fk_after.end(),
                                    fk_before.begin(), fk_before.end(),
                                    std::back_inserter(introduced));
                if (!introduced.empty()) {
                    throw std::runtime_error(std::format(
                        "0.15 -> 0.16 migration introduced {} orphaned foreign key "
                        "row(s) (first: {}); rolling back",
                        introduced.size(), introduced.front()));
                }
                if (!fk_before.empty()) {
                    Diskerror::Logger::warn(std::format(
                        "{} pre-existing foreign-key violation(s) carried through "
                        "the 0.16 migration unchanged (first: {}). Not caused by "
                        "the migration; worth investigating separately.",
                        fk_before.size(), *fk_before.begin()));
                }
            }

            Stmt(db, "COMMIT").exec();
            exec("PRAGMA legacy_alter_table = OFF");
            if (fk_was_on) exec("PRAGMA foreign_keys = ON");
            Diskerror::Logger::info("DB migration 0.15 -> 0.16 complete.");
        } catch (...) {
            try {
                Stmt(db, "ROLLBACK").exec();
            } catch (...) {}
            try {
                exec("PRAGMA legacy_alter_table = OFF");
                if (fk_was_on) exec("PRAGMA foreign_keys = ON");
            } catch (...) {}
            throw;
        }
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
