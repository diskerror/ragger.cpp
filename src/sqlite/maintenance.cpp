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


    // PHASE8 semantic shift: the old count() counted summaries only; notes
    // now live in `knowledge`, so count() is every knowledge row (decisions,
    // insights, notes). Same value table_row_counts() reports for knowledge.
    int Backend::Impl::count() const {
        Stmt s(db, "SELECT COUNT(*) FROM knowledge");
        int c = 0;
        if (s.step())
            c = s.column_int(0);
        return c;
    }


    std::vector<std::pair<std::string, int64_t>> Backend::Impl::table_row_counts() const {
        // User-facing tables for the dashboard status pane, in display order.
        static const char* kTables[] = {
            "turns", "turn_summaries", "episodes", "knowledge", "sessions",
            "documents", "models",
        };
        std::vector<std::pair<std::string, int64_t>> out;
        for (const char* t : kTables) {
            try {
                Stmt s(db, std::string("SELECT COUNT(*) FROM ") + t);
                if (s.step())
                    out.emplace_back(t, static_cast<int64_t>(s.column_int(0)));
            } catch (...) {
                // Table absent in this schema version — just skip it.
            }
        }
        return out;
    }


    // Total rows across the registry's embedded tables — i.e. how many rows
    // `rebuild_embeddings()` will re-encode. (count() alone is just
    // summaries, which understates the rebuild scope.)
    int Backend::Impl::count_embeddable_rows(const std::string& table) const {
        int total = 0;
        for (const auto& t : kTextTables) {
            if (table != "all" && table != t.table) continue;
            Stmt s(db, std::format("SELECT COUNT(*) FROM {}", t.table));
            if (s.step()) total += s.column_int(0);
        }
        return total;
    }


    // True if any embedded table holds a non-NULL embedding. EXISTS short-
    // circuits on the first hit. Deferred (NULL-embedding) rows don't count —
    // they get the current model on backfill, so they aren't incompatible.
    bool Backend::Impl::has_embeddings() const {
        std::string sql = "SELECT EXISTS(";
        bool first = true;
        for (const auto& t : kTextTables) {
            sql += first ? "SELECT 1 FROM " : " UNION ALL SELECT 1 FROM ";
            sql += t.table;
            sql += " WHERE embedding IS NOT NULL";
            first = false;
        }
        sql += " LIMIT 1)";
        Stmt s(db, sql);
        return s.step() && s.column_int(0) != 0;
    }


    // Lean v2 summaries have no collection column; the `collection` argument
    // is ignored (kept for API compat). PHASE8: returns every knowledge row
    // (kind/status/tags in metadata), score 0.
    std::vector<SearchResult> Backend::Impl::load_all(const std::string& /*collection*/) {
        std::vector<SearchResult> results;
        Stmt s(db,
            "SELECT knowledge_id, text, kind, status, tags, created_at "
            "FROM knowledge ORDER BY knowledge_id");

        while (s.step()) {
            int id        = s.column_int(0);
            auto text     = s.column_text(1);
            auto kind     = s.column_text(2);
            auto status   = s.column_text(3);
            auto tag      = s.column_text(4);
            auto ts       = db_timestamp(static_cast<std::time_t>(s.column_int64(5)));

            json meta = json::object();
            if (!kind.empty())   meta["kind"]   = kind;
            if (!status.empty()) meta["status"] = status;
            if (!tag.empty())    meta["tags"]   = tag;

            results.push_back({id, std::move(text), 0.0f, std::move(meta), std::move(ts)});
        }
        return results;
    }


    int Backend::Impl::embed_tables(Embedder& emb_ref, bool only_missing, bool progress,
                      const std::string& table_filter) {
        struct TableSpec {
            const char* table;
            const char* id_col;
            const char* text_col;
            const char* extra_col;  // if set, combined with text_col via turn_embed_text
        };
        static constexpr std::size_t kSpecCount = kTextTableCount;
        std::array<TableSpec, kSpecCount> kAllTables{};
        for (std::size_t i = 0; i < kSpecCount; ++i) {
            const auto& t = kTextTables[i];
            if (std::string_view(t.table) == "turns")
                kAllTables[i] = { t.table, t.pk, "user_text", "assistant_text" };
            else
                kAllTables[i] = { t.table, t.pk,
                                  t.embed_sql ? t.embed_sql : "text", nullptr };
        }

        // Filter to one table, or run the full fixed set when "all"/empty.
        std::vector<TableSpec> selected;
        for (auto& t : kAllTables) {
            if (table_filter == "all" || table_filter.empty() || table_filter == t.table)
                selected.push_back(t);
        }
        const auto& tables = selected;

        int done = 0;

        // Key-paginated scan. The obvious implementation — open one SELECT
        // over the table and UPDATE each row as the cursor walks it — writes
        // to the very b-tree being scanned on the same connection. SQLite
        // leaves that undefined: the cursor can be invalidated mid-scan and
        // sqlite3_step() returns an error, which the old `while (s.step())`
        // read as "no more rows". The loop exited cleanly, the caller reported
        // success, and the rebuild had silently stopped partway (on 2026-08-26
        // and again on 2026-08-29 it quit at documents id 1632 of 5654, leaving
        // two vector spaces in one table). Reading a bounded batch, finalizing
        // the statement, and only then writing keeps every write outside an
        // open scan. step_checked() makes a genuine failure throw instead of
        // masquerading as end-of-data.
        constexpr int kBatch = 500;

        struct Row { int id; std::string text; };

        // Per-table progress counter: resets to 0 at the start of each table
        // (not a running total across tables) so the printed number climbs
        // from 0 up to that table's own row count -- e.g. "turns: 3363/3363"
        // -- before moving to the next table's line, matching reindex_table's
        // "\rReindexing {table}: {done}/{total}" style.
        auto process = [&](const TableSpec& t, bool stale_only, int table_total) {
            int table_done = 0;
            // stale_only's filter is a plain embedding_version column
            // comparison (db_version 0.15+) — no blob decode needed. Rows
            // with embedding IS NULL were already handled in pass 1, so this
            // targets NOT-NULL rows whose version doesn't match current
            // (embedding_version IS NULL is included defensively — it
            // should never happen alongside a non-NULL embedding, since
            // bind_embedding/bind_embedding_version always write both
            // together, but a row that reaches this state some other way
            // should still be treated as stale rather than skipped forever).
            const char* filter = stale_only
                ? "embedding IS NOT NULL AND (embedding_version IS NULL OR embedding_version != ?)"
                : only_missing ? "embedding IS NULL"
                              : "1";
            std::string cols = t.extra_col
                ? std::format("{}, {}, {}", t.id_col, t.text_col, t.extra_col)
                : std::format("{}, {}", t.id_col, t.text_col);

            std::string sel = std::format(
                "SELECT {} FROM {} WHERE {} AND {} > ? ORDER BY {} LIMIT {}",
                cols, t.table, filter, t.id_col, t.id_col, kBatch);
            std::string upd = std::format(
                "UPDATE {} SET embedding_version = ?, embedding = ? WHERE {} = ?",
                t.table, t.id_col);

            // stale_only's filter has an extra bound parameter (current
            // version) ahead of the id-pagination cursor; only_missing/full
            // have just the cursor at position 1.
            const int id_param = stale_only ? 2 : 1;

            int last_id = 0;
            for (;;) {
                std::vector<Row> batch;
                batch.reserve(kBatch);
                {
                    Stmt select_stmt(db, sel);
                    if (stale_only) select_stmt.bind(1, static_cast<int>(embedding_version_));
                    select_stmt.bind(id_param, last_id);
                    while (select_stmt.step_checked()) {
                        int id = select_stmt.column_int(0);
                        last_id = id;

                        // Build the text FIRST, then test it. Testing the
                        // first column alone skipped assistant-only turns
                        // (empty user_text) even though turn_embed_text()
                        // would happily use assistant_text — those rows kept
                        // whatever blob they had forever, which no rebuild or
                        // backfill would ever revisit.
                        auto col1 = select_stmt.column_text(1);
                        std::string embed_text = t.extra_col
                            ? turn_embed_text(col1, select_stmt.column_text(2))
                            : std::move(col1);
                        if (embed_text.empty()) continue;

                        batch.push_back(Row{id, std::move(embed_text)});
                    }
                }   // select_stmt finalized before any write below
                if (batch.empty()) {
                    // A batch can come back empty while rows remain (every row
                    // in it was skipped as up-to-date or textless), so only
                    // stop when the scan itself is exhausted.
                    Stmt more(db, std::format(
                        "SELECT 1 FROM {} WHERE {} AND {} > ? LIMIT 1",
                        t.table, filter, t.id_col));
                    if (stale_only) more.bind(1, static_cast<int>(embedding_version_));
                    more.bind(id_param, last_id);
                    if (!more.step_checked()) break;
                    continue;
                }

                for (auto& r : batch) {
                    auto emb = emb_ref.encode(r.text);
                    Stmt update(db, upd);
                    bind_embedding_version(update.raw(), 1, emb);
                    bind_embedding(update.raw(), 2, emb);
                    update.bind(3, r.id);
                    if (!update.exec()) {
                        throw std::runtime_error(std::format(
                            ragger::lang::ERR_EMBED_UPDATE_FAILED,
                            t.table, r.id, sqlite3_errmsg(db)));
                    }
                    ++done;
                    ++table_done;
                    if (progress) {
                        std::cout << std::format(
                            ragger::lang::MSG_REBUILD_EMBEDDINGS_PROGRESS,
                            t.table, table_done, table_total);
                        std::cout.flush();
                    }
                    else if (done % 1000 == 0) {
                        // Non-interactive callers (the daemon) get an audit
                        // trail instead of \r spam, so a run that dies partway
                        // leaves a record of how far it got.
                        Diskerror::Logger::info(std::format(
                            ragger::lang::MSG_REBUILD_EMBEDDINGS_LOG,
                            t.table, table_done, table_total));
                    }
                }
            }
            if (progress) std::cout << "\n";
        };

        // Pass 1: rows with no embedding (or every row, on a full rebuild).
        for (auto& t : tables) {
            Stmt c(db, std::format("SELECT COUNT(*) FROM {}", t.table));
            int table_total = c.step() ? c.column_int(0) : 0;
            process(t, /*stale_only=*/false, table_total);
        }

        // Pass 2 (backfill only): re-embed rows with stale version bytes.
        // A full rebuild already covered everything in pass 1.
        if (only_missing) {
            for (auto& t : tables) {
                Stmt c(db, std::format("SELECT COUNT(*) FROM {}", t.table));
                int table_total = c.step() ? c.column_int(0) : 0;
                process(t, /*stale_only=*/true, table_total);
            }
        }

        if (done > 0 || !only_missing) {
            // Backfill/re-embed can touch every corpus (turn_summaries and
            // insights included) — drop ALL caches, not just three (B §1.6).
            invalidate_all_caches();
        }
        return done;
    }


    // Full re-encode of every embedded row (interactive, with progress),
    // or one table when `table` != "all".
    int Backend::Impl::rebuild_embeddings(Embedder& emb_ref, bool progress, const std::string& table) {
        if (!embedder_usable(emb_ref)) return 0;
        int n = embed_tables(emb_ref, /*only_missing=*/false, progress, table);
        // Belt and braces: a full rebuild touches every corpus and a dtype
        // change makes every cached decoded vector stale — drop all.
        invalidate_all_caches();
        return n;
    }


    // Cheap backfill: embed rows left NULL or with stale version byte,
    // scoped to one table when `table` != "all".
    int Backend::Impl::backfill_embeddings(Embedder& emb_ref, const std::string& table) {
        if (!embedder_usable(emb_ref)) return 0;
        return embed_tables(emb_ref, /*only_missing=*/true, /*progress=*/false, table);
    }


    // A disabled embedder returns {} from encode(), which bind_embedding()
    // turns into NULL. That is right for a single store, but catastrophic for
    // a bulk pass: a full rebuild would walk every table replacing good
    // vectors with NULL. Refuse instead.
    bool Backend::Impl::embedder_usable(const Embedder& emb_ref) const {
        if (emb_ref.ready()) return true;
        Diskerror::Logger::error(ragger::lang::ERR_EMBED_NO_MODEL_BULK);
        return false;
    }

    Embedder& Backend::Impl::require_embedder() {
        if (!embedder)
            throw std::runtime_error("embedder not available");
        return *embedder;
    }


    uint8_t Backend::Impl::get_embedding_version() const {
        return embedding_version_;
    }


    uint8_t Backend::Impl::increment_embedding_version() {
        // Cycle 1–255; 0 is reserved for empty/placeholder blobs.
        int next = static_cast<int>(embedding_version_) + 1;
        if (next > 255) next = 1;
        embedding_version_ = static_cast<uint8_t>(next);
        Stmt s(db,
            "INSERT INTO settings (key, value) VALUES ('embedding_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value");
        s.bind(1, std::to_string(static_cast<int>(embedding_version_)));
        s.exec();
        return embedding_version_;
    }


    // (Re)build the custom index for a single text table. Delegates to the
    // TextIndex engine (schema + tokenize + TF-IDF live there now).
    int Backend::Impl::reindex_table(const std::string& table_name, bool progress) {
        return text_index_->reindex_table(table_name, progress);
    }


    // Truncate the shared `terms` table + reset its AUTOINCREMENT counter.
    // Delegates to TextIndex; see its header comment for the "only
    // safe immediately before reindexing ALL five tables" caveat.
    void Backend::Impl::reset_terms_table() {
        text_index_->reset_terms_table();
    }


    // PHASE3 Step 6: one embedding write-back. The four public
    // update_*_embedding entry points forward here; the UPDATE shape and the
    // cache invalidation live in one place. `turns` invalidates nothing:
    // turns are not in a summaries cache (the old one there
    // was the wrong table), and the turn cache holds text rows, not vectors
    // (B §1.6). Insights share the decision write-back path (no separate
    // insight update existed pre-0.17; the switch covers it for symmetry).
    void Backend::Impl::invalidate_cache_for(const TextTable& t) {
        const std::string_view name = t.table;
        if (name == "documents")            invalidate_doc_cache();
        else if (name == "knowledge")       invalidate_knowledge_cache();
        else if (name == "episodes")        invalidate_episode_cache();
        // turns / turn_summaries: no vector cache keyed on these blobs.
    }

    bool Backend::Impl::update_embedding(const TextTable& t, int id,
                                         const std::vector<float>& emb) {
        Stmt s(db, std::format("UPDATE {} SET embedding_version = ?, "
                               "embedding = ? WHERE {} = ?", t.table, t.pk));
        bind_embedding_version(s.raw(), 1, emb);
        bind_embedding(s.raw(), 2, emb);
        s.bind(3, id);
        if (s.exec() && sqlite3_changes(db) > 0) {
            invalidate_cache_for(t);
            return true;
        }
        return false;
    }

    bool Backend::Impl::update_document_embedding(int document_id, const std::vector<float>& emb) {
        return update_embedding(*find_text_table("documents"), document_id, emb);
    }

    bool Backend::Impl::update_decision_embedding(int decision_id, const std::vector<float>& emb) {
        return update_embedding(*find_text_table("knowledge"), decision_id, emb);
    }

    // PHASE8: the old summary embedding write-back is gone with its table;
    // re-embedding goes through update_embedding(knowledge).

    bool Backend::Impl::update_turn_embedding(int turn_id, const std::vector<float>& emb) {
        return update_embedding(*find_text_table("turns"), turn_id, emb);
    }

    void Backend::Impl::invalidate_all_caches() {
        invalidate_doc_cache();
        invalidate_knowledge_cache();
        invalidate_turn_cache();
        invalidate_episode_cache();
    }

    std::vector<std::string> Backend::Impl::collections() const {
        std::vector<std::string> result;
        // Lean v2 summaries have no collection column — collections are not a
        // v2 concept. Return empty (kept for API compat).
        return result;
    }


    // True if the knowledge row exists and has a "keep" tag.
    bool Backend::Impl::has_keep_tag(int knowledge_id) {
        Stmt s(db, "SELECT tags FROM knowledge WHERE knowledge_id = ?");
        s.bind(1, knowledge_id);
        if (!s.step()) return false;
        return s.column_text(0).find("keep") != std::string::npos;
    }


    // PHASE8: HARD deletes (Phase 11 removes these from the HTTP/MCP
    // surface, TODO Q2). Restricted to kind='note' rows that no other row
    // `supersedes` — deleting a superseded parent would dangle the chain —
    // and that carry no `keep` tag. Returns whether a row was deleted.
    bool Backend::Impl::delete_memory(int memory_id) {
        Stmt stmt(db,
            "DELETE FROM knowledge WHERE knowledge_id = ? AND kind = 'note'"
            "  AND (tags IS NULL OR tags NOT LIKE '%keep%')"
            "  AND NOT EXISTS (SELECT 1 FROM knowledge c WHERE c.supersedes = knowledge.knowledge_id)");
        stmt.bind(1, memory_id);
        stmt.step();
        int changes = sqlite3_changes(db);
        if (changes > 0) {
            invalidate_knowledge_cache();
            return true;
        }
        return false;
    }


    int Backend::Impl::delete_batch(const std::vector<int>& memory_ids) {
        if (memory_ids.empty()) return 0;
        // Single atomic DELETE filtered by kind/keep/supersedes, entirely in
        // SQL. RAII: a throw between BEGIN and COMMIT used to leave the
        // connection inside an open transaction.
        std::string sql = "DELETE FROM knowledge WHERE kind = 'note'"
                          "  AND (tags IS NULL OR tags NOT LIKE '%keep%')"
                          "  AND NOT EXISTS (SELECT 1 FROM knowledge c WHERE c.supersedes = knowledge.knowledge_id)"
                          "  AND knowledge_id IN (";
        for (size_t i = 0; i < memory_ids.size(); ++i) {
            if (i > 0) sql += ",";
            sql += "?";
        }
        sql += ")";
        int changes = 0;
        {
            Transaction tx(db);
            Stmt stmt(db, sql);
            for (size_t i = 0; i < memory_ids.size(); ++i)
                stmt.bind(static_cast<int>(i + 1), memory_ids[i]);
            stmt.step();
            changes = sqlite3_changes(db);
            tx.commit();
        }
        if (changes > 0) invalidate_knowledge_cache();
        return changes;
    }


    std::vector<SearchResult> Backend::Impl::search_by_metadata(const json& metadata_filter, int limit,
                                                 const std::string& after,
                                                 const std::string& before) {
        std::vector<SearchResult> results;

        // PHASE8: knowledge rows expose kind / status / tags. Filter on
        // those columns; any other requested key matches nothing (there is
        // no free-form metadata blob in the lean schema). The old `level`
        // key matches the `level:<value>` tag store() appends for callers
        // that still send a level.
        // created_at is an INTEGER epoch: after/before arrive as "%F %T"
        // strings and must be resolved to ints (raw-string comparison never
        // matched — same bug class as note_exists_exact). The returned
        // timestamp renders as a datetime string like every other read.
        std::string sql = "SELECT knowledge_id, text, kind, status, tags, "
                          "datetime(created_at,'unixepoch','localtime') "
                          "FROM knowledge";
        std::string where;
        std::vector<std::string> binds;
        std::vector<int64_t> epoch_binds;

        for (auto it = metadata_filter.begin(); it != metadata_filter.end(); ++it) {
            const std::string& k = it.key();
            if (k == "level") {
                where += (where.empty() ? " WHERE " : " AND ") + std::string("tags LIKE ?");
                binds.push_back("%level:" + it.value().get<std::string>() + "%");
            } else if (k == "kind") {
                where += (where.empty() ? " WHERE " : " AND ") + k + " = ?";
                binds.push_back(it.value().get<std::string>());
            } else if (k == "status") {
                where += (where.empty() ? " WHERE " : " AND ") + k + " = ?";
                binds.push_back(it.value().get<std::string>());
            } else if (k == "tags") {
                where += (where.empty() ? " WHERE " : " AND ") + std::string("tags LIKE ?");
                binds.push_back("%" + it.value().get<std::string>() + "%");
            } else {
                // Unsupported key under the lean schema → no rows match.
                return results;
            }
        }

        if (!after.empty()) {
            where += (where.empty() ? " WHERE " : " AND ") + std::string("created_at >= ?");
            epoch_binds.push_back(resolve_epoch(after));
        }
        if (!before.empty()) {
            where += (where.empty() ? " WHERE " : " AND ") + std::string("created_at < ?");
            epoch_binds.push_back(resolve_epoch(before));
        }

        sql += where + " ORDER BY created_at DESC";
        if (limit > 0) sql += " LIMIT " + std::to_string(limit);

        Stmt stmt(db, sql);
        for (size_t i = 0; i < binds.size(); ++i) {
            stmt.bind(static_cast<int>(i + 1), binds[i]);
        }
        // after/before always trail the metadata binds (their WHERE fragments
        // are appended last), so binding them after keeps positional order.
        for (size_t i = 0; i < epoch_binds.size(); ++i) {
            stmt.bind(static_cast<int>(binds.size() + i + 1), epoch_binds[i]);
        }

        while (stmt.step()) {
            int id        = stmt.column_int(0);
            auto text     = stmt.column_text(1);
            auto kind     = stmt.column_text(2);
            auto status   = stmt.column_text(3);
            auto tag      = stmt.column_text(4);
            auto ts       = stmt.column_text(5);

            json metadata = json::object();
            if (!kind.empty())   metadata["kind"]   = kind;
            if (!status.empty()) metadata["status"] = status;
            if (!tag.empty())    metadata["tags"]   = tag;

            results.push_back({id, std::move(text), 0.0f, std::move(metadata), std::move(ts)});
        }
        return results;
    }


    void Backend::Impl::close() {
        if (db) {
            // TextIndex owns prepared statements against db and is declared
            // after it: destroy the index FIRST or sqlite3_close returns
            // SQLITE_BUSY and leaks the connection + WAL checkpoint (B §1.8).
            text_index_.reset();
            const int rc = sqlite3_close(db);
            if (rc != SQLITE_OK)
                Diskerror::Logger::warn(std::format(
                    "sqlite3_close failed: {} (connection leaked)", sqlite3_errstr(rc)));
            db = nullptr;
        }
    }


} // namespace ragger::sqlite
