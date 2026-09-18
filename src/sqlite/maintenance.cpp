#include "backend_impl.h"
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


    int SqliteBackend::Impl::count() const {
        Stmt s(db, "SELECT COUNT(*) FROM summaries");
        int c = 0;
        if (s.step())
            c = s.column_int(0);
        return c;
    }


    std::vector<std::pair<std::string, int64_t>> SqliteBackend::Impl::table_row_counts() const {
        // User-facing tables for the dashboard status pane, in display order.
        static const char* kTables[] = {
            "turns", "summaries", "turn_summaries", "sessions",
            "documents", "decisions", "statements", "models",
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


    // Total rows across the five embedded tables (turns, turn_summaries,
    // summaries, decisions, documents) — i.e. how many rows
    // `rebuild_embeddings()` will re-encode. (count() alone is just
    // summaries, which understates the rebuild scope.)
    int SqliteBackend::Impl::count_embeddable_rows(const std::string& table) const {
        int total = 0;
        for (const char* tbl : {"turns", "turn_summaries", "summaries", "decisions", "documents"}) {
            if (table != "all" && table != tbl) continue;
            Stmt s(db, std::format("SELECT COUNT(*) FROM {}", tbl));
            if (s.step()) total += s.column_int(0);
        }
        return total;
    }


    // True if any embedded table holds a non-NULL embedding. EXISTS short-
    // circuits on the first hit. Deferred (NULL-embedding) rows don't count —
    // they get the current model on backfill, so they aren't incompatible.
    bool SqliteBackend::Impl::has_embeddings() const {
        Stmt s(db,
            "SELECT EXISTS("
            "  SELECT 1 FROM summaries  WHERE embedding IS NOT NULL "
            "  UNION ALL SELECT 1 FROM turns     WHERE embedding IS NOT NULL "
            "  UNION ALL SELECT 1 FROM documents WHERE embedding IS NOT NULL "
            "  UNION ALL SELECT 1 FROM decisions WHERE embedding IS NOT NULL "
            "  LIMIT 1)");
        return s.step() && s.column_int(0) != 0;
    }


    // Lean v2 summaries have no collection column; the `collection` argument
    // is ignored (kept for API compat). Returns every summary, score 0.
    std::vector<SearchResult> SqliteBackend::Impl::load_all(const std::string& /*collection*/) {
        std::vector<SearchResult> results;
        Stmt s(db,
            "SELECT summary_id, text, level, tags, created_at "
            "FROM summaries ORDER BY summary_id");

        while (s.step()) {
            int id        = s.column_int(0);
            auto text     = s.column_text(1);
            auto lvl      = s.column_text(2);
            auto tag      = s.column_text(3);
            auto ts       = s.column_text(4);

            json meta = json::object();
            if (!lvl.empty()) meta["level"]  = lvl;
            if (!tag.empty()) meta["tags"]   = tag;

            results.push_back({id, std::move(text), 0.0f, std::move(meta), std::move(ts)});
        }
        return results;
    }


    int SqliteBackend::Impl::embed_tables(Embedder& emb_ref, bool only_missing, bool progress,
                      const std::string& table_filter) {
        struct TableSpec {
            const char* table;
            const char* id_col;
            const char* text_col;
            const char* extra_col;  // if set, combined with text_col via turn_embed_text
        };
        static constexpr TableSpec kAllTables[] = {
            { "turns",           "turn_id",         "user_text", "assistant_text" },
            { "summaries",       "summary_id",      "text",      nullptr          },
            { "decisions",       "decision_id",     "text",      nullptr          },
            { "documents",       "document_id",     kDocEmbedTextSQL, nullptr       },
            { "turn_summaries",  "turn_summary_id", "text",      nullptr          },
        };

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
            invalidate_cache();
            invalidate_doc_cache();
            invalidate_dec_cache();
        }
        return done;
    }


    // Full re-encode of every embedded row (interactive, with progress),
    // or one table when `table` != "all".
    int SqliteBackend::Impl::rebuild_embeddings(Embedder& emb_ref, bool progress, const std::string& table) {
        if (!embedder_usable(emb_ref)) return 0;
        return embed_tables(emb_ref, /*only_missing=*/false, progress, table);
    }


    // Cheap backfill: embed rows left NULL or with stale version byte,
    // scoped to one table when `table` != "all".
    int SqliteBackend::Impl::backfill_embeddings(Embedder& emb_ref, const std::string& table) {
        if (!embedder_usable(emb_ref)) return 0;
        return embed_tables(emb_ref, /*only_missing=*/true, /*progress=*/false, table);
    }


    // A disabled embedder returns {} from encode(), which bind_embedding()
    // turns into NULL. That is right for a single store, but catastrophic for
    // a bulk pass: a full rebuild would walk every table replacing good
    // vectors with NULL. Refuse instead.
    bool SqliteBackend::Impl::embedder_usable(const Embedder& emb_ref) const {
        if (emb_ref.ready()) return true;
        Diskerror::Logger::error(ragger::lang::ERR_EMBED_NO_MODEL_BULK);
        return false;
    }


    uint8_t SqliteBackend::Impl::get_embedding_version() const {
        return embedding_version_;
    }


    uint8_t SqliteBackend::Impl::increment_embedding_version() {
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
    // SqliteTextIndex engine (schema + tokenize + TF-IDF live there now).
    int SqliteBackend::Impl::reindex_table(const std::string& table_name, bool progress) {
        return text_index_->reindex_table(table_name, progress);
    }


    // Truncate the shared `terms` table + reset its AUTOINCREMENT counter.
    // Delegates to SqliteTextIndex; see its header comment for the "only
    // safe immediately before reindexing ALL five tables" caveat.
    void SqliteBackend::Impl::reset_terms_table() {
        text_index_->reset_terms_table();
    }


    // Set a document's embedding (used by the import path after embedding
    // chunks via the subprocess executor). Returns true on a row update.
    bool SqliteBackend::Impl::update_document_embedding(int document_id, const std::vector<float>& emb) {
        Stmt s(db,
            "UPDATE documents SET embedding_version = ?, embedding = ? WHERE document_id = ?");
        bind_embedding_version(s.raw(), 1, emb);
        bind_embedding(s.raw(), 2, emb);
        s.bind(3, document_id);
        if (s.exec() && sqlite3_changes(db) > 0) {
            invalidate_doc_cache();
            return true;
        }
        return false;
    }


    // Per-row embedding write-back for the other three context tables.
    // Mirrors update_document_embedding; each invalidates the cache that
    // backs its table's vector search.
    bool SqliteBackend::Impl::update_decision_embedding(int decision_id, const std::vector<float>& emb) {
        Stmt s(db, "UPDATE decisions SET embedding_version = ?, embedding = ? WHERE decision_id = ?");
        bind_embedding_version(s.raw(), 1, emb);
        bind_embedding(s.raw(), 2, emb);
        s.bind(3, decision_id);
        if (s.exec() && sqlite3_changes(db) > 0) {
            invalidate_dec_cache();
            return true;
        }
        return false;
    }


    bool SqliteBackend::Impl::update_summary_embedding(int summary_id, const std::vector<float>& emb) {
        Stmt s(db, "UPDATE summaries SET embedding_version = ?, embedding = ? WHERE summary_id = ?");
        bind_embedding_version(s.raw(), 1, emb);
        bind_embedding(s.raw(), 2, emb);
        s.bind(3, summary_id);
        if (s.exec() && sqlite3_changes(db) > 0) {
            invalidate_cache();
            return true;
        }
        return false;
    }


    bool SqliteBackend::Impl::update_turn_embedding(int turn_id, const std::vector<float>& emb) {
        Stmt s(db, "UPDATE turns SET embedding_version = ?, embedding = ? WHERE turn_id = ?");
        bind_embedding_version(s.raw(), 1, emb);
        bind_embedding(s.raw(), 2, emb);
        s.bind(3, turn_id);
        if (s.exec() && sqlite3_changes(db) > 0) {
            invalidate_cache();
            return true;
        }
        return false;
    }


    std::vector<std::string> SqliteBackend::Impl::collections() const {
        std::vector<std::string> result;
        // Lean v2 summaries have no collection column — collections are not a
        // v2 concept. Return empty (kept for API compat).
        return result;
    }


    // Returns true if the summaries row exists and has a "keep" tag.
    bool SqliteBackend::Impl::has_keep_tag(int summary_id) {
        Stmt s(db, "SELECT tags FROM summaries WHERE summary_id = ?");
        s.bind(1, summary_id);
        if (!s.step()) return false;
        return s.column_text(0).find("keep") != std::string::npos;
    }


    bool SqliteBackend::Impl::delete_memory(int memory_id) {
        if (has_keep_tag(memory_id)) return false;
        Stmt stmt(db, "DELETE FROM summaries WHERE summary_id = ?");
        stmt.bind(1, memory_id);
        stmt.step();
        int changes = sqlite3_changes(db);
        if (changes > 0) {
            invalidate_cache();
            return true;
        }
        return false;
    }


    int SqliteBackend::Impl::delete_batch(const std::vector<int>& memory_ids) {
        if (memory_ids.empty()) return 0;
        // Single atomic DELETE filtered by keep-tag, entirely in SQL.
        std::string sql = "DELETE FROM summaries WHERE summary_id IN (";
        for (size_t i = 0; i < memory_ids.size(); ++i) {
            if (i > 0) sql += ",";
            sql += "?";
        }
        sql += ") AND (tags IS NULL OR tags NOT LIKE '%keep%')";
        Stmt(db, "BEGIN").exec();
        Stmt stmt(db, sql);
        for (size_t i = 0; i < memory_ids.size(); ++i)
            stmt.bind(static_cast<int>(i + 1), memory_ids[i]);
        stmt.step();
        int changes = sqlite3_changes(db);
        Stmt(db, "COMMIT").exec();
        if (changes > 0) invalidate_cache();
        return changes;
    }


    std::vector<SearchResult> SqliteBackend::Impl::search_by_metadata(const json& metadata_filter, int limit,
                                                 const std::string& after,
                                                 const std::string& before) {
        std::vector<SearchResult> results;

        // Lean v2 summaries expose level / tags / timestamp. Filter
        // on those columns; any other requested key matches nothing (there is
        // no free-form metadata blob in the lean schema).
        std::string sql = "SELECT summary_id, text, level, tags, created_at "
                          "FROM summaries";
        std::string where;
        std::vector<std::string> binds;

        for (auto it = metadata_filter.begin(); it != metadata_filter.end(); ++it) {
            const std::string& k = it.key();
            if (k == "level") {
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
            binds.push_back(after);
        }
        if (!before.empty()) {
            where += (where.empty() ? " WHERE " : " AND ") + std::string("created_at < ?");
            binds.push_back(before);
        }

        sql += where + " ORDER BY created_at DESC";
        if (limit > 0) sql += " LIMIT " + std::to_string(limit);

        Stmt stmt(db, sql);
        for (size_t i = 0; i < binds.size(); ++i) {
            stmt.bind(static_cast<int>(i + 1), binds[i]);
        }

        while (stmt.step()) {
            int id        = stmt.column_int(0);
            auto text     = stmt.column_text(1);
            auto lvl      = stmt.column_text(2);
            auto tag      = stmt.column_text(3);
            auto ts       = stmt.column_text(4);

            json metadata = json::object();
            if (!lvl.empty()) metadata["level"]  = lvl;
            if (!tag.empty()) metadata["tags"]   = tag;

            results.push_back({id, std::move(text), 0.0f, std::move(metadata), std::move(ts)});
        }
        return results;
    }


    void SqliteBackend::Impl::close() {
        if (db) {
            sqlite3_close(db);
            db = nullptr;
        }
    }


} // namespace ragger::sqlite
