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

// Build the tags string from a metadata JSON object. Accepts metadata["tags"]
// as a comma-separated string or a JSON array. `keep`/`bad` flag fields are
// folded in (they protect a row from deletion / mark it as low-quality).
static std::string tags_from_metadata(const json& metadata) {
    std::string tags_str;
    if (metadata.contains("tags")) {
        const auto& tv = metadata["tags"];
        if (tv.is_array()) {
            for (size_t i = 0; i < tv.size(); ++i) {
                if (i > 0) tags_str += ",";
                tags_str += tv[i].get<std::string>();
            }
        } else if (tv.is_string()) {
            tags_str = tv.get<std::string>();
        }
    }
    if (metadata.value("keep", false) &&
        tags_str.find("keep") == std::string::npos)
        tags_str += (tags_str.empty() ? "" : ",") + std::string("keep");
    if (metadata.value("bad", false) &&
        tags_str.find("bad") == std::string::npos)
        tags_str += (tags_str.empty() ? "" : ",") + std::string("bad");
    return tags_str;
}



    // ---- path normalization -------------------------------------------
    std::string SqliteBackend::Impl::normalize_path(const std::string& text) {
        if (!config().normalize_home_path) return text;
        std::string home = home_dir();
        if (home.empty()) return text;
        std::string prefix = home + "/";
        std::string out = text;
        size_t pos = 0;
        while ((pos = out.find(prefix, pos)) != std::string::npos) {
            out.replace(pos, prefix.size(), "~/");
            pos += 2;
        }
        return out;
    }


    // Strip a leading "Decision #NNN", "Design Decision #NNN", or
    // "Decision Log #NNN" label (optionally with a "(context)" aside
    // and/or the word "Logged") from decision text, e.g.
    // "Decision #122 (Vienna, June 2026): Ragger adds ..." ->
    // "Ragger adds ...". The row's own decision_id + created_at already
    // carry this information — a hand/agent-authored numeric label baked
    // into the text is pure duplication that dilutes embedding/FTS search
    // (and drifts out of sync with the real decision_id over time).
    // Markdown emphasis markers ("**"/"*") wrapping the label are also
    // consumed. No-op if the text doesn't start with this pattern.
    std::string SqliteBackend::Impl::strip_decision_number_prefix(const std::string& text) {
        static const std::regex prefix_re(
            R"(^\s*\*{0,2}(?:Design\s+)?Decision(?:\s+Log)?\s*#\d+\s*)"
            R"((?:Logged)?\s*(?:\([^)]*\))?\s*\*{0,2}\s*:\s*\*{0,2}\s*)",
            std::regex::icase);
        std::smatch m;
        if (!std::regex_search(text, m, prefix_re) || m.position(0) != 0)
            return text;
        std::string out = text.substr(m.length(0));
        // If the consumed prefix opened with "**"/"*" (bold/italic wrapping
        // the whole label), the matching close marker sits at the very end
        // of the string, past what the prefix regex could reach — trim it
        // too so we don't leave a dangling "**"/"*" behind.
        std::string opened = m.str(0);
        size_t nstars = 0;
        for (char c : opened) {
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') continue;
            if (c == '*') { ++nstars; continue; }
            break;
        }
        if (nstars > 0) {
            while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' '))
                out.pop_back();
            size_t trailing = 0;
            for (size_t i = out.size(); i > 0 && out[i - 1] == '*'; --i) ++trailing;
            if (trailing >= nstars) out.erase(out.size() - nstars);
        }
        return out;
    }


    // ---- local timestamp ("%F %T" == "YYYY-MM-DD HH:MM:SS") ----------
    // Thin aliases over the shared formatter (ragger/util/time.h) so every
    // DB timestamp uses one local-time format.
    std::string SqliteBackend::Impl::local_timestamp(std::time_t tt) { return db_timestamp(tt); }

    std::string SqliteBackend::Impl::local_timestamp() { return db_timestamp(); }


    // ---- public API ---------------------------------------------------

    // v2: the generic store API writes a summary (L2/L3/L4). A memory-only
    // install records agent memories here. `level`/`status` come from
    // metadata when supplied; defaults suit a settled session-level note.
    // collection/category and the free-form metadata blob are gone in v2 —
    // metadata fields are either promoted to columns or dropped. FTS5 sync
    // triggers index the row, so there is no explicit BM25 step.
    std::string SqliteBackend::Impl::store(const std::string& raw_text, json metadata, bool defer_embedding) {
        if (metadata.is_null()) metadata = json::object();

        std::string level  = metadata.value("level",  std::string("session"));

        // Optional historical timestamp override (imports of past
        // conversations). Must be an ISO-8601 UTC string; else falls to now.
        std::string ts_override;
        if (metadata.contains("timestamp") && metadata["timestamp"].is_string()) {
            ts_override = metadata["timestamp"].get<std::string>();
        }

        // tags: accept a JSON array or a plain string. `keep`/`bad` are
        // flag-tags folded into the tags column — a row tagged "keep" is
        // protected from delete/update (see delete_memory / update_text).
        std::string tags_str = tags_from_metadata(metadata);

        std::string text = normalize_path(raw_text);

        std::vector<float> emb;
        if (!defer_embedding) {
            emb = embedder->encode(text);
        }

        auto ts = resolve_epoch(ts_override);

        // Every summary should record the model that produced it. Callers pass
        // the live model via metadata["model"]; empty → 0 → NULL (the raw-turn
        // sentinel), which a non-turn summary should never be. get_or_create_model
        // interns the name in the models table.
        std::string model_name = metadata.value("model", std::string(""));
        int model_id = get_or_create_model(model_name);

        // Custom text index (v0.16): index_record() below tokenizes text and
        // maintains terms/*_terms/count columns — no FTS5 triggers involved.
        Stmt s(db,
            "INSERT INTO summaries (text, embedding_version, embedding, level, tags, created_at, model_id, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?)");

        s.bind(1, text);
        if (defer_embedding) {
            s.bind_null(2);
            s.bind_null(3);
        } else {
            bind_embedding_version(s.raw(), 2, emb);
            bind_embedding(s.raw(), 3, emb);
        }
        s.bind(4, level).bind(5, tags_str).bind(6, ts);
        if (model_id) s.bind(7, model_id); else s.bind_null(7);
        s.bind(8, ts);  // updated_at == created_at on insert

        if (!s.exec()) {
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        }

        int summary_id = static_cast<int>(sqlite3_last_insert_rowid(db));
        invalidate_cache();
        text_index_->index_record("summaries", summary_id, text);
        return std::to_string(summary_id);
    }


    // ---- store_document: write Level 5 RAG chunk ----------------------

    // Resolve a (path, title) document to its document_sources row, inserting
    // one on first sighting. Returns {document_source_id, imported_at_epoch}.
    // imported_at_text is the caller's timestamp (TEXT 'YYYY-MM-DD HH:MM:SS'
    // or 'YYYYMMDD', interpreted LOCAL); empty -> now. On an existing source
    // the stored epoch is returned (the first import's time wins), so every
    // chunk of a document shares one imported_at.
    std::pair<long long, long long> SqliteBackend::Impl::get_or_create_document_source(
            const std::string& path, const std::string& title,
            int year, const std::string& tags,
            const std::string& imported_at_text) {
        {
            Stmt s(db,
                "SELECT document_source_id, imported_at FROM document_sources "
                "WHERE ifnull(path,'') = ifnull(?,'') "
                "  AND ifnull(title,'') = ifnull(?,'')");
            if (path.empty()) s.bind_null(1); else s.bind(1, path);
            if (title.empty()) s.bind_null(2); else s.bind(2, title);
            if (s.step())
                return { s.column_int64(0), s.column_int64(1) };
        }
        // Convert the caller's local-wall-clock text to a unix epoch via SQLite
        // (matches the migration's TEXT->epoch reading); fall back to now.
        long long epoch = static_cast<long long>(std::time(nullptr));
        if (!imported_at_text.empty()) {
            Stmt c(db, "SELECT CAST(strftime('%s', ?, 'utc') AS INTEGER)");
            c.bind(1, imported_at_text);
            if (c.step() && !c.is_null(0)) epoch = c.column_int64(0);
        }
        Stmt ins(db,
            "INSERT INTO document_sources(title, path, year, tags, imported_at) "
            "VALUES (?,?,?,?,?)");
        if (title.empty()) ins.bind_null(1); else ins.bind(1, title);
        if (path.empty())  ins.bind_null(2); else ins.bind(2, path);
        if (year <= 0) ins.bind_null(3); else ins.bind(3, year);
        ins.bind(4, tags);
        ins.bind(5, epoch);
        if (!ins.exec())
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        return { static_cast<long long>(sqlite3_last_insert_rowid(db)), epoch };
    }


    int SqliteBackend::Impl::store_document(const DocumentChunk& chunk, bool defer_embedding) {
        // Path-normalise the body text for consistency with store().
        std::string text = normalize_path(chunk.text);

        // Title rides the vector + phon signals (not just FTS): the embedder
        // and phonizer see `text + '\n' + title`. Title goes LAST so the chunk
        // body leads each signal (chunks of one doc don't share a phon prefix)
        // and the primary content dominates the embedding.
        std::string signal_text = chunk.title.empty()
            ? text
            : text + "\n" + chunk.title;

        std::vector<float> emb;
        if (!defer_embedding) {
            emb = embedder->encode(signal_text);
        }

        // Resolve/insert the source; every chunk of a (path,title) doc shares
        // one document_sources row and its imported_at epoch. modified_on
        // starts equal to that epoch (until the chunk text is later edited).
        auto [source_id, epoch] = get_or_create_document_source(
            chunk.path, chunk.title, chunk.year, chunk.tags, chunk.imported_at);

        // Slim chunk row: no path/title/year/imported_at (they live on the
        // source). documents_fts sync triggers index text/tags only.
        Stmt s(db,
            "INSERT INTO documents "
            "(text, tags, chunk_index, document_source_id, modified_on, "
            " embedding_version, embedding) "
            "VALUES (?,?,?,?,?,?,?)");

        s.bind(1, text);
        s.bind(2, chunk.tags);
        if (chunk.chunk_index <= 0) s.bind_null(3);
        else s.bind(3, chunk.chunk_index);
        s.bind(4, source_id);
        s.bind(5, epoch);
        if (defer_embedding) {
            s.bind_null(6);
            s.bind_null(7);
        } else {
            bind_embedding_version(s.raw(), 6, emb);
            bind_embedding(s.raw(), 7, emb);
        }

        if (!s.exec()) {
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        }

        invalidate_doc_cache();
        int doc_id = static_cast<int>(sqlite3_last_insert_rowid(db));
        text_index_->index_record("documents", doc_id, signal_text);
        return doc_id;
    }


    // ---- turns (L1) raw exchange capture ------------------------------
    // Resolve a model name to its models.model_id, creating the row if it
    // doesn't exist. Empty name → 0 (callers bind NULL).
    int SqliteBackend::Impl::get_or_create_model(const std::string& name) {
        if (name.empty()) return 0;
        Stmt s(db, "SELECT model_id FROM models WHERE name = ?");
        s.bind(1, name);
        int id = 0;
        if (s.step()) id = s.column_int(0);
        if (id) return id;
        Stmt ins(db, "INSERT INTO models (name) VALUES (?)");
        ins.bind(1, name);
        ins.step();
        return static_cast<int>(sqlite3_last_insert_rowid(db));
    }


    /// Resolve a session GUID to its compact integer id, inserting on first
    /// sighting. Empty guid → 0 (NULL session_id), mirroring get_or_create_model.
    /// If session_name is non-empty, sets sessions.name (and name_source) on
    /// first sighting or updates it if the title has changed since last capture.
    int SqliteBackend::Impl::get_or_create_session(const std::string& guid,
                              const std::string& session_name,
                              const std::string& name_source) {
        if (guid.empty()) return 0;
        Stmt s(db, "SELECT session_id FROM sessions WHERE guid = ?");
        s.bind(1, guid);
        int id = 0;
        if (s.step()) id = s.column_int(0);
        if (!id) {
            Stmt ins(db, "INSERT INTO sessions (guid, name, name_source) VALUES (?, ?, ?)");
            ins.bind(1, guid);
            if (session_name.empty()) ins.bind_null(2);
            else                     ins.bind(2, session_name);
            if (session_name.empty()) ins.bind_null(3);
            else                      ins.bind(3, name_source);
            ins.step();
            id = static_cast<int>(sqlite3_last_insert_rowid(db));
        } else if (!session_name.empty()) {
            // Update name if changed (retitled session, or first time a name
            // is available for a previously-unnamed session).
            Stmt u(db, "UPDATE sessions SET name = ?, name_source = ? "
                       "WHERE session_id = ? AND (name IS NULL OR name != ?)");
            u.bind(1, session_name);
            u.bind(2, name_source);
            u.bind(3, id);
            u.bind(4, session_name);
            u.exec();
        }
        return id;
    }


    // Embedding for a turn is over the joined exchange (user + assistant),
    // using the same U+001F unit-separator as chat's summary turns.
    std::string SqliteBackend::Impl::turn_embed_text(const std::string& u, const std::string& a) {
        return a.empty() ? u : (u + "\n\x1F\n" + a);
    }


    // Store a raw L1 turn. An empty assistant_text writes a *partial* row
    // (assistant + embedding NULL) for the prompt-arrival/finalize flow;
    // otherwise the exchange is embedded unless defer_embedding. FTS5 sync
    // triggers index user_text/assistant_text. Returns turn_id.
    // `source_timestamp` (non-empty, db format) overrides created_at for
    // historical imports; the regeneration-dedup window is skipped in that
    // case (it reasons about "now", which is meaningless for old turns).
    int SqliteBackend::Impl::store_turn(const std::string& user_text, const std::string& assistant_text,
                   const std::string& model_name, bool defer_embedding,
                   const std::string& session_guid,
                   const std::string& source_timestamp,
                   const std::string& session_name,
                   const std::string& name_source) {
        std::string u = normalize_path(user_text);
        std::string a = normalize_path(assistant_text);
        int model_id   = get_or_create_model(model_name);
        int session_id = get_or_create_session(session_guid, session_name, name_source);

        // Retry/regeneration dedup. When the agent re-answers the *same* user
        // prompt without an intervening new prompt, we get a second turn whose
        // user_text duplicates the immediately-preceding turn. In a healthy
        // agent a regeneration stays in one session; a session change between
        // two identical prompts is itself a breakage signal (observed when the
        // TUI crash-restarted and minted a fresh session GUID per turn). So the
        // match is intentionally cross-session: identical user_text in the most
        // recent turn, within a short time window. We keep-latest — update that
        // row's assistant_text (+ re-embed) in place rather than inserting a
        // duplicate. The FTS au-trigger keeps the index consistent. Empty
        // assistant_text (partial/prompt-arrival rows) never dedups.
        // Skipped entirely for historical imports (source_timestamp set).
        if (!a.empty() && source_timestamp.empty()) {
            Stmt p(db,
                "SELECT turn_id, user_text, created_at, session_id FROM turns "
                "ORDER BY turn_id DESC LIMIT 1");
            if (p.step()) {
                const int prev_id          = p.column_int(0);
                const std::string prev_u   = p.column_text(1);
                const int64_t prev_ts_ep   = p.column_int64(2);
                // 5-minute window: wide enough for a slow regeneration, narrow
                // enough that a genuinely re-typed identical prompt much later
                // is still recorded as its own turn.
                bool within_window = false;
                if (prev_u == u) {
                    int64_t now_ep = db_epoch();
                    within_window = (now_ep - prev_ts_ep) <= 300;
                }
                if (within_window) {
                    std::vector<float> emb2 = embedder->encode(turn_embed_text(u, a));
                    // Single timestamp value shared by the turns UPDATE and the
                    // placeholder cleanup/insert below — computed once, never
                    // re-read from the clock, so the two writes can't drift
                    // apart and strand the old (session_id, prev_ts) slot.
                    const int64_t new_ts = db_epoch();
                    Stmt up(db,
                        "UPDATE turns SET assistant_text = ?, embedding_version = ?, embedding = ?, "
                        "model_id = COALESCE(?, model_id), "
                        "session_id = COALESCE(?, session_id), created_at = ? "
                        "WHERE turn_id = ?");
                    up.bind(1, a);
                    bind_embedding_version(up.raw(), 2, emb2);
                    bind_embedding(up.raw(), 3, emb2);
                    if (model_id) up.bind(4, model_id); else up.bind_null(4);
                    if (session_id) up.bind(5, session_id); else up.bind_null(5);
                    up.bind(6, new_ts);
                    up.bind(7, prev_id);
                    if (!up.exec())
                        if (!up.exec()) throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
                        text_index_->index_record("turns", prev_id, u + "\n" + a);
                        return prev_id;
                }
            }
        }

        std::vector<float> emb;
        bool have_emb = false;
        if (!defer_embedding && !a.empty()) {
            emb = embedder->encode(turn_embed_text(u, a));
            have_emb = true;
        }

        Stmt s(db,
            "INSERT INTO turns (model_id, session_id, user_text, assistant_text, embedding_version, embedding, created_at) "
            "VALUES (?,?,?,?,?,?,?)");
        if (model_id) s.bind(1, model_id); else s.bind_null(1);
        if (session_id) s.bind(2, session_id); else s.bind_null(2);
        s.bind(3, u);
        if (a.empty()) s.bind_null(4);
        else s.bind(4, a);
        if (have_emb) {
            bind_embedding_version(s.raw(), 5, emb);
            bind_embedding(s.raw(), 6, emb);
        } else {
            s.bind_null(5);
            s.bind_null(6);
        }
        s.bind(7, resolve_epoch(source_timestamp));

        if (!s.exec())
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        int new_id = static_cast<int>(sqlite3_last_insert_rowid(db));
        text_index_->index_record("turns", new_id, a.empty() ? u : (u + "\n" + a));
        return new_id;
    }


    // Shared implementation for turn queries over a session GUID.
    // asc=true → oldest-first (ORDER BY turn_id ASC, no limit).
    // asc=false → newest-first (ORDER BY timestamp DESC, turn_id DESC) with optional limit.
    std::vector<TurnRecord> SqliteBackend::Impl::turns_by_session_impl(
            const std::string& session_guid, bool asc, int limit) {
        std::vector<TurnRecord> out;
        if (session_guid.empty()) return out;
        std::string sql =
            "SELECT t.turn_id, t.user_text, t.assistant_text, m.name, "
            "       datetime(t.created_at,'unixepoch','localtime') "
            "FROM turns t "
            "JOIN sessions ss ON t.session_id = ss.session_id "
            "LEFT JOIN models m ON t.model_id = m.model_id "
            "WHERE ss.guid = ? ";
        sql += asc ? "ORDER BY t.turn_id ASC"
                   : "ORDER BY t.created_at DESC, t.turn_id DESC";
        if (limit > 0) sql += " LIMIT ?";
        Stmt s(db, sql);
        s.bind(1, session_guid);
        if (limit > 0) s.bind(2, limit);
        while (s.step()) {
            out.push_back({
                s.column_int(0),
                s.column_text(1),
                s.column_text(2),
                s.column_text(3),
                s.column_text(4),
                session_guid
            });
        }
        return out;
    }


    /// All turns belonging to a session GUID, oldest first.
    std::vector<TurnRecord> SqliteBackend::Impl::turns_by_session(const std::string& session_guid) {
        return turns_by_session_impl(session_guid, true);
    }


    // Finalize a partial turn: set assistant_text, (re)embed the exchange,
    // and record the model. Returns false if the turn doesn't exist.
    bool SqliteBackend::Impl::finalize_turn(int turn_id, const std::string& assistant_text,
                       const std::string& model_name) {
        Stmt g(db, "SELECT user_text FROM turns WHERE turn_id = ?");
        g.bind(1, turn_id);
        std::string u;
        bool found = false;
        if (g.step()) {
            found = true;
            u = g.column_text(0);
        }
        if (!found) return false;

        std::string a = normalize_path(assistant_text);
        int model_id  = get_or_create_model(model_name);
        auto emb = embedder->encode(turn_embed_text(u, a));

        Stmt s(db,
            "UPDATE turns SET assistant_text = ?, embedding_version = ?, embedding = ?, "
            "model_id = COALESCE(?, model_id) WHERE turn_id = ?");
        s.bind(1, a);
        bind_embedding_version(s.raw(), 2, emb);
        bind_embedding(s.raw(), 3, emb);
        if (model_id) s.bind(4, model_id); else s.bind_null(4);
        s.bind(5, turn_id);
        if (!s.exec()) return false;
        text_index_->index_record("turns", turn_id, u + "\n" + a);
        return true;
    }


    // ---- summaries (L2/L3) pipeline primitives (issue #22) ------------
    // Insert a summary row. level: 'turn' (L2) | 'session' (L3) | 'project'.
    // Embeds text, records model.
    // source_timestamp (non-empty) overrides the row's timestamp: L2 turn
    // summaries inherit the source turn's timestamp so the (session_id,
    // timestamp) pair links a turn to its summary (no FK column needed) and
    // stays stable across embedding-model changes.
    int SqliteBackend::Impl::store_summary(const std::string& text, const std::string& level,
                      const std::string& model_name,
                      const std::string& session_guid,
                      const std::string& source_timestamp,
                      const std::string& tags) {
        std::string t = normalize_path(text);
        int model_id   = get_or_create_model(model_name);
        int session_id = get_or_create_session(session_guid);
        auto emb = embedder->encode(t);

        Stmt s(db,
            "INSERT INTO summaries (model_id, session_id, text, embedding_version, embedding, level, tags, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)");
        if (model_id) s.bind(1, model_id); else s.bind_null(1);
        if (session_id) s.bind(2, session_id); else s.bind_null(2);
        s.bind(3, t);
        bind_embedding_version(s.raw(), 4, emb);
        bind_embedding(s.raw(), 5, emb);
        s.bind(6, level).bind(7, tags);
        {
            const int64_t ts = resolve_epoch(source_timestamp);
            s.bind(8, ts);
            s.bind(9, ts);  // updated_at == created_at on insert
        }
        if (!s.exec())
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        invalidate_cache();
        int sid = static_cast<int>(sqlite3_last_insert_rowid(db));
        text_index_->index_record("summaries", sid, t);
        return sid;
    }


    // ---- store_decision: write Level 6 curated decision/lesson ----------
    // Mirrors store_summary but the decisions table has no level/model/session
    // columns — just text/embedding/status/tags/timestamp. defer_embedding
    // leaves embedding NULL for a later backfill pass.
    int SqliteBackend::Impl::store_decision(const std::string& text, const std::string& status,
                       const std::string& tags,
                       const std::string& source_timestamp,
                       bool defer_embedding) {
        std::string t = normalize_path(strip_decision_number_prefix(text));

        Stmt s(db,
            "INSERT INTO decisions (text, embedding_version, embedding, status, tags, created_at) "
            "VALUES (?,?,?,?,?,?)");
        s.bind(1, t);
        if (defer_embedding) {
            s.bind_null(2);
            s.bind_null(3);
        } else {
            auto emb = embedder->encode(t);
            bind_embedding_version(s.raw(), 2, emb);
            bind_embedding(s.raw(), 3, emb);
        }
        s.bind(4, status.empty() ? "current" : status);
        s.bind(5, tags);
        s.bind(6, resolve_epoch(source_timestamp));
        if (!s.exec())
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        invalidate_dec_cache();
        int did = static_cast<int>(sqlite3_last_insert_rowid(db));
        text_index_->index_record("decisions", did, t);
        return did;
    }


    // ---- catch-up / recipe helpers ------------------------------------
    // Turns lacking an L2 summary, linked by (session_id, timestamp).
    // LEFT JOIN keeps turns whose summary row doesn't exist; ordered
    // newest-first so the summarizer works on the most recently active
    // turns first. This matters most for a manual resummarize (nulling
    // model_id on a batch of `summaries` rows via direct SQL): the turns
    // the user actually cares about right now are usually the recent
    // ones, and with catch_up_batch_size capping each tick, newest-first
    // means those show up with real summaries soonest instead of waiting
    // behind a long tail of old backlog.
    std::vector<TurnRecord> SqliteBackend::Impl::unsummarized_turns(int limit) {
        std::vector<TurnRecord> out;
        std::string sql =
            "SELECT t.turn_id, t.user_text, t.assistant_text, m.name, "
            "       datetime(t.created_at,'unixepoch','localtime'), COALESCE(ss.guid, '') "
            "FROM turns t "
            "LEFT JOIN turn_summaries ts ON ts.turn_id = t.turn_id "
            "LEFT JOIN models m ON t.model_id = m.model_id "
            "LEFT JOIN sessions ss ON t.session_id = ss.session_id "
            "WHERE ts.turn_id IS NULL "
            "  AND t.assistant_text IS NOT NULL "
            "ORDER BY t.created_at DESC, t.turn_id DESC";
        if (limit > 0) sql += " LIMIT ?";
        Stmt s(db, sql);
        if (limit > 0) s.bind(1, limit);
        while (s.step()) {
            out.push_back({
                s.column_int(0),
                s.column_text(1),
                s.column_text(2),
                s.column_text(3),
                s.column_text(4),
                s.column_text(5)
            });
        }
        return out;
    }


    // ---- turn_id-based turn_summaries helpers (v0.12.0) ----------------
    bool SqliteBackend::Impl::turn_summary_exists(int turn_id) {
        Stmt s(db, "SELECT 1 FROM turn_summaries WHERE turn_id = ? LIMIT 1");
        s.bind(1, turn_id);
        return s.step();
    }


    // Housekeeping retry for poison-abandoned turns: mark_turn_summarized()
    // permanently stamps a turn_summaries row with model "bad" and text=''
    // when a turn hits max_turn_failures (summarizer_service.cpp). That row
    // deliberately blocks re-enqueue via unsummarized_turns()'s LEFT JOIN
    // exclusion -- otherwise a persistently-poison turn would spin forever.
    // But whatever caused the failure (inference outage, config regression)
    // may since be fixed, and there's no other path back for these turns.
    // Deleting the abandoned row here lets the turn fall back into
    // unsummarized_turns()'s normal candidate set on the very next
    // enqueue_catch_up() pass, giving it a fresh set of max_turn_failures
    // attempts. Trivial-turn skips are NOT touched -- those are tagged with
    // the real summarizer model name, never "bad", so this only targets
    // genuine poison-abandon rows.
    int SqliteBackend::Impl::reset_abandoned_turn_summaries(int limit) {
        // SQLite's DELETE doesn't support ORDER BY/LIMIT without a special
        // build flag (SQLITE_ENABLE_UPDATE_DELETE_LIMIT), which we don't
        // assume is compiled in. Select target ids first, then delete by id.
        std::vector<int> ids;
        {
            std::string sel =
                "SELECT ts.turn_summary_id FROM turn_summaries ts "
                "JOIN models m ON m.model_id = ts.summary_model_id "
                "WHERE m.name = 'bad' AND ts.text = '' "
                "ORDER BY ts.turn_summary_id ASC";
            if (limit > 0) sel += " LIMIT ?";
            Stmt s(db, sel);
            if (limit > 0) s.bind(1, limit);
            while (s.step()) ids.push_back(s.column_int(0));
        }
        if (ids.empty()) return 0;

        std::string placeholders;
        for (size_t i = 0; i < ids.size(); ++i)
            placeholders += (i ? ",?" : "?");
        Stmt d(db, "DELETE FROM turn_summaries WHERE turn_summary_id IN (" + placeholders + ")");
        for (size_t i = 0; i < ids.size(); ++i)
            d.bind(static_cast<int>(i + 1), ids[i]);
        d.exec();
        invalidate_turn_cache();
        return static_cast<int>(ids.size());
    }


    bool SqliteBackend::Impl::finalize_turn_summary(int turn_id, const std::string& text,
                                const std::string& summary_model_name) {
        std::string t = normalize_path(text);
        int model_id = get_or_create_model(summary_model_name);
        auto emb = embedder->encode(t);

        Stmt g(db, "SELECT created_at, model_id, session_id FROM turns WHERE turn_id = ?");
        g.bind(1, turn_id);
        if (!g.step()) return false;  // turn_id must exist
        int64_t turn_dt = g.column_int64(0);
        bool turn_model_null = g.is_null(1);
        int turn_model_id = turn_model_null ? 0 : g.column_int(1);
        bool session_null = g.is_null(2);
        int session_id = session_null ? 0 : g.column_int(2);

        // Plain insert — no placeholder row exists to conflict with anymore
        // (Reid's design decision). The unique partial index
        // idx_turn_summaries_turn_id_uniq still guards against an accidental
        // duplicate finalize for the same turn_id: Stmt::exec() just checks
        // sqlite3_step() == SQLITE_DONE, so a UNIQUE-constraint violation on
        // a second finalize for the same turn_id naturally falls through to
        // `return false` below (no exception, no special-case needed) —
        // matching this file's existing convention of surfacing insert
        // failures via the bool return rather than throwing.
        Stmt s(db,
            "INSERT INTO turn_summaries "
            "(turn_id, text, embedding_version, embedding, session_id, turn_model_id, "
            " summary_model_id, turn_datetime, summarized_on) "
            "VALUES (?,?,?,?,?,?,?,?,unixepoch())");
        s.bind(1, turn_id);
        s.bind(2, t);
        bind_embedding_version(s.raw(), 3, emb);
        bind_embedding(s.raw(), 4, emb);
        if (session_null) s.bind_null(5); else s.bind(5, session_id);
        if (turn_model_null) s.bind_null(6); else s.bind(6, turn_model_id);
        if (model_id) s.bind(7, model_id); else s.bind_null(7);
        s.bind(8, turn_dt);
        bool ok = s.exec();
        if (ok) {
            invalidate_turn_cache();
            text_index_->index_record("turn_summaries", turn_id, t);
        }
        return ok;
    }


    // Create a turn_summaries row marking a turn as "done" with no summary
    // text (trivial-turn skip, or poison-turn abandonment) — mirrors
    // finalize_turn_summary's turns lookup, just with text left as ''
    // (empty string, not NULL -- an empty string unambiguously means
    // "intentionally blank"; NULL in this column would look like a data
    // integrity error rather than a deliberate skip/abandon marker).
    // Reid's design decision: no placeholder mechanism, so a turn_summaries
    // row now always represents completed work (real summary, trivial-skip,
    // or poison-abandon), never an in-progress sentinel.
    bool SqliteBackend::Impl::mark_turn_summarized(int turn_id, const std::string& model_name) {
        int model_id = get_or_create_model(model_name);
        if (!model_id) return false;

        Stmt g(db, "SELECT created_at, model_id, session_id FROM turns WHERE turn_id = ?");
        g.bind(1, turn_id);
        if (!g.step()) return false;  // turn_id must exist
        int64_t turn_dt = g.column_int64(0);
        bool turn_model_null = g.is_null(1);
        int turn_model_id = turn_model_null ? 0 : g.column_int(1);
        bool session_null = g.is_null(2);
        int session_id = session_null ? 0 : g.column_int(2);

        Stmt s(db,
            "INSERT INTO turn_summaries "
            "(turn_id, text, session_id, turn_model_id, summary_model_id, turn_datetime, summarized_on) "
            "VALUES (?,'',?,?,?,?,unixepoch())");
        s.bind(1, turn_id);
        if (session_null) s.bind_null(2); else s.bind(2, session_id);
        if (turn_model_null) s.bind_null(3); else s.bind(3, turn_model_id);
        s.bind(4, model_id);
        s.bind(5, turn_dt);
        bool ok = s.exec();
        if (ok) {
            invalidate_turn_cache();
            text_index_->index_record("turn_summaries", turn_id, "");
        }
        return ok;
    }


    // Draft-tagged summary rows for re-summarization (housekeeping retry).
    std::vector<DraftSummary> SqliteBackend::Impl::draft_summaries(int limit) {
        std::vector<DraftSummary> out;
        std::string sql =
            "SELECT s.summary_id, s.level, COALESCE(ss.guid, ''), "
            "       datetime(s.created_at,'unixepoch','localtime') "
            "FROM summaries s "
            "LEFT JOIN sessions ss ON s.session_id = ss.session_id "
            "WHERE s.tags LIKE '%draft%' "
            "ORDER BY s.created_at ASC, s.summary_id ASC";
        if (limit > 0) sql += " LIMIT ?";
        Stmt s(db, sql);
        if (limit > 0) s.bind(1, limit);
        while (s.step()) {
            out.push_back({
                s.column_int(0),
                s.column_text(1),
                s.column_text(2),
                s.column_text(3)
            });
        }
        return out;
    }


    // Sessions whose newest turn is older than (now - pause_minutes) AND
    // that have at least one non-draft L2 summary but no complete L3 yet.
    // The summarizer's pause timer treats this set as "ready to finalize."
    // Returns session GUIDs; anonymous (session_id NULL) turns are skipped.
    std::vector<std::string> SqliteBackend::Impl::sessions_needing_close(int pause_minutes) {
        std::vector<std::string> out;
        if (pause_minutes <= 0) return out;
        const std::string cutoff =
            std::format("datetime('now','localtime','-{} minutes')", pause_minutes);
        std::string sql =
            "SELECT ss.guid "
            "FROM sessions ss "
            // NOTE: sessions_needing_close() has no live callers as of the
            // Phase-2 episode/session-rollup rework (see summarizer_service.cpp's
            // "sessions_needing_close() path is retired" comment) -- this query
            // is dead code today. Redirected to turn_summaries anyway (rather
            // than left pointing at the now-empty summaries/level='turn' rows)
            // so the function stays correct if it's ever revived.
            "WHERE EXISTS (SELECT 1 FROM turn_summaries s2 "
            "               WHERE s2.session_id = ss.session_id) "
            "  AND (SELECT MAX(t.created_at) FROM turns t "
            "        WHERE t.session_id = ss.session_id) < " + cutoff + " "
            "  AND NOT EXISTS (SELECT 1 FROM summaries s "
            "                   WHERE s.session_id = ss.session_id "
            "                     AND s.level = 'session') "
            "ORDER BY ss.session_id ASC";
        Stmt s(db, sql);
        while (s.step()) {
            const auto g = s.column_text(0);
            if (!g.empty()) out.push_back(g);
        }
        return out;
    }


    // ---- episode layer (EPISODE_PLAN Phase 1) --------------------------
    // Most recent episode's span-end (stored in updated_at) for a session.
    // Empty when the session has no episode rows yet.
    std::string SqliteBackend::Impl::last_episode_end(const std::string& session_guid) {
        if (session_guid.empty()) return {};
        Stmt s(db,
            "SELECT datetime(COALESCE(s.updated_at, s.created_at),'unixepoch','localtime') "
            "FROM summaries s "
            "JOIN sessions ss ON s.session_id = ss.session_id "
            "WHERE ss.guid = ? AND s.level = 'episode' "
            "ORDER BY s.created_at DESC, s.summary_id DESC LIMIT 1");
        s.bind(1, session_guid);
        if (s.step()) return s.column_text(0);
        return {};
    }


    // Non-draft L2 turn summaries for a session with timestamp > since_ts
    // (empty since_ts = all), oldest-first. Composes the closing episode.
    // since_ts arrives as a "%F %T" string (from last_episode_end(), which
    // renders the epoch column back to that format) -- parse it back to
    // epoch seconds to compare against the INTEGER created_at column.
    std::vector<SummaryRecord> SqliteBackend::Impl::l2_summaries_since(
            const std::string& session_guid, const std::string& since_ts) {
        std::vector<SummaryRecord> out;
        if (session_guid.empty()) return out;
        // L2 turn summaries now live entirely in turn_summaries (db_version
        // 0.12) -- redirected from the old summaries/level='turn' rows.
        // turn_summaries has no `status`/`tags` columns (no draft state for
        // turn-level rows under the new design), so `status` is left empty
        // and the old `tags != 'draft'` filter is simply dropped.
        std::string sql =
            "SELECT s.turn_summary_id, s.text, '', "
            "       datetime(s.turn_datetime,'unixepoch','localtime') "
            "FROM turn_summaries s "
            "JOIN sessions ss ON s.session_id = ss.session_id "
            "WHERE ss.guid = ? AND s.text != '' ";
        if (!since_ts.empty()) sql += "AND s.turn_datetime > ? ";
        sql += "ORDER BY s.turn_datetime ASC, s.turn_summary_id ASC";
        Stmt s(db, sql);
        s.bind(1, session_guid);
        if (!since_ts.empty()) s.bind(2, resolve_epoch(since_ts));
        while (s.step()) {
            out.push_back({s.column_int(0), s.column_text(1),
                           s.column_text(2), s.column_text(3)});
        }
        return out;
    }


    // Candidate turns for similarity-based episode detection: joins
    // turn_summaries against turns to fetch both embeddings (turn + summary)
    // plus the turn-summary text. Rows where EITHER embedding is NULL are
    // skipped (boundary detection requires both signals).
    std::vector<EpisodeCandidateTurn> SqliteBackend::Impl::episode_candidate_turns(
            const std::string& session_guid, const std::string& since_ts) {
        std::vector<EpisodeCandidateTurn> out;
        if (session_guid.empty()) return out;
        const int dims = config().embedding_dimensions;

        std::string sql =
            "SELECT ts.turn_summary_id, ts.text, "
            "       datetime(ts.turn_datetime,'unixepoch','localtime'), "
            "       ts.turn_datetime, "
            "       t.embedding,  "  // raw turn embedding
            "       ts.embedding "   // turn-summary embedding
            "FROM turn_summaries ts "
            "JOIN turns t ON ts.turn_id = t.turn_id "
            "JOIN sessions ss ON ts.session_id = ss.session_id "
            "WHERE ss.guid = ? "
            "  AND t.embedding IS NOT NULL "
            "  AND ts.embedding IS NOT NULL ";
        if (!since_ts.empty()) sql += "AND ts.turn_datetime > ? ";
        sql += "ORDER BY ts.turn_datetime ASC, ts.turn_summary_id ASC";

        Stmt s(db, sql);
        s.bind(1, session_guid);
        if (!since_ts.empty()) s.bind(2, resolve_epoch(since_ts));

        bool warned = false;
        while (s.step()) {
            EpisodeCandidateTurn ct;
            ct.turn_summary_id   = s.column_int(0);
            ct.summary_text      = s.column_text(1);
            ct.timestamp         = s.column_text(2);
            ct.epoch             = s.column_int64(3);
            ct.turn_embedding    = decode_embedding_blob(
                s.column_blob(4), s.column_bytes(4), dims, "turns", ct.turn_summary_id, warned);
            ct.summary_embedding = decode_embedding_blob(
                s.column_blob(5), s.column_bytes(5), dims, "turn_summaries", ct.turn_summary_id, warned);
            out.push_back(std::move(ct));
        }
        return out;
    }


    // Insert one immutable level='episode' row: timestamp=first_ts (span
    // start), updated_at=last_ts (span end). Mirrors store_summary.
    int SqliteBackend::Impl::store_episode(const std::string& text, const std::string& model_name,
                      const std::string& session_guid,
                      const std::string& first_ts, const std::string& last_ts) {
        std::string t = normalize_path(text);
        int model_id   = get_or_create_model(model_name);
        int session_id = get_or_create_session(session_guid);
        auto emb = embedder->encode(t);

        Stmt s(db,
            "INSERT INTO summaries (model_id, session_id, text, embedding_version, embedding, "
            "level, tags, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)");
        if (model_id) s.bind(1, model_id); else s.bind_null(1);
        if (session_id) s.bind(2, session_id); else s.bind_null(2);
        s.bind(3, t);
        bind_embedding_version(s.raw(), 4, emb);
        bind_embedding(s.raw(), 5, emb);
        s.bind(6, std::string("episode"))
         .bind(7, std::string(""));
        s.bind(8, resolve_epoch(first_ts));
        s.bind(9, resolve_epoch(last_ts.empty() ? first_ts : last_ts));
        if (!s.exec())
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        invalidate_cache();
        int eid = static_cast<int>(sqlite3_last_insert_rowid(db));
        text_index_->index_record("summaries", eid, t);
        return eid;
    }


    // Sessions whose open episode is ready to close: they have >=1 non-draft
    // L2 turn summary past the last episode's end, and their newest turn is
    // older than idle_minutes. Returns session GUIDs.
    std::vector<std::string> SqliteBackend::Impl::episodes_needing_close(int idle_minutes) {
        std::vector<std::string> out;
        if (idle_minutes <= 0) return out;
        const std::string cutoff =
            std::format("(unixepoch('now','-{} minutes'))", idle_minutes);
        // For each session: last episode end (updated_at of newest episode, or
        // '' when none). "Turns needing an episode" = L2 turn_summaries rows
        // with timestamp > that end (turn_summaries has no tags/'draft'
        // state under the new design, so the old tags != 'draft' filter is
        // simply dropped here). Idle = MAX(turn.timestamp) < cutoff.
        std::string sql =
            "SELECT ss.guid "
            "FROM sessions ss "
            "WHERE EXISTS ("
            "  SELECT 1 FROM turn_summaries s "
            "  WHERE s.session_id = ss.session_id "
            "    AND s.turn_datetime > COALESCE(("
            "        SELECT COALESCE(e.updated_at, e.created_at) FROM summaries e "
            "        WHERE e.session_id = ss.session_id AND e.level = 'episode' "
            "        ORDER BY e.created_at DESC, e.summary_id DESC LIMIT 1), 0)) "
            "  AND (SELECT MAX(t.created_at) FROM turns t "
            "        WHERE t.session_id = ss.session_id) < " + cutoff + " "
            "ORDER BY ss.session_id ASC";
        Stmt s(db, sql);
        while (s.step()) {
            const auto g = s.column_text(0);
            if (!g.empty()) out.push_back(g);
        }
        return out;
    }


    // ---- Phase 2: boundary-triggered session/project rollups -----------
    // All episode texts for a session, oldest-first. Corpus for the session
    // rollup (episodes + any tail L2 turns are combined by the caller).
    std::vector<std::string> SqliteBackend::Impl::episode_texts(const std::string& session_guid) {
        std::vector<std::string> out;
        if (session_guid.empty()) return out;
        Stmt s(db,
            "SELECT s.text FROM summaries s "
            "JOIN sessions ss ON s.session_id = ss.session_id "
            "WHERE ss.guid = ? AND s.level = 'episode' "
            "ORDER BY s.created_at ASC, s.summary_id ASC");
        s.bind(1, session_guid);
        while (s.step()) {
            auto t = s.column_text(0);
            if (!t.empty()) out.push_back(std::move(t));
        }
        return out;
    }


    // Stamp updated_at = now for a running rollup row.
    bool SqliteBackend::Impl::set_summary_updated_at(int summary_id) {
        Stmt s(db, "UPDATE summaries SET updated_at = ? WHERE summary_id = ?");
        s.bind(1, db_epoch()).bind(2, summary_id);
        return s.exec() && sqlite3_changes(db) > 0;
    }


    // ---- boundary-detection watermark (settings key/value table) --------
    int64_t SqliteBackend::Impl::get_watermark(const std::string& key) {
        Stmt s(db, "SELECT value FROM settings WHERE key = ?");
        s.bind(1, key);
        if (s.step()) {
            try { return std::stoll(s.column_text(0)); } catch (...) { return 0; }
        }
        return 0;
    }


    void SqliteBackend::Impl::set_watermark(const std::string& key, int64_t turn_id) {
        Stmt s(db, "INSERT INTO settings (key, value) VALUES (?, ?) "
                   "ON CONFLICT(key) DO UPDATE SET value = excluded.value");
        s.bind(1, key);
        s.bind(2, std::to_string(turn_id));
        s.exec();
    }


    // Maximal runs of consecutive (by turn_id) turns sharing the same
    // non-NULL session_id, edge-detected via a running group id that
    // increments whenever session_id changes from the previous row. The
    // final/currently-open group (grp == MAX(grp)) is always excluded —
    // it's still open because a future turn could extend it.
    std::vector<ClosedRun> SqliteBackend::Impl::sessions_needing_close_boundary() {
        std::vector<ClosedRun> out;
        int64_t watermark = get_watermark(std::string(kSessionBoundaryWatermarkKey));
        // Structural floor (Reid's invariant: "no automatic lookback to do
        // any session summary before the last session summary"). A run is
        // eligible only if NO level='session' summary already covers it --
        // i.e. none exists for this session whose stored span end
        // (updated_at == the run's last_ts at store time) reaches this run's
        // last turn timestamp. This makes correctness independent of the
        // watermark: even with watermark=0 (e.g. a migration that never
        // stamped the key -> get_watermark() defaults to 0), an
        // already-summarized run is never re-closed. The watermark remains as
        // a cheap first-pass filter (`last_turn_id > ?`) so the common
        // steady-state scan stays bounded, but it is no longer the sole
        // guard against re-summarizing history.
        Stmt s(db, R"(
            WITH lagged AS (
                SELECT t.turn_id, t.session_id, t.created_at,
                    LAG(t.session_id) OVER (ORDER BY t.turn_id) AS prev_sid
                FROM turns t
            ),
            numbered AS (
                SELECT turn_id, session_id, created_at,
                    SUM(CASE WHEN session_id IS prev_sid THEN 0 ELSE 1 END)
                        OVER (ORDER BY turn_id) AS grp
                FROM lagged
            ),
            runs AS (
                SELECT session_id, grp,
                       MIN(turn_id) AS first_turn_id, MAX(turn_id) AS last_turn_id,
                       MIN(created_at) AS first_ts, MAX(created_at) AS last_ts
                FROM numbered
                GROUP BY grp
            )
            SELECT ss.guid, r.first_ts, r.last_ts, r.first_turn_id, r.last_turn_id
            FROM runs r
            JOIN sessions ss ON r.session_id = ss.session_id
            WHERE r.session_id IS NOT NULL
              AND r.last_turn_id > ?
              AND r.grp < (SELECT MAX(grp) FROM numbered)
              AND NOT EXISTS (
                  SELECT 1 FROM summaries sm
                  WHERE sm.level = 'session'
                    AND sm.session_id = r.session_id
                    AND sm.updated_at >= r.last_ts
              )
            ORDER BY r.first_turn_id ASC
        )");
        s.bind(1, watermark);
        while (s.step()) {
            out.push_back({s.column_text(0), s.column_int64(1), s.column_int64(2),
                            s.column_int(3), s.column_int(4)});
        }
        return out;
    }


    // Same shape as sessions_needing_close_boundary(), but grouped by a
    // time gap (>= gap_days days) between consecutive turns (by turn_id)
    // rather than a session_id change, and NOT restricted to non-NULL
    // session_id (project runs span all turns, anonymous included).
    std::vector<ClosedRun> SqliteBackend::Impl::projects_needing_close_boundary(int gap_days) {
        std::vector<ClosedRun> out;
        if (gap_days <= 0) return out;
        int64_t gap_seconds = static_cast<int64_t>(gap_days) * 86400;
        int64_t watermark = get_watermark(std::string(kProjectBoundaryWatermarkKey));
        Stmt s(db, R"(
            WITH lagged AS (
                SELECT t.turn_id, t.created_at,
                    LAG(t.created_at) OVER (ORDER BY t.turn_id) AS prev_ts
                FROM turns t
            ),
            numbered AS (
                SELECT turn_id, created_at,
                    SUM(CASE
                            WHEN prev_ts IS NULL THEN 1
                            WHEN (created_at - prev_ts) >= ? THEN 1
                            ELSE 0
                        END) OVER (ORDER BY turn_id) AS grp
                FROM lagged
            ),
            runs AS (
                SELECT grp,
                       MIN(turn_id) AS first_turn_id, MAX(turn_id) AS last_turn_id,
                       MIN(created_at) AS first_ts, MAX(created_at) AS last_ts
                FROM numbered
                GROUP BY grp
            )
            SELECT r.first_ts, r.last_ts, r.first_turn_id, r.last_turn_id
            FROM runs r
            WHERE r.last_turn_id > ?
              AND r.grp < (SELECT MAX(grp) FROM numbered)
            ORDER BY r.first_turn_id ASC
        )");
        s.bind(1, gap_seconds);
        s.bind(2, watermark);
        while (s.step()) {
            out.push_back({"", s.column_int64(0), s.column_int64(1),
                            s.column_int(2), s.column_int(3)});
        }
        return out;
    }


    void SqliteBackend::Impl::advance_session_boundary_watermark(int turn_id) {
        set_watermark(std::string(kSessionBoundaryWatermarkKey), turn_id);
    }


    void SqliteBackend::Impl::advance_project_boundary_watermark(int turn_id) {
        set_watermark(std::string(kProjectBoundaryWatermarkKey), turn_id);
    }


    // Bounded text-gathering for a closed session run: episodes whose full
    // span [created_at, COALESCE(updated_at,created_at)] lies entirely
    // within [first_ts, last_ts] for this session_guid, plus trailing
    // turn_summaries after the newest such episode's end (or from
    // first_ts if there are none) up to last_ts. Unlike episode_texts()/
    // l2_summaries_since() (unbounded above), this never reaches into a
    // LATER run of the same session_guid.
    std::vector<std::string> SqliteBackend::Impl::bounded_session_rollup_texts(
            const std::string& session_guid, int64_t first_ts, int64_t last_ts) {
        std::vector<std::string> out;
        if (session_guid.empty()) return out;

        // Episodes whose full span lies within the run, oldest-first.
        Stmt e(db,
            "SELECT s.text, COALESCE(s.updated_at, s.created_at) FROM summaries s "
            "JOIN sessions ss ON s.session_id = ss.session_id "
            "WHERE ss.guid = ? AND s.level = 'episode' "
            "  AND s.created_at >= ? AND COALESCE(s.updated_at, s.created_at) <= ? "
            "ORDER BY s.created_at ASC, s.summary_id ASC");
        e.bind(1, session_guid);
        e.bind(2, first_ts);
        e.bind(3, last_ts);
        int64_t newest_episode_end = first_ts - 1;  // sentinel: no episodes yet
        while (e.step()) {
            auto t = e.column_text(0);
            int64_t end_ts = e.column_int64(1);
            if (!t.empty()) out.push_back(std::move(t));
            if (end_ts > newest_episode_end) newest_episode_end = end_ts;
        }

        // Trailing turn_summaries strictly after the newest episode's end
        // (or from first_ts-1 if no episodes), up to and including last_ts.
        Stmt ts_q(db,
            "SELECT ts.text FROM turn_summaries ts "
            "JOIN sessions ss ON ts.session_id = ss.session_id "
            "WHERE ss.guid = ? AND ts.text != '' "
            "  AND ts.turn_datetime > ? AND ts.turn_datetime <= ? "
            "ORDER BY ts.turn_datetime ASC, ts.turn_summary_id ASC");
        ts_q.bind(1, session_guid);
        ts_q.bind(2, newest_episode_end);
        ts_q.bind(3, last_ts);
        while (ts_q.step()) {
            auto t = ts_q.column_text(0);
            if (!t.empty()) out.push_back(std::move(t));
        }
        return out;
    }


    // Bounded text-gathering for a closed project run: every level='session'
    // row whose full span [created_at, COALESCE(updated_at,created_at)]
    // lies entirely within [first_ts, last_ts], session-unscoped (a project
    // run spans sessions), oldest-first.
    std::vector<std::string> SqliteBackend::Impl::bounded_project_rollup_texts(
            int64_t first_ts, int64_t last_ts) {
        std::vector<std::string> out;
        Stmt s(db,
            "SELECT text FROM summaries "
            "WHERE level = 'session' "
            "  AND created_at >= ? AND COALESCE(updated_at, created_at) <= ? "
            "ORDER BY created_at ASC, summary_id ASC");
        s.bind(1, first_ts);
        s.bind(2, last_ts);
        while (s.step()) {
            auto t = s.column_text(0);
            if (!t.empty()) out.push_back(std::move(t));
        }
        return out;
    }


    // Insert one immutable level='session' row spanning a closed run.
    // Mirrors store_episode's shape exactly, except created_at/updated_at
    // are bound directly as epoch ints (ClosedRun already carries real
    // epoch seconds — no resolve_epoch() round-trip needed).
    int SqliteBackend::Impl::store_session_summary(const std::string& text, const std::string& model_name,
                              const std::string& session_guid,
                              int64_t first_ts, int64_t last_ts) {
        std::string t = normalize_path(text);
        int model_id   = get_or_create_model(model_name);
        int session_id = get_or_create_session(session_guid);
        auto emb = embedder->encode(t);

        Stmt s(db,
            "INSERT INTO summaries (model_id, session_id, text, embedding_version, embedding, "
            "level, tags, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)");
        if (model_id) s.bind(1, model_id); else s.bind_null(1);
        if (session_id) s.bind(2, session_id); else s.bind_null(2);
        s.bind(3, t);
        bind_embedding_version(s.raw(), 4, emb);
        bind_embedding(s.raw(), 5, emb);
        s.bind(6, std::string("session"))
         .bind(7, std::string(""));
        s.bind(8, first_ts);
        s.bind(9, last_ts);
        if (!s.exec())
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        invalidate_cache();
        int sid = static_cast<int>(sqlite3_last_insert_rowid(db));
        text_index_->index_record("summaries", sid, t);
        return sid;
    }


    // Insert one immutable level='project' row spanning a closed run.
    // Session-unscoped (session_id NULL). Otherwise mirrors store_episode.
    int SqliteBackend::Impl::store_project_summary(const std::string& text, const std::string& model_name,
                              int64_t first_ts, int64_t last_ts) {
        std::string t = normalize_path(text);
        int model_id   = get_or_create_model(model_name);
        auto emb = embedder->encode(t);

        Stmt s(db,
            "INSERT INTO summaries (model_id, session_id, text, embedding_version, embedding, "
            "level, tags, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)");
        if (model_id) s.bind(1, model_id); else s.bind_null(1);
        s.bind_null(2);
        s.bind(3, t);
        bind_embedding_version(s.raw(), 4, emb);
        bind_embedding(s.raw(), 5, emb);
        s.bind(6, std::string("project"))
         .bind(7, std::string(""));
        s.bind(8, first_ts);
        s.bind(9, last_ts);
        if (!s.exec())
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        invalidate_cache();
        int pid = static_cast<int>(sqlite3_last_insert_rowid(db));
        text_index_->index_record("summaries", pid, t);
        return pid;
    }

    std::vector<TurnRecord> SqliteBackend::Impl::turns_by_session_desc(
            const std::string& session_guid, int limit) {
        return turns_by_session_impl(session_guid, false, limit);
    }


    // Shared query: summaries for a session filtered by level, newest-first.
    std::vector<SummaryRecord> SqliteBackend::Impl::summaries_by_level_desc(
            const std::string& session_guid, const std::string& level, int limit) {
        std::vector<SummaryRecord> out;
        if (session_guid.empty()) return out;
        std::string sql =
            "SELECT s.summary_id, s.text, s.created_at "
            "FROM summaries s "
            "JOIN sessions ss ON s.session_id = ss.session_id "
            "WHERE ss.guid = ? AND s.level = ? "
            "ORDER BY s.created_at DESC, s.summary_id DESC";
        if (limit > 0) sql += " LIMIT ?";
        Stmt s(db, sql);
        s.bind(1, session_guid).bind(2, level);
        if (limit > 0) s.bind(3, limit);
        while (s.step()) {
            out.push_back({s.column_int(0), s.column_text(1),
                           std::string(), s.column_text(2)});
        }
        return out;
    }


    // L2 (turn) summaries for a session, newest-first.
    std::vector<SummaryRecord> SqliteBackend::Impl::turn_summaries_by_session_desc(
            const std::string& session_guid, int limit) {
        return summaries_by_level_desc(session_guid, "turn", limit);
    }


    // L3 (session) summaries for a session, newest-first.
    std::vector<SummaryRecord> SqliteBackend::Impl::session_summaries_desc(
            const std::string& session_guid, int limit) {
        return summaries_by_level_desc(session_guid, "session", limit);
    }


    // Exact-match existence check for import idempotency: same text +
    // same created_at timestamp already present in summaries. Used by
    // `ragger import conversations`/`import summaries` so re-running an
    // import (or feeding overlapping exports) doesn't duplicate rows.
    // Normalizes `text` the same way store_summary() does before storing
    // (absolute home-dir paths -> "~/") — otherwise a chunk containing
    // e.g. "/Users/reid/..." never matches what's actually in the table
    // (already normalized to "~/...") and gets re-inserted on every run.
    bool SqliteBackend::Impl::summary_exists_exact(const std::string& text, const std::string& created_at) {
        std::string t = normalize_path(text);
        Stmt s(db, "SELECT 1 FROM summaries WHERE text = ? AND created_at = ? LIMIT 1");
        s.bind(1, t);
        s.bind(2, created_at);
        return s.step();
    }


    // Mirrors summary_exists_exact for the decisions table (same
    // normalize-before-compare fix applies here too — plus the
    // decision-number-prefix strip, since store_decision() strips that
    // before storing too).
    bool SqliteBackend::Impl::decision_exists_exact(const std::string& text, const std::string& created_at) {
        std::string t = normalize_path(strip_decision_number_prefix(text));
        Stmt s(db, "SELECT 1 FROM decisions WHERE text = ? AND created_at = ? LIMIT 1");
        s.bind(1, t);
        s.bind(2, created_at);
        return s.step();
    }


    // Fuzzy dedup against live-captured turns: exact user_text, timestamp
    // within ±window_seconds. Catches import overlap with turns Ragger
    // captured live (identical content, a few seconds of clock skew —
    // message send time vs. capture time, plus any tz-conversion rounding).
    bool SqliteBackend::Impl::turn_exists_fuzzy(const std::string& user_text, const std::string& ts,
                           int window_seconds) {
        Stmt s(db,
            "SELECT 1 FROM turns WHERE user_text = ? "
            "AND created_at BETWEEN datetime(?, ?) AND datetime(?, ?) LIMIT 1");
        std::string neg = "-" + std::to_string(window_seconds) + " seconds";
        std::string pos = "+" + std::to_string(window_seconds) + " seconds";
        s.bind(1, user_text);
        s.bind(2, ts).bind(3, neg);
        s.bind(4, ts).bind(5, pos);
        return s.step();
    }


    // Collapse whitespace, drop zero-width/variation-selector/control
    // characters, and case-fold so the same exchange pasted through two
    // different clients (Telegram markdown escaping vs. Claude.ai's raw
    // text) compares equal. Comparison-only — never used for storage.
    std::string SqliteBackend::Impl::normalize_for_compare(const std::string& s) {
        std::string out;
        out.reserve(s.size());
        bool prev_space = false;
        for (unsigned char c : s) {
            // Drop ASCII control chars outright (keep the rest of UTF-8
            // multibyte sequences as-is; this is a coarse cross-source
            // comparison, not a full Unicode normalizer).
            if (c < 0x20 && c != '\n') continue;
            if (c == '\n' || c == '\t' || c == ' ') {
                if (!prev_space && !out.empty()) out.push_back(' ');
                prev_space = true;
                continue;
            }
            prev_space = false;
            out.push_back(static_cast<char>(std::tolower(c)));
        }
        while (!out.empty() && out.back() == ' ') out.pop_back();
        return out;
    }


    // Cross-source overlap lookup: scan turns for a user_text that
    // normalizes-equal (see normalize_for_compare), ignoring timestamp
    // entirely. Used to reconcile the same exchange captured from two
    // different importers whose timestamps have no reason to agree.
    std::optional<TurnRecord> SqliteBackend::Impl::find_turn_by_text(const std::string& user_text) {
        std::string target = normalize_for_compare(user_text);
        if (target.empty()) return std::nullopt;
        Stmt s(db,
            "SELECT t.turn_id, t.user_text, t.assistant_text, m.name, "
            "t.created_at, ss.guid "
            "FROM turns t "
            "LEFT JOIN models m ON t.model_id = m.model_id "
            "LEFT JOIN sessions ss ON t.session_id = ss.session_id "
            "WHERE t.user_text LIKE '%' || substr(?, 1, 40) || '%'");
        // Narrow with a cheap substring prefilter (avoids a full table
        // scan through normalize_for_compare on every row), then confirm
        // with the real normalized comparison in C++.
        s.bind(1, user_text.substr(0, 40));
        while (s.step()) {
            std::string cand = s.column_text(1);
            if (normalize_for_compare(cand) == target) {
                return TurnRecord{
                    s.column_int(0), cand, s.column_text(2),
                    s.column_text(3), s.column_text(4), s.column_text(5)
                };
            }
        }
        return std::nullopt;
    }


    // Upgrade an existing turn's timestamp/session_guid in place. Empty
    // args leave that field untouched. session_guid resolves/creates a
    // sessions row same as store_turn.
    bool SqliteBackend::Impl::update_turn_meta(int turn_id, const std::string& timestamp,
                          const std::string& session_guid) {
        if (timestamp.empty() && session_guid.empty()) return true;
        std::string sql = "UPDATE turns SET ";
        bool first = true;
        if (!timestamp.empty()) { sql += "created_at = ?"; first = false; }
        if (!session_guid.empty()) {
            if (!first) sql += ", ";
            sql += "session_id = ?";
        }
        sql += " WHERE turn_id = ?";
        Stmt s(db, sql);
        int idx = 1;
        if (!timestamp.empty()) s.bind(idx++, timestamp);
        if (!session_guid.empty())
            s.bind(idx++, get_or_create_session(session_guid));
        s.bind(idx, turn_id);
        return s.exec() && sqlite3_changes(db) > 0;
    }


    // Recipe ingredients (issue #23): recent summaries of a given level, and
    // current decisions — fetched by recency (not semantic search) for the
    // default tiered payload. Returned newest-first.
    std::vector<std::string> SqliteBackend::Impl::recent_summaries(const std::string& level, int limit) {
        std::vector<std::string> out;
        if (limit <= 0) return out;
        Stmt s(db,
            "SELECT text FROM summaries WHERE level = ? "
            "ORDER BY created_at DESC, summary_id DESC LIMIT ?");
        s.bind(1, level).bind(2, limit);
        while (s.step()) {
            auto text = s.column_text(0);
            if (!text.empty()) out.push_back(std::move(text));
        }
        return out;
    }


    std::vector<std::string> SqliteBackend::Impl::current_decisions(int limit) {
        std::vector<std::string> out;
        if (limit <= 0) return out;
        Stmt s(db,
            "SELECT text FROM decisions WHERE status = 'current' "
            "ORDER BY created_at DESC, decision_id DESC LIMIT ?");
        s.bind(1, limit);
        while (s.step()) {
            auto text = s.column_text(0);
            if (!text.empty()) out.push_back(std::move(text));
        }
        return out;
    }


    // Set a decision's status (e.g. "roadmap" -> "current" once planned
    // work is done, or -> "superseded"/"deprecated" once stale).
    bool SqliteBackend::Impl::set_decision_status(int decision_id, const std::string& status) {
        Stmt s(db, "UPDATE decisions SET status = ? WHERE decision_id = ?");
        s.bind(1, status).bind(2, decision_id);
        bool ok = s.exec() && sqlite3_changes(db) > 0;
        if (ok) invalidate_dec_cache();
        return ok;
    }


    // Decisions with an arbitrary status, most recent first — e.g.
    // status="roadmap" to list planned/future work explicitly (roadmap
    // entries are deliberately excluded from current_decisions()'s
    // recall-pipeline query so unfinished plans don't clutter every
    // session's context).
    std::vector<std::string> SqliteBackend::Impl::decisions_by_status(const std::string& status, int limit) {
        std::vector<std::string> out;
        if (limit <= 0) return out;
        Stmt s(db,
            "SELECT text FROM decisions WHERE status = ? "
            "ORDER BY created_at DESC, decision_id DESC LIMIT ?");
        s.bind(1, status);
        s.bind(2, limit);
        while (s.step()) {
            auto text = s.column_text(0);
            if (!text.empty()) out.push_back(std::move(text));
        }
        return out;
    }


    // Replace a summary's text + embedding, update its model. False if absent.
    bool SqliteBackend::Impl::update_summary_text(int summary_id, const std::string& text,
                             const std::string& model_name) {
        std::string t = normalize_path(text);
        int model_id  = get_or_create_model(model_name);
        auto emb = embedder->encode(t);

        Stmt s(db,
            "UPDATE summaries SET text = ?, embedding_version = ?, embedding = ?, "
            "model_id = COALESCE(?, model_id) WHERE summary_id = ?");
        s.bind(1, t);
        bind_embedding_version(s.raw(), 2, emb);
        bind_embedding(s.raw(), 3, emb);
        if (model_id) s.bind(4, model_id); else s.bind_null(4);
        s.bind(5, summary_id);
        bool ok = s.exec() && sqlite3_changes(db) > 0;
        if (ok) {
            invalidate_cache();
            text_index_->index_record("summaries", summary_id, t);
        }
        return ok;
    }


    // Replace a summary's tags column. Used by the summarizer to clear
    // "draft" once a row has been rewritten with a real summary.
    bool SqliteBackend::Impl::set_summary_tags(int summary_id, const std::string& tags) {
        Stmt s(db,
            "UPDATE summaries SET tags = ? WHERE summary_id = ?");
        s.bind(1, tags).bind(2, summary_id);
        return s.exec() && sqlite3_changes(db) > 0;
    }


    bool SqliteBackend::Impl::update_text(int memory_id, const std::string& raw_text, json metadata, bool defer_embedding) {
        if (metadata.is_null()) metadata = json::object();

        // Refuse to mutate protected rows for parity with delete_memory.
        Stmt check_stmt(db, "SELECT tags FROM summaries WHERE summary_id = ?");
        check_stmt.bind(1, memory_id);
        if (!check_stmt.step()) return false;  // row not found
        if (check_stmt.column_text(0).find("keep") != std::string::npos) return false;

        // Lean v2 summaries: only text/embedding/tags are mutable here.
        // keep/bad flags fold into tags (parity with store()).
        std::string tags_str = tags_from_metadata(metadata);

        std::string text = normalize_path(raw_text);
        std::vector<float> emb;
        if (!defer_embedding) {
            emb = embedder->encode(text);
        }

        // Custom index: re-tokenize the record on the new text (v0.16).
        Stmt stmt(db,
            "UPDATE summaries SET text = ?, embedding_version = ?, embedding = ?, tags = ? "
            "WHERE summary_id = ?");

        stmt.bind(1, text);
        if (defer_embedding) {
            stmt.bind_null(2);
            stmt.bind_null(3);
        } else {
            bind_embedding_version(stmt.raw(), 2, emb);
            bind_embedding(stmt.raw(), 3, emb);
        }
        stmt.bind(4, tags_str);
        stmt.bind(5, memory_id);

        if (!stmt.exec()) return false;

        invalidate_cache();
        text_index_->index_record("summaries", memory_id, text);
        return true;
    }


} // namespace ragger::sqlite
