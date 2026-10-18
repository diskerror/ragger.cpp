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
    std::string Backend::Impl::normalize_path(const std::string& text) {
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
    std::string Backend::Impl::strip_decision_number_prefix(const std::string& text) {
        // Fast reject (review 3C): the pattern is anchored, so a match
        // requires the first non-whitespace token (after at most two '*')
        // to begin with "Decision" or "Design" (case-insensitive). Most
        // stored texts don't — skip the regex engine for them.
        {
            size_t i = 0;
            while (i < text.size() && (text[i] == ' ' || text[i] == '\t' ||
                                       text[i] == '\n' || text[i] == '\r'))
                ++i;
            for (size_t stars = 0; i < text.size() && text[i] == '*' && stars < 2;
                 ++i, ++stars) {}
            auto ci_at = [&text](size_t at, const char* word) {
                for (const char* w = word; *w; ++w, ++at) {
                    if (at >= text.size()) return false;
                    char c = text[at];
                    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
                    if (c != *w) return false;
                }
                return true;
            };
            if (!ci_at(i, "decision") && !ci_at(i, "design"))
                return text;
        }
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
    std::string Backend::Impl::local_timestamp(std::time_t tt) { return db_timestamp(tt); }

    std::string Backend::Impl::local_timestamp() { return db_timestamp(); }


    // ---- public API ---------------------------------------------------

    // v2: the generic store API (behind /store, MCP `store`, CLI `ragger
    // store`). PHASE8: it writes a knowledge row of kind='note' — the
    // summaries table is gone. Caller fields map onto the note:
    //   status   <- metadata["status"] (default current; must be creatable)
    //   tags     <- metadata["tags"] (+ keep/bad flag folding, as ever)
    //   level    <- metadata["level"]: the column no longer exists, but
    //               nothing a caller sent is discarded — a level other than
    //               "note" is appended as a `level:<value>` tag
    //   created  <- metadata["timestamp"] (historical imports)
    //   session  <- metadata["session"] GUID, lookup only (unknown -> NULL)
    //   model    <- metadata["model"]
    // The returned id is a knowledge_id (string, as before).
    std::string Backend::Impl::store(const std::string& raw_text, json metadata, bool defer_embedding) {
        if (metadata.is_null()) metadata = json::object();

        std::string status = metadata.value("status", std::string("current"));
        {
            bool ok = false;
            for (auto sv : status::kCreatable) if (status == sv) ok = true;
            if (!ok)
                throw std::invalid_argument(
                    "status must be one of: current, provisional, pending");
        }

        // tags: accept a JSON array or a plain string. `keep`/`bad` are
        // flag-tags folded into the tags column — a row tagged "keep" is
        // protected from deletion (see delete_memory).
        std::string tags_str = tags_from_metadata(metadata);
        const std::string level = metadata.value("level", std::string());
        if (!level.empty() && level != "note") {
            if (!tags_str.empty()) tags_str += ",";
            tags_str += "level:" + level;
        }

        // Optional historical timestamp override (imports of past
        // conversations). Must be an ISO-8601 UTC string; else falls to now.
        std::string ts_override;
        if (metadata.contains("timestamp") && metadata["timestamp"].is_string())
            ts_override = metadata["timestamp"].get<std::string>();

        // Session provenance by GUID, lookup only — never create a session
        // from here (same rule as store_knowledge).
        std::optional<int> session_id;
        const std::string guid = metadata.value("session", std::string());
        if (!guid.empty()) {
            Stmt st(db, "SELECT session_id FROM sessions WHERE guid = ?");
            st.bind(1, guid);
            if (st.step()) session_id = st.column_int(0);
        }

        std::string text = normalize_path(raw_text);
        if (text.empty())
            throw std::invalid_argument("note text is required");

        std::vector<float> emb;
        if (!defer_embedding)
            emb = require_embedder().encode(text);   // trigger NULL -> text

        std::optional<int> model_id;
        const std::string model_name = metadata.value("model", std::string());
        if (!model_name.empty())
            model_id = get_or_create_model(model_name);

        Transaction txn(db);
        const int id = insert_knowledge_row(
            "note", text, status, tags_str,
            /*trigger=*/"", /*rationale=*/"", /*scope=*/"",
            /*supersedes=*/std::nullopt, session_id, model_id,
            resolve_epoch(ts_override), emb, defer_embedding);
        text_index_->index_record("knowledge", id, text);
        txn.commit();
        invalidate_knowledge_cache();
        return std::to_string(id);
    }


    // ---- store_document: write Level 5 RAG chunk ----------------------

    // Imported documents with chunk counts, by primary key. Gaps left by
    // deletions are shown as they are (the key is never renumbered).
    json Backend::Impl::list_document_sources() {
        Stmt s(db,
            "SELECT ds.document_source_id, ds.title, ds.path, ds.year, ds.tags, "
            "       datetime(ds.imported_at,'unixepoch','localtime'), "
            "       (SELECT COUNT(*) FROM documents d "
            "        WHERE d.document_source_id = ds.document_source_id) "
            "FROM document_sources ds ORDER BY ds.document_source_id");
        json out = json::array();
        while (s.step()) {
            out.push_back({
                {"id",          s.column_int64(0)},
                {"title",       s.is_null(1) ? json(nullptr) : json(s.column_text(1))},
                {"path",        s.is_null(2) ? json(nullptr) : json(s.column_text(2))},
                {"year",        s.is_null(3) ? json(nullptr) : json(s.column_int(3))},
                {"tags",        s.column_text(4)},
                {"imported_at", s.column_text(5)},
                {"chunks",      s.column_int64(6)},
            });
        }
        return out;
    }

    // Delete a document: every chunk (documents_terms rows go with them via
    // ON DELETE CASCADE; shared `terms` entries stay by design) and then the
    // source row, in ONE transaction so a failure leaves nothing half-removed.
    json Backend::Impl::delete_document_source(int document_source_id) {
        json info;
        {
            Stmt s(db,
                "SELECT title, path, "
                "  (SELECT COUNT(*) FROM documents WHERE document_source_id = ?1) "
                "FROM document_sources WHERE document_source_id = ?1");
            s.bind(1, document_source_id);
            if (!s.step())
                throw std::invalid_argument(std::format(
                    "no document with id {}", document_source_id));
            info = {{"id", document_source_id},
                    {"title", s.is_null(0) ? json(nullptr) : json(s.column_text(0))},
                    {"path",  s.is_null(1) ? json(nullptr) : json(s.column_text(1))},
                    {"chunks", s.column_int64(2)}};
        }
        Stmt(db, "BEGIN").exec();
        try {
            Stmt dc(db, "DELETE FROM documents WHERE document_source_id = ?");
            dc.bind(1, document_source_id);
            if (!dc.exec())
                throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
            Stmt ds(db, "DELETE FROM document_sources WHERE document_source_id = ?");
            ds.bind(1, document_source_id);
            if (!ds.exec())
                throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
            Stmt(db, "COMMIT").exec();
        } catch (...) {
            Stmt(db, "ROLLBACK").exec();
            throw;
        }
        invalidate_doc_cache();
        return info;
    }

    // Document source by its unique path (0.17: path alone identifies a
    // document; title is label metadata). Null json when absent.
    json Backend::Impl::find_document_source(const std::string& path) {
        Stmt s(db,
            "SELECT document_source_id, ifnull(title,''), "
            "       datetime(imported_at,'unixepoch','localtime') "
            "FROM document_sources WHERE path = ?");
        s.bind(1, path);
        if (!s.step()) return json(nullptr);
        return json{{"id", s.column_int64(0)}, {"title", s.column_text(1)},
                    {"imported_at", s.column_text(2)}};
    }

    // Resolve a document (by its unique path) to its document_sources row,
    // inserting one on first sighting. Returns {document_source_id, imported_at_epoch}.
    // imported_at_text is the caller's timestamp (TEXT 'YYYY-MM-DD HH:MM:SS'
    // or 'YYYYMMDD', interpreted LOCAL); empty -> now. On an existing source
    // the stored epoch is returned (the first import's time wins), so every
    // chunk of a document shares one imported_at.
    std::pair<long long, long long> Backend::Impl::get_or_create_document_source(
            const std::string& path, const std::string& title,
            int year, const std::string& tags,
            const std::string& imported_at_text) {
        {
            Stmt s(db,
                "SELECT document_source_id, imported_at FROM document_sources "
                "WHERE path = ?");
            s.bind(1, path);
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
        ins.bind(2, path);
        if (year <= 0) ins.bind_null(3); else ins.bind(3, year);
        ins.bind(4, tags);
        ins.bind(5, epoch);
        if (!ins.exec())
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        return { static_cast<long long>(sqlite3_last_insert_rowid(db)), epoch };
    }


    int Backend::Impl::store_document(const DocumentChunk& chunk, bool defer_embedding) {
        // documents index real files: path is the unique identity (0.17).
        if (chunk.path.empty())
            throw std::invalid_argument("document path is required");

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
            emb = require_embedder().encode(signal_text);
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

        int doc_id = -1;
        {
            Transaction tx(db);  // row + postings atomic (B §1.3)
            if (!s.exec()) {
                throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
            }
            doc_id = static_cast<int>(sqlite3_last_insert_rowid(db));
            text_index_->index_record("documents", doc_id, signal_text);
            tx.commit();
        }
        invalidate_doc_cache();  // after commit, not before (B §1.3)
        return doc_id;
    }


    // ---- turns (L1) raw exchange capture ------------------------------
    // Resolve a model name to its models.model_id, creating the row if it
    // doesn't exist. Empty name → 0 (callers bind NULL).
    int Backend::Impl::get_or_create_model(const std::string& name) {
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
    int Backend::Impl::get_or_create_session(const std::string& guid,
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
    std::string Backend::Impl::turn_embed_text(const std::string& u, const std::string& a) {
        return a.empty() ? u : (u + "\n\x1F\n" + a);
    }


    // Store a raw L1 turn. An empty assistant_text writes a *partial* row
    // (assistant + embedding NULL) for the prompt-arrival/finalize flow;
    // otherwise the exchange is embedded unless defer_embedding. FTS5 sync
    // triggers index user_text/assistant_text. Returns turn_id.
    // `source_timestamp` (non-empty, db format) overrides created_at for
    // historical imports; the regeneration-dedup window is skipped in that
    // case (it reasons about "now", which is meaningless for old turns).
    void Backend::Impl::touch_session_span(int session_id, int64_t epoch) {
        if (session_id <= 0) return;   // turn has no session
        // NOTE: two-arg scalar MIN()/MAX() are NOT used — they return NULL
        // when either side is NULL, which would pin a NULL bound forever.
        Stmt s(db,
            "UPDATE sessions SET "
            "started_on = CASE WHEN started_on IS NULL OR ?1 < started_on THEN ?1 ELSE started_on END, "
            "ended_on   = CASE WHEN ended_on IS NULL OR ?1 > ended_on   THEN ?1 ELSE ended_on END "
            "WHERE session_id = ?2");
        s.bind(1, epoch);
        s.bind(2, session_id);
        if (!s.exec())
            throw std::runtime_error(std::format(
                "touch_session_span (session {}): {}", session_id, sqlite3_errmsg(db)));
    }


    int Backend::Impl::store_turn(const std::string& user_text, const std::string& assistant_text,
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
                    std::vector<float> emb2 = require_embedder().encode(turn_embed_text(u, a));
                    // Single timestamp value shared by the turns UPDATE and the
                    // placeholder cleanup/insert below — computed once, never
                    // re-read from the clock, so the two writes can't drift
                    // apart and strand the old (session_id, prev_ts) slot.
                    const int64_t new_ts = db_epoch();
                    // Row write + postings in one transaction (B §1.3); the
                    // embedder call above stays outside on purpose.
                    Transaction tx(db);
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
                        throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
                    text_index_->index_record("turns", prev_id, u + "\n" + a);
                    // Span follows the row's effective session (the UPDATE
                    // keeps the old session when no new guid was given).
                    touch_session_span(session_id ? session_id : p.column_int(3), new_ts);
                    tx.commit();
                    return prev_id;
                }
            }
        }

        std::vector<float> emb;
        bool have_emb = false;
        if (!defer_embedding && !a.empty()) {
            emb = require_embedder().encode(turn_embed_text(u, a));
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
        const int64_t ts = resolve_epoch(source_timestamp);
        s.bind(7, ts);

        int new_id = -1;
        {
            Transaction tx(db);  // row + postings atomic (B §1.3)
            if (!s.exec())
                throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
            new_id = static_cast<int>(sqlite3_last_insert_rowid(db));
            text_index_->index_record("turns", new_id, a.empty() ? u : (u + "\n" + a));
            touch_session_span(session_id, ts);
            tx.commit();
        }
        return new_id;
    }


    // Shared implementation for turn queries over a session GUID.
    // asc=true → oldest-first (ORDER BY turn_id ASC, no limit).
    // asc=false → newest-first (ORDER BY timestamp DESC, turn_id DESC) with optional limit.
    std::vector<TurnRecord> Backend::Impl::turns_by_session_impl(
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
    std::vector<TurnRecord> Backend::Impl::turns_by_session(const std::string& session_guid) {
        return turns_by_session_impl(session_guid, true);
    }


    // Finalize a partial turn: set assistant_text, (re)embed the exchange,
    // and record the model. Returns false if the turn doesn't exist.
    bool Backend::Impl::finalize_turn(int turn_id, const std::string& assistant_text,
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
        auto emb = require_embedder().encode(turn_embed_text(u, a));

        Transaction tx(db);  // row + postings atomic (B §1.3)
        Stmt s(db,
            "UPDATE turns SET assistant_text = ?, embedding_version = ?, embedding = ?, "
            "model_id = COALESCE(?, model_id) WHERE turn_id = ?");
        s.bind(1, a);
        bind_embedding_version(s.raw(), 2, emb);
        bind_embedding(s.raw(), 3, emb);
        if (model_id) s.bind(4, model_id); else s.bind_null(4);
        s.bind(5, turn_id);
        if (!s.exec()) return false;  // tx rolls back on scope exit
        text_index_->index_record("turns", turn_id, u + "\n" + a);
        tx.commit();
        return true;
    }


    // PHASE8: the old summary-store entry point is gone — importers write rows of
    // kind='note' through store_knowledge (caller tags + `imported`); the
    // generic store() above covers the free-text path.


    // Shared INSERT for both write paths (PHASE7 B.3): the structured path
    // (store_knowledge) and the legacy store_decision wrapper. Returns the new
    // knowledge_id. `trigger` NULL for the legacy path — the trigger-required
    // rule lives in store_knowledge's validation, NOT here, exactly as the
    // old store_decision bypassed store_structured_row's checks.
    int Backend::Impl::insert_knowledge_row(const std::string& kind,
                                            const std::string& text,
                                            const std::string& status,
                                            const std::string& tags,
                                            const std::string& trigger,
                                            const std::string& rationale,
                                            const std::string& scope,
                                            std::optional<int> supersedes,
                                            std::optional<int> session_id,
                                            std::optional<int> model_id,
                                            int64_t created_at,
                                            const std::vector<float>& emb,
                                            bool defer_embedding) {
        Stmt ins(db,
            "INSERT INTO knowledge (kind, text, status, tags, trigger, rationale, scope,"
            " supersedes, session_id, model_id, created_at, embedding_version, embedding)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?, ?,?)");
        ins.bind(1, kind);
        ins.bind(2, text);
        ins.bind(3, status);
        ins.bind(4, tags);
        if (trigger.empty()) ins.bind_null(5); else ins.bind(5, trigger);
        if (rationale.empty()) ins.bind_null(6); else ins.bind(6, rationale);
        if (scope.empty())     ins.bind_null(7); else ins.bind(7, scope);
        if (supersedes)  ins.bind(8, *supersedes); else ins.bind_null(8);
        if (session_id)  ins.bind(9, *session_id); else ins.bind_null(9);
        if (model_id)    ins.bind(10, *model_id);  else ins.bind_null(10);
        ins.bind(11, created_at);
        if (defer_embedding) {
            ins.bind_null(12);
            ins.bind_null(13);
        } else {
            bind_embedding_version(ins.raw(), 12, emb);
            bind_embedding(ins.raw(), 13, emb);
        }
        if (!ins.exec())
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        return static_cast<int>(sqlite3_last_insert_rowid(db));
    }


    // ---- store_decision: legacy Level 6 curated decision/lesson ----------
    // Thin wrapper over knowledge(kind='decision') WITHOUT the trigger rule
    // (used by /store category=decision, the CLI, and importers — none of
    // which ever supplied a trigger). defer_embedding leaves embedding NULL.
    int Backend::Impl::store_decision(const std::string& text, const std::string& status,
                       const std::string& tags,
                       const std::string& source_timestamp,
                       bool defer_embedding) {
        std::string t = normalize_path(strip_decision_number_prefix(text));
        std::vector<float> emb;
        if (!defer_embedding) emb = require_embedder().encode(t);
        const int did = insert_knowledge_row(
            "decision", t, status.empty() ? "current" : status, tags,
            /*trigger=*/"", /*rationale=*/"", /*scope=*/"",
            /*supersedes=*/std::nullopt, /*session_id=*/std::nullopt,
            /*model_id=*/std::nullopt, resolve_epoch(source_timestamp),
            emb, defer_embedding);
        invalidate_knowledge_cache();
        text_index_->index_record("knowledge", did, t);
        return did;
    }


    // ---- store_knowledge: the ONE structured write path (PHASE7 B.1) ------
    // Replaces store_structured_decision + store_insight (and their
    // StructuredRowSpec): kind selects the rules via status::kKinds —
    // decision requires a trigger, insight requires evidence, note neither.
    // supersedes may point at ANY kind (insight -> decision promotion).
    // Order matters (EXACTLY as the old path):
    //   1. validate everything that can be checked without writing,
    //   2. embed OUTSIDE the transaction (the embedder can take 100+ ms and
    //      we must not hold a write lock across it),
    //   3. Transaction: INSERT; flip the superseded row; link evidence;
    //      index; commit().
    // Any throw inside step 3 rolls back, so a failed evidence link never
    // leaves a half-written row or an orphaned 'superseded' flip.
    KnowledgeResult Backend::Impl::store_knowledge(const KnowledgeInput& in) {
        auto trim = [](const std::string& s) {
            size_t a = s.find_first_not_of(" \t\r\n");
            if (a == std::string::npos) return std::string();
            size_t b = s.find_last_not_of(" \t\r\n");
            return s.substr(a, b - a + 1);
        };
        const std::string text    = normalize_path(strip_decision_number_prefix(trim(in.text)));
        const std::string trigger = trim(in.trigger);
        const std::string status  = in.status.empty() ? "current" : in.status;

        // ---- 1. validate ----------------------------------------------
        const status::KindRule* rule = status::find_kind(in.kind);
        if (!rule)
            throw std::invalid_argument("kind must be one of: decision, insight, note");
        const std::string_view noun = rule->noun;
        if (text.empty())
            throw std::invalid_argument(std::format("{} text is required", noun));
        if (rule->trigger_required && trigger.empty())
            throw std::invalid_argument(std::format("{} trigger is required", noun));
        if (rule->evidence_required && in.evidence.empty())
            throw std::invalid_argument(std::format(
                "{} requires at least one evidence turn_summary_id", noun));
        // Statuses a NEW row may start in. 'superseded' and 'retired' are
        // reached only by curation, never on create.
        {
            bool ok = false;
            for (auto sv : ragger::sqlite::status::kCreatable) if (status == sv) ok = true;
            if (!ok)
                throw std::invalid_argument(
                    "status must be one of: current, provisional, pending");
        }

        if (in.supersedes) {
            Stmt st(db, "SELECT status FROM knowledge WHERE knowledge_id = ?");
            st.bind(1, *in.supersedes);
            if (!st.step())
                throw std::invalid_argument(std::format(
                    "supersedes: knowledge {} does not exist", *in.supersedes));
            const std::string old_status = st.column_text(0);
            if (old_status == "superseded" || old_status == "retired")
                throw std::invalid_argument(std::format(
                    "supersedes: knowledge {} is already {}", *in.supersedes, old_status));
        }

        for (int ts_id : in.evidence) {
            Stmt st(db, "SELECT 1 FROM turn_summaries "
                        "WHERE turn_summary_id = ? AND text != ''");
            st.bind(1, ts_id);
            if (!st.step())
                throw std::invalid_argument(std::format(
                    "evidence: turn_summary {} does not exist or is empty", ts_id));
        }

        std::optional<int> session_id;
        if (!in.session_guid.empty()) {
            // Look up only -- never create a session from here. An unknown
            // GUID just means no provenance link (NULL), not an error.
            Stmt st(db, "SELECT session_id FROM sessions WHERE guid = ?");
            st.bind(1, in.session_guid);
            if (st.step()) session_id = st.column_int(0);
        }

        // ---- 2. embed (outside the transaction) ------------------------
        // MUST equal kKnowledgeEmbedTextSQL's result for this row, so a
        // housekeeping re-embed produces the same vector. (The SQL CASE
        // collapses to bare `text` when trigger is NULL/empty — insights
        // and notes allow that; decisions never reach the empty branch.)
        const std::string embed_text =
            trigger.empty() ? text : trigger + "\n" + text;
        auto emb = require_embedder().encode(embed_text);

        // ---- 3. write ---------------------------------------------------
        KnowledgeResult out;
        out.kind = in.kind;
        {
            Transaction txn(db);
            out.id = insert_knowledge_row(
                in.kind, text, status, in.tags, trigger, trim(in.rationale),
                trim(in.scope), in.supersedes, session_id,
                in.model_name.empty() ? std::optional<int>{}
                                      : std::optional<int>(get_or_create_model(in.model_name)),
                resolve_epoch(in.source_timestamp), emb, /*defer=*/false);

            if (in.supersedes) {
                Stmt up(db, "UPDATE knowledge SET status = 'superseded' WHERE knowledge_id = ?");
                up.bind(1, *in.supersedes);
                if (!up.exec())
                    throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
                out.superseded = *in.supersedes;
            }

            for (int ts_id : in.evidence) {
                // OR IGNORE: a caller listing the same id twice is harmless.
                Stmt ev(db, "INSERT OR IGNORE INTO knowledge_turn_evidence "
                            "(knowledge_id, turn_summary_id) VALUES (?,?)");
                ev.bind(1, out.id);
                ev.bind(2, ts_id);
                if (!ev.exec())
                    throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
            }

            text_index_->index_record("knowledge", out.id, embed_text);
            txn.commit();
        }
        invalidate_knowledge_cache();

        // ---- possible_supersedes hint (Phase 2 step 5) -------------------
        // Flag only: up to 3 OTHER active rows of kind decision|insight (a
        // note is never a hint target) whose embedding cosine with the new
        // row reaches decision_supersede_threshold. The row the call already
        // superseded is skipped. NO row is ever changed here.
        if (in.kind != "note") {
            std::vector<int> exclude{out.id};
            if (out.superseded) exclude.push_back(*in.supersedes);
            out.possible_supersedes = supersede_hints(
                emb, exclude, "kind IN ('decision','insight')");
        }
        return out;
    }


    // Cosine near-duplicate scan over active knowledge rows (Phase 2 step 5;
    // PHASE7: one table, optional kind restriction). Purely
    // vector-based on purpose: a near-duplicate ruling is exactly what this
    // looks like in embedding space, and lexical overlap is already guaranteed
    // by the shared trigger vocabulary. Returns up to 3 (id, cosine) pairs at
    // or above decision_supersede_threshold, highest first. Never changes a row.
    std::vector<std::pair<int,float>> Backend::Impl::supersede_hints(
            const std::vector<float>& emb,
            const std::vector<int>& exclude_ids,
            const char* kind_filter) {
        std::vector<std::pair<int,float>> scored;
        if (emb.empty()) return scored;
        Eigen::Map<const Eigen::VectorXf> v(emb.data(), static_cast<int>(emb.size()));
        float vnorm = v.norm();
        if (vnorm <= 1e-12f) return scored;
        const float threshold = config().decision_supersede_threshold;
        const int expected_dims = config().embedding_dimensions;
        bool blob_warned = false;
        std::string sql =
            "SELECT knowledge_id, embedding FROM knowledge WHERE status IN (" +
            status::in_list_sql(status::kSearchActive) + ") ";
        if (kind_filter) sql += std::format("  AND ({}) ", kind_filter);
        for (size_t i = 0; i < exclude_ids.size(); ++i)
            sql += std::format("  AND knowledge_id != ? ");
        sql += "  AND embedding IS NOT NULL";
        Stmt s(db, sql);
        for (size_t i = 0; i < exclude_ids.size(); ++i)
            s.bind(static_cast<int>(i) + 1, exclude_ids[i]);
        while (s.step()) {
            std::vector<float> other = decode_embedding_blob(
                s.column_blob(1), s.column_bytes(1), expected_dims,
                "knowledge", s.column_int(0), blob_warned);
            if (static_cast<int>(other.size()) != expected_dims) continue;
            Eigen::Map<Eigen::VectorXf> o(other.data(), expected_dims);
            float onorm = o.norm();
            if (onorm <= 1e-12f) continue;   // zero/undecodable vector
            float cos = (v.dot(o)) / (vnorm * onorm);
            if (cos >= threshold) scored.emplace_back(s.column_int(0), cos);
        }
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        if (scored.size() > 3) scored.resize(3);
        return scored;
    }


    std::vector<std::pair<int,float>> Backend::Impl::possible_supersedes_for_text(
            const std::string& text) {
        if (!embedder || !embedder->ready() || text.empty())
            return {};                       // degraded: no hints (per P2 rule)
        return supersede_hints(require_embedder().encode(text), {});
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
    std::vector<TurnRecord> Backend::Impl::unsummarized_turns(int limit) {
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
    bool Backend::Impl::turn_summary_exists(int turn_id) {
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
    // a real model name or the "(skipped)" pseudo-model (PHASE6 3), never
    // "bad", so this only targets genuine poison-abandon rows.
    int Backend::Impl::reset_abandoned_turn_summaries(int limit) {
        // SQLite's DELETE doesn't support ORDER BY/LIMIT without a special
        // build flag (SQLITE_ENABLE_UPDATE_DELETE_LIMIT), which we don't
        // assume is compiled in. Select target ids first, then delete by id.
        std::vector<int> ids;
        {
            std::string sel =
                "SELECT ts.turn_summary_id FROM turn_summaries ts "
                "JOIN models m ON m.model_id = ts.summary_model_id "
                "WHERE m.name = 'bad' AND ts.text = '' AND ts.turn_id IS NOT NULL "
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
        if (!d.exec())
            throw std::runtime_error(std::format("reset_abandoned_turn_summaries delete: {}",
                                                 sqlite3_errmsg(db)));
        invalidate_turn_cache();
        return static_cast<int>(ids.size());
    }


    bool Backend::Impl::finalize_turn_summary(int turn_id, const std::string& text,
                                const std::string& summary_model_name) {
        std::string t = normalize_path(text);
        int model_id = get_or_create_model(summary_model_name);
        auto emb = require_embedder().encode(t);

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
        // The index addresses turn_summaries by turn_summary_id, NOT turn_id
        // (they diverge on any real DB). The INSERT+index pair must be atomic:
        // a failed index rolls the summary row back instead of leaving an
        // unindexed or mis-indexed row.
        Transaction tx(db);
        bool ok = s.exec();
        if (ok) {
            // Read the new rowid BEFORE any other statement runs on db.
            const int ts_id = static_cast<int>(sqlite3_last_insert_rowid(db));
            text_index_->index_record("turn_summaries", ts_id, t);
            tx.commit();
            invalidate_turn_cache();
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
    bool Backend::Impl::mark_turn_summarized(int turn_id, const std::string& model_name) {
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
        // Keying + atomicity: see finalize_turn_summary above. The empty text
        // indexes to zero terms but set_counts must still land on THIS row.
        Transaction tx(db);
        bool ok = s.exec();
        if (ok) {
            // Read the new rowid BEFORE any other statement runs on db.
            const int ts_id = static_cast<int>(sqlite3_last_insert_rowid(db));
            text_index_->index_record("turn_summaries", ts_id, "");
            tx.commit();
            invalidate_turn_cache();
        }
        return ok;
    }


    // ---- episode layer (db 0.17) ----------------------------------------
    // Most recent episode's span_end for a session, rendered "%F %T" local
    // so callers/tests keep the historic shape. Empty when no episodes.
    std::string Backend::Impl::last_episode_end(const std::string& session_guid) {
        if (session_guid.empty()) return {};
        Stmt s(db,
            "SELECT datetime(e.span_end,'unixepoch','localtime') "
            "FROM episodes e "
            "JOIN sessions ss ON e.session_id = ss.session_id "
            "WHERE ss.guid = ? "
            "ORDER BY e.span_start DESC, e.episode_id DESC LIMIT 1");
        s.bind(1, session_guid);
        if (s.step()) return s.column_text(0);
        return {};
    }


    // Non-draft L2 turn summaries for a session with timestamp > since_ts
    // (empty since_ts = all), oldest-first. Composes the closing episode.
    // since_ts arrives as a "%F %T" string (from last_episode_end(), which
    // renders the epoch column back to that format) -- parse it back to
    // epoch seconds to compare against the INTEGER created_at column.
    std::vector<SummaryRecord> Backend::Impl::l2_summaries_since(
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
    // PHASE6 6: ONE predicate for "pending content L2 of session X after
    // epoch Y". Both episodes_needing_close() and episode_candidate_turns()
    // build their WHERE clauses from this, so they cannot drift apart again
    // (the old disagreement: the scan counted any L2 past the last episode,
    // the candidate query additionally required embeddings — a session whose
    // newest L2 had no embedding yet was re-queued every tick forever).
    // Requires `turn_summaries ts JOIN turns t ON t.turn_id = ts.turn_id`.
    // Rows whose source turn was culled (ts.turn_id NULL) can never be
    // candidates — the JOIN drops them; they cannot close an episode either.
    std::string pending_episode_l2_conditions(const std::string& epoch_expr) {
        return " AND ts.text != '' "
               " AND ts.episode_id IS NULL "
               " AND ts.turn_datetime > " + epoch_expr + " "
               " AND ts.embedding IS NOT NULL "
               " AND t.embedding IS NOT NULL ";
    }

    std::vector<EpisodeCandidateTurn> Backend::Impl::episode_candidate_turns(
            const std::string& session_guid, const std::string& since_ts) {
        std::vector<EpisodeCandidateTurn> out;
        if (session_guid.empty()) return out;
        const int dims = config().embedding_dimensions;

        // PHASE6 6: WHERE clause from the shared predicate. episode_id IS
        // NULL means rows already inside a closed episode are never returned
        // again, even when their turn_datetime falls after `since` (a late
        // L2 landing inside a span that was closed around it).
        const std::string epoch_expr = since_ts.empty() ? "0" : "?";
        std::string sql =
            "SELECT ts.turn_summary_id, ts.text, "
            "       datetime(ts.turn_datetime,'unixepoch','localtime'), "
            "       ts.turn_datetime, "
            "       t.embedding,  "  // raw turn embedding
            "       ts.embedding "   // turn-summary embedding
            "FROM turn_summaries ts "
            "JOIN turns t ON ts.turn_id = t.turn_id "
            "JOIN sessions ss ON ts.session_id = ss.session_id "
            "WHERE ss.guid = ? " +
            pending_episode_l2_conditions(epoch_expr) +
            "ORDER BY ts.turn_datetime ASC, ts.turn_summary_id ASC";

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


    // Close an episode (PHASE5 B.1): one atomic episode + membership record.
    // The contract ORDER is the point: validate -> embed (outside the txn) ->
    // INSERT -> per-member UPDATE (all-or-nothing) -> index -> commit ->
    // invalidate. See storage_backend.h for the full contract.
    int Backend::Impl::store_episode_close(const EpisodeClose& close) {
        if (close.text.empty())
            throw std::invalid_argument("store_episode_close: empty text");
        if (close.member_turn_summary_ids.empty())
            throw std::invalid_argument("store_episode_close: no members");
        if (close.span_end < close.span_start)
            throw std::invalid_argument("store_episode_close: span_end < span_start");
        const int session_id = get_or_create_session(close.session_guid);
        if (!session_id)
            throw std::invalid_argument("store_episode_close: unknown session '" +
                                        close.session_guid + "'");

        const std::string t = normalize_path(close.text);
        const int model_id = get_or_create_model(close.model_name);

        // Compute the embedding OUTSIDE the transaction: encoding is slow and
        // holding the write lock across a model call would block every other
        // writer.
        auto emb = require_embedder().encode(t);

        Transaction tx(db);
        Stmt ins(db,
            "INSERT INTO episodes (text, tags, session_id, model_id, "
            "span_start, span_end, embedding_version, embedding) "
            "VALUES (?,?,?,?,?,?,?,?)");
        ins.bind(1, t).bind(2, std::string(""));
        ins.bind(3, session_id);
        if (model_id) ins.bind(4, model_id); else ins.bind_null(4);
        ins.bind(5, close.span_start).bind(6, close.span_end);
        bind_embedding_version(ins.raw(), 7, emb);
        bind_embedding(ins.raw(), 8, emb);
        if (!ins.exec()) {
            if (sqlite3_extended_errcode(db) == SQLITE_CONSTRAINT_UNIQUE) {
                // Someone already closed this (session, span_start): not an
                // error. The Transaction destructor rolls back; the caller
                // logs "already closed" and does not retry.
                return 0;
            }
            throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
        }
        const int episode_id = static_cast<int>(sqlite3_last_insert_rowid(db));

        // Membership: each member must exist in THIS session, be unassigned,
        // and not be a marker row. Any straggler (foreign session, already
        // closed, marker, missing) comes up short on the change count and the
        // whole close rolls back — nothing half-written survives.
        int assigned = 0;
        for (const int ts_id : close.member_turn_summary_ids) {
            Stmt u(db,
                "UPDATE turn_summaries SET episode_id = ? "
                "WHERE turn_summary_id = ? AND session_id = ? "
                "  AND episode_id IS NULL AND text != ''");
            u.bind(1, episode_id).bind(2, ts_id).bind(3, session_id);
            if (!u.exec())
                throw std::runtime_error(std::format(lang::ERR_STORE_FAILED, sqlite3_errmsg(db)));
            assigned += sqlite3_changes(db);
        }
        if (assigned != static_cast<int>(close.member_turn_summary_ids.size()))
            throw std::runtime_error(std::format(
                "store_episode_close: {} of {} members assignable "
                "(foreign session / already closed / marker / missing); rolled back",
                assigned, close.member_turn_summary_ids.size()));

        text_index_->index_record("episodes", episode_id, t);
        tx.commit();
        invalidate_episode_cache();
        return episode_id;
    }


    // Sessions whose open episode is ready to close: >=1 non-empty L2 turn
    // summary past the last episode's span_end (0 when none), and their
    // newest turn is older than idle_minutes. Returns session GUIDs.
    std::vector<std::string> Backend::Impl::episodes_needing_close(int idle_minutes) {
        std::vector<std::string> out;
        if (idle_minutes <= 0) return out;
        const std::string cutoff =
            std::format("(unixepoch('now','-{} minutes'))", idle_minutes);
        // EXISTS(shared predicate): content L2 past the last episode end,
        // embeddings present on both sides (verified-bug fix for ''-text
        // marker rows included in the predicate). PHASE6 6 additionally
        // requires that NO turn of the session is still unsummarized — an
        // episode must not close while a late L2 could still land inside
        // the span it just wrote. A bad/skip marker row counts as
        // summarized (its turn_summaries row exists).
        std::string sql =
            "SELECT ss.guid "
            "FROM sessions ss "
            "WHERE EXISTS ("
            "  SELECT 1 FROM turn_summaries ts "
            "  JOIN turns t ON t.turn_id = ts.turn_id "
            "  WHERE ts.session_id = ss.session_id " +
            pending_episode_l2_conditions(
                "COALESCE((SELECT MAX(e.span_end) FROM episodes e "
                "WHERE e.session_id = ss.session_id), 0)") +
            ") "
            "  AND NOT EXISTS ("
            "    SELECT 1 FROM turns t2 "
            "    WHERE t2.session_id = ss.session_id "
            "      AND t2.assistant_text IS NOT NULL AND t2.assistant_text <> '' "
            "      AND NOT EXISTS (SELECT 1 FROM turn_summaries x "
            "                      WHERE x.turn_id = t2.turn_id)) "
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



    std::vector<TurnRecord> Backend::Impl::turns_by_session_desc(
            const std::string& session_guid, int limit) {
        return turns_by_session_impl(session_guid, false, limit);
    }




    // L2 (turn) summaries for a session, newest-first (PHASE5 B.5: reads
    // turn_summaries directly — the summaries.level='turn' rows this used to
    // query have been empty since 0.12).
    std::vector<SummaryRecord> Backend::Impl::turn_summaries_by_session_desc(
            const std::string& session_guid, int limit) {
        std::vector<SummaryRecord> out;
        if (session_guid.empty()) return out;
        std::string sql =
            "SELECT ts.turn_summary_id, ts.text, '', "
            "       datetime(ts.turn_datetime,'unixepoch','localtime') "
            "FROM turn_summaries ts "
            "JOIN sessions ss ON ts.session_id = ss.session_id "
            "WHERE ss.guid = ? AND ts.text != '' "
            "ORDER BY ts.turn_datetime DESC, ts.turn_summary_id DESC";
        if (limit > 0) sql += " LIMIT ?";
        Stmt s(db, sql);
        s.bind(1, session_guid);
        if (limit > 0) s.bind(2, limit);
        while (s.step()) {
            out.push_back({s.column_int(0), s.column_text(1),
                           s.column_text(2), s.column_text(3)});
        }
        return out;
    }


    // This session's episodes, newest-first (span_start DESC). Timestamp
    // carries the rendered span_end (the episode's "when it ended").
    std::vector<SummaryRecord> Backend::Impl::episodes_by_session_desc(
            const std::string& session_guid, int limit) {
        std::vector<SummaryRecord> out;
        if (session_guid.empty()) return out;
        std::string sql =
            "SELECT e.episode_id, e.text, '', "
            "       datetime(e.span_end,'unixepoch','localtime') "
            "FROM episodes e "
            "JOIN sessions ss ON e.session_id = ss.session_id "
            "WHERE ss.guid = ? AND e.text != '' "
            "ORDER BY e.span_start DESC, e.episode_id DESC";
        if (limit > 0) sql += " LIMIT ?";
        Stmt s(db, sql);
        s.bind(1, session_guid);
        if (limit > 0) s.bind(2, limit);
        while (s.step()) {
            out.push_back({s.column_int(0), s.column_text(1),
                           s.column_text(2), s.column_text(3)});
        }
        return out;
    }



    // Exact-match existence check for import idempotency: same text +
    // same created_at timestamp already present among knowledge notes.
    // Used by the importers so re-running an import (or feeding overlapping
    // exports) doesn't duplicate rows. Normalizes `text` the same way the
    // note write path does (strip the Decision/Design-number label, trim,
    // then absolute home-dir paths -> "~/") — otherwise a chunk containing
    // e.g. "/Users/reid/..." never matches what's actually in the table
    // (already normalized to "~/...") and gets re-inserted on every run.
    bool Backend::Impl::note_exists_exact(const std::string& text, const std::string& created_at) {
        std::string t = normalize_path(strip_decision_number_prefix(text));
        // created_at is an INTEGER epoch; binding the "%F %T" string never
        // matched (SQLite does no implicit conversion), so the probe was
        // always false and every re-import duplicated rows. Use the same
        // resolve_epoch the store path uses, so probe and insert agree.
        Stmt s(db, "SELECT 1 FROM knowledge WHERE kind = 'note' AND text = ? AND created_at = ? LIMIT 1");
        s.bind(1, t);
        s.bind(2, resolve_epoch(created_at));
        return s.step();
    }


    // Mirrors note_exists_exact for the decisions table (same
    // normalize-before-compare fix applies here too — plus the
    // decision-number-prefix strip, since store_decision() strips that
    // before storing too).
    bool Backend::Impl::decision_exists_exact(const std::string& text, const std::string& created_at) {
        std::string t = normalize_path(strip_decision_number_prefix(text));
        Stmt s(db, "SELECT 1 FROM knowledge WHERE kind = 'decision' AND text = ? AND created_at = ? LIMIT 1");
        s.bind(1, t);
        s.bind(2, resolve_epoch(created_at));  // INTEGER epoch, see note_exists_exact
        return s.step();
    }


    // Fuzzy dedup against live-captured turns: exact user_text, timestamp
    // within ±window_seconds. Catches import overlap with turns Ragger
    // captured live (identical content, a few seconds of clock skew —
    // message send time vs. capture time, plus any tz-conversion rounding).
    bool Backend::Impl::turn_exists_fuzzy(const std::string& user_text, const std::string& ts,
                           int window_seconds) {
        // created_at is an INTEGER epoch — compare against resolved integer
        // bounds (the old datetime(?,?) form compared TEXT to INTEGER and
        // never matched).
        const int64_t ep = resolve_epoch(ts);
        Stmt s(db,
            "SELECT 1 FROM turns WHERE user_text = ? "
            "AND created_at BETWEEN ? AND ? LIMIT 1");
        s.bind(1, user_text);
        s.bind(2, ep - window_seconds);
        s.bind(3, ep + window_seconds);
        return s.step();
    }


    // Collapse whitespace, drop zero-width/variation-selector/control
    // characters, and case-fold so the same exchange pasted through two
    // different clients (Telegram markdown escaping vs. Claude.ai's raw
    // text) compares equal. Comparison-only — never used for storage.
    std::string Backend::Impl::normalize_for_compare(const std::string& s) {
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
    std::optional<TurnRecord> Backend::Impl::find_turn_by_text(const std::string& user_text) {
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
    bool Backend::Impl::update_turn_meta(int turn_id, const std::string& timestamp,
                          const std::string& session_guid) {
        if (timestamp.empty() && session_guid.empty()) return true;

        // Bounds are DERIVED here, not touched: a timestamp move can leave
        // the old span stale in either direction, so both affected sessions
        // get a full MIN/MAX recompute from their remaining turns. A session
        // left with no turns goes back to NULL/NULL.
        auto recompute_span = [&](int sid) {
            if (sid <= 0) return;
            Stmt q(db, "SELECT MIN(created_at), MAX(created_at) FROM turns WHERE session_id = ?");
            q.bind(1, sid);
            Stmt u(db, "UPDATE sessions SET started_on = ?, ended_on = ? WHERE session_id = ?");
            if (q.step() && !q.is_null(0)) {
                u.bind(1, q.column_int64(0));
                u.bind(2, q.column_int64(1));
            } else {
                u.bind_null(1);
                u.bind_null(2);
            }
            u.bind(3, sid);
            if (!u.exec())
                throw std::runtime_error(std::format(
                    "update_turn_meta recompute (session {}): {}", sid, sqlite3_errmsg(db)));
        };

        int old_session_id = 0;
        {
            Stmt f(db, "SELECT session_id FROM turns WHERE turn_id = ?");
            f.bind(1, turn_id);
            if (f.step() && !f.is_null(0)) old_session_id = f.column_int(0);
        }

        std::string sql = "UPDATE turns SET ";
        bool first = true;
        int new_session_id = 0;
        if (!timestamp.empty()) { sql += "created_at = ?"; first = false; }
        if (!session_guid.empty()) {
            if (!first) sql += ", ";
            sql += "session_id = ?";
        }
        sql += " WHERE turn_id = ?";
        Transaction tx(db);
        Stmt s(db, sql);
        int idx = 1;
        if (!timestamp.empty()) s.bind(idx++, resolve_epoch(timestamp));  // INTEGER epoch
        if (!session_guid.empty()) {
            new_session_id = get_or_create_session(session_guid);
            s.bind(idx++, new_session_id);
        }
        s.bind(idx, turn_id);
        const bool changed = s.exec() && sqlite3_changes(db) > 0;
        if (changed) {
            recompute_span(old_session_id);
            if (new_session_id && new_session_id != old_session_id)
                recompute_span(new_session_id);
        }
        tx.commit();
        return changed;
    }


    // Recipe ingredients (issue #23): current decisions — fetched by recency
    // (not semantic search) for the default tiered payload. Returned
    // newest-first. (PHASE8: the old recent-summary fetch is gone with its
    // table; recipes read episodes since PHASE5.)
    std::vector<std::string> Backend::Impl::current_decisions(int limit) {
        std::vector<std::string> out;
        if (limit <= 0) return out;
        Stmt s(db,
            "SELECT text FROM knowledge WHERE kind = 'decision' AND status = 'current' "
            "ORDER BY created_at DESC, knowledge_id DESC LIMIT ?");
        s.bind(1, limit);
        while (s.step()) {
            auto text = s.column_text(0);
            if (!text.empty()) out.push_back(std::move(text));
        }
        return out;
    }


    // Set a decision's status (e.g. "roadmap" -> "current" once planned
    // work is done, or -> "superseded"/"deprecated" once stale).
    bool Backend::Impl::set_decision_status(int decision_id, const std::string& status) {
        Stmt s(db, "UPDATE knowledge SET status = ? WHERE kind = 'decision' AND knowledge_id = ?");
        s.bind(1, status).bind(2, decision_id);
        bool ok = s.exec() && sqlite3_changes(db) > 0;
        if (ok) invalidate_knowledge_cache();
        return ok;
    }


    // Decisions with an arbitrary status, most recent first — e.g.
    // status="roadmap" to list planned/future work explicitly (roadmap
    // entries are deliberately excluded from current_decisions()'s
    // recall-pipeline query so unfinished plans don't clutter every
    // session's context).
    std::vector<std::string> Backend::Impl::decisions_by_status(const std::string& status, int limit) {
        std::vector<std::string> out;
        if (limit <= 0) return out;
        Stmt s(db,
            "SELECT text FROM knowledge WHERE kind = 'decision' AND status = ? "
            "ORDER BY created_at DESC, knowledge_id DESC LIMIT ?");
        s.bind(1, status);
        s.bind(2, limit);
        while (s.step()) {
            auto text = s.column_text(0);
            if (!text.empty()) out.push_back(std::move(text));
        }
        return out;
    }


    // PHASE8: the old summary text/tag update paths are gone.
    // Their only live callers were the draft path (removed in PHASE5) and the
    // public facade; honest history wins — text is never rewritten in place
    // (store a new row with supersedes), and tags/status change through
    // agent_update on knowledge:<id>.

} // namespace ragger::sqlite
