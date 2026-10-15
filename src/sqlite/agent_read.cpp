// Agent API Phase 2 reads: layered search (memory_search) and full-row reads
// with hierarchy expansion / conversation-history paging (memory_get).
//
// Design contract lives in StorageBackend's docs and PHASE2-STEPS.md; this
// file implements Backend::Impl::agent_search / agent_get only. The hybrid
// scoring itself is reused untouched (Impl::search); everything here is
// post-filtering, handle formatting, and row fetches.
#include "sqlite/backend_impl.h"
#include "embedder.h"
#include "config.h"
#include "util/time.h"
#include "util/sqlite.h"
#include "sqlite/text_index.h"
#include <format>
#include "nlohmann_json.hpp"

#include <sqlite3.h>
#include <Eigen/Dense>

#include <algorithm>
#include <ctime>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace ragger::sqlite {

using json = nlohmann::json;

namespace {

// ---- since/until resolution -------------------------------------------
// Accepted formats (PHASE2-STEPS step 3): "YYYY-MM-DD",
// "YYYY-MM-DD HH:MM:SS", or relative "<N>d" / "<N>h" (ago). Returns a local
// "YYYY-MM-DD HH:MM:SS" string so the comparison against the rendered
// datetime() columns is a plain lexicographic compare (same format, same
// local timezone). A bare date for `until` expands to end-of-day so the
// day itself is inclusive. Anything else -> invalid_argument.
std::string resolve_agent_time_bound(const std::string& s, bool end_of_day) {
    auto fail = [&s]() -> std::string {
        throw std::invalid_argument("bad since/until value: '" + s + "'");
    };
    static const std::regex re_date(R"(^(\d{4})-(\d{2})-(\d{2})$)");
    static const std::regex re_dt(R"(^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})$)");
    static const std::regex re_rel(R"(^(\d+)([dh])$)");

    auto valid_date = [](int mo, int d) { return mo >= 1 && mo <= 12 && d >= 1 && d <= 31; };
    auto valid_time = [](int h, int mi, int se) { return h <= 23 && mi <= 59 && se <= 59; };

    std::smatch m;
    if (std::regex_match(s, m, re_date)) {
        if (!valid_date(std::stoi(m[2]), std::stoi(m[3]))) fail();
        return s + (end_of_day ? " 23:59:59" : " 00:00:00");
    }
    if (std::regex_match(s, m, re_dt)) {
        if (!valid_date(std::stoi(m[2]), std::stoi(m[3]))) fail();
        if (!valid_time(std::stoi(m[4]), std::stoi(m[5]), std::stoi(m[6]))) fail();
        return s;
    }
    if (std::regex_match(s, m, re_rel)) {
        long n = std::stol(m[1].str());
        if (n > 100000000L) fail();   // absurd offset; reject rather than overflow
        long long secs = (m[2].str() == "d") ? n * 86400LL : n * 3600LL;
        std::time_t tt = static_cast<std::time_t>(db_epoch() - secs);
        return db_timestamp(tt);
    }
    return fail();
}

// ---- snippet -----------------------------------------------------------
// First <=240 BYTES, cut at a whitespace boundary; "…" (3 bytes UTF-8) is
// appended only when the text was actually cut. When no whitespace exists in
// the window (one giant token), back off to a UTF-8 lead boundary so a
// multi-byte character is never split.
std::string agent_snippet(const std::string& text) {
    constexpr size_t kMax = 240;
    if (text.size() <= kMax) return text;
    size_t cut = kMax;
    size_t sp = text.find_last_of(" \t\n\r", kMax);
    if (sp != std::string::npos && sp > 0) {
        cut = sp;
    } else {
        while (cut > 0 && (static_cast<unsigned char>(text[cut]) & 0xC0) == 0x80)
            --cut;
    }
    return text.substr(0, cut) + "…";
}

// ---- typed handle ------------------------------------------------------
// "<layer>:<rowid>" (API format only, never stored). Strict parse: exactly
// one colon, known layer, all-digits positive rowid.
bool parse_agent_handle(const std::string& handle, std::string& layer, int& rowid) {
    static const std::regex re(R"(^(turn|turn_summary|summary|document|decision|insight):([1-9][0-9]*)$)");
    std::smatch m;
    if (!std::regex_match(handle, m, re)) return false;
    layer = m[1].str();
    rowid = std::stoi(m[2].str());
    return true;
}

} // anonymous namespace


// =======================================================================
// agent_search — one hybrid search over a wide pool, then post-filters:
// layer, time window, session/exclude_session, decision status. Returns
// handles + snippets only (full text via agent_get).
// =======================================================================
std::vector<AgentHit> Backend::Impl::agent_search(const AgentSearchQuery& q) {
    // ---- 1. validate ----------------------------------------------------
    static const std::set<std::string> kKnownLayers = {
        "turn", "turn_summary", "summary", "document", "decision", "insight"};
    for (const auto& l : q.layers)
        if (!kKnownLayers.count(l))
            throw std::invalid_argument("unknown layer: '" + l + "'");
    const std::string since_b = q.since.empty() ? "" : resolve_agent_time_bound(q.since, false);
    const std::string until_b = q.until.empty() ? "" : resolve_agent_time_bound(q.until, true);
    const int limit = std::clamp(q.limit, 1, 50);

    // ---- 2. wide-pool hybrid search --------------------------------------
    // Degraded (no usable embedder) mirrors RaggerMemory::search's fallback:
    // text-only scoring across the same four corpora.
    const int pool = std::max(limit * 5, 50);
    std::vector<SearchResult> cands;
    if (embedder && embedder->ready())
        cands = search(q.query, pool, 0.0f, {}).results;
    else
        cands = search_text_only(q.query, pool).results;

    // Decision/insight status comes from the row, never from cache metadata:
    // the hybrid cache only holds active rows and the text-only path carries
    // no status at all, so one authoritative lookup per hit keeps every path
    // consistent. (Pool is <= 50 hits; the extra queries are trivial.)
    auto status_of = [this](const char* table, const char* pk,
                            int id) -> std::string {
        Stmt s(db, std::format("SELECT status FROM {} WHERE {} = ?", table, pk));
        s.bind(1, id);
        return s.step() ? s.column_text(0) : std::string();
    };

    // 'active' / exact-status needs rows the hybrid cache deliberately hides
    // (superseded/retired/pending). SEPARATE query path — ensure_dec_cache()
    // / ensure_ins_cache() keep their narrow allow-lists for normal search.
    // Lexical-only signal; ids already scored by the hybrid pass are not
    // duplicated. Applied to both structured corpora (decisions, insights).
    if (q.status != "active") {
        text_index_->set_weights({config().fts_w_unigram, config().fts_w_bigram,
                                  config().fts_w_metaphone, config().fts_literal_enabled});
        auto supplement = [&](const char* table, const char* pk,
                              const char* source) {
            std::unordered_set<int> seen;
            for (const auto& r : cands)
                if (r.metadata.value("source", "") == source) seen.insert(r.id);
            for (const auto& hit : text_index_->score_query(table, q.query)) {
                if (seen.count(hit.id)) continue;
                Stmt s(db, std::format(
                    "SELECT text, status, ifnull(trigger,''), "
                    "datetime(created_at,'unixepoch','localtime') "
                    "FROM {} WHERE {} = ?", table, pk));
                s.bind(1, hit.id);
                if (!s.step()) continue;
                const std::string status = s.column_text(1);
                if (q.status != "any" && status != q.status) continue;
                json meta = json::object();
                meta["source"] = source;
                meta["status"] = status;
                SearchResult sr{hit.id, s.column_text(0), hit.score, std::move(meta), s.column_text(3)};
                sr.vec_score  = -1.0f;
                sr.bm25_score = hit.score;
                sr.blended    = hit.score;
                cands.push_back(std::move(sr));
            }
        };
        supplement("decisions", "decision_id", "decision");
        supplement("insights", "insight_id", "insight");
    }

    // ---- session filter helpers ------------------------------------------
    // GUID -> sessions.session_id once per filter (unknown GUID -> nullopt:
    // `session` then matches nothing, `exclude_session` excludes nothing).
    auto guid_to_id = [this](const std::string& guid) -> std::optional<int> {
        Stmt s(db, "SELECT session_id FROM sessions WHERE guid = ?");
        s.bind(1, guid);
        if (!s.step()) return std::nullopt;
        return s.column_int(0);
    };
    // Owning session id per layer (documents have none; turn layer is never
    // produced by the search corpora).
    auto owning_session = [this](const std::string& layer, int id) -> std::optional<int> {
        const char* sql = nullptr;
        if (layer == "turn_summary")
            sql = "SELECT session_id FROM turn_summaries WHERE turn_summary_id = ?";
        else if (layer == "summary")
            sql = "SELECT session_id FROM summaries WHERE summary_id = ?";
        else if (layer == "decision")
            sql = "SELECT session_id FROM decisions WHERE decision_id = ?";
        else if (layer == "insight")
            sql = "SELECT session_id FROM insights WHERE insight_id = ?";
        else
            return std::nullopt;
        Stmt s(db, sql);
        s.bind(1, id);
        if (!s.step() || s.is_null(0)) return std::nullopt;
        return s.column_int(0);
    };
    std::optional<int> sess_id, excl_id;
    if (!q.session.empty())         sess_id = guid_to_id(q.session);
    if (!q.exclude_session.empty()) excl_id = guid_to_id(q.exclude_session);

    // ---- project filter (Phase 3) ----------------------------------------
    // An explicitly named project that doesn't resolve is an error (400);
    // cwd INFERENCE happens at the HTTP layer and leaves the filter empty
    // when nothing matches, per PHASE3-STEPS step 4.
    std::optional<int> proj_id;
    if (!q.project.empty()) {
        proj_id = resolve_project(q.project);
        if (!proj_id)
            throw std::invalid_argument(std::format(
                "project '{}' does not exist", q.project));
    }
    // Owning project id per layer: decisions/summaries carry it directly;
    // turn_summaries reach it through their session. Documents belong to no
    // project (shared reference material), so they have no owning project and
    // never match an explicit project filter.
    auto project_id_of = [this](const std::string& layer, int id) -> std::optional<int> {
        const char* sql = nullptr;
        if (layer == "decision")
            sql = "SELECT project_id FROM decisions WHERE decision_id = ?";
        else if (layer == "insight")
            sql = "SELECT project_id FROM insights WHERE insight_id = ?";
        else if (layer == "summary")
            sql = "SELECT project_id FROM summaries WHERE summary_id = ?";
        else if (layer == "turn_summary")
            sql = "SELECT s.project_id FROM turn_summaries ts "
                  "JOIN sessions s ON s.session_id = ts.session_id "
                  "WHERE ts.turn_summary_id = ?";
        else
            return std::nullopt;   // document: no project
        Stmt s(db, sql);
        s.bind(1, id);
        if (!s.step() || s.is_null(0)) return std::nullopt;
        return s.column_int(0);
    };

    auto session_id_of = [this](int sid) -> std::string {
        Stmt s(db, "SELECT guid FROM sessions WHERE session_id = ?");
        s.bind(1, sid);
        return s.step() ? s.column_text(0) : std::string();
    };

    // ---- 3. post-filter + shape ------------------------------------------
    std::vector<AgentHit> hits;
    for (const auto& r : cands) {
        const std::string layer = r.metadata.value("source", "");
        if (!q.layers.empty() &&
            std::find(q.layers.begin(), q.layers.end(), layer) == q.layers.end())
            continue;

        std::string row_status;
        if (layer == "decision" || layer == "insight") {
            const bool is_dec = layer == "decision";
            row_status = status_of(is_dec ? "decisions" : "insights",
                                   is_dec ? "decision_id" : "insight_id", r.id);
            if (row_status.empty()) continue;   // row vanished mid-search
            if (q.status == "active") {
                // NOTE: deliberately narrower than the search cache's
                // allow-list: 'roadmap' is planned work, not an active ruling.
                if (row_status != "current" && row_status != "provisional") continue;
            } else if (q.status != "any" && row_status != q.status) {
                continue;
            }
        }

        if (!since_b.empty() && r.timestamp <  since_b) continue;
        if (!until_b.empty() && r.timestamp >  until_b) continue;

        if (sess_id || excl_id) {
            auto sid = owning_session(layer, r.id);
            if (sess_id && (!sid || *sid != *sess_id)) continue;
            if (excl_id && sid && *sid == *excl_id) continue;
        }

        if (proj_id) {
            auto pid = project_id_of(layer, r.id);
            if (!pid || *pid != *proj_id) continue;
        }

        AgentHit h;
        h.id    = layer + ":" + std::to_string(r.id);
        h.layer = layer;
        h.score = r.blended;   // the value that actually ranked this hit
        if (r.vec_score > 0.0f && r.bm25_score > 0.0f)      h.matched_by = "both";
        else if (r.bm25_score > 0.0f)                       h.matched_by = "lexical";
        else                                                h.matched_by = "vector";
        h.created = r.timestamp;
        h.snippet = agent_snippet(r.text);
        if (layer == "decision" || layer == "insight") {
            const bool is_dec = layer == "decision";
            Stmt s(db, std::format("SELECT ifnull(trigger,'') FROM {} WHERE {} = ?",
                                   is_dec ? "decisions" : "insights",
                                   is_dec ? "decision_id" : "insight_id"));
            s.bind(1, r.id);
            h.extra = json{{"status", row_status}};
            if (s.step() && !s.column_text(0).empty())
                h.extra["trigger"] = s.column_text(0);
        } else if (layer == "summary") {
            h.extra = json{{"level", r.metadata.value("level", "")}};
        } else if (layer == "turn_summary") {
            h.extra = json::object();
            if (r.metadata.contains("turn_id"))
                h.extra["turn_id"] = r.metadata["turn_id"];
            if (r.metadata.contains("session_id"))
                h.extra["session"] = session_id_of(r.metadata["session_id"].get<int>());
        } else if (layer == "document") {
            h.extra = json{{"title", r.metadata.value("title", "")}};
        }
        hits.push_back(std::move(h));
    }

    // ---- 4. rank, budget, truncate ----------------------------------------
    std::sort(hits.begin(), hits.end(),
              [](const AgentHit& a, const AgentHit& b) { return a.score > b.score; });

    std::vector<AgentHit> out;
    long long token_sum = 0;   // running chars/4 over accepted snippets
    for (auto& h : hits) {
        if (static_cast<int>(out.size()) >= limit) break;
        // Budget check happens BEFORE adding, but the first accepted hit is
        // always kept ("always return at least 1 hit if any matched").
        if (q.max_tokens > 0 && !out.empty() &&
            (token_sum + static_cast<long long>(h.snippet.size())) / 4 > q.max_tokens)
            break;
        token_sum += static_cast<long long>(h.snippet.size());
        out.push_back(std::move(h));
    }
    return out;
}


// =======================================================================
// agent_get — full rows by typed handle (mode 1) or paged session history
// (mode 2). Exactly one mode per call.
// =======================================================================
namespace {

// Row fetchers. Each returns the full row as JSON, or json(nullptr) when the
// row does not exist. Shapes follow PHASE2-STEPS step 4: id/layer/text/created
// plus the layer's own columns.
json fetch_agent_row(sqlite3* db, const std::string& layer, int rowid) {
    if (layer == "decision") {
        Stmt s(db, "SELECT text, status, tags, trigger, rationale, scope, supersedes, "
                   "datetime(created_at,'unixepoch','localtime') "
                   "FROM decisions WHERE decision_id = ?");
        s.bind(1, rowid);
        if (!s.step()) return json(nullptr);
        json item = {
            {"id",      std::format("decision:{}", rowid)},
            {"layer",   "decision"},
            {"text",    s.column_text(0)},
            {"status",  s.column_text(1)},
            {"tags",    s.column_text(2)},
            {"trigger", s.is_null(3) ? json(nullptr) : json(s.column_text(3))},
            {"rationale", s.is_null(4) ? json(nullptr) : json(s.column_text(4))},
            {"scope",   s.is_null(5) ? json(nullptr) : json(s.column_text(5))},
            {"supersedes", s.is_null(6) ? json(nullptr) : json(s.column_int(6))},
            {"created", s.column_text(7)},
        };
        json ev = json::array();
        Stmt e(db, "SELECT turn_summary_id FROM decision_evidence WHERE decision_id = ? "
                   "ORDER BY turn_summary_id");
        e.bind(1, rowid);
        while (e.step()) ev.push_back(e.column_int(0));
        item["evidence"] = std::move(ev);
        return item;
    }
    if (layer == "insight") {
        // Same shape as a decision row (Phase 5: shared vocabulary).
        Stmt s(db, "SELECT text, status, tags, trigger, rationale, scope, supersedes, "
                   "datetime(created_at,'unixepoch','localtime') "
                   "FROM insights WHERE insight_id = ?");
        s.bind(1, rowid);
        if (!s.step()) return json(nullptr);
        json item = {
            {"id",      std::format("insight:{}", rowid)},
            {"layer",   "insight"},
            {"text",    s.column_text(0)},
            {"status",  s.column_text(1)},
            {"tags",    s.column_text(2)},
            {"trigger", s.is_null(3) ? json(nullptr) : json(s.column_text(3))},
            {"rationale", s.is_null(4) ? json(nullptr) : json(s.column_text(4))},
            {"scope",   s.is_null(5) ? json(nullptr) : json(s.column_text(5))},
            {"supersedes", s.is_null(6) ? json(nullptr) : json(s.column_int(6))},
            {"created", s.column_text(7)},
        };
        json ev = json::array();
        Stmt e(db, "SELECT turn_summary_id FROM insight_evidence WHERE insight_id = ? "
                   "ORDER BY turn_summary_id");
        e.bind(1, rowid);
        while (e.step()) ev.push_back(e.column_int(0));
        item["evidence"] = std::move(ev);
        return item;
    }
    if (layer == "turn") {
        Stmt s(db, "SELECT user_text, assistant_text, "
                   "datetime(created_at,'unixepoch','localtime'), "
                   "COALESCE((SELECT guid FROM sessions WHERE session_id = turns.session_id), ''), "
                   "COALESCE((SELECT name FROM models WHERE model_id = turns.model_id), '') "
                   "FROM turns WHERE turn_id = ?");
        s.bind(1, rowid);
        if (!s.step()) return json(nullptr);
        return {
            {"id",            std::format("turn:{}", rowid)},
            {"layer",         "turn"},
            {"text",          s.column_text(0)},   // user_text is the turn's primary text
            {"user_text",     s.column_text(0)},
            {"assistant_text", s.is_null(1) ? json(nullptr) : json(s.column_text(1))},
            {"created",       s.column_text(2)},
            {"session",       s.column_text(3)},
            {"model",         s.column_text(4)},
        };
    }
    if (layer == "turn_summary") {
        Stmt s(db, "SELECT text, turn_id, "
                   "datetime(turn_datetime,'unixepoch','localtime'), "
                   "COALESCE((SELECT guid FROM sessions WHERE session_id = turn_summaries.session_id), '') "
                   "FROM turn_summaries WHERE turn_summary_id = ?");
        s.bind(1, rowid);
        if (!s.step()) return json(nullptr);
        return {
            {"id",      std::format("turn_summary:{}", rowid)},
            {"layer",   "turn_summary"},
            {"text",    s.column_text(0)},
            {"turn_id", s.is_null(1) ? json(nullptr) : json(s.column_int(1))},
            {"created", s.column_text(2)},
            {"session", s.column_text(3)},
        };
    }
    if (layer == "summary") {
        Stmt s(db, "SELECT text, level, tags, "
                   "datetime(created_at,'unixepoch','localtime'), "
                   "datetime(updated_at,'unixepoch','localtime'), "
                   "COALESCE((SELECT guid FROM sessions WHERE session_id = summaries.session_id), '') "
                   "FROM summaries WHERE summary_id = ?");
        s.bind(1, rowid);
        if (!s.step()) return json(nullptr);
        return {
            {"id",      std::format("summary:{}", rowid)},
            {"layer",   "summary"},
            {"text",    s.column_text(0)},
            {"level",   s.column_text(1)},
            {"tags",    s.column_text(2)},
            {"created", s.column_text(3)},
            {"updated", s.column_text(4)},
            {"session", s.column_text(5)},
        };
    }
    if (layer == "document") {
        Stmt s(db, "SELECT d.text, d.tags, d.chunk_index, d.document_source_id, "
                   "COALESCE(ds.title, ''), "
                   "datetime(COALESCE(d.modified_on, ds.imported_at, 0),'unixepoch','localtime') "
                   "FROM documents d "
                   "LEFT JOIN document_sources ds ON ds.document_source_id = d.document_source_id "
                   "WHERE d.document_id = ?");
        s.bind(1, rowid);
        if (!s.step()) return json(nullptr);
        return {
            {"id",                 std::format("document:{}", rowid)},
            {"layer",              "document"},
            {"text",               s.column_text(0)},
            {"tags",               s.column_text(1)},
            {"chunk_index",        s.is_null(2) ? json(nullptr) : json(s.column_int(2))},
            {"document_source_id", s.is_null(3) ? json(nullptr) : json(s.column_int(3))},
            {"title",              s.column_text(4)},
            {"created",            s.column_text(5)},
        };
    }
    return json(nullptr);
}

} // anonymous namespace


// =======================================================================
// agent_timeline — deterministic newest-first range scan (Phase 4).
// No embeddings: plain indexed reads of turn_summaries / summaries,
// key-paginated by "<created_epoch>:<rowid>" cursors (the embed_tables()
// pattern: bounded read, strict "<" continuation, no OFFSET).
// =======================================================================
namespace {

// Parse a timeline cursor "<epoch>:<rowid>". Both parts must be all-digits
// non-negative; anything else -> invalid_argument.
void parse_timeline_cursor(const std::string& cursor, long long& epoch, long long& rowid) {
    auto fail = [&cursor]() -> std::invalid_argument {
        return std::invalid_argument("bad cursor: '" + cursor + "'");
    };
    auto colon = cursor.find(':');
    if (colon == std::string::npos || colon + 1 >= cursor.size())
        throw fail();
    const std::string a = cursor.substr(0, colon), b = cursor.substr(colon + 1);
    auto all_digits = [](const std::string& s) {
        return !s.empty() && std::all_of(s.begin(), s.end(),
                                         [](unsigned char c) { return c >= '0' && c <= '9'; });
    };
    if (!all_digits(a) || !all_digits(b))
        throw fail();
    epoch = std::stoll(a);
    rowid = std::stoll(b);
}

// Convert a resolved "YYYY-MM-DD HH:MM:SS" bound to epoch seconds for
// comparison against the INTEGER timestamp columns (same conversion as
// resolve_epoch()).
long long bound_to_epoch(const std::string& bound) {
    auto tt = parse_db_timestamp(bound);
    if (!tt)
        throw std::invalid_argument("bad since/until value: '" + bound + "'");
    return static_cast<long long>(*tt);
}

} // anonymous namespace

json Backend::Impl::agent_timeline(const TimelineQuery& q) {
    static const std::set<std::string> kGranularities = {"turn", "episode", "session"};
    if (!kGranularities.count(q.granularity))
        throw std::invalid_argument("granularity must be one of: turn, episode, session");
    const int limit = std::clamp(q.limit, 1, 100);

    long long since_e = 0, until_e = 0;
    if (!q.since.empty()) since_e = bound_to_epoch(resolve_agent_time_bound(q.since, false));
    if (!q.until.empty()) until_e = bound_to_epoch(resolve_agent_time_bound(q.until, true));

    std::optional<long long> cur_epoch, cur_row;
    if (!q.cursor.empty()) {
        long long e = 0, r = 0;
        parse_timeline_cursor(q.cursor, e, r);
        cur_epoch = e;
        cur_row = r;
    }

    // Project filter: explicit name/path must resolve (400), same contract as
    // agent_search. Session filter: unknown GUID matches nothing (nullopt).
    std::optional<int> proj_id;
    if (!q.project.empty()) {
        auto p = resolve_project(q.project);
        if (!p)
            throw std::invalid_argument(std::format("project '{}' does not exist", q.project));
        proj_id = *p;
    }
    std::optional<int> sess_id;
    if (!q.session.empty()) {
        Stmt s(db, "SELECT session_id FROM sessions WHERE guid = ?");
        s.bind(1, q.session);
        if (s.step()) sess_id = s.column_int(0);
    }

    // Each granularity builds one SELECT over (created_epoch, rowid, text,
    // session_guid) with the same filter/cursor plumbing, ordered newest
    // first. The continuation predicate is strict "<" on the (created, rowid)
    // pair — never OFFSET. `first_expr` is the created-epoch expression; it
    // must match the first projected column so the cursor and the ORDER BY
    // agree.
    std::string sql;
    std::string first_expr, row_expr;
    std::vector<std::pair<int, long long>> int_binds;   // positional binds (1-based)
    if (q.granularity == "turn") {
        // One entry per turn: its turn_summary text when one exists, else the
        // raw user text (turns that haven't been summarized yet). Keyed by
        // (turn_datetime, turn_id) — turn_summaries.turn_id is unique, and
        // fallback rows are exactly the turns without a summary row.
        first_expr =
            "COALESCE((SELECT ts.turn_datetime FROM turn_summaries ts "
            "WHERE ts.turn_id = t.turn_id AND ts.text != ''), t.created_at)";
        row_expr = "t.turn_id";
        sql =
            "SELECT " + first_expr + ", " + row_expr + ", "
            "  COALESCE((SELECT ts.text FROM turn_summaries ts "
            "            WHERE ts.turn_id = t.turn_id AND ts.text != ''), t.user_text), "
            "  COALESCE((SELECT guid FROM sessions WHERE session_id = t.session_id), '') "
            "FROM turns t "
            "WHERE (t.user_text != '' OR EXISTS "
            "  (SELECT 1 FROM turn_summaries ts WHERE ts.turn_id = t.turn_id AND ts.text != ''))";
        if (sess_id) {
            sql += " AND t.session_id = ?";
            int_binds.push_back({static_cast<int>(int_binds.size()) + 1, *sess_id});
        }
        if (proj_id) {
            sql += " AND t.session_id IN (SELECT session_id FROM sessions WHERE project_id = ?)";
            int_binds.push_back({static_cast<int>(int_binds.size()) + 1, *proj_id});
        }
    } else if (q.granularity == "episode") {
        first_expr = "sm.created_at";
        row_expr = "sm.summary_id";
        sql =
            "SELECT sm.created_at, sm.summary_id, sm.text, "
            "  COALESCE((SELECT guid FROM sessions WHERE session_id = sm.session_id), '') "
            "FROM summaries sm WHERE sm.level = 'episode' AND sm.text != ''";
        if (sess_id) {
            sql += " AND sm.session_id = ?";
            int_binds.push_back({static_cast<int>(int_binds.size()) + 1, *sess_id});
        }
        if (proj_id) {
            // Own project_id or the owning session's (parity with the
            // project briefing's recent_episodes query).
            sql += " AND (sm.project_id = ? OR sm.session_id IN "
                   "  (SELECT session_id FROM sessions WHERE project_id = ?))";
            int_binds.push_back({static_cast<int>(int_binds.size()) + 1, *proj_id});
            int_binds.push_back({static_cast<int>(int_binds.size()) + 1, *proj_id});
        }
    } else { // session
        // Per session: its session summary if one exists, else its newest
        // episode. Sessions with neither contribute nothing (never require
        // session summaries — design §10.2.5).
        first_expr = "sm.created_at";
        row_expr = "sm.summary_id";
        sql =
            "SELECT sm.created_at, sm.summary_id, sm.text, s.guid "
            "FROM sessions s "
            "JOIN summaries sm ON sm.summary_id = ("
            "  SELECT summary_id FROM summaries "
            "  WHERE session_id = s.session_id AND text != '' AND ("
            "    level = 'session' OR summary_id = ("
            "      SELECT summary_id FROM summaries "
            "      WHERE session_id = s.session_id AND level = 'episode' AND text != '' "
            "      ORDER BY created_at DESC, summary_id DESC LIMIT 1)) "
            "  ORDER BY CASE level WHEN 'session' THEN 0 ELSE 1 END, "
            "           created_at DESC, summary_id DESC LIMIT 1)";
        if (sess_id) {
            sql += " AND s.session_id = ?";
            int_binds.push_back({static_cast<int>(int_binds.size()) + 1, *sess_id});
        }
        if (proj_id) {
            sql += " AND s.project_id = ?";
            int_binds.push_back({static_cast<int>(int_binds.size()) + 1, *proj_id});
        }
    }

    // Time bounds and the cursor apply to the first projected column pair
    // (created epoch, rowid) — one continuation predicate for all three
    // granularities: strictly older in (created, rowid) order.
    if (since_e) sql += " AND " + first_expr + " >= ?";
    if (until_e) sql += " AND " + first_expr + " <= ?";
    if (cur_epoch)
        sql += " AND (" + first_expr + " < ? OR (" + first_expr + " = ? AND " + row_expr + " < ?))";
    sql += " ORDER BY 1 DESC, 2 DESC LIMIT ?";

    Stmt s(db, sql);
    int bind = 1;
    for (auto& [idx, v] : int_binds) { (void)idx; s.bind(bind++, v); }
    if (since_e) s.bind(bind++, since_e);
    if (until_e) s.bind(bind++, until_e);
    if (cur_epoch) {
        s.bind(bind++, *cur_epoch);
        s.bind(bind++, *cur_epoch);
        s.bind(bind++, *cur_row);
    }
    s.bind(bind, limit + 1);   // one extra row decides next_cursor

    json items = json::array();
    long long last_epoch = 0, last_row = 0;
    bool more = false;
    int rows = 0;
    while (s.step()) {
        ++rows;
        if (rows > limit) { more = true; break; }
        last_epoch = s.column_int64(0);
        last_row   = s.column_int64(1);
        const std::string text = s.column_text(2);
        json item = {
            // Turn-granularity entries are keyed by turn_id (the summary is
            // looked up per turn), so the handle is turn:<id> — resolvable
            // by agent_get even when the entry shows summary text.
            {"id",      q.granularity == "turn"
                           ? std::format("turn:{}", last_row)
                           : std::format("summary:{}", last_row)},
            {"created", db_timestamp(static_cast<std::time_t>(last_epoch))},
            {"session", s.column_text(3)},
            {"text",    agent_snippet(text)},
        };
        items.push_back(std::move(item));
    }

    return json{
        {"items",       std::move(items)},
        {"next_cursor", more
            ? json(std::format("{}:{}", last_epoch, last_row))
            : json(nullptr)},
    };
}


// =======================================================================
// agent_get — full rows by typed handle (mode 1) or paged session history
// (mode 2). Exactly one mode per call.
// =======================================================================
json Backend::Impl::agent_get(const AgentGetQuery& q) {
    const bool has_ids  = !q.ids.empty();
    const bool has_sess = !q.session.empty();
    if (has_ids == has_sess)
        throw std::invalid_argument(
            "agent_get: exactly one of ids or session must be set");

    // ---- mode 1: full rows by handle -------------------------------------
    if (has_ids) {
        json items = json::array();
        for (const auto& handle : q.ids) {
            std::string layer;
            int rowid = 0;
            if (!parse_agent_handle(handle, layer, rowid)) {
                items.push_back({{"id", handle}, {"error", "not found"}});
                continue;
            }
            json item = fetch_agent_row(db, layer, rowid);
            if (item.is_null()) {
                items.push_back({{"id", handle}, {"error", "not found"}});
                continue;
            }

            if (q.expand == "context") {
                json expanded = json::array();
                if (layer == "turn" || layer == "turn_summary") {
                    // The 2 turns before and after in the same session. A
                    // turn_summary anchors on its source turn; a pruned
                    // (turn_id NULL) summary or an anonymous turn has no
                    // context window.
                    int anchor = rowid;
                    if (layer == "turn_summary") {
                        Stmt t(db, "SELECT turn_id FROM turn_summaries WHERE turn_summary_id = ?");
                        t.bind(1, rowid);
                        if (!t.step() || t.is_null(0)) anchor = -1;
                        else anchor = t.column_int(0);
                    }
                    if (anchor >= 0) {
                        Stmt c(db, "SELECT turn_id FROM turns "
                                   "WHERE session_id = (SELECT session_id FROM turns WHERE turn_id = ?) "
                                   "  AND session_id IS NOT NULL "
                                   "  AND turn_id BETWEEN ? AND ? AND turn_id != ? "
                                   "ORDER BY turn_id");
                        c.bind(1, anchor);
                        c.bind(2, anchor - 2);
                        c.bind(3, anchor + 2);
                        c.bind(4, anchor);
                        while (c.step())
                            expanded.push_back(fetch_agent_row(db, "turn", c.column_int(0)));
                    }
                } else if (layer == "document") {
                    // chunk_index ±1 from the same document_source_id.
                    Stmt c(db, "SELECT document_id FROM documents "
                               "WHERE document_source_id = ("
                               "  SELECT document_source_id FROM documents WHERE document_id = ?) "
                               "  AND document_source_id IS NOT NULL "
                               "  AND chunk_index IN ("
                               "    (SELECT chunk_index FROM documents WHERE document_id = ?) - 1, "
                               "    (SELECT chunk_index FROM documents WHERE document_id = ?) + 1) "
                               "ORDER BY chunk_index");
                    c.bind(1, rowid);
                    c.bind(2, rowid);
                    c.bind(3, rowid);
                    while (c.step())
                        expanded.push_back(fetch_agent_row(db, "document", c.column_int(0)));
                }
                if (!expanded.empty()) item["expanded"] = std::move(expanded);
            } else if (q.expand == "sources") {
                json expanded = json::array();
                if (layer == "decision" || layer == "insight") {
                    // Its evidence turn_summaries, full rows. The join filter
                    // keeps done-marker rows (text='') out of expansion even
                    // if a source turn was later abandoned (RULES 6).
                    const bool is_dec = layer == "decision";
                    Stmt e(db, std::format(
                        "SELECT de.turn_summary_id FROM {} de "
                        "JOIN turn_summaries ts ON ts.turn_summary_id = de.turn_summary_id "
                        "WHERE de.{} = ? AND ts.text != '' "
                        "ORDER BY de.turn_summary_id",
                        is_dec ? "decision_evidence" : "insight_evidence",
                        is_dec ? "decision_id" : "insight_id"));
                    e.bind(1, rowid);
                    while (e.step())
                        expanded.push_back(fetch_agent_row(db, "turn_summary", e.column_int(0)));
                } else if (layer == "summary") {
                    // The turn_summaries in its session within its span.
                    // Episodes/turn_summaries only — never depends on session
                    // summaries existing (design §10.2.5).
                    Stmt c(db, "SELECT turn_summary_id FROM turn_summaries ts "
                               "WHERE ts.session_id = ("
                               "  SELECT session_id FROM summaries WHERE summary_id = ?) "
                               "  AND ts.session_id IS NOT NULL "
                               "  AND ts.text != '' "
                               "  AND ts.turn_datetime BETWEEN "
                               "    (SELECT created_at FROM summaries WHERE summary_id = ?) "
                               "    AND (SELECT updated_at FROM summaries WHERE summary_id = ?) "
                               "ORDER BY ts.turn_datetime");
                    c.bind(1, rowid);
                    c.bind(2, rowid);
                    c.bind(3, rowid);
                    while (c.step())
                        expanded.push_back(fetch_agent_row(db, "turn_summary", c.column_int(0)));
                }
                if (!expanded.empty()) item["expanded"] = std::move(expanded);
            } else if (q.expand == "chain" &&
                       (layer == "decision" || layer == "insight")) {
                // Full supersede lineage oldest -> newest: walk `supersedes`
                // backwards to the root, then forwards from the anchor. The
                // visited set makes a corrupt cycle terminate.
                const bool is_dec = layer == "decision";
                const char* table = is_dec ? "decisions" : "insights";
                const char* pk    = is_dec ? "decision_id" : "insight_id";
                std::vector<int> chain_ids;
                std::unordered_set<int> visited{rowid};
                {
                    std::vector<int> up;
                    int cur = rowid;
                    while (true) {
                        Stmt p(db, std::format("SELECT supersedes FROM {} WHERE {} = ?",
                                               table, pk));
                        p.bind(1, cur);
                        if (!p.step() || p.is_null(0)) break;
                        int parent = p.column_int(0);
                        if (visited.count(parent)) break;
                        visited.insert(parent);
                        up.push_back(parent);
                        cur = parent;
                    }
                    std::reverse(up.begin(), up.end());
                    chain_ids = std::move(up);
                    chain_ids.push_back(rowid);
                }
                {
                    std::vector<int> queue{rowid};
                    while (!queue.empty()) {
                        std::vector<int> next;
                        for (int cur : queue) {
                            Stmt c(db, std::format("SELECT {} FROM {} WHERE supersedes = ? "
                                                   "ORDER BY {}", pk, table, pk));
                            c.bind(1, cur);
                            while (c.step()) {
                                int child = c.column_int(0);
                                if (visited.count(child)) continue;
                                visited.insert(child);
                                chain_ids.push_back(child);
                                next.push_back(child);
                            }
                        }
                        queue = std::move(next);
                    }
                }
                json expanded = json::array();
                for (int id : chain_ids)
                    expanded.push_back(fetch_agent_row(db, layer, id));
                item["expanded"] = std::move(expanded);
            }

            items.push_back(std::move(item));
        }
        return json{{"items", std::move(items)}};
    }

    // ---- mode 2: paged conversation history for one session --------------
    // "current" is NOT resolved here — the HTTP/MCP layer passes a real GUID.
    const int limit = std::clamp(q.limit, 1, 100);
    std::string sql =
        "SELECT t.turn_id, t.user_text, t.assistant_text, "
        "       datetime(t.created_at,'unixepoch','localtime'), "
        "       (SELECT text FROM turn_summaries ts "
        "         WHERE ts.turn_id = t.turn_id AND ts.text != '' LIMIT 1) "
        "FROM turns t JOIN sessions s ON s.session_id = t.session_id "
        "WHERE s.guid = ?";
    if (q.after_turn  > 0) sql += " AND t.turn_id > ?";
    if (q.before_turn > 0) sql += " AND t.turn_id < ?";
    sql += " ORDER BY t.turn_id ASC LIMIT ?";

    Stmt s(db, sql);
    int bind = 1;
    s.bind(bind++, q.session);
    if (q.after_turn  > 0) s.bind(bind++, q.after_turn);
    if (q.before_turn > 0) s.bind(bind++, q.before_turn);
    s.bind(bind, limit + 1);   // one extra row tells us whether more exist

    json turns = json::array();
    int last_turn_id = 0;
    bool more = false;
    int rows = 0;
    while (s.step()) {
        ++rows;
        if (rows > limit) { more = true; break; }   // the probe row exists
        json t = {
            {"id",        std::format("turn:{}", s.column_int(0))},
            {"user",      s.column_text(1)},
            {"assistant", s.is_null(2) ? json(nullptr) : json(s.column_text(2))},
            {"created",   s.column_text(3)},
            {"summary",   s.is_null(4) ? json(nullptr) : json(s.column_text(4))},
        };
        last_turn_id = s.column_int(0);
        turns.push_back(std::move(t));
    }

    return json{
        {"session",    q.session},
        {"turns",      std::move(turns)},
        {"next_after", more ? json(last_turn_id) : json(nullptr)},
    };
}


// =======================================================================
// agent_update — metadata-only curation (Phase 4).
// Text never changes in place; a text change is a new row with supersedes.
// =======================================================================
namespace {

// Decision status transition table (PHASE4-STEPS step 2). 'superseded' is
// reachable only through `supersedes` on a new decision, never here.
bool decision_transition_ok(const std::string& from, const std::string& to) {
    static const std::map<std::string, std::set<std::string>> kAllowed = {
        {"pending",     {"current", "provisional", "retired"}},
        {"provisional", {"current", "retired"}},
        {"current",     {"provisional", "retired"}},
        {"roadmap",     {"current", "provisional", "retired"}},
        {"superseded",  {}},
        {"retired",     {}},
    };
    auto it = kAllowed.find(from);
    if (it == kAllowed.end()) return false;
    return it->second.count(to) > 0;
}

} // anonymous namespace

json Backend::Impl::agent_update(const UpdateRequest& req) {
    // Rule 1: text is never edited in place.
    if (req.text.has_value())
        throw std::invalid_argument(
            "text cannot be edited in place; store a new decision/note with supersedes");

    std::string layer;
    int rowid = 0;
    if (!parse_agent_handle(req.id, layer, rowid))
        throw std::invalid_argument("bad handle: '" + req.id + "'");

    // Rule 2: agents can't modify turn / turn_summary / document rows.
    if (layer == "turn" || layer == "turn_summary" || layer == "document")
        throw std::invalid_argument(
            layer + " rows are not agent-editable");

    if (layer == "summary") {
        // Rule 3: only agent-written notes (no model_id) are editable;
        // summarizer rollups carry the summarizer's model and are history.
        Stmt f(db, "SELECT model_id FROM summaries WHERE summary_id = ?");
        f.bind(1, rowid);
        if (!f.step())
            throw std::invalid_argument("summary " + std::to_string(rowid) + " not found");
        if (!f.is_null(0))
            throw std::invalid_argument(
                "summary rows written by the summarizer are not editable; only agent notes are");
        if (req.status.has_value())
            throw std::invalid_argument("summaries have no status");

        std::string set;
        if (req.tags.has_value())
            set += "tags = ?, ";
        if (req.project.has_value()) {
            if (!req.project->empty() && !resolve_project(*req.project))
                throw std::invalid_argument(
                    std::format("project '{}' does not exist", *req.project));
            set += "project_id = ?, ";
        }
        if (set.empty())
            throw std::invalid_argument("nothing to update");
        set.erase(set.end() - 2, set.end());   // trailing ", "
        Stmt u(db, "UPDATE summaries SET " + set +
                   ", updated_at = unixepoch() WHERE summary_id = ?");
        int b = 1;
        if (req.tags.has_value()) u.bind(b++, *req.tags);
        if (req.project.has_value()) {
            if (req.project->empty()) u.bind_null(b++);
            else u.bind(b++, *resolve_project(*req.project));
        }
        u.bind(b, rowid);
        u.exec();
        invalidate_cache();
    } else if (layer == "decision" || layer == "insight") {
        // Rule 4: status transitions per the table; tags/project free.
        // Insights share the decision transition table verbatim (P5 step 4).
        const bool is_dec = layer == "decision";
        const char* table = is_dec ? "decisions" : "insights";
        const char* pk    = is_dec ? "decision_id" : "insight_id";
        Stmt f(db, std::format("SELECT status FROM {} WHERE {} = ?", table, pk));
        f.bind(1, rowid);
        if (!f.step())
            throw std::invalid_argument(layer + " " + std::to_string(rowid) + " not found");
        const std::string old_status = f.column_text(0);

        std::string set;
        int b = 1;
        std::optional<int> pid_bind;      // tri-state: nullopt=absent, -1=NULL
        bool clear_project = false;
        if (req.status.has_value()) {
            static const std::set<std::string> kStatuses = {
                "current", "provisional", "pending", "roadmap", "retired"};
            if (!kStatuses.count(*req.status))
                throw std::invalid_argument("unknown " + layer + " status: '" + *req.status + "'");
            if (*req.status == "superseded")
                throw std::invalid_argument(std::format(
                    "status 'superseded' is set only by storing a superseding {}", layer));
            // Strict table: same-status is not a listed transition either
            // (the review queue relies on re-approving failing).
            if (!decision_transition_ok(old_status, *req.status))
                throw std::invalid_argument(std::format(
                    "illegal {} status transition: {} -> {}", layer, old_status, *req.status));
            set += "status = ?, ";
        }
        if (req.tags.has_value()) set += "tags = ?, ";
        if (req.project.has_value()) {
            set += "project_id = ?, ";
            if (req.project->empty()) clear_project = true;
            else {
                pid_bind = resolve_project(*req.project);
                if (!pid_bind)
                    throw std::invalid_argument(
                        std::format("project '{}' does not exist", *req.project));
            }
        }
        if (set.empty())
            throw std::invalid_argument("nothing to update");
        set.erase(set.end() - 2, set.end());   // trailing ", "
        Stmt u(db, std::format("UPDATE {} SET {} WHERE {} = ?", table, set, pk));
        if (req.status.has_value()) u.bind(b++, *req.status);
        if (req.tags.has_value())   u.bind(b++, *req.tags);
        if (req.project.has_value()) {
            if (clear_project) u.bind_null(b++);
            else u.bind(b++, *pid_bind);
        }
        u.bind(b, rowid);
        u.exec();
        // rule 5: any structured-row change refreshes its search cache
        if (is_dec) invalidate_dec_cache();
        else        invalidate_ins_cache();
    } else {
        throw std::invalid_argument("layer not updatable: " + layer);
    }

    return fetch_agent_row(db, layer, rowid);
}


// =======================================================================
// agent_review — pending-decision queue (Phase 4). Reuses agent_update's
// transition code; a non-pending id fails only that id.
// =======================================================================
json Backend::Impl::agent_review(const std::string& action,
                                 const std::vector<std::string>& ids) {
    if (action == "list") {
        // Pending decisions AND pending insights (P5 step 4), oldest first
        // across both queues.
        std::vector<std::pair<long long, json>> rows;
        auto collect = [&](const char* table, const char* pk, const char* source) {
            Stmt s(db, std::format(
                "SELECT {}, ifnull(trigger,''), text, created_at FROM {} "
                "WHERE status = 'pending'", pk, table));
            while (s.step()) {
                rows.emplace_back(s.column_int64(3), json{
                    {"id",      std::format("{}:{}", source, s.column_int(0))},
                    {"trigger", s.column_text(1)},
                    {"snippet", agent_snippet(s.column_text(2))},
                });
            }
        };
        collect("decisions", "decision_id", "decision");
        collect("insights", "insight_id", "insight");
        std::stable_sort(rows.begin(), rows.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
        json items = json::array();
        for (auto& [created, item] : rows)
            items.push_back(std::move(item));
        return json{{"pending", std::move(items)}};
    }
    if (action != "approve" && action != "reject")
        throw std::invalid_argument("action must be one of: list, approve, reject");
    if (ids.empty())
        throw std::invalid_argument("ids required for approve/reject");

    const std::string to = (action == "approve") ? "current" : "retired";
    json done = json::array();
    json failed = json::array();
    for (const auto& raw : ids) {
        // Accept "decision:<id>" or a bare integer.
        std::string handle = raw;
        if (handle.find(':') == std::string::npos)
            handle = "decision:" + handle;
        try {
            UpdateRequest req;
            req.id = handle;
            req.status = to;
            agent_update(req);
            done.push_back(raw);
        } catch (const std::invalid_argument& e) {
            failed.push_back({{"id", raw}, {"error", e.what()}});
        }
    }
    return json{{"done", std::move(done)}, {"failed", std::move(failed)}};
}

} // namespace ragger::sqlite
