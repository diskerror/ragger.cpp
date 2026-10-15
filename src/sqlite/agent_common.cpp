// Agent API shared helpers — definitions moved verbatim from agent_read.cpp
// by review 5A so the read surface and the write surface share one copy.
#include "sqlite/agent_common.h"

#include "util/sqlite.h"

#include <sqlite3.h>

#include <format>
#include <limits>
#include <set>
#include <string>

namespace ragger::sqlite {

using json = nlohmann::json;

namespace {

// The six searchable layers, shared by agent_search's validation (via
// parse_agent_handle's layer check) and kept here as the one copy.
const std::set<std::string> kKnownLayers = {
    "turn", "turn_summary", "summary", "document", "decision", "insight"};

} // anonymous namespace

bool is_known_agent_layer(const std::string& layer) {
    return kKnownLayers.count(layer) > 0;
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
// one colon, known layer, all-digits positive rowid without leading zeros.
// Hand-rolled (review 3B): runs on every agent_get/update call. Overflowing
// ids reject as false — the regex version let 10+ digit ids reach std::stoi
// and throw out_of_range.
bool parse_agent_handle(const std::string& handle, std::string& layer, int& rowid) {
    const size_t colon = handle.find(':');
    if (colon == std::string::npos ||
        handle.find(':', colon + 1) != std::string::npos)
        return false;                       // not exactly one colon
    layer = handle.substr(0, colon);
    if (!kKnownLayers.count(layer)) return false;
    if (colon + 1 >= handle.size() || handle[colon + 1] == '0') return false;
    long long v = 0;
    for (size_t i = colon + 1; i < handle.size(); ++i) {
        const char c = handle[i];
        if (c < '0' || c > '9') return false;
        v = v * 10 + (c - '0');
        if (v > std::numeric_limits<int>::max()) return false;
    }
    rowid = static_cast<int>(v);
    return true;
}

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

} // namespace ragger::sqlite
