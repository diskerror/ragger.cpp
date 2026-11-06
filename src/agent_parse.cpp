// Agent API request parsing — shared by server.cpp (HTTP) and mcp.cpp
// (MCP tool_call). See agent_parse.h for the contract. Field defaults here
// are the documented API defaults; keep them in sync with the tool schemas
// in mcp.cpp and the endpoint docs in server.cpp handler comments.
#include "agent_parse.h"

#include <string>
#include <utility>
#include <vector>

namespace ragger {

using json = nlohmann::json;

std::string tags_to_comma_string(const json& a, const char* key) {
    if (!a.contains(key)) return "";
    const auto& t = a[key];
    if (t.is_string()) return t.get<std::string>();
    if (t.is_array()) {
        std::string out;
        for (const auto& x : t) {
            if (!out.empty()) out += ',';
            out += x.get<std::string>();
        }
        return out;
    }
    return "";
}

namespace {
// LR S80 (D2): evidence references EPISODES. Bare ints and "episode:N"
// handles are accepted; "turn_summary:N" / "turn:N" handles (the old
// evidence level) and anything else are client errors pointing at episode
// ids. Shared by the decision and note-insight branches.
std::vector<int> parse_evidence_ids(const json& v) {
    if (!v.is_array())
        throw std::invalid_argument("evidence must be an array of episode ids");
    std::vector<int> out;
    for (const auto& e : v) {
        if (e.is_number_integer()) {
            out.push_back(e.get<int>());
            continue;
        }
        if (e.is_string()) {
            const std::string s = e.get<std::string>();
            constexpr std::string_view pfx = "episode:";
            if (s.rfind(pfx, 0) == 0 && s.size() > pfx.size()) {
                int id = 0;
                bool ok = true;
                for (char c : s.substr(pfx.size())) {
                    if (c < '0' || c > '9') { ok = false; break; }
                    id = id * 10 + (c - '0');
                }
                if (ok && id > 0) { out.push_back(id); continue; }
            }
            throw std::invalid_argument(std::string(
                "evidence must reference episodes: pass episode ids or "
                "\"episode:<id>\" handles (turn_summary/turn handles no longer carry evidence)"));
        }
        throw std::invalid_argument("evidence entries must be episode ids");
    }
    return out;
}
} // namespace

AgentSearchQuery parse_agent_search_query(const json& a) {
    AgentSearchQuery q;
    q.query           = a.value("query", "");
    q.since           = a.value("since", "");
    q.until           = a.value("until", "");
    q.status          = a.value("status", "active");
    q.session         = a.value("session", "");
    q.exclude_session = a.value("exclude_session", "");
    q.limit           = a.value("limit", 10);
    q.max_tokens      = a.value("max_tokens", 0);
    q.mode            = a.value("mode", "hybrid");
    if (a.contains("layers"))
        q.layers = a["layers"].get<std::vector<std::string>>();
    if (a.contains("kinds"))
        q.kinds = a["kinds"].get<std::vector<std::string>>();
    // LR S78 (logic §2): tags filter — array form, or a comma string for
    // compatibility (same accepted shapes as tags_to_comma_string on write).
    if (a.contains("tags") && !a["tags"].is_null()) {
        const auto& t = a["tags"];
        if (t.is_array()) {
            for (const auto& x : t)
                if (x.is_string()) q.tags.push_back(x.get<std::string>());
        } else if (t.is_string()) {
            const std::string s = t.get<std::string>();
            std::string::size_type start = 0;
            while (start <= s.size()) {
                auto comma = s.find(',', start);
                const auto end = comma == std::string::npos ? s.size() : comma;
                if (end > start) q.tags.push_back(s.substr(start, end - start));
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
        }
    }
    return q;
}

DocumentSearchQuery parse_document_search_query(const json& a) {
    DocumentSearchQuery q;
    q.query  = a.value("query", "");
    q.source = a.value("source", "");
    q.limit  = a.value("limit", 10);
    q.mode   = a.value("mode", "hybrid");
    return q;
}

AgentGetQuery parse_agent_get_query(const json& a) {
    AgentGetQuery q;
    q.expand      = a.value("expand", "none");
    q.session     = a.value("session", "");
    q.after_turn  = a.value("after_turn", 0);
    q.before_turn = a.value("before_turn", 0);
    q.limit       = a.value("limit", 20);
    if (a.contains("ids"))
        q.ids = a["ids"].get<std::vector<std::string>>();
    return q;
}

TimelineQuery parse_timeline_query(const json& a) {
    TimelineQuery q;
    q.session     = a.value("session", "");
    q.since       = a.value("since", "");
    q.until       = a.value("until", "");
    q.granularity = a.value("granularity", "episode");
    q.limit       = a.value("limit", 20);
    q.cursor      = a.value("cursor", "");
    return q;
}

UpdateRequest parse_update_request(const json& a) {
    UpdateRequest u;
    u.id = a.value("id", "");
    if (a.contains("status") && !a["status"].is_null())
        u.status = a["status"].get<std::string>();
    // LR S78: memory_update accepts the same array/string forms as store.
    if (a.contains("tags") && !a["tags"].is_null())
        u.tags = tags_to_comma_string(a);
    // Forwarded when present; agent_update rejects any text change with the
    // canonical message. (MCP previously dropped the field silently, letting
    // a caller's text pass without effect -- now both surfaces reject.)
    if (a.contains("text") && !a["text"].is_null())
        u.text = a["text"].get<std::string>();
    return u;
}

KnowledgeInput parse_decision_input(const json& a) {
    KnowledgeInput in;
    in.kind         = "decision";
    in.text         = a.value("text", "");
    in.trigger      = a.value("trigger", "");
    in.rationale    = a.value("rationale", "");
    in.scope        = a.value("scope", "");
    in.status       = a.value("status", "current");
    in.session_guid = a.value("session", "");
    in.tags         = tags_to_comma_string(a);
    if (a.contains("supersedes") && !a["supersedes"].is_null())
        in.supersedes = a["supersedes"].get<int>();
    if (a.contains("evidence"))
        in.evidence = parse_evidence_ids(a["evidence"]);
    return in;
}

KnowledgeInput parse_note_insight(const json& a, const std::string& text) {
    KnowledgeInput in;
    in.kind         = "insight";
    in.text         = text;
    in.evidence     = parse_evidence_ids(a["evidence"]);
    in.tags         = tags_to_comma_string(a);
    in.trigger      = a.value("trigger", "");        // LR S59 (Q6)
    in.session_guid = a.value("session", "");
    return in;
}

void validate_note_request(const json& a) {
    // LR S59 (D-3): a trigger on a plain note is a client error; insights
    // (evidence present) are the only rows that carry one.
    if (a.contains("trigger") && !a.value("trigger", std::string{}).empty() &&
        !(a.contains("evidence") && a["evidence"].is_array()))
        throw std::invalid_argument("trigger is only valid on an insight (pass evidence)");
}

std::pair<std::string, std::vector<std::string>> parse_review_request(const json& a) {
    std::vector<std::string> ids;
    if (a.contains("ids"))
        ids = a["ids"].get<std::vector<std::string>>();
    return {a.value("action", ""), std::move(ids)};
}

json supersedes_json(const std::vector<std::pair<int, float>>& ps) {
    json out = json::array();
    for (const auto& [id, score] : ps)
        out.push_back({{"id", id}, {"score", score}});
    return out;
}

} // namespace ragger
