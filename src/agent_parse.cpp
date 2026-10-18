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
    if (a.contains("layers"))
        q.layers = a["layers"].get<std::vector<std::string>>();
    if (a.contains("kinds"))
        q.kinds = a["kinds"].get<std::vector<std::string>>();
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
    if (a.contains("tags") && !a["tags"].is_null())
        u.tags = a["tags"].get<std::string>();
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
        in.evidence = a["evidence"].get<std::vector<int>>();
    return in;
}

KnowledgeInput parse_note_insight(const json& a, const std::string& text) {
    KnowledgeInput in;
    in.kind     = "insight";
    in.text     = text;
    in.evidence = a["evidence"].get<std::vector<int>>();
    in.tags     = tags_to_comma_string(a);
    return in;
}

json supersedes_json(const std::vector<std::pair<int, float>>& ps) {
    json out = json::array();
    for (const auto& [id, score] : ps)
        out.push_back({{"id", id}, {"score", score}});
    return out;
}

} // namespace ragger
