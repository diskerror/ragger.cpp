/**
 * agent_api.cpp — shared bodies of the agent API verbs (see agent_api.h).
 *
 * S70: `store`. Before this step HTTP /store carried the collection/source
 * validation and defaults while MCP `store` had neither, and each surface
 * shaped its own reply; the CLI had no category path at all. The body below
 * is now the only place the verb's arguments, defaults, kind rule and
 * response shape are defined.
 */
#include "agent_api.h"

#include "agent_parse.h"
#include "config.h"
#include "lang.h"
#include "Logger.h"
#include "store_kind.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace ragger {

namespace {
/// AgentHit -> JSON (the shaping was duplicated in server.cpp and mcp.cpp;
/// inventory D13). Key set unchanged.
nlohmann::json agent_hit_to_json(const AgentHit& h) {
    return nlohmann::json{{"id", h.id},
                          {"layer", h.layer},
                          {"score", h.score},
                          {"matched_by", h.matched_by},
                          {"created", h.created},
                          {"snippet", h.snippet},
                          {"extra", h.extra}};
}
} // namespace

nlohmann::json agent_store(RaggerMemory& mem, const nlohmann::json& args,
                           const std::string& default_source) {
    if (!args.is_object())
        throw std::invalid_argument("arguments must be an object");
    const std::string text = args.value("text", std::string());
    if (text.empty())
        throw std::invalid_argument("text is required");

    nlohmann::json metadata = args.value("metadata", nlohmann::json::object());
    if (metadata.is_null())
        metadata = nlohmann::json::object();
    if (!metadata.is_object())
        throw std::invalid_argument("metadata must be an object");
    // S27 (C §4): a non-string collection/source is a client error with a
    // clear message. This validation used to live in the HTTP handler only;
    // every surface now rejects it the same way.
    if (metadata.contains("collection") && !metadata["collection"].is_string())
        throw std::invalid_argument("metadata.collection must be a string");
    if (metadata.contains("source") && !metadata["source"].is_string())
        throw std::invalid_argument("metadata.source must be a string");
    if (!metadata.contains("collection") || metadata["collection"].get<std::string>().empty())
        metadata["collection"] = "memory";
    if (!metadata.contains("source") || metadata["source"].get<std::string>().empty())
        metadata["source"] = default_source;

    const std::string id = mem.store(text, metadata);
    return nlohmann::json{{"id", id},
                          {"status", "stored"},
                          {"kind", std::string(store_kind_for(metadata))}};
}

nlohmann::json agent_search(RaggerMemory& mem, const nlohmann::json& args) {
    // One message now (inventory D4: was HTTP "Missing 'query' field" /
    // MCP "Error: query parameter required").
    const std::string query = args.value("query", std::string());
    if (query.empty())
        throw std::invalid_argument("query is required");
    const int limit = args.value("limit", config().default_search_limit);
    const float min_score = args.value("min_score", config().default_min_score);
    const auto collections = args.value("collections", std::vector<std::string>{});

    const auto t0 = std::chrono::high_resolution_clock::now();
    auto sr = mem.search(query, limit, min_score, collections);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t0).count();
    nlohmann::json timing = sr.timing;
    timing["total_ms"] = ms;
    Diskerror::Logger::trace(std::format(lang::DBG_QUERY_LOG, query, sr.results.size(), ms));

    nlohmann::json arr = nlohmann::json::array();
    for (const auto& r : sr.results)
        arr.push_back({{"id", r.id}, {"text", r.text}, {"score", r.score},
                       {"metadata", r.metadata}, {"timestamp", r.timestamp}});
    return nlohmann::json{{"results", std::move(arr)},
                          {"timing", std::move(timing)}};
}

nlohmann::json agent_memory_search(RaggerMemory& mem, const nlohmann::json& args) {
    // D12 resolved: the empty-query check existed only on MCP; every
    // surface now rejects it the same way (HTTP used to run a garbage
    // empty-string search).
    AgentSearchQuery q = parse_agent_search_query(args);
    if (q.query.empty())
        throw std::invalid_argument("query is required");
    const auto hits = mem.agent_search(q);
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& h : hits)
        arr.push_back(agent_hit_to_json(h));
    return nlohmann::json{{"results", std::move(arr)}};
}

nlohmann::json agent_memory_get(RaggerMemory& mem, const nlohmann::json& args) {
    return mem.agent_get(parse_agent_get_query(args));
}

nlohmann::json agent_memory_timeline(RaggerMemory& mem, const nlohmann::json& args) {
    return mem.agent_timeline(parse_timeline_query(args));
}

nlohmann::json agent_memory_note(RaggerMemory& mem, const nlohmann::json& args,
                                 const std::string& default_source) {
    const std::string text = args.value("text", std::string());
    if (text.empty())
        throw std::invalid_argument("text is required");
    validate_note_request(args);   // LR S59 (D-3): trigger needs evidence

    // ---- insight routing (Phase 5) --------------------------------------
    // evidence array -> store_knowledge(kind='insight'). This branch used to
    // be written twice (server.cpp and mcp.cpp, inventory D14); it exists
    // once here. Response keys unchanged on both paths.
    if (args.contains("evidence") && args["evidence"].is_array()) {
        const auto r = mem.store_knowledge(parse_note_insight(args, text));
        return nlohmann::json{{"id", r.id},
                              {"kind", r.kind},
                              {"status", "stored"},
                              {"superseded", r.superseded ? nlohmann::json(*r.superseded)
                                                          : nlohmann::json(nullptr)},
                              {"possible_supersedes", supersedes_json(r.possible_supersedes)}};
    }

    // ---- plain note ------------------------------------------------------
    nlohmann::json metadata = nlohmann::json::object();
    metadata["source"] = default_source;
    if (args.contains("tags"))
        metadata["tags"] = args["tags"];
    // LR S59: plain notes carry session provenance (GUID lookup only,
    // unknown GUID stores with NULL session — Impl::store's rule).
    if (args.contains("session") && args["session"].is_string())
        metadata["session"] = args["session"];
    const std::string id = mem.store(text, metadata);
    return nlohmann::json{{"id", id},
                          {"status", "stored"},
                          {"possible_supersedes",
                           supersedes_json(mem.possible_supersedes_for_text(text))}};
}

nlohmann::json agent_memory_decision(RaggerMemory& mem, const nlohmann::json& args) {
    const auto r = mem.store_knowledge(parse_decision_input(args));
    return nlohmann::json{{"id", r.id},
                          {"kind", r.kind},
                          {"status", "stored"},
                          {"superseded", r.superseded ? nlohmann::json(*r.superseded)
                                                      : nlohmann::json(nullptr)},
                          {"possible_supersedes", supersedes_json(r.possible_supersedes)}};
}

nlohmann::json agent_memory_update(RaggerMemory& mem, const nlohmann::json& args) {
    return mem.agent_update(parse_update_request(args));
}

nlohmann::json agent_memory_review(RaggerMemory& mem, const nlohmann::json& args) {
    const auto [action, ids] = parse_review_request(args);
    return mem.agent_review(action, ids);
}

CaptureResult agent_capture_turn(RaggerMemory& mem, const nlohmann::json& args) {
    // One message now (D16: was HTTP {"error":"missing required field: user"}
    // as a JSON body / MCP "Error: user parameter required").
    const std::string user = args.value("user", std::string());
    if (user.empty())
        throw std::invalid_argument("user is required");
    return capture_turn(mem, user,
                        args.value("assistant", std::string()),
                        args.value("model", std::string()),
                        args.value("session_id", std::string()),
                        args.value("session_name", std::string()),
                        args.value("name_source", std::string()));
}

nlohmann::json capture_turn_payload(const CaptureResult& result) {
    return nlohmann::json{{"status", result.captured ? "captured" : "disabled"},
                          {"turn_id", result.turn_id}};
}

nlohmann::json agent_build_context(RaggerMemory& mem, const nlohmann::json& args) {
    const std::string session = args.value("session_id", std::string());
    const auto ctx = build_context(mem, session, args.value("recipe", std::string()));
    if (!ctx.enabled)
        return nlohmann::json{{"status", "disabled"}};
    nlohmann::json chunks = nlohmann::json::array();
    for (const auto& c : ctx.chunks)
        chunks.push_back({{"kind", c.kind}, {"text", c.text}, {"timestamp", c.timestamp}});
    return nlohmann::json{{"status", "ok"},
                          {"session_id", session},
                          {"recipe", ctx.recipe_name},
                          {"chunks", std::move(chunks)}};
}

} // namespace ragger
