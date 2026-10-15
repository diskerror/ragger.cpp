/**
 * Agent API request parsing — the single JSON->struct translation shared by
 * the HTTP surface (server.cpp) and the MCP surface (mcp.cpp) (review 2B).
 * Both surfaces used to hand-extract every field independently (~200 lines
 * of parallel maintenance); the defaults here are the documented API
 * defaults, so a field addition lands once and both surfaces stay in parity.
 *
 * Validation is deliberately thin: these parsers only lift JSON into the
 * query structs. Semantic validation (known layer, legal status
 * transition, resolvable project, ...) lives in the backend, which throws
 * std::invalid_argument; each surface maps that to its own error channel
 * (HTTP 400 / MCP isError).
 */
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "storage_types.h"
#include "nlohmann_json.hpp"

namespace ragger {

class RaggerMemory;

/// Project resolution (Phase 3): an explicit name/path wins; else a `cwd`
/// string that resolves to a known project is used; no match leaves it
/// empty (not an error).
std::string infer_project(RaggerMemory& mem, const nlohmann::json& a,
                          const std::string& explicit_project);

/// tags field -> comma string. Accepts an array of strings or a plain
/// string (the documented contract); absent or other types yield "".
std::string tags_to_comma_string(const nlohmann::json& a,
                                 const char* key = "tags");

/// POST /memory/search body / memory_search arguments -> AgentSearchQuery,
/// including cwd inference for the project filter.
AgentSearchQuery parse_agent_search_query(const nlohmann::json& a,
                                          RaggerMemory& mem);

/// POST /memory/get body / memory_get arguments -> AgentGetQuery.
AgentGetQuery parse_agent_get_query(const nlohmann::json& a);

/// POST /memory/timeline body / memory_timeline arguments -> TimelineQuery,
/// including cwd inference.
TimelineQuery parse_timeline_query(const nlohmann::json& a,
                                   RaggerMemory& mem);

/// POST /memory/update body / memory_update arguments -> UpdateRequest.
/// `text` is forwarded when present; agent_update rejects it with the
/// canonical "store a new row with supersedes" message.
UpdateRequest parse_update_request(const nlohmann::json& a);

/// POST /decision body / memory_decision arguments -> DecisionInput,
/// including cwd inference.
DecisionInput parse_decision_input(const nlohmann::json& a,
                                   RaggerMemory& mem);

/// /memory/note insight routing (Phase 5): the evidence-array branch of
/// the note body -> DecisionInput for store_insight. `text` and the
/// already-resolved `project` come from the caller (the note body shares
/// them with the plain-store branch).
DecisionInput parse_note_insight(const nlohmann::json& a,
                                 const std::string& text,
                                 const std::string& project);

/// possible_supersedes hint list -> the shared response shape
/// [{"id":N,"score":f}...].
nlohmann::json supersedes_json(const std::vector<std::pair<int, float>>& ps);

} // namespace ragger
