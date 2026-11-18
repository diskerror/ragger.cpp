#pragma once
/**
 * store_kind.h — the ONE rule for what `store` means (PHASE9B step 6).
 *
 * HTTP /store and the MCP `store` tool are the same verb; which knowledge
 * kind a call means must be decided in exactly one place so the two
 * transports can never disagree. metadata.category == "decision" selects
 * kind='decision'; everything else (absent, empty, "fact", ...) is a note.
 * X19: store() rejects the kind this rule selects for "decision" —
 * every decision, trigger-bearing or not, goes through memory_decision /
 * POST /decision. The rule itself stays here so the surfaces agree.
 */
#include "nlohmann_json.hpp"

#include <string>
#include <string_view>

namespace ragger {

inline std::string_view store_kind_for(const nlohmann::json& metadata) {
    if (!metadata.is_object()) return "note";
    return metadata.value("category", std::string()) == "decision" ? "decision" : "note";
}

} // namespace ragger
