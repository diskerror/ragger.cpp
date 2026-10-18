/**
 * Agent API shared helpers — used by both the read surface (agent_read.cpp:
 * agent_search / agent_get / agent_timeline) and the write surface
 * (agent_write.cpp: agent_update / agent_review). Moved out of agent_read.cpp
 * by review 5A; behavior unchanged.
 */
#pragma once

#include <sqlite3.h>

#include <string>

#include "nlohmann_json.hpp"

namespace ragger::sqlite {

/// First <=240 BYTES of text, cut at a whitespace boundary; "…" appended
/// only when the text was actually cut. See agent_common.cpp for the
/// UTF-8-safety rules.
std::string agent_snippet(const std::string& text);

/// "<layer>:<rowid>" strict parse (API format only, never stored): exactly
/// one colon, known layer, all-digits positive rowid without leading zeros.
/// Overflowing ids return false (never throw).
bool parse_agent_handle(const std::string& handle, std::string& layer, int& rowid);

// PHASE7: true for legacy 'decision:<n>' / 'insight:<n>' handles (strict
// positive-integer suffix). agent_get/agent_update reject these with a
// pointer to knowledge:<id>; anything looser ("decision:", "decision:-3")
// stays an ordinary parse failure ("not found").
bool is_legacy_knowledge_handle(const std::string& handle);

/// True for the six searchable layer names (the typed-handle vocabulary).
bool is_known_agent_layer(const std::string& layer);

/// Full row by layer + rowid in the agent API shape (PHASE2-STEPS step 4:
/// id/layer/text/created plus the layer's own columns); json(nullptr) when
/// the row does not exist.
nlohmann::json fetch_agent_row(sqlite3* db, const std::string& layer, int rowid);

} // namespace ragger::sqlite
