/**
 * agent_api.h — one function per verb (R4): the transport-independent body
 * of each agent API verb, shared by the HTTP surface (server.cpp), the MCP
 * surface (mcp.cpp) and the CLI (main.cpp).
 *
 * Envelopes stay per-transport (LR ruling D-4): HTTP answers JSON with a
 * status code, MCP answers content[]/isError, the CLI prints one line. Only
 * the *mapping* is shared: a validation failure throws
 * std::invalid_argument and each surface routes it to its own error channel
 * (HTTP 400 "ERROR: <msg>" / MCP error_result / CLI stderr + exit 1).
 *
 * S70 starts the move with `store`; S71/S72 move the read and write verbs.
 */
#pragma once

#include "memory.h"
#include "nlohmann_json.hpp"

#include <string>

namespace ragger {

struct Config;            // config.h
class InferenceClient;    // inference.h

/// The `store` verb (HTTP POST /store, MCP tool `store`, CLI `ragger store`).
/// args: {text (required), metadata?: object}. metadata.category == "decision"
/// is rejected X19 (Q9/L6): a decision needs a trigger, which this verb cannot
/// carry — the decision verb (memory_decision / POST /decision) is the only way
/// to write kind='decision'.
/// Everything else is a note. metadata.collection defaults to "memory";
/// metadata.source defaults to `default_source` — per-transport provenance
/// (HTTP: the authenticated user; MCP: "mcp"; CLI: "cli").
/// Returns the response object {id, status: "stored", kind}.
/// Throws std::invalid_argument on validation failure.
nlohmann::json agent_store(RaggerMemory& mem, const nlohmann::json& args,
                           const std::string& default_source);

/// Flat hybrid search (HTTP POST /search, MCP tool `search`, CLI
/// `ragger search`). Returns {results:[{id,text,score,metadata,timestamp}],
/// timing:{...,total_ms}}. The MCP surface answers the bare `results` array
/// and the CLI prints its own summary — envelope differences preserved from
/// before this step (S71 is a pure move; the pins prove it).
nlohmann::json agent_search(RaggerMemory& mem, const nlohmann::json& args);

/// Layered agent search (HTTP POST /memory/search, MCP tool memory_search).
/// Returns {results:[AgentHit...]}. The empty-query check used to exist only
/// on MCP (inventory D12); it is now one rule on every surface.
/// X25: with args.split == true the response instead is
/// {groups:{both,vector_only,fts_only}, vector_available, min_vector_score,
/// min_fts_score} — two ordinary agent_search runs (semantic + keyword)
/// partitioned by handle, native scores, absent score = null. The default
/// (split absent/false) path is unchanged.
nlohmann::json agent_memory_search(RaggerMemory& mem, const nlohmann::json& args);

/// LR S76 (D-7): documents' own verb (HTTP POST /memory/documents, MCP tool
/// document_search). {query, source?, limit?} -> {results: [hits]}.
nlohmann::json agent_document_search(RaggerMemory& mem, const nlohmann::json& args);

/// Agent row reads (HTTP POST /memory/get, MCP tool memory_get).
nlohmann::json agent_memory_get(RaggerMemory& mem, const nlohmann::json& args);

/// Timeline (HTTP POST /memory/timeline, MCP tool memory_timeline).
nlohmann::json agent_memory_timeline(RaggerMemory& mem, const nlohmann::json& args);

/// Note/insight (HTTP POST /memory/note, MCP tool memory_note). args:
/// {text (required), tags?, session?, trigger?, evidence?}. evidence array
/// routes to kind='insight' (the routing was written twice, inventory D14;
/// it exists once here). `default_source` is the plain-note provenance
/// (HTTP: the authenticated user; MCP: "mcp").
nlohmann::json agent_memory_note(RaggerMemory& mem, const nlohmann::json& args,
                                 const std::string& default_source);

/// Capture / summarizer readiness flags shared by GET /health and the MCP
/// `status` tool, so an agent plugin can tell why nothing is being captured or
/// summarized without guessing:
///   capture_turns      bool    [server] capture_turns (false: /turn is a no-op)
///   summarizer_model   string  "" when no inference endpoint is configured
///   summarizer_engine  string  endpoint name serving it; "" when none does
///   summarizer_reachable bool  that endpoint answers GET /v1/models (cached
///                              15 s, 1 s probe timeout; false when none)
/// Config reads plus at most one cached probe; no DB.
nlohmann::json capture_summarizer_flags(const Config& cfg, const InferenceClient& inference);

/// Structured decision (HTTP POST /decision, MCP tool memory_decision).
nlohmann::json agent_memory_decision(RaggerMemory& mem, const nlohmann::json& args);

/// Metadata-only curation (HTTP POST /memory/update, MCP tool memory_update).
nlohmann::json agent_memory_update(RaggerMemory& mem, const nlohmann::json& args);

/// Pending queue (HTTP POST /memory/review only — memory_review left the
/// MCP tool list at LR S88, and there is no CLI memory_review verb; the
/// CLI path is `decision list --status pending` + `decision set-status`).
nlohmann::json agent_memory_review(RaggerMemory& mem, const nlohmann::json& args);

/// Turn capture (HTTP POST /turn, MCP tool capture_turn). Thin wrapper over
/// the shared capture_turn(): one argument rule ('user' required, was two
/// different messages, inventory D16). Returns the capture result so the
/// HTTP surface can hand the fresh turn to its summarizer (D17: HTTP eager,
/// MCP lazy via the daemon tick — architectural, kept).
CaptureResult agent_capture_turn(RaggerMemory& mem, const nlohmann::json& args);

/// The shared {status, turn_id} response payload of the turn verb.
nlohmann::json capture_turn_payload(const CaptureResult& result);

/// Session context (HTTP GET /session/<guid>[?recipe=], MCP tool
/// build_context). The HTTP surface maps path+query onto the same args
/// object (inventory D21: arg placement differs, payload never did).
nlohmann::json agent_build_context(RaggerMemory& mem, const nlohmann::json& args);

} // namespace ragger
