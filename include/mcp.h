/**
 * MCP (Model Context Protocol) server for Ragger Memory.
 *
 * Implements a JSON-RPC 2.0 server over stdin/stdout following the
 * MCP specification. Exposes the Ragger memory API (store, search)
 * as MCP tools so any MCP-capable agent can use Ragger as its
 * memory backend without an HTTP server.
 *
 * Usage:  ragger mcp      (blocks until stdin is closed)
 */
#pragma once

#include "nlohmann_json.hpp"

namespace ragger {

class RaggerMemory;

/// Start the MCP JSON-RPC server over stdin/stdout.
/// Spawns a housekeeping thread if no HTTP server process is running,
/// then blocks in the read loop until stdin closes.
void run_mcp(RaggerMemory& memory);

/// Dispatch one MCP tools/call request and return the tool result.
/// Malformed arguments (a nlohmann type error, e.g. {"limit":"10"}) come
/// back as an isError tool result, never as an exception (C §1.12).
/// Exposed for testing; run_mcp is the only production caller.
nlohmann::json mcp_tool_call(RaggerMemory& memory, const nlohmann::json& params);

} // namespace ragger
