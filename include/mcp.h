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

namespace ragger {

class RaggerMemory;

/// Start the MCP JSON-RPC server over stdin/stdout.
/// Spawns a housekeeping thread if no HTTP server process is running,
/// then blocks in the read loop until stdin closes.
void run_mcp(RaggerMemory& memory);

} // namespace ragger
