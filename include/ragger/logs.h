#pragma once
#include <string>

namespace ragger {

/// Initialize logging. Call once at startup after config is loaded.
/// @param verbose  Enable DEBUG on stderr
/// @param server_mode  Enable INFO on stderr, use system log_dir
void setup_logging(bool verbose = false, bool server_mode = false);

/// Log a query (if query logging enabled)
void log_query(const std::string& message);

/// Log an HTTP event (if HTTP logging enabled)
void log_http(const std::string& message);

/// Log an MCP event (if MCP logging enabled)
void log_mcp(const std::string& message);

/// Log an error (always on)
void log_error(const std::string& message);

/// Log an info message to stderr
void log_info(const std::string& message);

} // namespace ragger
