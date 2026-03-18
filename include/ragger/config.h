/**
 * Configuration for Ragger Memory (C++ port)
 *
 * Mirrors ragger_memory/config.py from the Python version.
 * All values are compile-time defaults; override via config file or CLI.
 */
#pragma once

#include <string>

namespace ragger {

// --- Server ---
constexpr const char* DEFAULT_HOST = "127.0.0.1";
constexpr int         DEFAULT_PORT = 8432;

// --- SQLite backend ---
// Expanded at runtime; "~" resolved to $HOME
constexpr const char* SQLITE_PATH           = "~/.ragger/memories.db";
constexpr const char* SQLITE_MEMORIES_TABLE  = "memories";
constexpr const char* SQLITE_USAGE_TABLE     = "memory_usage";

// --- Embedding model ---
constexpr const char* EMBEDDING_MODEL      = "all-MiniLM-L6-v2";
constexpr int         EMBEDDING_DIMENSIONS = 384;

// --- Logging ---
constexpr const char* LOG_DIR             = "~/.ragger";
constexpr bool        QUERY_LOG_ENABLED   = true;
constexpr bool        HTTP_LOG_ENABLED    = true;

// --- Usage tracking ---
constexpr bool USAGE_TRACKING_ENABLED = true;

// --- Path normalization ---
constexpr bool NORMALIZE_HOME_PATH = true;

// --- Hybrid search (BM25 + vector) ---
constexpr bool  BM25_ENABLED   = true;
constexpr float BM25_WEIGHT    = 0.3f;
constexpr float VECTOR_WEIGHT  = 0.7f;
constexpr float BM25_K1        = 1.5f;
constexpr float BM25_B         = 0.75f;

// --- Search defaults ---
constexpr int         DEFAULT_SEARCH_LIMIT = 5;
constexpr float       DEFAULT_MIN_SCORE    = 0.4f;
constexpr const char* DEFAULT_COLLECTION   = "memory";

// --- Runtime helpers ---
/// Expand ~ to $HOME in a path string.
std::string expand_path(const std::string& path);

/// Get the resolved SQLite database path.
std::string sqlite_db_path();

/// Get the resolved log directory.
std::string log_dir_path();

} // namespace ragger
