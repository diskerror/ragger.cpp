/**
 * English language strings for Ragger
 *
 * To add a new language: copy this file to xx.h, translate values,
 * and include that file in lang.h instead.
 */
#pragma once

namespace ragger::lang {

// --- CLI ---
constexpr const char* CLI_DESCRIPTION         = "ragger — Ragger Memory";
constexpr const char* CLI_HELP                = "Show help";
constexpr const char* CLI_VERSION             = "Show version";
constexpr const char* CLI_CONFIG_FILE         = "Path to config file";
constexpr const char* CLI_HOST                = "Server bind address (overrides config)";
constexpr const char* CLI_PORT                = "Server port (overrides config)";
constexpr const char* CLI_DB                  = "SQLite database path (overrides config)";
constexpr const char* CLI_MODEL_DIR           = "Model directory path (overrides config)";
constexpr const char* CLI_COMMAND             = "Command";
constexpr const char* CLI_ARGS                = "Command arguments";
constexpr const char* CLI_USAGE_SEARCH        = "Usage: ragger search <query>";
constexpr const char* CLI_USAGE_STORE         = "Usage: ragger store <text>";
constexpr const char* CLI_UNKNOWN_COMMAND     = "Unknown command: ";

// --- Status messages ---
constexpr const char* MSG_LOADED_MEMORIES     = "Loaded %d memories";
constexpr const char* MSG_STORED_WITH_ID      = "Stored with id: ";
constexpr const char* MSG_SERVER_STARTING     = "Starting Ragger server on ";
constexpr const char* ERR_PORT_IN_USE_1       = "Error: port ";
constexpr const char* ERR_PORT_IN_USE_2       =
    " is already in use. Another instance may be running.\n"
    "To run multiple instances, set a different port in your config file\n"
    "(~/.ragger/ragger.ini) and update the OpenClaw plugin's serverUrl to match.";
constexpr const char* MSG_CONFIG_LOADED       = "Config loaded from ";
constexpr const char* MSG_CONFIG_CREATED      = "Created default config: ";

// --- Errors: config ---
constexpr const char* ERR_CONFIG_NOT_FOUND    = "No config file found.\n"
                                                 "Searched: /etc/ragger.ini, ~/.ragger/ragger.ini\n"
                                                 "Use --config-file=<path> to specify one.";
constexpr const char* ERR_CONFIG_FILE_MISSING = "Config file not found: ";
constexpr const char* ERR_CONFIG_OPEN         = "Cannot open config file: ";
constexpr const char* ERR_CONFIG_NOT_INIT     = "Config not initialized — call init_config() first";

// --- Errors: database ---
constexpr const char* ERR_SQLITE_OPEN         = "SQLite open failed: ";
constexpr const char* ERR_SQL                 = "SQL error: ";
constexpr const char* ERR_STORE_FAILED        = "Failed to store: ";

// --- Errors: embedder ---
constexpr const char* ERR_MODEL_NOT_FOUND     = "Model file not found: ";
constexpr const char* ERR_TOKENIZER_NOT_FOUND = "Tokenizer file not found: ";
constexpr const char* ERR_EMPTY_TOKENIZATION  = "Empty tokenization result";
constexpr const char* ERR_OUTPUT_SHAPE        = "Unexpected output shape from model";

// --- Errors: tokenizer wrapper ---
constexpr const char* ERR_TOKENIZER_OPEN      = "Failed to open tokenizer.json: ";
constexpr const char* ERR_TOKENIZER_EMPTY     = "Empty tokenizer.json file: ";
constexpr const char* ERR_TOKENIZER_CREATE    = "Failed to create tokenizer from JSON";
constexpr const char* ERR_TOKENIZER_NOT_INIT  = "Tokenizer not initialized";

} // namespace ragger::lang
