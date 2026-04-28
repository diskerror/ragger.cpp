/**
 * English language strings for Ragger
 *
 * To add a new language: copy this file to xx.h, translate values,
 * and include that file in lang.h instead.
 */
#pragma once

namespace ragger::lang {

// --- CLI options ---
constexpr const char* CLI_DESCRIPTION         = "ragger — Ragger Memory";
constexpr const char* CLI_HELP                = "Show help";
constexpr const char* CLI_VERSION             = "Show version";
constexpr const char* CLI_CONFIG_FILE         = "Path to config file";
constexpr const char* CLI_HOST                = "Server bind address (overrides config)";
constexpr const char* CLI_PORT                = "Server port (overrides config)";
constexpr const char* CLI_DB                  = "SQLite database path (overrides config)";
constexpr const char* CLI_MODEL_DIR           = "Model directory path (overrides config)";
constexpr const char* CLI_LM_PROXY_URL        = "LM proxy upstream URL (OpenAI-compatible pass-through)";
constexpr const char* CLI_COMMAND             = "Command";
constexpr const char* CLI_ARGS                = "Command arguments";
constexpr const char* CLI_USAGE_SEARCH        = "Usage: ragger search <query>";
constexpr const char* CLI_USAGE_STORE         = "Usage: ragger store <text>";
constexpr const char* CLI_USAGE_USERADD       = "Usage: ragger useradd <username>";
constexpr const char* CLI_USAGE_USERMOD       = "Usage: ragger usermod <username>";
constexpr const char* CLI_USAGE_USERDEL       = "Usage: ragger userdel <username>";
constexpr const char* CLI_USAGE_PASSWD        = "Usage: ragger passwd <username>";
constexpr const char* CLI_USAGE_IMPORT        = "Usage: ragger import <file> [file...] [--collection name]";
constexpr const char* CLI_UNKNOWN_COMMAND     = "Unknown command: ";

// --- Status messages ---
constexpr const char* MSG_LOADED_MEMORIES     = "Loaded %d memories";
constexpr const char* MSG_BACKFILLED_EMBEDDINGS  = "Backfilled embeddings for {} row(s).";

// --- Migration / maintenance ---
constexpr const char* MSG_MIGRATE_EMBEDDING_NULLABLE = "Migrating: dropping NOT NULL on memories.embedding...";
constexpr const char* MSG_MIGRATE_USER_ID            = "Migrated memories: added user_id column";
constexpr const char* MSG_MIGRATE_DEDICATED_COLUMNS  = "Migrating: adding collection, category, tags columns...";
constexpr const char* MSG_MIGRATE_DEDICATED_BACKFILL = "Migrated {} rows: collection/category/tags extracted";
constexpr const char* MSG_MIGRATE_TOKEN_ROTATED_AT   = "Migrated users: added token_rotated_at column";
constexpr const char* MSG_MIGRATE_PREFERRED_MODEL    = "Migrated users: added preferred_model column";
constexpr const char* MSG_MIGRATE_PASSWORD_HASH      = "Migrated users: added password_hash column";
constexpr const char* MSG_REBUILD_EMBEDDINGS_PROGRESS = "\rRebuilding embeddings: {}/{}";
constexpr const char* WARN_FORMAT_LOAD_FAILED        = "Failed to load format {}: {}";
constexpr const char* MSG_STORED_WITH_ID      = "Stored with id: ";
constexpr const char* MSG_SERVER_STARTING     = "Starting Ragger server on ";
constexpr const char* ERR_PORT_IN_USE         = "Error: port {} is already in use";
constexpr const char* MSG_CONFIG_LOADED       = "Config loaded from ";
constexpr const char* MSG_CONFIG_CREATED      = "Created default config: ";

// --- Import ---
constexpr const char* MSG_IMPORTING_CHUNKS    = "Importing {} chunks from {}...";
constexpr const char* MSG_IMPORT_CHUNK        = "  Chunk {}/{}: {}";
constexpr const char* MSG_IMPORT_DONE         = "✓ Imported {} chunks";
constexpr const char* ERR_PAYLOAD_DUMP_DIR    = "Error: cannot create payload dump directory '{}': {}";
constexpr const char* MSG_PAYLOAD_DUMP_CREATED= "Created payload dump directory: {}";

// --- Inference / chat setup ---
constexpr const char* ERR_NO_ENDPOINTS        = "Error: no inference endpoints configured.";
constexpr const char* MSG_INFERENCE_HINT      = "Add to settings.ini:\n"
                                                "\n"
                                                "  [inference]\n"
                                                "  api_url = http://localhost:1234/v1\n"
                                                "  api_key = lmstudio-local\n"
                                                "\n"
                                                "Or for multiple endpoints:\n"
                                                "\n"
                                                "  [inference.local]\n"
                                                "  api_url = http://localhost:1234/v1\n"
                                                "  api_key = lmstudio-local\n"
                                                "  models = qwen/*, llama/*";
constexpr const char* ERR_ENDPOINT_UNREACHABLE= "Error: cannot reach inference endpoint '{}' at {}";
constexpr const char* ERR_ENDPOINT_HTTP       = "Error: inference endpoint '{}' returned HTTP {}";

// --- Daemon control ---
constexpr const char* ERR_HOME_NOT_FOUND      = "Error: cannot resolve $HOME";
constexpr const char* ERR_PLIST_NOT_FOUND     = "Error: {} not found. Run ./install.sh first.";
constexpr const char* ERR_LAUNCHCTL_KICKSTART = "Error: launchctl kickstart failed";
constexpr const char* ERR_LAUNCHCTL_BOOTSTRAP = "Error: launchctl bootstrap failed";
constexpr const char* ERR_LAUNCHCTL_BOOTOUT   = "Error: launchctl bootout failed";
constexpr const char* ERR_SYSTEMCTL_START     = "Error: systemctl start failed";
constexpr const char* ERR_SYSTEMCTL_STOP      = "Error: systemctl stop failed";
constexpr const char* ERR_UNKNOWN_ACTION      = "Error: unknown daemon action '{}'";
constexpr const char* ERR_DAEMON_UNSUPPORTED  = "Error: daemon control not supported on this platform.\n"
                                                "       Run 'ragger serve' manually.";
constexpr const char* MSG_ALREADY_RUNNING     = "ragger is already running (pid {})";
constexpr const char* MSG_STARTED             = "ragger started";
constexpr const char* MSG_STARTED_PID         = "ragger started (pid {})";
constexpr const char* MSG_RESTARTED           = "ragger restarted";
constexpr const char* MSG_RESTARTED_PID       = "ragger restarted (pid {})";
constexpr const char* MSG_STOPPED             = "ragger stopped";
// Singular/plural split — natural language can't append 's' universally
constexpr const char* MSG_STOPPED_EXTRA_1     = "ragger stopped (and {} other instance)";
constexpr const char* MSG_STOPPED_EXTRA_N     = "ragger stopped (and {} other instances)";
constexpr const char* MSG_EXTRAS_ONLY_1       = "ragger daemon was not running; stopped {} other instance";
constexpr const char* MSG_EXTRAS_ONLY_N       = "ragger daemon was not running; stopped {} other instances";
constexpr const char* MSG_NOT_RUNNING         = "ragger is not running";
constexpr const char* MSG_NOT_LOADED          = "ragger is not loaded. Run: ragger start";
constexpr const char* MSG_LOADED_NOT_RUNNING  = "ragger is loaded but not running";
constexpr const char* MSG_RUNNING_PID         = "ragger is running (pid {})";
constexpr const char* MSG_IS_RUNNING          = "ragger is running";

// --- User management ---
constexpr const char* ERR_USERADD_EXISTS      = "Error: user '{}' already exists.";
constexpr const char* ERR_USERADD_EXISTS_HINT = "       Use `ragger usermod {}` to rotate their token.";
constexpr const char* ERR_USERMOD_MISSING     = "Error: user '{}' does not exist.";
constexpr const char* ERR_USERMOD_MISSING_HINT= "       Use `ragger useradd {}` to create them.";
constexpr const char* ERR_PASSWD_MISSING_HINT = "       Create them first: ragger useradd {}";
constexpr const char* ERR_UNKNOWN_USER        = "Error: cannot determine username";
constexpr const char* MSG_USER_ADDED          = "✓ User {} added";
constexpr const char* MSG_USER_REMOVED        = "✓ User {} removed";
constexpr const char* MSG_TOKEN_ROTATED       = "✓ Token rotated for {}";
constexpr const char* MSG_TOKEN_VALUE         = "Token: {}";
constexpr const char* MSG_TOKEN_SAVE_WARNING  = "Save this now — it will not be shown again.";
constexpr const char* MSG_PASSWORD_SET        = "✓ Password set for {}";
constexpr const char* MSG_PASSWORD_CLEARED    = "✓ Password cleared for {} (web-UI login disabled)";
constexpr const char* ERR_PASSWORDS_DIFFER    = "Error: passwords do not match";
constexpr const char* PROMPT_NEW_PASSWORD     = "New password (empty to clear): ";
constexpr const char* PROMPT_CONFIRM_PASSWORD = "Confirm password: ";

// --- Token bootstrap (add-self) ---
constexpr const char* MSG_TOKEN_CREATED       = "✓ Created ~/.ragger/token for {}";
constexpr const char* MSG_TOKEN_EXISTS        = "Token already exists for {}";
constexpr const char* MSG_YOUR_TOKEN          = "\nYour token: {}";
constexpr const char* MSG_TOKEN_USE_HINT      = "Use this in your client config (OpenClaw, Claude Desktop, etc).";
constexpr const char* MSG_TOKEN_FILE_HINT     = "Token file: ~/.ragger/token";
constexpr const char* MSG_USER_IN_DB          = "✓ User exists in database (id: {})";
constexpr const char* MSG_USER_REGISTERED     = "✓ Registered in database (user_id: {})";
constexpr const char* WARN_DB_DEFERRED        = "Warning: DB registration deferred ({})";

// --- Rebuild ---
constexpr const char* MSG_BM25_REBUILT        = "✓ BM25 index rebuilt: {} documents";
constexpr const char* MSG_REBUILD_CONFIRM     = "This will re-embed all {} memories. The server should be stopped first.";
constexpr const char* PROMPT_CONTINUE         = "Continue? [y/N] ";
constexpr const char* MSG_ABORTED             = "Aborted.";
constexpr const char* MSG_DB_BACKED_UP        = "Database backed up to: {}";
constexpr const char* WARN_BACKUP_FAILED      = "Warning: Failed to create backup: {}";
constexpr const char* MSG_EMBEDDINGS_REBUILT  = "✓ Embeddings rebuilt: {} documents";

// --- Chat prompts ---
constexpr const char* CHAT_PROMPT_USER        = "You: ";
constexpr const char* CHAT_PROMPT_ASSISTANT   = "Assistant: ";
constexpr const char* WARN_STORE_TURN         = "Warning: failed to store turn: {}";
constexpr const char* WARN_ORPHAN_CHECK       = "Warning: orphan check failed: {}";
constexpr const char* WARN_SUMMARY            = "Warning: summary generation failed: {}";
constexpr const char* ERR_NO_SYSTEM_PROMPT    = "Error: no system prompt files found. ";
constexpr const char* MSG_NO_SYSTEM_PROMPT_HINT = "Create ~/.ragger/SYSTEM.md or add SOUL.md / USER.md / MEMORY.md.";
constexpr const char* WARN_ENDPOINT_DOWN      = "Warning: {} is not reachable";
constexpr const char* WARN_CHAT_ERROR         = "Warning: {}";
constexpr const char* WARN_MEMORY_SEARCH      = "Warning: memory search failed: {}";

// --- Chat REPL banner / status ---
constexpr const char* MSG_CHAT_BANNER         = "Ragger Chat (model: {})";
constexpr const char* MSG_CHAT_TURN_STORAGE   = "Turn storage: {}";
constexpr const char* MSG_CHAT_CONTEXT_SIZED  = "Context: {} tokens ({}) → {}% = {} chars persona";
constexpr const char* MSG_CHAT_CONTEXT_OPEN   = "Context: {} ({}) | Persona: {}";
constexpr const char* MSG_CHAT_CTX_UNKNOWN    = "unknown";
constexpr const char* MSG_CHAT_CTX_TOKENS     = "{} tokens";
constexpr const char* MSG_CHAT_PERSONA_CHARS  = "{} chars";
constexpr const char* MSG_CHAT_PERSONA_NONE   = "unlimited";
constexpr const char* MSG_CHAT_QUIT_HINT      = "Type '/quit' or Ctrl+D to exit";
constexpr const char* MSG_CHAT_GOODBYE        = "Goodbye!";

// --- Chat recovery ---
constexpr const char* MSG_ORPHAN_FOUND        = "Found {} orphaned turns from previous session...";
constexpr const char* MSG_ORPHAN_RECOVERED    = "Recovered {} orphaned turns (deleted {} raw entries)";
constexpr const char* MSG_SUMMARIZING         = "Summarizing in background...";

// --- Chat /models ---
constexpr const char* MSG_MODELS_QUERYING     = "Querying {} ({})...";
constexpr const char* MSG_MODELS_NONE         = "  (no models returned or endpoint unreachable)";
constexpr const char* MSG_MODELS_COUNT        = "{} model(s)";
constexpr const char* MSG_MODELS_NO_ENDPOINTS = "No endpoints configured.";
constexpr const char* MSG_MODELS_ALIASES      = "Aliases:";
constexpr const char* MSG_MODELS_ALIAS_LINE   = "  {} → {}";
constexpr const char* MSG_MODELS_CURRENT      = "Current: {}";
constexpr const char* MSG_MODEL_DEFAULT       = "(default)";

// --- Chat /model ---
constexpr const char* MSG_MODEL_CURRENT       = "Current model: {}";
constexpr const char* MSG_MODEL_USAGE_SWITCH  = "Use /model <name> to switch";
constexpr const char* MSG_MODEL_USAGE         = "Usage: /model <name>";
constexpr const char* MSG_MODEL_SWITCHED      = "Switched to: {}";
constexpr const char* MSG_MODEL_ALIAS_SUFFIX  = " (alias: {})";

// --- Chat /endpoints ---
constexpr const char* MSG_ENDPOINTS_HEADER    = "Endpoints:";
constexpr const char* MSG_ENDPOINT_LINE       = "  {} {} — {} [{}]{}";
constexpr const char* MSG_ENDPOINT_ACTIVE     = " (active)";
constexpr const char* MSG_ENDPOINT_ROUTING    = "Routing: {}";
constexpr const char* MSG_ENDPOINT_USAGE      = "Use /endpoint <name> to force, /endpoint auto to auto-route";
constexpr const char* MSG_ENDPOINT_CURRENT    = "Current endpoint: {}";
constexpr const char* MSG_ENDPOINT_USAGE_ARG  = "Usage: /endpoint <name|auto>";
constexpr const char* MSG_ENDPOINT_AUTO       = "Routing: auto (model-based)";
constexpr const char* MSG_ENDPOINT_FORCED     = "Forced endpoint: {}";
constexpr const char* MSG_ENDPOINT_AVAILABLE  = "Available: {}";

// --- Chat /help ---
constexpr const char* MSG_HELP_HEADER         = "Commands:";
constexpr const char* MSG_HELP_MODELS         = "  /models          — list models available on active endpoint";
constexpr const char* MSG_HELP_MODEL          = "  /model [name]    — show or switch model (alias or full name)";
constexpr const char* MSG_HELP_ENDPOINTS      = "  /endpoints       — list inference endpoints with status";
constexpr const char* MSG_HELP_ENDPOINT       = "  /endpoint [name] — show, force, or auto-route endpoint";
constexpr const char* MSG_HELP_HELP           = "  /help            — show this help";
constexpr const char* MSG_HELP_QUIT           = "  /quit            — exit chat (also /exit, Ctrl+D)";

// --- Chat inference error ---
constexpr const char* ERR_INFERENCE           = "Error: {}";

// --- MCP ---
constexpr const char* ERR_MCP_TEXT_REQUIRED   = "Error: text parameter required";
constexpr const char* ERR_MCP_QUERY_REQUIRED  = "Error: query parameter required";
constexpr const char* MSG_MCP_NO_RESULTS      = "No results found.";

// --- Errors: config ---
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
