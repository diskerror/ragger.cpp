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
constexpr const char* CLI_MODEL_DIR           = "Model directory path (overrides config)";
constexpr const char* CLI_COMMAND             = "Command";
constexpr const char* CLI_ARGS                = "Command arguments";
constexpr const char* CLI_USAGE_SEARCH        = "Usage: ragger search <query>";
constexpr const char* CLI_USAGE_STORE         = "Usage: ragger store <text>";
constexpr const char* CLI_USAGE_USERADD       = "Usage: ragger useradd <username>";
constexpr const char* CLI_USAGE_USERMOD       = "Usage: ragger usermod <username>";
constexpr const char* CLI_USAGE_USERDEL       = "Usage: ragger userdel <username>";
constexpr const char* CLI_USAGE_PASSWD        = "Usage: ragger passwd <username>";
constexpr const char* CLI_USAGE_IMPORT        =
    "Usage:\n"
    "  ragger import <file> [file...]                  # markdown/text → L5 documents\n"
    "  ragger import conversations --format=code|web PATH\n"
    "                                                  # Claude Code JSONL or claude.ai export\n"
    "                                                  # [--since YYYY-MM-DD] [--until YYYY-MM-DD] [--session ID]\n"
    "  ragger import summaries <file> [file...]        # one L4 project summary per file\n"
    "  ragger import summaries --jsonl=FILE            # {text, tags?, timestamp?} per line";
constexpr const char* CLI_USAGE_EXPORT        = "Usage: ragger export <all|table> [-e|--embeddings] [--output file]";
constexpr const char* CLI_USAGE_EMBED           = "Usage: ragger embed";
constexpr const char* CLI_UNKNOWN_COMMAND     = "Unknown command: {}";

// --- Status messages ---
constexpr const char* MSG_LOADED_MEMORIES     = "Loaded {} memories";
constexpr const char* MSG_BACKFILLED_EMBEDDINGS  = "Backfilled embeddings for {} row(s).";

// --- Migration / maintenance ---
// (The schema is declarative — there is no in-place migration. Pre-v2 data is
//  copied into a fresh v2 DB out-of-band.)
constexpr const char* MSG_REBUILD_EMBEDDINGS_PROGRESS = "\rRebuilding embeddings: {}/{}";
constexpr const char* WARN_FORMAT_LOAD_FAILED        = "Failed to load format {}: {}";
constexpr const char* MSG_STORED_WITH_ID      = "Stored with id: {}";
constexpr const char* MSG_SERVER_STARTING     = "Starting Ragger server on {}";
constexpr const char* ERR_PORT_IN_USE         = "Error: port {} is already in use";
constexpr const char* MSG_CONFIG_LOADED       = "Config loaded from {}";
constexpr const char* MSG_CONFIG_CREATED      = "Created default config: {}";

// --- Import ---
constexpr const char* MSG_IMPORTING_CHUNKS    = "Importing {} chunks from {}...";
constexpr const char* MSG_IMPORT_CHUNK        = "  Chunk {}/{}: {}";
constexpr const char* MSG_IMPORT_DONE         = "✓ Imported {} chunks";

// --- Export ---
constexpr const char* MSG_EXPORT_DONE         = "✓ Exported {} rows";
constexpr const char* MSG_EXPORT_TABLE_NOT_FOUND = "Error: table '{}' not found. Available: {}";
constexpr const char* MSG_EXPORT_WROTE_FILE   = "Wrote {}";
constexpr const char* CLI_EMBEDDINGS          = "Include embedding column in export";
constexpr const char* ERR_PAYLOAD_DUMP_DIR    = "Error: cannot create payload dump directory '{}': {}";
constexpr const char* MSG_PAYLOAD_DUMP_CREATED= "Created payload dump directory: {}";

// --- Inference / chat setup ---
constexpr const char* ERR_NO_ENDPOINTS_HINT    = R"(Error: no inference endpoints configured.
Add to settings.ini:

  [inference]
  api_url = http://localhost:1234/v1
  api_key = lmstudio-local

Or for multiple endpoints:

  [inference.local]
  api_url = http://localhost:1234/v1
  api_key = lmstudio-local
  models = qwen/*, llama/*)";
constexpr const char* ERR_ENDPOINT_UNREACHABLE= "Error: cannot reach inference endpoint '{}' at {}\n    {}";
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
constexpr const char* ERR_DAEMON_UNSUPPORTED  = R"(Error: daemon control not supported on this platform.
       Run 'ragger serve' manually.)";
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
constexpr const char* MSG_REBUILD_CONFIRM     = "This will re-embed all {} stored rows (turns, summaries, decisions, documents). The server should be stopped first.";
constexpr const char* PROMPT_CONTINUE         = "Continue? [y/N] ";
constexpr const char* MSG_ABORTED             = "Aborted.";
constexpr const char* MSG_DB_BACKED_UP        = "Database backed up to: {}";
constexpr const char* WARN_BACKUP_FAILED      = "Warning: Failed to create backup: {}";
constexpr const char* MSG_EMBEDDINGS_REBUILT  = "✓ Embeddings rebuilt: {} rows";

// --- Chat prompts ---
constexpr const char* CHAT_PROMPT_USER        = "You: ";
constexpr const char* CHAT_PROMPT_ASSISTANT   = "Assistant: ";
constexpr const char* WARN_STORE_TURN         = "Warning: failed to store turn: {}";
constexpr const char* WARN_EMBED_SUBPROCESS   = "Warning: embed subprocess failed (attempt {}/{}); embedding skipped";
constexpr const char* WARN_IMPORT_EMBED_SKIPPED = "Warning: {}/{} chunks left unembedded (embed subprocess failed or timed out); re-import to retry";
constexpr const char* WARN_ORPHAN_CHECK       = "Warning: orphan check failed: {}";
constexpr const char* WARN_SUMMARY            = "Warning: summary generation failed: {}";

// --- Summarizer service (daemon-resident L2/L3 worker) -------------
constexpr const char* MSG_SUMMARIZER_START    = "Summarizer started (catch-up queued: {} turn(s), {} draft(s), {} session(s) to close)";
constexpr const char* MSG_SUMMARIZER_STOP     = "Summarizer stopped";
constexpr const char* MSG_SUMMARIZER_L2       = "L2 turn summary written for session {} at {}";
constexpr const char* MSG_SUMMARIZER_L3       = "L3 session summary finalized for session {}";
constexpr const char* MSG_SUMMARIZER_DRAFT    = "Stored draft L2 (inference unreachable) for session {} at {}";
constexpr const char* MSG_SUMMARIZER_REDRAFT  = "Rewrote draft L2 summary_id {} with real summary";
constexpr const char* WARN_SUMMARIZER_L2      = "Summarizer L2 failed for turn_id {}: {}";
constexpr const char* WARN_SUMMARIZER_L3      = "Summarizer L3 failed for session {}: {}";
constexpr const char* WARN_SUMMARIZER_DRAFT   = "Summarizer draft retry failed for summary_id {}: {}";
constexpr const char* ERR_NO_SYSTEM_PROMPT    = "Error: no system prompt files found. ";
constexpr const char* MSG_NO_SYSTEM_PROMPT_HINT = "Create ~/.ragger/SYSTEM.md or add SOUL.md / USER.md / MEMORY.md.";
constexpr const char* MSG_NO_PERSONA_FILES    = "No persona files found in {}. Chat will use an empty system prompt.";
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

// --- Errors: config ---
constexpr const char* ERR_CONFIG_OPEN         = "Cannot open config file: {}";
constexpr const char* ERR_CONFIG_NOT_INIT     = "Config not initialized — call init_config() first";
constexpr const char* ERR_CONFIG_SOCKET_OR_BIND = "Config error: either [server] socket or bind must be configured";
constexpr const char* ERR_CONFIG_SYSTEM_NOT_FOUND = "Failed to find system config";
constexpr const char* ERR_CONFIG_SYSTEM_LOAD  = "Failed to load system config: {}";
constexpr const char* ERR_CONFIG_SYSTEM_PARSE = "Failed to parse system config: {}";
constexpr const char* ERR_CONFIG_SYSTEM_UNKNOWN = "Unknown error loading system config: {}";

// --- Errors: database ---
constexpr const char* ERR_SQLITE_OPEN         = "SQLite open failed: {}";
constexpr const char* ERR_SQL                 = "SQL error: {}";
constexpr const char* ERR_STORE_FAILED        = "Failed to store: {}";

// --- Errors: embedder ---
constexpr const char* ERR_MODEL_NOT_FOUND     = "Model file not found: {}";
constexpr const char* ERR_TOKENIZER_NOT_FOUND = "Tokenizer file not found: {}";
constexpr const char* ERR_EMPTY_TOKENIZATION  = "Empty tokenization result";
constexpr const char* ERR_OUTPUT_SHAPE        = "Unexpected output shape from model";
constexpr const char* ERR_EMBEDDING_MISMATCH  = "Embedding model mismatch: database was built with '{}' but config specifies '{}'. Reorganise your models directory and run 'ragger rebuild' to re-embed.";
constexpr const char* ERR_VECTOR_TYPE_MISMATCH = "Vector dtype mismatch: database was built with '{}' embeddings but config specifies '{}'. Run 'ragger rebuild-embeddings' to re-encode at the new precision.";
constexpr const char* ERR_DIMENSIONS_MISMATCH = "Vector length mismatch: database was built with {}-dim embeddings but config specifies {}. Run 'ragger rebuild-embeddings' to re-encode at the new dimensions.";

// --- Errors: tokenizer wrapper ---
constexpr const char* ERR_TOKENIZER_OPEN      = "Failed to open tokenizer.json: {}";
constexpr const char* ERR_TOKENIZER_EMPTY     = "Empty tokenizer.json file: {}";
constexpr const char* ERR_TOKENIZER_CREATE    = "Failed to create tokenizer from JSON";
constexpr const char* ERR_TOKENIZER_NOT_INIT  = "Tokenizer not initialized";

// --- Help: main usage ---
constexpr const char* HELP_VERSION_HEADER     = "ragger {}";
constexpr const char* HELP_SCREEN             = R"(

Usage: ragger <command> [options] [args]
Commands:
  onboard            Guided first-run setup — capture/build flags, recipe,
                     inference endpoint + model, daemon start. Re-run any
                     time to change one section. Start here.
  start              Start the background daemon (user LaunchAgent / systemd --user)
  stop               Stop the background daemon
  restart            Restart the background daemon
  status             Show daemon status
  serve              Run the server in the foreground (what the daemon invokes)
  search <query>     Search memories by meaning
  store <text>       Store a new memory
  count              Show number of stored memories
  import <file...>   Import files (paragraph-aware chunking)
  export <all|table> Dump database as SQL (like mysqldump)
                       -e, --embeddings  include embedding blobs
                       --output <file>   write to file instead of stdout
  chat               Interactive chat with memory context
  mcp                Start MCP server (JSON-RPC over stdin/stdout)
  recipe [name]      List/inspect build_context recipes (interactive picker)
  useradd <name>     Create a user and issue a bearer token (printed once)
  usermod <name>     Rotate an existing user's bearer token (printed once)
  userdel <name>     Remove a user and revoke their token
  passwd <name>      Set (or clear) a user's web-UI login password
  add-self           Bootstrap ~/.ragger/token for the current user
  housekeeping       Trigger housekeeping on running daemon
  reload             Reload config on running daemon (SIGHUP)
                     options: --user <name>, --dry-run
  rebuild-embeddings Rebuild embeddings for all memories
  show-embedding-model  Show current embedding model info
  embed               Embed text from stdin, write JSON array to stdout
  help               Show this help
  version            Show version
Options:
)";

constexpr const char* VERSION_FORMAT          = "ragger {}\ncommit {}\nbuilt  {}";

// --- Command execution ---
constexpr const char* ERR_DAEMON_NOT_FOUND    = "Error: no running ragger daemon found";
constexpr const char* ERR_SIGNAL_FAILED       = "Error: failed to signal process: {}";
constexpr const char* MSG_CONFIG_RELOAD_OK    = "✓ Config reload triggered (pid {})";

// --- Config reload warnings ---
constexpr const char* WARN_CONFIG_RESTART     = " [WARN] Config reload: '{}' changed but requires restart";

// --- Server logging (user-facing strings in logger calls) ---
constexpr const char* MSG_WARMUP_EMBEDDING_CACHE = "Warmup: embedding cache loaded ({})";
constexpr const char* MSG_WARMUP_ERROR          = "Warmup: {}";
constexpr const char* MSG_PRELOADED_MODEL       = "Preloaded model: {}";
constexpr const char* MSG_MODEL_PRELOAD_SKIPPED = "Model preload skipped: {}";
constexpr const char* MSG_SUMMARIZED_SESSION    = "Summarized session {} ({} turns)";
constexpr const char* ERR_SESSION_SUMMARY_FAIL  = "Session summarization failed: {}";
constexpr const char* MSG_CLEANED               = "Cleaned {} expired conversations from main DB";
constexpr const char* ERR_CLEANUP_DB            = "Cleanup failed for main DB: {}";
constexpr const char* MSG_HOUSEKEEPING_EXPIRED  = "Housekeeping: {} sessions expired, {} conversations cleaned";
constexpr const char* MSG_HOUSEKEEPING_OWNER    = "Housekeeping owner for user '{}'";
constexpr const char* MSG_HOUSEKEEPING_DISABLED = "Housekeeping: disabled (interval = 0)";
constexpr const char* MSG_HOUSEKEEPING_SIGNAL   = "Housekeeping triggered by signal";
constexpr const char* MSG_CONFIG_RELOADED_N     = "Config reloaded: {} value(s) changed";
constexpr const char* MSG_INFERENCE_CLIENT_OK   = "Inference client reloaded";
constexpr const char* ERR_INFERENCE_CLIENT_FAIL = "Inference client reload failed: {}";
constexpr const char* MSG_CONFIG_RELOADED_NONE  = "Config reloaded: no changes";
constexpr const char* MSG_INFERENCE_ENABLED     = "Inference: enabled ({} endpoint(s))";
constexpr const char* MSG_CREATED_USER          = "Created user: {} (id={})";
constexpr const char* MSG_SINGLE_USER_MODE      = "Single-user mode initialized";
constexpr const char* ERR_MEMORY_SEARCH_FAIL    = "Memory search failed for /chat: {}";
constexpr const char* ERR_NO_SYSTEM_PROMPT_FILES = "No system prompt files found (SYSTEM.md, SOUL.md, USER.md, etc.) — ";
constexpr const char* ERR_LOGIN                 = "Login error: {}";
constexpr const char* MSG_BIND_UNIX_SOCKET      = "Binding AF_UNIX socket: {}";
constexpr const char* ERR_LISTEN_UNIX           = "listen() on unix socket returned false: {} (errno={})";
constexpr const char* MSG_BIND_TCP              = "Binding TCP {}:{}";
constexpr const char* ERR_LISTEN_TCP            = "listen() on TCP returned false (errno={})";

// --- Server: debug HTTP logging ---
constexpr const char* DBG_HTTP                  = "{} {} {}";
constexpr const char* DBG_HTTP_DETAIL           = "{} {} {} ({})";
constexpr const char* DBG_QUERY_LOG             = "query=\"{}\" results={} time={}ms";

// --- Server: HTTP response bodies ---
constexpr const char* HTTP_UNAUTHORIZED         = "Unauthorized";
constexpr const char* HTTP_MISSING_TEXT         = "Missing 'text' field";
constexpr const char* HTTP_MISSING_QUERY        = "Missing 'query' field";
constexpr const char* HTTP_MISSING_IDS          = "Missing or invalid 'ids' field";
constexpr const char* HTTP_MISSING_METADATA     = "Missing or invalid 'metadata' field";
constexpr const char* HTTP_MISSING_MODEL        = "Missing 'model' field";
constexpr const char* HTTP_JSON_ERROR           = "JSON error: {}";
constexpr const char* HTTP_MEMORY_NOT_FOUND     = "Memory not found";
constexpr const char* HTTP_INFERENCE_NOT_CONFIGURED = "inference not configured";
constexpr const char* HTTP_MESSAGE_REQUIRED     = "message required";
constexpr const char* HTTP_SYSTEM_USER_NOT_FOUND = "system user not found";
constexpr const char* HTTP_NO_TOKEN_FILE        = "no token file";
constexpr const char* HTTP_LOGIN_CREDENTIALS_REQ = "username and password required";
constexpr const char* HTTP_LOGIN_INVALID        = "invalid credentials";
constexpr const char* HTTP_LOGIN_NO_PASSWORD    = "no password set — use 'ragger passwd' first";
constexpr const char* HTTP_LOGIN_FAILED         = "login failed";

// --- Server: startup/runtime ---
constexpr const char* MSG_HEALTH_CHECK_TCP      = "  Health check: curl http://{}/health";
constexpr const char* MSG_HEALTH_CHECK_UNIX     = "  Health check: curl --unix-socket {} http://localhost/health";
constexpr const char* MSG_TLS_NOT_SUPPORTED     = "TLS config present — native TLS not yet supported with httplib. Use reverse proxy.";
constexpr const char* MSG_PID_FILE              = "PID file: {}";
constexpr const char* MSG_UNIX_CHMOD_TIMEOUT    = "Unix socket chmod 0600 timed out; check {}";

// --- Server: error logging ---
constexpr const char* ERR_ROUTE_FAILED          = "{} {} failed: {}";
constexpr const char* ERR_SESSION_SAVE_FAIL     = "Session save failed: {}";
constexpr const char* ERR_TURN_STORAGE_FAIL     = "Turn storage failed: {}";

// --- MCP ---
constexpr const char* ERR_MCP_TEXT_REQUIRED     = "Error: text parameter required";
constexpr const char* ERR_MCP_USER_REQUIRED     = "Error: user parameter required";
constexpr const char* ERR_MCP_QUERY_REQUIRED    = "Error: query parameter required";
constexpr const char* MSG_MCP_NO_RESULTS        = "No results found.";
constexpr const char* MSG_MCP_HOUSEKEEPING      = "MCP housekeeping: cleaned {} expired conversations";
constexpr const char* ERR_MCP_METHOD_NOT_FOUND  = "Method not found: {}";

// --- Client errors ---
constexpr const char* ERR_CLIENT_STORE        = "Store failed: HTTP {}";
constexpr const char* ERR_CLIENT_SEARCH       = "Search failed: HTTP {}";
constexpr const char* ERR_CLIENT_COUNT        = "Count failed: HTTP {}";
constexpr const char* ERR_CLIENT_DELETE_BATCH = "Delete batch failed: HTTP {}";
constexpr const char* ERR_CLIENT_SEARCH_META  = "Search by metadata failed: HTTP {}";
constexpr const char* ERR_CLIENT_REGISTER     = "Register failed: HTTP {}";
constexpr const char* ERR_CLIENT_SOCKET       = "Failed to create socket: {}";
constexpr const char* ERR_CLIENT_ADDRESS      = "Invalid address: {}";
constexpr const char* ERR_CLIENT_CONNECT      = "Connection failed: {}";
constexpr const char* ERR_CLIENT_SEND         = "Failed to send request: {}";
constexpr const char* ERR_CLIENT_READ         = "Failed to read response: {}";
constexpr const char* ERR_CLIENT_NO_STATUS    = "Invalid HTTP response: no status line";
constexpr const char* ERR_CLIENT_BAD_STATUS   = "Invalid HTTP response: malformed status line";

// --- Inference errors ---
constexpr const char* ERR_CURL_INIT           = "Failed to initialize libcurl";
constexpr const char* ERR_HTTP_REQUEST        = "HTTP request failed: {}";
constexpr const char* ERR_INFERENCE_API       = "Inference API error {}: {}";
constexpr const char* ERR_INFERENCE_API_STATUS = "Inference API error {}";
constexpr const char* ERR_PARSE_RESPONSE      = "Failed to parse response: {}";
constexpr const char* ERR_UNKNOWN_ENDPOINT    = "Unknown endpoint: {}";
constexpr const char* ERR_NO_ENDPOINTS        = "No inference endpoints configured";
constexpr const char* ERR_ENGINE_UNREACHABLE  = "Inference engine not reachable at {}";
constexpr const char* ERR_MODEL_LOAD_TIMEOUT  = "Model {} load timed out (POST {})";
constexpr const char* ERR_MODEL_LOAD_FAILED   = "Failed to load model {} (POST {}: {})";

// --- Auth errors ---
constexpr const char* ERR_URANDOM_OPEN        = "Failed to open /dev/urandom";
constexpr const char* ERR_URANDOM_READ        = "Failed to read enough random bytes";
constexpr const char* ERR_TOKEN_WRITE         = "Failed to write token file: {}";
constexpr const char* ERR_RANDOM_BYTES        = "Failed to generate random bytes";
constexpr const char* ERR_RANDOM_SALT         = "Failed to generate random salt";
constexpr const char* ERR_PBKDF2              = "PBKDF2 failed";

// --- API format errors ---
constexpr const char* ERR_UNKNOWN_FORMAT      = "Unknown API format '{}'. No {}.json found in search dirs.";

// --- MCP errors ---
constexpr const char* ERR_MCP_UNKNOWN_TOOL    = "Unknown tool: {}";

// --- Import / file errors ---
constexpr const char* ERR_FILE_NOT_FOUND      = "File not found: {}";
constexpr const char* ERR_USER_NOT_FOUND      = "User not found: {}";

// --- Logger errors ---
constexpr const char* ERR_LOG_OPEN            = "failed to open log file: \"{}\"";

// --- CLI option descriptions ---
constexpr const char* CLI_MIN_CHUNK_SIZE      = "Min chunk size for import";
constexpr const char* CLI_TITLE               = "Document title (import; default: filename)";
constexpr const char* CLI_YEAR                = "Document publish year (import)";
constexpr const char* CLI_TAGS                = "Document tags/subjects, comma-separated (import)";
constexpr const char* CLI_YES                 = "Skip confirmation prompts (for scripting)";
constexpr const char* CLI_DUMP_PAYLOADS       = "Write raw request JSON to this directory (one file per prompt)";

// --- Main.cpp specific strings (useradd/userdel/add-self) ---
constexpr const char* MSG_ADD_USER_PROMPT       = "ragger useradd <username>";
constexpr const char* MSG_ADD_USER_USAGE        = "Usage: ragger add-self [--force]";
constexpr const char* MSG_ADDED_TO_GROUP        = "✓ Added {} to ragger group";
constexpr const char* MSG_EMBEDDING_DIMENSIONS  = "Dimensions: {}";
constexpr const char* MSG_EMBEDDING_MODEL_NAME  = "Model name: {}";
constexpr const char* MSG_EMBEDDING_PATH        = "Path: {}";
constexpr const char* MSG_EMBEDDING_PATH_DEFAULT= "(default)";
constexpr const char* MSG_GROUP_SKIPPED         = "{} was not in ragger group (skipped)";
constexpr const char* MSG_HOUSEKEEPING_TRIGGERED= "Housekeeping triggered";
constexpr const char* MSG_NOT_IN_DB             = "User '{}' not in database";
constexpr const char* MSG_PROCESSED_USERS       = "Processed {} users";
constexpr const char* MSG_REMOVED_FROM_DB       = "Removed user '{}' from database";
constexpr const char* MSG_REMOVED_FROM_GROUP    = "✓ Removed {} from ragger group";
constexpr const char* MSG_REMOVED_TOKEN_FILE    = "Removed token file for {}";
constexpr const char* MSG_TOKEN_CREATED_FOR     = "✓ Created ~/.ragger/token for {}";
constexpr const char* MSG_TOKEN_EXISTS_FOR      = "Token already exists for {}";
constexpr const char* MSG_TOKEN_FILE_USER       = "Token file: ~/.ragger/users/{}/token";
constexpr const char* MSG_TOKEN_SKIPPED         = "Skipped {} (no token)";
constexpr const char* MSG_TOKEN_UPDATE_HINT     = "Use this in your client config.";
constexpr const char* MSG_USER_ERROR            = "Error processing {}: {}";
constexpr const char* MSG_USER_NOT_IN_PASSWD    = "User {} not found in system passwd (skipped token)";
constexpr const char* MSG_USER_REMOVED_FINAL    = "✓ User {} removed (final)";
constexpr const char* MSG_USER_SKIPPED          = "{}: skipped";
constexpr const char* MSG_USER_STATUS           = "{}: {}";
constexpr const char* MSG_WARNING               = "Warning";
constexpr const char* MSG_WARNING_SUDO          = "[SUDO] {} requires elevated privileges";

// --- Main.cpp specific (daemon control) ---
constexpr const char* ERR_DAEMON_PID_NOT_RUNNING= "Daemon not running (pid {} not found)";
constexpr const char* ERR_PERMISSION_DENIED_SIGNAL= "Permission denied: cannot send signal to daemon process";

} // namespace ragger::lang
