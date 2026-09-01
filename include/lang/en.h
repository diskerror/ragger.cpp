/**
 * English language strings for Ragger
 *
 * To add a new language: copy this file to xx.h, translate values,
 * and include that file in lang.h instead.
 *
 * SCOPE — this file is NOT the whole UI.
 *
 * It covers the CLI and the daemon: help screens, status and progress
 * messages, warnings, and errors. It does NOT cover the browser dashboard.
 *
 * `web/dashboard.html` carries its own user-visible English inline — field
 * labels, section headings, button captions, banner text ("Up to date.",
 * "Needs re-embedding — desired settings differ from current.", "Re-embedding
 * in progress…"), and status strings built in JavaScript. Those are static by
 * design; there is no lookup mechanism on the browser side. Translating this
 * file alone therefore leaves the dashboard in English, which is the more
 * visible half for a non-technical user.
 *
 * So a translation is two files: this one, and the literals in
 * `web/dashboard.html`. Anyone ADDING user-visible text should put it here if
 * the CLI or daemon prints it, and inline in the dashboard if the browser
 * renders it — and keep this note accurate if that ever changes.
 */
#pragma once

#include "config_schema.h"
#include <array>

namespace ragger::lang {

// --- CLI options ---
constexpr const char* CLI_DESCRIPTION         = "ragger — Ragger Memory";
constexpr const char* CLI_HELP                = "Show help";
constexpr const char* CLI_VERSION             = "Show version";
constexpr const char* CLI_HOST                = "Server bind address (overrides config)";
constexpr const char* CLI_PORT                = "Server port (overrides config)";
constexpr const char* CLI_COMMAND             = "Command";
constexpr const char* CLI_ARGS                = "Command arguments";
constexpr const char* CLI_USAGE_SEARCH        = "Usage: ragger search <query>";
constexpr const char* CLI_USAGE_STORE         = "Usage: ragger store <text>";
constexpr const char* CLI_USAGE_DECISION      =
    "Usage:\n"
    "  ragger decision add <text> [--status=current|roadmap] [--tags=a,b,c]\n"
    "      Store a curated L6 decision/lesson. --status defaults to\n"
    "      \"current\" (the only status the recall pipeline surfaces).\n"
    "      Use --status=roadmap for planned/future work you want tracked\n"
    "      but don't want cluttering every session's recall until it's\n"
    "      actually done.\n"
    "  ragger decision list [--status=current|roadmap|superseded|deprecated] [-n N]\n"
    "      List decisions by status, newest first. Defaults to \"current\".\n"
    "  ragger decision set-status <decision_id> <status>\n"
    "      Change a decision's status (e.g. promote a roadmap item to\n"
    "      \"current\" once it's done, or mark one \"superseded\"/\"deprecated\").";
constexpr const char* CLI_USAGE_USERADD       = "Usage: ragger useradd <username>";
constexpr const char* CLI_USAGE_USERMOD       = "Usage: ragger usermod <username>";
constexpr const char* CLI_USAGE_USERDEL       = "Usage: ragger userdel <username>";
constexpr const char* CLI_USAGE_PASSWD        = "Usage: ragger passwd <username>";
constexpr const char* CLI_USAGE_IMPORT        =
    "Usage: ragger import-docs <file> [file...]      # markdown/text → L5 documents\n"
    "       [--title T] [--year YYYY] [--tags a,b,c]\n"
    "Deprecated alias: ragger import <file> [file...] (same behavior, warns)\n"
    "\n"
    "See also: ragger import-conversations --help, ragger import-conversations summaries";
constexpr const char* CLI_USAGE_IMPORT_CONVERSATIONS =
    "Usage:\n"
    "  ragger import-conversations PATH\n"
    "      Import Claude/Telegram/Claude-Code conversation history into turns\n"
    "      (L1/L2), session summaries (L3/L4), and memories (L5/L6 decisions).\n"
    "      Format is auto-detected from PATH's JSON structure.\n"
    "\n"
    "  Options:\n"
    "    --all                     Required when PATH is a directory, or a file\n"
    "                              with sibling export files (memories.json,\n"
    "                              projects/) next to it — treats the whole set\n"
    "                              as one import unit.\n"
    "    --format=code|web|telegram\n"
    "                              Override auto-detection (rarely needed).\n"
    "    --self=\"Name\"             Required for --format=telegram: your display\n"
    "                              name, to split user/assistant turns.\n"
    "    --since=YYYY-MM-DD        Only turns on/after this date.\n"
    "    --until=YYYY-MM-DD        Only turns strictly before this date.\n"
    "    --session=ID              Restrict to one session id.\n"
    "    --fdate                   Standalone memories.json only: use the file's\n"
    "                              mtime as the timestamp (no derivable date\n"
    "                              otherwise). Ignored when --all is used with\n"
    "                              conversations.json present.\n"
    "    --date=[CC]YYMMDD         Standalone memories.json only: explicit\n"
    "                              timestamp override. Same --all exception.\n"
    "\n"
    "  ragger import-conversations summaries <file> [file...]\n"
    "      One hand-authored L4 project summary per file.\n"
    "  ragger import-conversations summaries --jsonl=FILE\n"
    "      {text, tags?, timestamp?} per line.\n"
    "\n"
    "Examples:\n"
    "  ragger import-conversations ~/.claude/projects/-my-project --format=code\n"
    "  ragger import-conversations ~/Downloads/claude-export/conversations.json\n"
    "  ragger import-conversations ~/Downloads/claude-export --all\n"
    "  ragger import-conversations tg_export.json --format=telegram --self=\"Jane Doe\"\n"
    "  ragger import-conversations ~/Downloads/memories.json --fdate";
constexpr const char* CLI_USAGE_EXPORT        = "Usage: ragger export <all|table> [-e|--embeddings] [--output file]";
constexpr const char* CLI_UNKNOWN_COMMAND     = "Unknown command: {}";

// --- Status messages ---
constexpr const char* MSG_LOADED_MEMORIES     = "Loaded {} memories";
constexpr const char* MSG_BACKFILLED_EMBEDDINGS  = "Backfilled embeddings for {} row(s).";

// --- Migration / maintenance ---
// (The schema is declarative — there is no in-place migration. Pre-v2 data is
//  copied into a fresh v2 DB out-of-band.)
constexpr const char* MSG_REBUILD_EMBEDDINGS_PROGRESS = "\rRebuilding embeddings: {}/{}";
// Same counter for non-interactive callers (the daemon): no leading \r, since
// it goes to the activity log one line at a time rather than over itself.
constexpr const char* MSG_REBUILD_EMBEDDINGS_LOG = "Rebuilding embeddings: {}/{}";

// --- Re-embed (staged model change: promote identity, re-encode, verify) ---
constexpr const char* MSG_REEMBED_STARTED     = "re-embed started: model '{}', engine '{}'";
constexpr const char* MSG_REEMBED_FINISHED    =
    "re-embed finished: {} row(s) re-encoded; identity promoted to '{}' {} {}-dim";
constexpr const char* WARN_REEMBED_REPAIRED   =
    "re-embed verification repaired {} row(s) the main pass left stale — please report this";
constexpr const char* ERR_REEMBED_FAILED      = "re-embed failed: {}";
constexpr const char* ERR_REEMBED_FAILED_UNKNOWN = "re-embed failed: unknown exception";
constexpr const char* WARN_REEMBED_STALE_FLAG =
    "Stale re-embed flag found (owner pid {} is gone) — a previous re-embed was "
    "interrupted. Clearing the flag; vectors may be a mix of the old and new "
    "model, so run the re-embed again to make them consistent.";
constexpr const char* MSG_REEMBED_RESUMING    =
    "Resuming an interrupted re-embed — re-encoding the rows it did not reach "
    "(semantic search stays keyword-only until this completes).";
constexpr const char* MSG_REEMBED_RESUME_DONE =
    "Re-embed resume complete: {} row(s) re-encoded.";
constexpr const char* ERR_REEMBED_RESUME_STUCK =
    "Re-embed resume did not converge after {} passes ({} row(s) re-encoded); "
    "leaving the repair marker set for the next tick.";
constexpr const char* WARN_FORMAT_LOAD_FAILED        = "Failed to load format {}: {}";
constexpr const char* MSG_STORED_WITH_ID      = "Stored with id: {}";
constexpr const char* MSG_SERVER_STARTING     = "Starting Ragger server on {}";
constexpr const char* ERR_PORT_IN_USE         = "Error: port {} is already in use";

// --- Import ---
constexpr const char* MSG_IMPORTING_CHUNKS    = "Importing {} chunks from {}...";
constexpr const char* MSG_IMPORT_CHUNK        = "  Chunk {}/{}: {}";
constexpr const char* MSG_IMPORT_DONE         = "✓ Imported {} chunks";

// --- Export ---
constexpr const char* MSG_EXPORT_DONE         = "✓ Exported {} rows";
constexpr const char* MSG_EXPORT_TABLE_NOT_FOUND = "Error: table '{}' not found. Available: {}";
constexpr const char* MSG_EXPORT_WROTE_FILE   = "Wrote {}";
constexpr const char* CLI_EMBEDDINGS          = "Include embedding column in export";

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
constexpr const char* MSG_REBUILD_CONFIRM     = "This will re-embed all {} stored rows (turns, summaries, decisions, documents). It's safe to run with the server up, but it blocks all other requests until it finishes — the server will appear to hang.";
constexpr const char* PROMPT_CONTINUE         = "Continue? [y/N] ";
constexpr const char* MSG_ABORTED             = "Aborted.";
constexpr const char* MSG_DB_BACKED_UP        = "Database backed up to: {}";
constexpr const char* WARN_BACKUP_FAILED      = "Warning: Failed to create backup: {}";
constexpr const char* MSG_EMBEDDINGS_REBUILT  = "✓ Embeddings rebuilt: {} rows";
// --- Import / embed warnings ---
constexpr const char* WARN_EMBED_SUBPROCESS   = "Warning: embed subprocess failed (attempt {}/{}); embedding skipped";
constexpr const char* WARN_IMPORT_EMBED_SKIPPED = "Warning: {}/{} chunks left unembedded (embed subprocess failed or timed out); re-import to retry";

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
constexpr const char* WARN_SUMMARY            = "Warning: summary generation failed: {}";

// --- Generic CLI error wrapper ---
constexpr const char* ERR_INFERENCE           = "Error: {}";

// --- Errors: config ---
constexpr const char* ERR_CONFIG_NOT_INIT     = "Config not initialized — call init_config() first";

// --- Errors: database ---
constexpr const char* ERR_SQLITE_OPEN         = "SQLite open failed: {}";
constexpr const char* ERR_SQL                 = "SQL error: {}";
constexpr const char* ERR_EMBED_UPDATE_FAILED = "embed_tables: UPDATE {} id {} failed: {}";
constexpr const char* ERR_STORE_FAILED        = "Failed to store: {}";

// --- Errors: embedder ---
constexpr const char* ERR_MODEL_NOT_FOUND     = "Model file not found: {}";
constexpr const char* ERR_TOKENIZER_NOT_FOUND = "Tokenizer file not found: {}";
constexpr const char* ERR_EMPTY_TOKENIZATION  = "Empty tokenization result";
constexpr const char* ERR_OUTPUT_SHAPE        = "Unexpected output shape from model";
constexpr const char* ERR_EMBEDDING_MISMATCH  = "Embedding model mismatch: database was built with '{}' but config specifies '{}'. Reorganise your models directory and run 'ragger rebuild' to re-embed.";
constexpr const char* ERR_VECTOR_TYPE_MISMATCH = "Vector dtype mismatch: database was built with '{}' embeddings but config specifies '{}'. Run 'ragger rebuild-embeddings' to re-encode at the new precision.";
// Ragger-managed ONNX models live at ~/.ragger/models/<provider>/<model>/, so
// a valid internal model name is always "provider/model". These report a bad
// value; nothing repairs one by guessing a provider.
constexpr const char* ERR_EMBED_MODEL_NO_PROVIDER =
    "Embedding model '{}' has no provider. Ragger models live in "
    "~/.ragger/models/<provider>/<model>, so the name must be the full "
    "'provider/model' path (for example 'sentence-transformers/{}'). Pick one "
    "from the dashboard's Embedding panel, which only lists models it can "
    "find. Conversations are still being recorded, but nothing will be "
    "embedded and semantic search stays keyword-only until this is fixed.";
constexpr const char* ERR_EMBED_MODEL_UNUSABLE =
    "Embedding model '{}' could not be loaded from '{}': {}. Conversations are "
    "still being recorded, but nothing will be embedded and semantic search "
    "stays keyword-only until this is fixed. Choose a model in the dashboard's "
    "Embedding panel; rows stored meanwhile are embedded automatically once it "
    "works.";
constexpr const char* ERR_EMBED_NO_MODEL_BULK =
    "Refusing to re-encode embeddings: no usable embedding model is loaded. "
    "Fix the embedding model first — running now would erase every stored "
    "vector.";
constexpr const char* MSG_EMBED_MODEL_RECOVERED =
    "Embedding model '{}' loaded — semantic search re-enabled; backfilling "
    "rows stored while it was unavailable.";

// Agent-facing (MCP): the user is far more likely to hear about a problem
// from their assistant than to read a log file, so tool results carry it.
constexpr const char* MCP_WARN_NO_EMBEDDINGS =
    "[ragger] Embeddings are unavailable — the configured embedding model "
    "cannot be loaded. Conversations are still being recorded, but new entries "
    "are stored without embeddings and this search used keyword matching only, "
    "so results may be less relevant. Tell the user to open the Ragger "
    "dashboard and set a valid embedding model in the Embedding panel; stored "
    "entries will be embedded automatically once it is fixed.";

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
  start              Start the background daemon (user LaunchAgent / systemd --user)
  stop               Stop the background daemon
  restart            Restart the background daemon
  status             Show daemon status
  serve              Run the server in the foreground (what the daemon invokes)
  search <query>     Search memories by meaning
  store <text>       Store a new memory
  decision <sub>      Curated L6 decisions/lessons (see `ragger decision`
                     with no args for full option list)
                       add <text> [--status=current|roadmap] [--tags=a,b,c]
                       list [--status=current|roadmap|...] [-n N]
                       set-status <decision_id> <status>
  count              Show number of stored memories
  import <file...>   Deprecated alias for import-docs (warns, same behavior)
  import-docs <file...>
                     Import markdown/text files (paragraph-aware chunking) → L5 documents
                       --title <T>  --year <YYYY>  --tags a,b,c
  import-conversations <PATH> [--all]
                     Import Claude web/Claude Code/Telegram conversation history
                     (format auto-detected) into turns, session summaries, and
                     memories/decisions. See `ragger import-conversations` with
                     no args for full option list.
                     Subcommand: import-conversations summaries <file...>
                       one hand-authored L4 project summary per file
  export <all|table> Dump database as SQL (like mysqldump)
                       -e, --embeddings  include embedding blobs
                       --output <file>   write to file instead of stdout
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
  rebuild-phon       Recompute the phonetic (sounds-like) index column
                     options: --missing  only fill rows with a NULL phon column
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
constexpr const char* MSG_BIND_UNIX_SOCKET      = "Binding AF_UNIX socket: {}";
constexpr const char* ERR_LISTEN_UNIX           = "listen() on unix socket returned false: {} (errno={})";
constexpr const char* MSG_BIND_TCP              = "Binding TCP {}:{}";
constexpr const char* ERR_LISTEN_TCP            = "listen() on TCP returned false (errno={})";

// --- Server: debug HTTP logging ---
constexpr const char* DBG_HTTP                  = "{} {} {}";
constexpr const char* DBG_QUERY_LOG             = "query=\"{}\" results={} time={}ms";

// --- Server: HTTP response bodies ---
constexpr const char* HTTP_UNAUTHORIZED         = "Unauthorized";
constexpr const char* HTTP_MISSING_TEXT         = "Missing 'text' field";
constexpr const char* HTTP_MISSING_QUERY        = "Missing 'query' field";
constexpr const char* HTTP_MISSING_IDS          = "Missing or invalid 'ids' field";
constexpr const char* HTTP_MISSING_METADATA     = "Missing or invalid 'metadata' field";
constexpr const char* HTTP_JSON_ERROR           = "JSON error: {}";
constexpr const char* HTTP_MEMORY_NOT_FOUND     = "Memory not found";
constexpr const char* HTTP_SYSTEM_USER_NOT_FOUND = "system user not found";
constexpr const char* HTTP_NO_TOKEN_FILE        = "no token file";

// --- Server: startup/runtime ---
constexpr const char* MSG_HEALTH_CHECK_TCP      = "  Health check: curl http://{}/health";
constexpr const char* MSG_HEALTH_CHECK_UNIX     = "  Health check: curl --unix-socket {} http://localhost/health";
constexpr const char* MSG_TLS_ENABLED           = "Native TLS enabled (cert: {})";
constexpr const char* WARN_TLS_PARTIAL_CONFIG   = "[server] TLS misconfigured: only '{}' is set — both cert and key are required. Serving plain HTTP.";
constexpr const char* WARN_TLS_INVALID_CERT     = "TLS setup failed: could not load cert '{}' / key '{}'. Falling back to plain HTTP — fix the cert/key to enable TLS.";
constexpr const char* WARN_TLS_SETUP_FAILED     = "TLS setup failed: {}. Falling back to plain HTTP — fix the cert/key to enable TLS.";
constexpr const char* MSG_PID_FILE              = "PID file: {}";
constexpr const char* MSG_UNIX_CHMOD_TIMEOUT    = "Unix socket chmod 0600 timed out; check {}";

// --- Server: error logging ---
constexpr const char* ERR_ROUTE_FAILED          = "{} {} failed: {}";

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
constexpr const char* ERR_CLIENT_CONNECT      = "Connection failed: {}";

// --- Inference errors ---
constexpr const char* ERR_CURL_INIT           = "Failed to initialize libcurl";
constexpr const char* ERR_HTTP_REQUEST        = "HTTP request failed: {}";
constexpr const char* ERR_INFERENCE_API       = "Inference API error {}: {}";
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

// --- CLI option descriptions ---
constexpr const char* CLI_MIN_CHUNK_SIZE      = "Min chunk size for import";
constexpr const char* CLI_TITLE               = "Document title (import; default: filename)";
constexpr const char* CLI_YEAR                = "Document publish year (import)";
constexpr const char* CLI_TAGS                = "Document tags/subjects, comma-separated (import)";
constexpr const char* CLI_YES                 = "Skip confirmation prompts (for scripting)";

// --- Main.cpp specific strings (useradd/userdel/add-self) ---
constexpr const char* MSG_EMBEDDING_DIMENSIONS  = "Dimensions: {}";
constexpr const char* MSG_EMBEDDING_MODEL_NAME  = "Model name: {}";
constexpr const char* MSG_EMBEDDING_PATH        = "Path: {}";
constexpr const char* MSG_EMBEDDING_PATH_DEFAULT= "(default)";
constexpr const char* MSG_HOUSEKEEPING_TRIGGERED= "Housekeeping triggered";

// --- Main.cpp specific (daemon control) ---
constexpr const char* ERR_DAEMON_PID_NOT_RUNNING= "Daemon not running (pid {} not found)";
constexpr const char* ERR_PERMISSION_DENIED_SIGNAL= "Permission denied: cannot send signal to daemon process";

// =====================================================================
// Config schema — single source of truth for CLI + dashboard.
// One entry per user-editable setting. See config_schema.h for field
// meanings. Keys match the DB `settings` rows and the Config struct.
// Sections drive the dashboard's left-hand tabs (server, embedding,
// search, summarizer, logging, housekeeping, import).
// =====================================================================
inline constexpr std::array<ConfigMeta, 56> kConfigSchema = {{
    // ---- server ----
    {"socket_enable", "server", "Socket Enable", CfgType::Boolean, CfgEdit::RestartRequired,
     "true", "",
     "Unix domain socket at ~/.ragger/ragger.sock. If disabled and no TCP bind is set, the daemon falls back to 127.0.0.1 so there is always at least one listener."},
    {"tcp_enable", "server", "TCP Enable", CfgType::Boolean, CfgEdit::RestartRequired,
     "true", "",
     "TCP listener on bind_address:port. If disabled and socket is also disabled, the daemon forces TCP on as a fallback."},
    {"bind", "server", "TCP Bind Address", CfgType::String, CfgEdit::RestartRequired,
     "127.0.0.1", "",
     "TCP bind address. Empty = no TCP listener. Both socket and TCP can run at once. A malformed value is a startup error."},
    {"port", "server", "TCP Port", CfgType::Integer, CfgEdit::Locked,
     "8432", "",
     "The port the running daemon is bound to. Read-only — set 'Desired Port' and restart to change it."},
    {"desired_port", "server", "Desired Port", CfgType::Integer, CfgEdit::RestartRequired,
     "", "",
     "The TCP port to bind on the next restart. When it differs from the running port the Server tab shows 'restart required'. Any restart (dashboard, service manager, or 'ragger serve') adopts it."},
    {"server_name", "server", "Server Name", CfgType::String, CfgEdit::RestartRequired,
     "ragger", "",
     "Name sent in the HTTP 'Server' response header. Cosmetic only — does not affect binding, routing, or DNS/mDNS resolution."},
    {"cert", "server", "TLS Certificate", CfgType::Path, CfgEdit::RestartRequired,
     "", "",
     "TLS certificate chain (PEM). Only relevant when exposing Ragger beyond localhost. Both cert and key must be set to enable HTTPS; leave both unset for plain HTTP."},
    {"key", "server", "TLS Private Key", CfgType::Path, CfgEdit::RestartRequired,
     "", "",
     "TLS private key (PEM). Paired with the TLS certificate."},
    {"capture_turns", "server", "Capture Turns", CfgType::Boolean, CfgEdit::Live,
     "true", "",
     "Accept agent-pushed turns (capture_turn tool / POST /turn) into the turns table. Agent-driven search/store are always available regardless."},
    {"build_context", "server", "Build Context", CfgType::Boolean, CfgEdit::Live,
     "false", "",
     "Enable the build_context entry point (GET /session/<id>) to assemble a session's turns into a context payload. Only meaningful when capture_turns is also true."},
    {"default_recipe", "server", "Default Recipe", CfgType::String, CfgEdit::Live,
     "natural_fading", "",
     "Recipe name applied when the caller doesn't specify one. Recipes are loaded from ~/.ragger/recipes."},

    // ---- embedding (+ embed subprocess) ----
    // Current identity: what the stored vectors ARE. Read-only display; the
    // drift guard / rebuild owns these. Editing happens on the desired_* keys.
    {"embedding_model", "embedding", "Current Model", CfgType::String, CfgEdit::Locked,
     "sentence-transformers/all-MiniLM-L6-v2", "",
     "The embedding model the stored vectors were built with. Read-only — change the desired model below and re-embed to switch."},
    {"embedding_engine", "embedding", "Current Engine", CfgType::Enum, CfgEdit::Locked,
     "internal", "internal,external",
     "Embedding engine the stored vectors were built with. Read-only — change the desired engine below."},
    {"embedding_dimensions", "embedding", "Current Dimensions", CfgType::Integer, CfgEdit::Locked,
     "384", "",
     "Vector dimensionality of the stored vectors. Read-only; model-determined."},
    {"embedding_vector_type", "embedding", "Current Vector Type", CfgType::Enum, CfgEdit::Locked,
     "f16", "f32,f16,bf16,int8",
     "On-disk precision of the stored vectors. Read-only. In-memory math is always f32; this only affects the stored blob. f32=lossless (4B/dim); f16=IEEE half (2B/dim); bf16=bfloat16, f32's range with less mantissa (2B/dim); int8=symmetric per-vector quantization (1B/dim)."},
    // Engine selection: internal (ONNX) or external (remote /v1/embeddings).
    {"desired_embedding_engine", "embedding", "Desired Engine", CfgType::Enum, CfgEdit::RebuildRequired,
     "", "internal,external",
     "Target engine. 'internal' runs ONNX models locally; 'external' calls a remote OpenAI-compatible /v1/embeddings endpoint."},
    // Model selection (internal = local ONNX, external = remote model name).
    {"desired_embedding_model", "embedding", "Desired Model", CfgType::Enum, CfgEdit::RebuildRequired,
     "", "",
     "The embedding model to switch to. Choices come from ~/.ragger/models/. Changing this stages a re-embed; nothing happens until you update."},
    {"embedding_external_model", "embedding", "External Model", CfgType::String, CfgEdit::RebuildRequired,
     "", "",
     "Model name served by the external embedding endpoint."},
    // Type and dimensions.
    {"desired_embedding_vector_type", "embedding", "Desired Vector Type", CfgType::Enum, CfgEdit::RebuildRequired,
     "", "f32,f16,bf16,int8",
     "Target on-disk vector precision. f32 (lossless, 4B/dim), f16 (half, 2B/dim), bf16 (bfloat16, 2B/dim), or int8 (quantized, 1B/dim). Changing this stages a re-embed; run 'Update now' or rebuild-embeddings to re-encode at the new precision."},
    {"desired_embedding_dimensions", "embedding", "Desired Dimensions", CfgType::Integer, CfgEdit::RebuildRequired,
     "", "",
     "Target vector dimensionality. Model-determined; usually leave as the model's native size."},
    // External connection details (shown only when engine=external).
    {"embedding_external_host", "embedding", "External Host", CfgType::String, CfgEdit::RebuildRequired,
     "", "",
     "IP address or hostname of the external embedding server."},
    {"embedding_external_port", "embedding", "External Port", CfgType::Integer, CfgEdit::RebuildRequired,
     "0", "",
     "Port of the external embedding server."},
    {"embedding_external_api_key", "embedding", "External API Key", CfgType::String, CfgEdit::RebuildRequired,
     "", "",
     "API key for the external embedding endpoint, if required."},
    // Embedding tweaks (always visible).
    {"embed_timeout_ms", "embedding", "Embed Timeout (ms)", CfgType::Integer, CfgEdit::Live,
     "10000", "",
     "Per-embed subprocess timeout in milliseconds."},
    {"embed_retries", "embedding", "Embed Retries", CfgType::Integer, CfgEdit::Live,
     "1", "",
     "Retries on a failed or timed-out embed."},
    {"embed_max_workers", "embedding", "Max Workers", CfgType::Integer, CfgEdit::Live,
     "8", "",
     "Cap on concurrent embed subprocesses."},

    // ---- search ----
    {"default_limit", "search", "Default Limit", CfgType::Integer, CfgEdit::Live,
     "5", "",
     "Number of search results returned by default."},
    {"default_min_score", "search", "Default Min Score", CfgType::Float, CfgEdit::Live,
     "0.4", "",
     "Minimum similarity score for results (0.0 - 1.0)."},
    {"bm25_enabled", "search", "BM25 Enabled", CfgType::Boolean, CfgEdit::Live,
     "true", "",
     "Enable BM25 keyword matching alongside vector search."},
    {"bm25_weight", "search", "BM25 Weight", CfgType::Float, CfgEdit::Live,
     "4", "",
     "Weight for keyword matching. Weights are ratios and need not sum to 1.0 (\"3 and 7\" == \"0.3 and 0.7\")."},
    {"vector_weight", "search", "Vector Weight", CfgType::Float, CfgEdit::Live,
     "8", "",
     "Weight for semantic (vector) similarity."},
    {"phon_weight", "search", "Phonetic Weight", CfgType::Float, CfgEdit::Live,
     "1", "",
     "\"Sounds-like\" (dolphining) weight — matches on how a phrase sounds (Double Metaphone) alongside meaning and keywords. 0 disables."},

    // ---- summarizer ----
    {"summarizer_model", "summarizer", "Model", CfgType::String, CfgEdit::Live,
     "qwen3-4b-instruct-2507", "",
     "Model used for L2/L3 summarization."},
    {"summarizer_api_url", "summarizer", "API URL", CfgType::String, CfgEdit::Live,
     "http://localhost:1234/v1", "",
     "Inference endpoint. Default follows the LM Studio convention."},
    {"summarizer_api_key", "summarizer", "API Key", CfgType::String, CfgEdit::Live,
     "", "",
     "API key for the summarizer endpoint, if required."},
    {"summarizer_max_tokens", "summarizer", "Max Tokens", CfgType::Integer, CfgEdit::Live,
     "1024", "",
     "Maximum tokens for a summary generation. 0 inherits the global inference default."},
    {"summarizer_target_pct", "summarizer", "Target %", CfgType::Integer, CfgEdit::Live,
     "40", "",
     "Target summary length as % of raw turn; 0 = 30%."},
    {"summarizer_max_pct", "summarizer", "Max %", CfgType::Integer, CfgEdit::Live,
     "60", "",
     "Hard cap as % of raw turn; 0 = 60%. Always >= target %."},
    {"summarizer_prompt", "summarizer", "Prompt", CfgType::Text, CfgEdit::Live,
     "", "",
     "System prompt sent to the summarizer. Empty uses the built-in default. A single space suppresses the system prompt entirely. Two {} placeholders are required (target chars, hard cap)."},

    // ---- logging ----
    {"log_level", "logging", "Log Level", CfgType::Enum, CfgEdit::Live,
     "warn", "trace,debug,info,warn,error,critical",
     "Verbosity of the single activity log. Everything (query/HTTP/MCP/general) goes to one log."},
    {"log_max_size_mb", "logging", "Log Max Size (MB)", CfgType::Integer, CfgEdit::Live,
     "1", "",
     "Rotate activity.log once it reaches this size (MB). 0 disables size-based rotation. Built in — no external tool or root needed."},
    {"log_max_age_days", "logging", "Log Max Age (days)", CfgType::Integer, CfgEdit::Live,
     "14", "",
     "Delete rotated backups older than this many days. Never touches the live log. 0 disables age-based cleanup."},

    // ---- housekeeping ----
    {"cleanup_max_age_hours", "housekeeping", "Cleanup Max Age (hrs)", CfgType::Float, CfgEdit::Live,
     "0", "",
     "Raw-turn lifetime (0 = keep forever). Set to hours to auto-purge."},
    {"housekeeping_interval", "housekeeping", "Housekeeping Interval (sec)", CfgType::Integer, CfgEdit::Live,
     "60", "",
     "Seconds between housekeeping passes (0 disables; values <10 clamp to 10)."},
    {"episode_idle_minutes", "housekeeping", "Episode Idle Minutes", CfgType::Integer, CfgEdit::Live,
     "15", "",
     "Idle gap (minutes) that closes an episode of work within a session. Also drives session/project rollups."},
    {"episode_threshold_base", "housekeeping", "Episode Threshold Base", CfgType::Float, CfgEdit::Live,
     "0.2", "",
     "Starting similarity threshold for mid-session episode boundaries. boundary fires when avg(turn_sim, summary_sim) <= threshold; threshold rises with idle time."},
    {"episode_threshold_cap", "housekeeping", "Episode Threshold Cap", CfgType::Float, CfgEdit::Live,
     "0.5", "",
     "Maximum threshold after time scaling."},
    {"episode_threshold_step", "housekeeping", "Episode Threshold Step", CfgType::Float, CfgEdit::Live,
     "0.01", "",
     "Threshold increment per step of idle time."},
    {"episode_step_minutes", "housekeeping", "Episode Step Minutes", CfgType::Float, CfgEdit::Live,
     "1.0", "",
     "Minutes of idle per threshold increment."},
    {"catch_up_batch_size", "housekeeping", "Catch-up Batch Size", CfgType::Integer, CfgEdit::Live,
     "8", "",
     "Per-tick cap on unsummarized turns / draft retries enqueued at once. Keeps a manual resummarize to a small deliberate slice per tick."},
    {"project_gap_days", "housekeeping", "Project Gap Days", CfgType::Integer, CfgEdit::Live,
     "7", "",
     "Minimum time-gap (days) between consecutive turns that triggers a project-boundary close."},
    {"max_turn_failures", "housekeeping", "Max Turn Failures", CfgType::Integer, CfgEdit::Live,
     "3", "",
     "Consecutive inference failures on a single turn before marking it 'bad' and removing it from the unsummarized queue. 0 = retry forever (old behaviour)."},

    // ---- import (+ paths) ----
    {"minimum_chunk_size", "import", "Minimum Chunk Size", CfgType::Integer, CfgEdit::Live,
     "300", "",
     "Minimum size of an imported document chunk."},
    {"normalize_home", "import", "Normalize Home", CfgType::Boolean, CfgEdit::Live,
     "true", "",
     "Replace /Users/<you> or /home/<user> with ~ in stored paths."},
}};

inline std::span<const ConfigMeta> config_schema() {
    return {kConfigSchema.data(), kConfigSchema.size()};
}

inline const ConfigMeta* config_meta(std::string_view key) {
    for (const auto& m : kConfigSchema)
        if (m.key == key) return &m;
    return nullptr;
}

} // namespace ragger::lang
