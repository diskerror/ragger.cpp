/**
 * Configuration for Ragger Memory (C++ port)
 *
 * Defaults are compiled in from default-settings.txt; the settings table in
 * memories.db overlays them at runtime. No config file is read or written.
 * 
 */
#pragma once

#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <expected>

namespace ragger {

// Configuration error types
enum class ConfigError {
    NotFound,
    ParseError,
    IOError
};

struct Config {
    // --- Server ---
    // Listener selection: socket_enabled controls the AF_UNIX listener
    // (path is always ragger_base_dir()/ragger.sock — not independently
    // configurable). bind_address controls the TCP listener; empty = off.
    // Both may be set simultaneously (dual listener). If both end up off
    // after config load, load_config() falls bind back to "127.0.0.1" so
    // the daemon always has at least one listener.
    bool        socket_enabled  = true;   // AF_UNIX listener at resolved_socket_path()
    bool        tcp_enabled    = true;    // TCP listener on bind_address:port
    std::string bind_address   = "127.0.0.1";  // TCP bind address
    int         port           = 8432;  // only meaningful when bind_address is set
    // Editable target port the dashboard writes. The daemon binds `port`, but
    // on startup (absent a --port CLI override) it rectifies `port` :=
    // `desired_port` so a UI port change is adopted by ANY restart path, not
    // just the dashboard button. 0 = "same as port" (seeded on first load).
    int         desired_port   = 0;
    // Sent as the "Server:" response header (see make_tcp_server() /
    // Impl::Impl() in server.cpp). Cosmetic only — cpp-httplib has no
    // hostname/virtual-host concept, so this does not affect routing,
    // binding, or DNS/mDNS resolution.
    std::string server_name = "ragger";

    // --- Storage ---
    // formats_dir, model_dir, recipes_dir, and log_dir are no longer
    // independently configurable — every Ragger path is now hardcoded
    // relative to ragger_base_dir() (see util/fs.h), so the entire on-disk
    // footprint moves together. The only override is the hidden
    // --ragger-base CLI flag (testing only; no env var equivalent).

    // --- Embedding ---
    std::string embedding_model = "sentence-transformers/all-MiniLM-L6-v2";
    int         embedding_dimensions = 384;
    // On-disk vector storage precision: "f16" (half, default — half the size)
    // or "f32" (full). In-memory math is always f32; this only affects the
    // blob written to the DB. Tracked in the settings table for drift
    // protection — a whole DB must use one type (model+type are paired).
    std::string embedding_vector_type = "f16";

    // --- Embedding engine: "internal" (ONNX, default) or "external" -------
    // When "external", embeddings are computed via an OpenAI-compatible
    // /v1/embeddings endpoint at embedding_external_host:port. The internal
    // ONNX embedder is not loaded; model management is the external engine's
    // concern.
    std::string embedding_engine = "internal";  // "internal" or "external"
    std::string embedding_external_host = "";   // IP or hostname
    int         embedding_external_port = 0;    // 0 = not configured
    std::string embedding_external_api_key = "";
    std::string embedding_external_model = "";  // model name from the endpoint

    // --- Desired embedding identity (staged target for re-embedding) -------
    // Editable via dashboard/CLI. The CURRENT identity above is what the
    // stored vectors actually are (written by the drift guard / rebuild) and
    // is read-only. When desired != current the DB "needs re-embedding"; an
    // explicit update re-encodes everything and promotes current := desired.
    // Empty string means "same as current" (seeded on first load).
    std::string desired_embedding_model       = "";
    int         desired_embedding_dimensions  = 0;    // 0 = same as current
    std::string desired_embedding_vector_type = "";   // "" = same as current
    std::string desired_embedding_engine      = "";   // "" = same as current

    // --- Search ---
    int   default_search_limit = 5;
    float default_min_score    = 0.4f;
    bool  bm25_enabled         = true;
    float bm25_weight          = 4.0f;
    float vector_weight        = 8.0f;
    // Phonetic ("dolphining" sounds-like) blend weight. Default 1 (low) so the
    // signal nudges rather than dominates; 0 disables it. See phon_scores().
    float phon_weight          = 1.0f;
    // Reserved: not yet wired to search behavior.
    bool  inject_data          = false;

    // --- Embed (subprocess) ---
    int embed_timeout_ms  = 10000;
    int embed_retries     = 1;
    int embed_max_workers = 8;   // cap concurrent `ragger embed` subprocesses

    // --- Inference ---
    struct InferenceEndpointConfig {
        std::string name;
        std::string api_url;
        std::string api_key;
        std::string models = "*";
        std::string format;  // API format name (openai, anthropic, etc.)
        int max_context = 0; // 0 = unknown. Context window in tokens.
        int max_tokens = 0;  // 0 = use global default
    };
    std::string inference_model = "";
    std::string inference_default = "";
    // Default to the LM Studio convention. Setting it lets the daemon's
    // summarizer find a local model out of the box; override per user.
    std::string inference_api_url = "http://localhost:1234/v1";
    std::string inference_api_key = "";
    int inference_max_tokens = 4096;
    std::vector<InferenceEndpointConfig> inference_endpoints;

    // --- [summarizer] section — model, endpoint, and prompt for L2/L3 ---
    std::string summarizer_model   = "";
    std::string summarizer_api_url = "";
    std::string summarizer_api_key = "";
    int         summarizer_max_tokens  = 0;  // 0 = inherit from [inference] max_tokens
    // Target summary length as a percentage of the raw turn size (1-100).
    // 0 = use built-in default of 30%.
    int         summarizer_target_pct  = 0;
    // Hard cap as a percentage of the raw turn size (1-100).
    // 0 = use built-in default of 60%. Always >= target_pct after clamping.
    int         summarizer_max_pct     = 0;
    // System-prompt sent to the summarizer model. Empty (or missing in the
    // INI) uses the built-in default. Any non-empty value is passed through
    // as-is — set to a single space " " to suppress the system prompt entirely.
    // Two positional placeholders {} {} are required (target chars, hard cap)
    // when overriding; the default already contains them.
    static constexpr std::string_view kDefaultSummarizerPrompt =
        "Summarize this conversation into a concise memory entry. "
        "Extract key facts, decisions, questions asked, topics discussed. "
        "Write in third person past tense. Target about {} characters; "
        "never exceed {}. If the exchange is trivial (a single command "
        "like /exit, a one-line greeting), respond with a short phrase, "
        "not a full sentence.";
    std::string summarizer_prompt = "";

    // --- Logging ---
    std::string log_file;   //  hardcoded to ~/.ragger/logs/activity.log
    std::string log_level   = "warn";   //  trace, debug, info, warn, error, and critical
    // Built-in log rotation (no external tool/root needed, cross-platform).
    // Checked on every append: once activity.log reaches this size it is
    // renamed to activity.log.<timestamp> and a fresh empty file is started.
    // 0 disables size-based rotation entirely.
    long log_max_size_mb    = 1;
    // Rotated backups (activity.log.<timestamp>) older than this many days
    // are deleted. Swept opportunistically (throttled, not on every append).
    // Never touches the live activity.log itself. 0 disables age cleanup
    // (rotated backups accumulate forever).
    int  log_max_age_days   = 14;

    // --- Paths ---
    bool normalize_home_path   = true;

    // --- TLS ---
    std::string tls_cert;      // path to certificate chain (PEM)
    std::string tls_key;       // path to private key (PEM)

    // --- Import ---
    int  minimum_chunk_size    = 300;

    // --- Turn handling ---
    // capture_turns (write side): when true, the capture_turn entry point
    // (MCP tool + HTTP POST /turn) ingests agent-pushed raw turns into the
    // `turns` table. When false, capture_turn is a no-op.
    bool capture_turns = true;
    // build_context (read side): when true, the build_context entry point
    // (MCP tool + HTTP GET /session/<id>) assembles a session's turns into a
    // context payload for the agent to inject. Only meaningful when
    // capture_turns is also true — there's nothing to build from otherwise.
    // Agent-driven search/store tools are unaffected by either flag.
    bool build_context = false;
    // auto_recall (recall/injection side): when true, agent-framework plugins
    // may prefetch memories and inject a <memory-context> block into prompts.
    // Purely a hint served to plugins via GET /config — the daemon itself does
    // no injection. Turn capture (capture_turns) is independent and unaffected.
    bool auto_recall = true;
    // Recipe name applied when the caller doesn't specify one. Recipes are
    // loaded from ~/.ragger/recipes (JSON files); built-ins cover the case
    // where the directory is missing or empty.
    std::string default_recipe = "natural_fading";

    // --- Housekeeping / retention ---
    float cleanup_max_age_hours  = 0.0f;  // 0 = keep forever (default)
    int   housekeeping_interval  = 60;      // seconds; 0 = disabled, <10 clamped to 10
    int   summary_pause_minutes  = 20;      // DEPRECATED alias for episode_idle_minutes (kept one release)
    // Idle gap (minutes) after a session's last turn that closes the current
    // episode (and, Phase 2, triggers the session/project rollups). Any
    // positive integer; no range clamp (ridiculous values are the user's
    // business — consistent with Ragger's retention philosophy). Default 15,
    // modelling the walk-away-and-return rhythm. Supersedes summary_pause_minutes,
    // which is accepted as a deprecated alias when this key is unset.
    int   episode_idle_minutes   = 15;

    // ---- Episode boundary: sliding average-similarity threshold ----
    // Mid-session episode boundaries are detected by comparing the average
    // of turn-embedding similarity and turn-summary-embedding similarity
    // against a threshold that rises with idle time between turns.
    //   threshold = min(cap, base + step * int(gap_minutes / step_minutes))
    //   boundary fires when  avg(turn_sim, summary_sim) <= threshold
    // Session starts are always implicit episode boundaries.
    float episode_threshold_base     = 0.2f;   // starting threshold
    float episode_threshold_cap      = 0.5f;   // maximum threshold
    float episode_threshold_step     = 0.01f;  // threshold increment per step
    float episode_step_minutes       = 1.0f;   // minutes of idle per increment
    // Minimum time-gap (days) between two consecutive turns that triggers a
    // project-boundary close, per the session/project boundary-detection
    // design (edge-detected on the gap, not an idle-scan like episodes).
    // Any positive integer; default 7.
    int   project_gap_days       = 7;
    // Per-housekeeping-tick cap on how many unsummarized L2 turns (and,
    // separately, how many draft-tagged retry rows) get enqueued in one
    // pass. Was a hardcoded constexpr of 200; made configurable so a manual
    // resummarize (nulling model_id on a batch of `summaries` rows via
    // direct SQL) can be throttled to a small, deliberate slice per tick
    // instead of redoing hundreds of inference calls at once. Any positive
    // integer; default 10.
    int   catch_up_batch_size    = 10;
    // Maximum consecutive inference failures on a single turn before it is
    // stamped with model "bad" (model_id 1) and removed from the unsummarized
    // queue. Prevents one poison turn from blocking the entire pipeline.
    // Default 3; set 0 to disable (retry forever — old behaviour).
    int   max_turn_failures      = 3;


    /// Resolved paths — all hardcoded relative to ragger_base_dir() (see
    /// util/fs.h), so every on-disk location Ragger ever touches is defined
    /// in exactly one place. Other modules call these instead of
    /// hand-building "~/.ragger/..." strings.
    std::string resolved_db_path() const;
    std::string resolved_model_dir() const;
    std::string resolved_recipes_dir() const;
    std::string resolved_formats_dir() const;
    std::string resolved_token_path() const;
    std::string resolved_stats_db_path() const;
    std::string resolved_agent_instructions_path() const;
    std::string resolved_log_file_path() const;
    std::string resolved_socket_path() const;
};

/// Expand a leading ~ to the real $HOME in a path string. This is for
/// user-supplied paths only (the CLI --db argument, socket_path/bind_address
/// values from the settings table, etc.) — it is NOT how Ragger's own hardcoded
/// paths are resolved; those go through Config::resolved_XX() /
/// ragger_base_dir() instead (see util/fs.h).
std::string expand_path(const std::string& path);

/// Parse config from an INI file. Ragger does not read a config file during
/// normal startup — defaults are compiled in and the settings table is the
/// only store. This remains for the one-time settings.ini -> DB migration.
std::expected<Config, ConfigError> load_config(const std::string& path);

/// Global config instance. Must call init_config() before use.
const Config& config();
/// Mutable access for CLI overrides applied at startup.
Config& mutable_config();

/// Initialize global config. Call once at startup.
void init_config();

/// Reload config from INI file(s). Updates hot-reloadable values in-place.
/// Returns number of values changed. Logs restart-required changes without applying.
int reload_config();

/// Absolute path to this `ragger` executable, used to spawn `ragger embed`
/// subprocesses. Set once at startup from argv[0]; falls back to "ragger"
/// (resolved via PATH) if never set or unresolvable.
void set_executable_path(const std::string& argv0);
const std::string& executable_path();

/// Full argv captured at startup, used to re-exec the daemon in place for an
/// in-process restart (e.g. after a port change from the dashboard). Set once
/// from main(); empty until then.
void set_full_argv(int argc, char** argv);
const std::vector<std::string>& full_argv();

} // namespace ragger
