/**
 * HTTP server for Ragger Memory implementation
 */

#include "server.h"
#include "memory.h"
#include "user_store.h"
#include "sqlite_backend.h"
#include "lang.h"
#include "Logger.h"
#include "auth.h"
#include "config.h"
#include "config_access.h"
#include "embedder.h"
#include "util/fs.h"
#include "inference.h"

#include <curl/curl.h>
#include "summarizer.h"
#include "summarizer_service.h"
#include "nlohmann_json.hpp"

#include "httplib.h"

// Dashboard HTML/JS, embedded at build time (web/dashboard.html -> generated
// dashboard_html.inc). Provides `static const char* DASHBOARD_HTML`.
#include "dashboard_html.inc"

#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <set>
#include <mutex>
#include <iostream>
#include <optional>
#include <pwd.h>
#include <format>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <atomic>
#include <signal.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

namespace ragger {
namespace fs = std::filesystem;

using json = nlohmann::json;

// Global signal flags (async-signal-safe)
static std::atomic<bool> g_housekeeping_requested{false};
static std::atomic<bool> g_config_reload_requested{false};
static std::atomic<bool> g_shutdown_requested{false};
// Set by POST /restart: like shutdown, but main() re-execs the daemon in
// place afterward (rebinding listeners — the only way a port change takes).
static std::atomic<bool> g_restart_requested{false};

static void sigusr1_handler(int) {
    g_housekeeping_requested.store(true, std::memory_order_relaxed);
}

static void sighup_handler(int) {
    g_config_reload_requested.store(true, std::memory_order_relaxed);
}

static void sigterm_handler(int) {
    g_shutdown_requested.store(true, std::memory_order_relaxed);
}

// Serialise a vector of SearchResults to a JSON array.
// score is always included (0.0 for metadata-only searches).
static json search_results_to_json(const std::vector<SearchResult>& results) {
    json arr = json::array();
    for (const auto& r : results)
        arr.push_back({{"id", r.id}, {"text", r.text}, {"score", r.score},
                       {"metadata", r.metadata}, {"timestamp", r.timestamp}});
    return arr;
}

struct Server::Impl {
    // Two listener instances so we can bind AF_UNIX and AF_INET at the same
    // time. cpp-httplib::Server::listen() blocks the calling thread and binds
    // exactly one socket, so each listener gets its own Server and its own
    // thread. Routes are registered on both via setup_routes(httplib::Server&).
    //
    // tcp_svr is a pointer because it's either a plain httplib::Server or an
    // httplib::SSLServer, decided once at construction time from [server]
    // cert/key. httplib::SSLServer derives from Server (virtual dtor), so a
    // base-class pointer works for both; setup_routes()/listen()/stop() are
    // all virtual or take the base type.
    httplib::Server                  unix_svr;
    std::unique_ptr<httplib::Server> tcp_svr;
    bool                              tls_enabled_ = false;
    RaggerMemory&   memory;
    UserStore       users_;
    std::string     host;
    int             port;
    std::string     server_token_;
    std::optional<UserInfo> default_user_;
    std::unique_ptr<InferenceClient>    inference_;
    std::unique_ptr<SummarizerService>  summarizer_;

    // Per-user memory cache (username → RaggerMemory)
    std::unordered_map<std::string, std::unique_ptr<RaggerMemory>> user_memories_;
    std::mutex user_memories_mutex_;

    // SSE activity-log tailing: byte offset already streamed to /events.
    std::uintmax_t activity_log_offset_ = 0;

    // Housekeeping timer
    std::atomic<bool> timer_running_{false};
    std::thread timer_thread_;
    std::string pid_file_;

    Impl(RaggerMemory& mem, const std::string& h, int p)
        : memory(mem), users_(config().resolved_db_path()), host(h), port(p)
    {
        bootstrap_auth();
        init_inference();
        tcp_svr = make_tcp_server();
        setup_routes(unix_svr);
        setup_routes(*tcp_svr);
        warmup();
    }

    /// Build the TCP listener — plain httplib::Server, or an SSLServer when
    /// [server] cert + key are both configured and load successfully. Any
    /// TLS problem (partial config, unloadable cert/key, construction
    /// failure) is logged as a warning (stderr + activity.log) and the
    /// daemon falls back to plain HTTP rather than refusing to start — a
    /// working insecure daemon beats a daemon that won't come up at all.
    std::unique_ptr<httplib::Server> make_tcp_server() {
        const auto& cfg = config();
        const bool have_cert = !cfg.tls_cert.empty();
        const bool have_key  = !cfg.tls_key.empty();

        if (have_cert != have_key) {
            // Only one of cert/key set — misconfiguration. Warn and fall
            // back to plain HTTP; enforcing TLS on a half-configured pair
            // isn't possible.
            std::string msg = std::format(lang::WARN_TLS_PARTIAL_CONFIG,
                have_cert ? "cert" : "key");
            Diskerror::Logger::warn(msg);
            std::cerr << msg << "\n";
            return std::make_unique<httplib::Server>();
        }

        if (!have_cert && !have_key) {
            return std::make_unique<httplib::Server>();
        }

        try {
            auto svr = std::make_unique<httplib::SSLServer>(
                cfg.tls_cert.c_str(), cfg.tls_key.c_str());
            if (!svr->is_valid()) {
                std::string msg = std::format(lang::WARN_TLS_INVALID_CERT,
                    cfg.tls_cert, cfg.tls_key);
                Diskerror::Logger::warn(msg);
                std::cerr << msg << "\n";
                return std::make_unique<httplib::Server>();
            }
            tls_enabled_ = true;
            Diskerror::Logger::info(std::format(lang::MSG_TLS_ENABLED, cfg.tls_cert));
            return svr;
        } catch (const std::exception& e) {
            std::string msg = std::format(lang::WARN_TLS_SETUP_FAILED, e.what());
            Diskerror::Logger::warn(msg);
            std::cerr << msg << "\n";
            return std::make_unique<httplib::Server>();
        }
    }

    void warmup() {
        // Pre-load embedding cache so first request isn't slow
        try {
            auto result = memory.search("warmup", 1, 0.0f);
            Diskerror::Logger::info(std::format(lang::MSG_WARMUP_EMBEDDING_CACHE, memory.count()));
        } catch (const std::exception& e) {
            Diskerror::Logger::info(std::format(lang::MSG_WARMUP_ERROR, e.what()));
        }
        // Preload default model on local inference engines
        if (inference_) {
            preload_local_model(inference_->model);
        }
    }

    /// Preload a model if its endpoint is local (non-commercial).
    /// Runs in background thread to avoid blocking startup.
    void preload_local_model(const std::string& model_name) {
        if (model_name.empty() || !inference_) return;
        try {
            auto& ep = inference_->resolve_endpoint(model_name);
            if (!ep.is_local()) return;
        } catch (...) {
            return;
        }
        std::thread([this, model_name]() {
            auto err = inference_->ensure_model_loaded(model_name);
            if (err.empty()) {
                Diskerror::Logger::info(std::format(lang::MSG_PRELOADED_MODEL, model_name));
            } else {
                // WARN, not info — preload is best-effort but a misconfigured or
                // unreachable management endpoint is the kind of thing the user
                // wants to see (issue #47). The err string already carries the
                // tried URL via ERR_ENGINE_UNREACHABLE / ERR_MODEL_LOAD_*.
                Diskerror::Logger::warn(std::format(lang::MSG_MODEL_PRELOAD_SKIPPED, err));
            }
        }).detach();
    }

    /// Run one housekeeping pass: purge old turns, catch up missing
    /// summaries, and backfill any NULL embeddings.
    void run_housekeeping() {
        const auto& cfg = config();
        float max_age_hours = cfg.cleanup_max_age_hours;

        // 1. Purge old conversation entries — use the main memory backend,
        //    not a separate SqliteBackend. Opening a second connection to the
        //    same file while the main backend holds it can produce SQLite
        //    BUSY/LOCKED contention that silently prevents the backfill UPDATE
        //    in step 3 from committing.
        if (max_age_hours > 0) {
            try {
                int deleted = memory.cleanup_old_conversations(max_age_hours);
                if (deleted > 0) {
                    Diskerror::Logger::info(std::format(
                        lang::MSG_HOUSEKEEPING_EXPIRED, 0, deleted));
                }
            } catch (const std::exception& e) {
                Diskerror::Logger::critical(std::format(lang::ERR_CLEANUP_DB, e.what()));
            }
        }

        // 2. Catch up unsummarized turns and draft summaries.
        if (summarizer_) {
            summarizer_->enqueue_catch_up();
        }

        // 3. Backfill NULL embeddings (deferred stores, partial rebuilds).
        //    If embeddings are degraded, try to recover first — a rebuild may
        //    have fixed the drift since the last housekeeping tick.
        if (memory.embeddings_degraded()) {
            memory.try_recover_embeddings();
        }
        if (!memory.embeddings_degraded()) {
            try {
                int filled = memory.backfill_embeddings();
                if (filled > 0) {
                    Diskerror::Logger::info(std::format(
                        lang::MSG_BACKFILLED_EMBEDDINGS, filled));
                }
            } catch (const std::exception& e) {
                Diskerror::Logger::warn(std::format(lang::ERR_CLEANUP_DB, e.what()));
            }
        }

        // 4. Backfill NULL phon (dolphining sounds-like) — self-heals rows that
        //    predate the phon column after the one-time ADD COLUMN migration.
        //    Pure string work (no embedder); NULL-only so it's a no-op once
        //    every row is populated.
        try {
            int phoned = memory.rebuild_phon(/*only_missing=*/true, /*progress=*/false);
            if (phoned > 0) {
                Diskerror::Logger::info(std::format(
                    "Backfilled phonetic codes for {} row(s)", phoned));
            }
        } catch (const std::exception& e) {
            Diskerror::Logger::warn(std::format(lang::ERR_CLEANUP_DB, e.what()));
        }
    }

    // Per-user housekeeping locks: username → fd
    std::unordered_map<std::string, int> housekeeping_locks_;
    std::mutex housekeeping_locks_mutex_;

    /// Try to acquire housekeeping lock for a specific user.
    /// Lock file: /tmp/ragger/housekeeping-{username}.lock
    /// Returns true if this instance now owns housekeeping for that user.
    bool acquire_user_housekeeping_lock(const std::string& username) {
        std::lock_guard<std::mutex> lock(housekeeping_locks_mutex_);
        if (housekeeping_locks_.count(username)) return true;  // already own it

        fs::create_directories("/tmp/ragger");
        std::string lock_path = "/tmp/ragger/housekeeping-" + username + ".lock";
        int fd = open(lock_path.c_str(), O_WRONLY | O_CREAT, 0644);
        if (fd < 0) return false;

        if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
            ::close(fd);
            return false;  // another instance owns it
        }

        // Write our PID
        ftruncate(fd, 0);
        auto pid_str = std::to_string(getpid());
        (void)write(fd, pid_str.c_str(), pid_str.size());
        housekeeping_locks_[username] = fd;
        Diskerror::Logger::info(std::format(lang::MSG_HOUSEKEEPING_OWNER, username));
        return true;
    }

    /// Start background timer for periodic housekeeping.
    /// Housekeeping only acts on users whose locks we hold.
    void start_housekeeping_timer() {
        int interval = config().housekeeping_interval;
        if (interval == 0) {
            Diskerror::Logger::info(lang::MSG_HOUSEKEEPING_DISABLED);
            return;
        }
        timer_running_ = true;
        timer_thread_ = std::thread([this, interval]() {
            while (timer_running_) {
                for (int i = 0; i < interval && timer_running_; ++i) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    // Check for shutdown (or restart, which stops listeners
                    // the same way — main() re-execs afterward).
                    if (g_shutdown_requested.load(std::memory_order_relaxed) ||
                        g_restart_requested.load(std::memory_order_relaxed)) {
                        Diskerror::Logger::info(
                            g_restart_requested.load(std::memory_order_relaxed)
                                ? "Restart signal received; stopping listeners to re-exec"
                                : "Shutdown signal received; stopping listeners");
                        timer_running_ = false;
                        if (unix_svr.is_running()) unix_svr.stop();
                        if (tcp_svr && tcp_svr->is_running()) tcp_svr->stop();
                        break;
                    }
                    // Check for signal-triggered housekeeping
                    if (g_housekeeping_requested.exchange(false)) {
                        Diskerror::Logger::info(lang::MSG_HOUSEKEEPING_SIGNAL);
                        run_housekeeping();
                    }
                    // Check for config reload (SIGHUP)
                    if (g_config_reload_requested.exchange(false)) {
                        int n = reload_config();
                        if (n > 0) {
                            Diskerror::Logger::info(std::format(ragger::lang::MSG_CONFIG_RELOADED_N, n));
                            // Re-initialize inference client if endpoints changed
                            try {
                                inference_ = std::make_unique<InferenceClient>(
                                    InferenceClient::from_config(config()));
                                Diskerror::Logger::info(lang::MSG_INFERENCE_CLIENT_OK);
                            } catch (const std::exception& e) {
                                Diskerror::Logger::critical(std::format(lang::ERR_INFERENCE_CLIENT_FAIL, e.what()));
                            }
                        } else {
                            Diskerror::Logger::info(lang::MSG_CONFIG_RELOADED_NONE);
                        }
                    }
                }
                if (timer_running_) {
                    run_housekeeping();
                }
            }
        });
    }

    void stop_housekeeping_timer() {
        timer_running_ = false;
        if (timer_thread_.joinable()) {
            timer_thread_.join();
        }
        // Release all user housekeeping locks
        std::lock_guard<std::mutex> lock(housekeeping_locks_mutex_);
        for (auto& [username, fd] : housekeeping_locks_) {
            flock(fd, LOCK_UN);
            ::close(fd);
        }
        housekeeping_locks_.clear();
    }

    void init_inference() {
        const auto& cfg = config();
        auto client = InferenceClient::from_config(cfg);
        if (!client.endpoints.empty()) {
            inference_ = std::make_unique<InferenceClient>(std::move(client));
            Diskerror::Logger::info(std::format(lang::MSG_INFERENCE_ENABLED, inference_->endpoints.size()));
        }
    }

    void bootstrap_auth() {
        // Single-user mode: ensure token exists
        server_token_ = ensure_token();
        
        if (!server_token_.empty()) {
            std::string token_hash = hash_token(server_token_);
            auto user = users_.get_user_by_token_hash(token_hash);

            if (!user) {
                // Auto-create the single user
                std::string username = "default";
                struct passwd* pw = getpwuid(getuid());
                if (pw) username = pw->pw_name;

                int user_id = users_.create_user(username, token_hash);
                user = UserInfo{user_id, username, token_hash};
                Diskerror::Logger::info(std::format(lang::MSG_CREATED_USER, username, user_id));
            }
            default_user_ = user;
        }
        Diskerror::Logger::info(lang::MSG_SINGLE_USER_MODE);
    }

    /// True if this request must present a valid token. False for requests
    /// arriving on the unix socket (empty remote_addr) or from localhost.
    bool request_requires_auth(const httplib::Request& req) const {
        if (req.remote_addr.empty()) return false;  // unix socket
        if (req.remote_addr == "127.0.0.1" || req.remote_addr == "::1") {
            return false;
        }
        return true;
    }

    std::optional<UserInfo> _check_auth(const httplib::Request& req) {
        const auto& cfg = config();

        // Auth bypass: unix socket and localhost requests are pre-authenticated
        // as the default (single) user.
        if (!request_requires_auth(req) && default_user_) {
            return default_user_;
        }

        // Parse "Bearer <token>" from the Authorization header.
        auto auth_header = req.get_header_value("Authorization");
        const std::string bearer_prefix = "Bearer ";
        std::string token;
        if (auth_header.substr(0, bearer_prefix.size()) == bearer_prefix) {
            token = auth_header.substr(bearer_prefix.size());
        } else if (req.has_param("token")) {
            // Browser navigation / EventSource can't set headers — accept the
            // token as a query parameter (?token=...) for the dashboard and
            // its SSE stream. Same trust model: holding the token = authorized.
            token = req.get_param_value("token");
        } else {
            return std::nullopt;
        }

        // Hash and lookup in database
        std::string token_hash = hash_token(token);
        auto user = users_.get_user_by_token_hash(token_hash);
        if (user) {
            return user;
        }

        // Fallback: direct comparison with server token
        if (token == server_token_ && default_user_) {
            return default_user_;
        }

        return std::nullopt;
    }

    /// Get per-user memory (or fallback to common).
    /// In single-user mode, always returns the main memory instance.
    RaggerMemory& _get_memory(const std::string& /*username*/) {
        // Single-user mode: always return the main memory
        return memory;
    }


    // ---- Route handlers (named for navigability) ---------------------------

    void handle_store(const UserInfo& user, const httplib::Request& req,
                      httplib::Response& res) {
        auto body = json::parse(req.body);
        std::string text = body.value("text", "");
        json metadata = body.value("metadata", json::object());
        if (text.empty()) {
            Diskerror::Logger::debug("POST /store 400");
            res.status = 400;
            res.set_content(lang::HTTP_MISSING_TEXT, "text/plain");
            return;
        }
        if (!metadata.contains("collection") || metadata["collection"].get<std::string>().empty())
            metadata["collection"] = "memory";
        if (!metadata.contains("source") || metadata["source"].get<std::string>().empty())
            metadata["source"] = user.username;
        auto& mem = _get_memory(user.username);

        // Route curated decisions to the L6 decisions table; everything else
        // (fact/preference/lesson/unspecified) lands in summaries via store().
        std::string category = metadata.value("category", "");
        if (category == "decision") {
            std::string tags = metadata.value("tags", "");
            // Default "current" (the only status the recall pipeline
            // surfaces). Callers can pass metadata.status="roadmap" for
            // planned/future work that shouldn't clutter every session's
            // recall until it's actually done.
            std::string status = metadata.value("status", "current");
            int did = mem.store_decision(text, status, tags);
            Diskerror::Logger::debug("POST /store 200 (decision)");
            res.set_content(json{{"id", std::to_string(did)},
                                 {"status", "stored"},
                                 {"table", "decisions"}}.dump(), "application/json");
            return;
        }

        std::string id = mem.store(text, metadata);
        Diskerror::Logger::debug("POST /store 200");
        res.set_content(json{{"id", id}, {"status", "stored"}}.dump(), "application/json");
    }

    void handle_search(const UserInfo& user, const httplib::Request& req,
                       httplib::Response& res) {
        auto body = json::parse(req.body);
        std::string query = body.value("query", "");
        if (query.empty()) {
            Diskerror::Logger::debug("POST /search 400");
            res.status = 400; res.set_content(lang::HTTP_MISSING_QUERY, "text/plain"); return;
        }
        int limit       = body.value("limit", config().default_search_limit);
        float min_score = body.value("min_score", config().default_min_score);
        auto collections = body.value("collections", std::vector<std::string>{});
        auto t0 = std::chrono::high_resolution_clock::now();
        auto& mem = _get_memory(user.username);
        auto sr = mem.search(query, limit, min_score, collections);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();
        json timing = sr.timing;
        timing["total_ms"] = ms;
        Diskerror::Logger::trace(std::format(lang::DBG_QUERY_LOG, query, sr.results.size(), ms));
        Diskerror::Logger::debug(std::format(lang::DBG_HTTP, "POST", "/search", "200"));
        res.set_content(json{{"results", search_results_to_json(sr.results)},
                             {"timing",  timing}}.dump(), "application/json");
    }

    // POST /turn: ingest one agent-pushed conversation turn for background
    // summarization. No-op unless [server] capture_turns is true.
    void handle_turn(const UserInfo& user, const httplib::Request& req,
                     httplib::Response& res) {
        auto body = json::parse(req.body);
        std::string user_text  = body.value("user", "");
        std::string assistant  = body.value("assistant", "");
        std::string model      = body.value("model", "");
        std::string session_id = body.value("session_id", "");
        std::string session_name = body.value("session_name", "");
        std::string name_source  = body.value("name_source", "");
        if (user_text.empty()) {
            Diskerror::Logger::debug("POST /turn 400");
            res.status = 400;
            res.set_content(json{{"error", "missing required field: user"}}.dump(),
                            "application/json");
            return;
        }
        auto& mem = _get_memory(user.username);
        auto result = capture_turn(mem, user_text, assistant, model, session_id,
                                   session_name, name_source);
        // Raw text is written to both turns and summaries (as a NULL-model_id
        // placeholder) by capture_turn. The summarizer picks it up on the next
        // housekeeping tick (every 60s) via enqueue_catch_up — no immediate
        // LLM call here.
        Diskerror::Logger::debug(std::format(lang::DBG_HTTP, "POST", "/turn", "200"));
        res.set_content(json{{"status",  result.captured ? "captured" : "disabled"},
                             {"turn_id", result.turn_id}}.dump(), "application/json");
    }

    // GET /session/<guid>[?recipe=name]: assemble a recipe-shaped context payload.
    void handle_session(const UserInfo& user, const httplib::Request& req,
                        httplib::Response& res) {
        std::string sid = req.matches[1];
        std::string recipe_name = req.get_param_value("recipe");
        auto& mem = _get_memory(user.username);
        auto ctx = build_context(mem, sid, recipe_name);
        if (!ctx.enabled) {
            res.set_content(json{{"status", "disabled"}}.dump(), "application/json");
            return;
        }
        json chunks = json::array();
        for (const auto& c : ctx.chunks)
            chunks.push_back({{"kind", c.kind}, {"text", c.text}, {"timestamp", c.timestamp}});
        Diskerror::Logger::debug(std::format(lang::DBG_HTTP, "GET", "/session", "200"));
        res.set_content(json{{"status",     "ok"},
                             {"session_id", sid},
                             {"recipe",     ctx.recipe_name},
                             {"chunks",     chunks}}.dump(), "application/json");
    }

    void handle_delete_memory(const UserInfo& user, const httplib::Request& req,
                               httplib::Response& res) {
        int id = std::stoi(req.matches[1]);
        auto& mem = _get_memory(user.username);
        if (mem.delete_memory(id)) {
            Diskerror::Logger::debug("DELETE /memory 200");
            res.set_content(json{{"id", id}, {"status", "deleted"}}.dump(), "application/json");
        } else {
            Diskerror::Logger::debug("DELETE /memory 404");
            res.status = 404; res.set_content(lang::HTTP_MEMORY_NOT_FOUND, "text/plain");
        }
    }

    void handle_delete_batch(const UserInfo& user, const httplib::Request& req,
                              httplib::Response& res) {
        auto body = json::parse(req.body);
        if (!body.contains("ids") || !body["ids"].is_array()) {
            Diskerror::Logger::debug("POST /delete_batch 400");
            res.status = 400; res.set_content(lang::HTTP_MISSING_IDS, "text/plain"); return;
        }
        auto ids = body["ids"].get<std::vector<int>>();
        auto& mem = _get_memory(user.username);
        int deleted = mem.delete_batch(ids);
        Diskerror::Logger::debug("POST /delete_batch 200");
        res.set_content(json{{"deleted", deleted}}.dump(), "application/json");
    }

    void handle_search_by_metadata(const UserInfo& user, const httplib::Request& req,
                                    httplib::Response& res) {
        auto body = json::parse(req.body);
        if (!body.contains("metadata") || !body["metadata"].is_object()) {
            Diskerror::Logger::debug("POST /search_by_metadata 400");
            res.status = 400; res.set_content(lang::HTTP_MISSING_METADATA, "text/plain"); return;
        }
        auto results = _get_memory(user.username).search_by_metadata(
            body["metadata"], body.value("limit", 0),
            body.value("after", ""), body.value("before", ""));
        Diskerror::Logger::debug("POST /search_by_metadata 200");
        res.set_content(json{{"results", search_results_to_json(results)},
                             {"count",   results.size()}}.dump(), "application/json");
    }

    // ---- Route registration -----------------------------------------------

    void setup_routes(httplib::Server& svr) {
        // GET /health — no auth required
        svr.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
            res.set_content(json{{"status",   "ok"},
                                 {"version",  RAGGER_VERSION},
                                 {"commit",   RAGGER_COMMIT},
                                 {"built",    RAGGER_BUILD_DATE},
                                 {"memories", memory.count()},
                                 {"embeddable_rows", memory.count_embeddable_rows()},
                                 {"degraded", memory.embeddings_degraded()}}.dump(), "application/json");
            Diskerror::Logger::debug("GET /health 200");
        });

        // Wrap a handler with bearer auth + uniform error handling.
        auto guarded = [this](auto handler) {
            return [this, handler](const httplib::Request& req, httplib::Response& res) {
                auto user = _check_auth(req);
                if (!user) {
                    Diskerror::Logger::debug(std::format("{} {} 401", req.method, req.path));
                    res.status = 401;
                    res.set_content("Unauthorized", "text/plain");
                    return;
                }
                try {
                    handler(*user, req, res);
                } catch (const json::exception& e) {
                    Diskerror::Logger::debug(std::format("{} {} 400", req.method, req.path));
                    res.status = 400;
                    res.set_content(std::string("JSON error: ") + e.what(), "text/plain");
                } catch (const std::exception& e) {
                    Diskerror::Logger::critical(std::format("{} {} failed: {}", req.method, req.path, e.what()));
                    res.status = 500;
                    res.set_content(std::string("ERROR: ") + e.what(), "text/plain");
                }
            };
        };

        svr.Get("/count", guarded([this](const UserInfo& user, const httplib::Request&,
                                         httplib::Response& res) {
            res.set_content(json{{"count", _get_memory(user.username).count()}}.dump(),
                            "application/json");
            Diskerror::Logger::debug("GET /count 200");
        }));

        svr.Post("/store",              guarded([this](const UserInfo& u, const httplib::Request& req, httplib::Response& res) { handle_store(u, req, res); }));
        svr.Post("/search",             guarded([this](const UserInfo& u, const httplib::Request& req, httplib::Response& res) { handle_search(u, req, res); }));
        svr.Post("/turn",               guarded([this](const UserInfo& u, const httplib::Request& req, httplib::Response& res) { handle_turn(u, req, res); }));
        svr.Get(R"(/session/([^/]+))",  guarded([this](const UserInfo& u, const httplib::Request& req, httplib::Response& res) { handle_session(u, req, res); }));
        svr.Delete(R"(/memory/(\d+))",  guarded([this](const UserInfo& u, const httplib::Request& req, httplib::Response& res) { handle_delete_memory(u, req, res); }));
        svr.Post("/delete_batch",       guarded([this](const UserInfo& u, const httplib::Request& req, httplib::Response& res) { handle_delete_batch(u, req, res); }));
        svr.Post("/search_by_metadata", guarded([this](const UserInfo& u, const httplib::Request& req, httplib::Response& res) { handle_search_by_metadata(u, req, res); }));

        svr.Get("/user/token", guarded([this](const UserInfo& user, const httplib::Request&,
                                              httplib::Response& res) {
            struct passwd* pw = getpwnam(user.username.c_str());
            if (!pw) { res.status = 404; res.set_content(R"({"error":"system user not found"})", "application/json"); return; }
            std::string token_file = std::string(pw->pw_dir) + "/.ragger/token";
            std::ifstream f(token_file);
            if (!f) { res.status = 404; res.set_content(R"({"error":"no token file"})", "application/json"); return; }
            std::string token;
            std::getline(f, token);
            size_t s = token.find_first_not_of(" \t\r\n");
            size_t e = token.find_last_not_of(" \t\r\n");
            if (s != std::string::npos) token = token.substr(s, e - s + 1);
            Diskerror::Logger::debug("GET /user/token 200");
            res.set_content(json{{"token", token}, {"username", user.username}}.dump(),
                            "application/json");
        }));

        // ---- Dashboard (embedded HTML/JS) --------------------------------
        // GET /dashboard[?token=...] -> the single-page control panel. The
        // page then talks to /config and /events with the same token.
        svr.Get("/dashboard", guarded([this](const UserInfo&, const httplib::Request&,
                                             httplib::Response& res) {
            res.set_content(DASHBOARD_HTML, "text/html; charset=utf-8");
            Diskerror::Logger::debug("GET /dashboard 200");
        }));

        // ---- Config API (dashboard + programmatic) -----------------------
        // GET /config          -> all settings as {key: {value, ...schema}}
        // GET /config/<key>    -> one setting
        // PUT /config/<key>    -> set one setting (body: {"value": "..."})
        //                         empty value reverts to schema default.
        svr.Get("/config", guarded([this](const UserInfo&, const httplib::Request&,
                                          httplib::Response& res) {
            res.set_content(build_config_json().dump(), "application/json");
            Diskerror::Logger::debug("GET /config 200");
        }));

        svr.Get(R"(/config/([A-Za-z0-9_]+))", guarded([this](const UserInfo&,
                const httplib::Request& req, httplib::Response& res) {
            std::string key = req.matches[1];
            const auto* meta = lang::config_meta(key);
            if (!meta) { res.status = 404; res.set_content(R"({"error":"unknown key"})", "application/json"); return; }
            auto v = get_config_value(config(), key);
            res.set_content(config_entry_json(*meta, v.value_or("")).dump(), "application/json");
        }));

        svr.Put(R"(/config/([A-Za-z0-9_]+))", guarded([this](const UserInfo&,
                const httplib::Request& req, httplib::Response& res) {
            std::string key = req.matches[1];
            std::string value;
            if (!req.body.empty()) {
                auto body = json::parse(req.body);
                value = body.value("value", "");
            }
            auto r = set_config_persisted(key, value);
            if (!r) {
                switch (r.error()) {
                    case ConfigSetError::UnknownKey:
                        res.status = 404; res.set_content(R"({"error":"unknown key"})", "application/json"); break;
                    case ConfigSetError::Locked:
                        res.status = 403; res.set_content(R"({"error":"key is locked"})", "application/json"); break;
                    case ConfigSetError::InvalidValue:
                        res.status = 400; res.set_content(R"({"error":"invalid value"})", "application/json"); break;
                }
                Diskerror::Logger::debug(std::format("PUT /config/{} {}", key, res.status));
                return;
            }
            // Echo the resolved (post-default) value plus apply flags.
            auto v = get_config_value(config(), key);
            json out = {{"key", key}, {"value", v.value_or("")},
                        {"restart_required", r->restart_required},
                        {"rebuild_required", r->rebuild_required}};
            res.set_content(out.dump(), "application/json");
            Diskerror::Logger::debug(std::format("PUT /config/{} 200", key));
        }));

        // GET /stats -> one-shot status snapshot (server + table sizes).
        svr.Get("/stats", guarded([this](const UserInfo&, const httplib::Request&,
                                         httplib::Response& res) {
            res.set_content(build_stats_json().dump(), "application/json");
        }));

        // GET /models -> ONNX embedding model choices from ~/.ragger/models/.
        // Scans for provider/model dirs (two-level: sentence-transformers/all-MiniLM-L6-v2)
        // and legacy flat dirs (all-MiniLM-L6-v2). A directory is a valid ONNX model
        // when it contains tokenizer.json AND (model.onnx or onnx/model.onnx).
        // Returns {models: [{name:"all-MiniLM-L6-v2", path:"sentence-transformers/all-MiniLM-L6-v2"}, ...]}
        // "name" is the display name (model only, unless two providers share a
        // model name — then "provider/model"). "path" is the canonical key to
        // store in the config and resolve with resolved_model_dir().
        svr.Get("/models", guarded([this](const UserInfo&, const httplib::Request&,
                                          httplib::Response& res) {
            namespace fs = std::filesystem;
            std::error_code ec;
            const std::string dir = ragger_base_dir() + "/models";

            auto is_onnx_model = [](const fs::path& p) -> bool {
                return fs::exists(p / "tokenizer.json") &&
                       (fs::exists(p / "model.onnx") || fs::exists(p / "onnx" / "model.onnx"));
            };

            // Collect all valid models as (display_name, canonical_path) pairs.
            // canonical_path is relative to the models dir.
            struct ModelInfo { std::string name; std::string path; };
            std::vector<ModelInfo> models;
            // Track model-name frequency for collision detection.
            std::map<std::string, int> name_count;

            for (auto& top : fs::directory_iterator(dir, ec)) {
                if (!top.is_directory(ec)) continue;
                std::string top_name = top.path().filename().string();
                if (!top_name.empty() && top_name[0] == '.') continue;

                // Only scan provider/model two-level layout. Legacy flat
                // model dirs (e.g. all-MiniLM-L6-v2/ at the top level) are
                // kept for backward compat but not offered as new choices.
                for (auto& sub : fs::directory_iterator(top.path(), ec)) {
                    if (!sub.is_directory(ec)) continue;
                    std::string sub_name = sub.path().filename().string();
                    if (!sub_name.empty() && sub_name[0] == '.') continue;
                    if (is_onnx_model(sub.path())) {
                        models.push_back({sub_name, top_name + "/" + sub_name});
                        name_count[sub_name]++;
                    }
                }
            }

            // Build response: disambiguate colliding display names with provider prefix.
            json arr = json::array();
            for (auto& m : models) {
                std::string display = (name_count[m.name] > 1) ? m.path : m.name;
                arr.push_back(json{{"name", display}, {"path", m.path}});
            }
            // Sort by display name for stable UI ordering.
            std::sort(arr.begin(), arr.end(),
                [](const json& a, const json& b) { return a["name"] < b["name"]; });
            res.set_content(json{{"models", arr}}.dump(), "application/json");
        }));

        // GET /models/external?host=X&port=Y&key=Z -> query a remote
        // /v1/models endpoint and return available model names.
        svr.Get("/models/external", guarded([this](const UserInfo&, const httplib::Request& req,
                                                    httplib::Response& res) {
            std::string host = req.get_param_value("host");
            std::string port_s = req.get_param_value("port");
            std::string key = req.get_param_value("key");
            if (host.empty() || port_s.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"host and port required"})", "application/json");
                return;
            }
            int port = std::stoi(port_s);
            Embedder probe_emb(host, port, "", key);
            auto models = probe_emb.list_remote_models();
            json arr = json::array();
            for (const auto& m : models) arr.push_back(m);
            res.set_content(json{{"models", arr}}.dump(), "application/json");
        }));

        // POST /models/external/probe -> send a test embedding request to a
        // remote endpoint and return the vector dimensionality.
        // Body: {"host":"...", "port":N, "model":"...", "key":"..."}
        svr.Post("/models/external/probe", guarded([this](const UserInfo&, const httplib::Request& req,
                                                          httplib::Response& res) {
            auto body = json::parse(req.body, nullptr, false);
            if (body.is_discarded()) {
                res.status = 400;
                res.set_content(R"({"error":"invalid JSON"})", "application/json");
                return;
            }
            std::string host = body.value("host", "");
            int port = body.value("port", 0);
            std::string model = body.value("model", "");
            std::string key = body.value("key", "");
            if (host.empty() || port == 0 || model.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"host, port, and model required"})", "application/json");
                return;
            }
            Embedder probe_emb(host, port, model, key);
            int dims = probe_emb.probe_dimensions();
            if (dims == 0) {
                res.status = 502;
                res.set_content(R"({"error":"probe failed — endpoint unreachable or returned no embedding"})",
                                "application/json");
                return;
            }
            res.set_content(json{{"dimensions", dims}}.dump(), "application/json");
        }));

        // GET /models/huggingface?q=<search> -> search HuggingFace for ONNX
        // embedding models. Returns {models: [{id, downloads}, ...]} sorted
        // by popularity. Combines sentence-similarity + feature-extraction.
        svr.Get("/models/huggingface", guarded([this](const UserInfo&, const httplib::Request& req,
                                                       httplib::Response& res) {
            std::string query = req.get_param_value("q");
            // Build HF API URL — search both pipeline tags for embedding models.
            auto fetch_hf = [&](const std::string& tag) -> json {
                std::string url = "https://huggingface.co/api/models?"
                    "pipeline_tag=" + tag +
                    "&library=onnx&sort=downloads&direction=-1&limit=20";
                if (!query.empty()) url += "&search=" + query;
                CURL* curl = curl_easy_init();
                if (!curl) return json::array();
                std::string buf;
                curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
                curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
                curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,
                    +[](char* p, size_t s, size_t n, void* ud) -> size_t {
                        static_cast<std::string*>(ud)->append(p, s * n);
                        return s * n;
                    });
                curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
                CURLcode rc = curl_easy_perform(curl);
                curl_easy_cleanup(curl);
                if (rc != CURLE_OK) return json::array();
                auto j = json::parse(buf, nullptr, false);
                return j.is_array() ? j : json::array();
            };

            // Merge results from both pipeline tags, dedup by modelId.
            json results = json::array();
            std::set<std::string> seen;
            for (const auto& tag : {"sentence-similarity", "feature-extraction"}) {
                for (const auto& m : fetch_hf(tag)) {
                    std::string mid = m.value("modelId", "");
                    if (mid.empty() || seen.count(mid)) continue;
                    seen.insert(mid);
                    results.push_back(json{
                        {"id", mid},
                        {"downloads", m.value("downloads", 0)}
                    });
                }
            }
            res.set_content(json{{"models", results}}.dump(), "application/json");
        }));

        // POST /models/download -> download an ONNX model from HuggingFace
        // to ~/.ragger/models/<provider>/<model>/.
        // Body: {"repo": "sentence-transformers/all-MiniLM-L6-v2"}
        svr.Post("/models/download", guarded([this](const UserInfo&, const httplib::Request& req,
                                                     httplib::Response& res) {
            auto body = json::parse(req.body, nullptr, false);
            if (body.is_discarded() || !body.contains("repo")) {
                res.status = 400;
                res.set_content(R"({"error":"'repo' required"})", "application/json");
                return;
            }
            std::string repo = body["repo"].get<std::string>();
            // Validate repo format: "provider/model"
            auto slash = repo.find('/');
            if (slash == std::string::npos || slash == 0 || slash == repo.size() - 1) {
                res.status = 400;
                res.set_content(R"({"error":"repo must be 'provider/model'"})", "application/json");
                return;
            }

            std::string dest = ragger_base_dir() + "/models/" + repo;
            std::string hf_base = "https://huggingface.co/" + repo + "/resolve/main";

            // Check if already present (model.onnx or onnx/model.onnx)
            if (std::filesystem::exists(dest + "/model.onnx") ||
                std::filesystem::exists(dest + "/onnx/model.onnx")) {
                res.set_content(json{{"status", "already_present"}, {"path", repo}}.dump(),
                                "application/json");
                return;
            }

            std::filesystem::create_directories(dest);
            std::filesystem::create_directories(dest + "/onnx");

            auto dl = [&](const std::string& remote, const std::string& local) -> bool {
                CURL* curl = curl_easy_init();
                if (!curl) return false;
                FILE* fp = fopen(local.c_str(), "wb");
                if (!fp) { curl_easy_cleanup(curl); return false; }
                curl_easy_setopt(curl, CURLOPT_URL, remote.c_str());
                curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
                curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
                curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);
                CURLcode rc = curl_easy_perform(curl);
                long http = 0;
                curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http);
                curl_easy_cleanup(curl);
                fclose(fp);
                if (rc != CURLE_OK || http != 200) {
                    std::filesystem::remove(local);  // clean up partial
                    return false;
                }
                return true;
            };

            // Download metadata files (small, quick).
            std::vector<std::string> meta = {
                "config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "vocab.txt"
            };
            for (const auto& f : meta) {
                dl(hf_base + "/" + f, dest + "/" + f);
                // Non-fatal — some models don't have all files.
            }

            // Download model.onnx — try onnx/model.onnx first, fall back to root.
            bool got_onnx = dl(hf_base + "/onnx/model.onnx", dest + "/onnx/model.onnx");
            if (!got_onnx) {
                // Try root-level model.onnx
                got_onnx = dl(hf_base + "/model.onnx", dest + "/model.onnx");
            }

            if (!got_onnx) {
                res.status = 502;
                res.set_content(json{{"error", "Failed to download model.onnx from " + repo}}.dump(),
                                "application/json");
                return;
            }

            res.set_content(json{{"status", "downloaded"}, {"path", repo}}.dump(),
                            "application/json");
        }));

        // GET /models/summarizer -> query the summarizer endpoint's /v1/models
        // (or the inference endpoint if summarizer URL is not separately set).
        // Returns {models: ["model-name", ...]} for the dashboard to populate
        // the summarizer_model dropdown.
        svr.Get("/models/summarizer", guarded([this](const UserInfo&, const httplib::Request&,
                                                      httplib::Response& res) {
            // Resolve which URL to probe: summarizer_api_url if set, else
            // inference_api_url (the summarizer falls back to it at runtime).
            std::string url = config().summarizer_api_url;
            if (url.empty()) url = config().inference_api_url;
            if (url.empty()) {
                res.set_content(R"({"models":[]})", "application/json");
                return;
            }
            // Use an Endpoint object to probe — it already handles /v1/models.
            Endpoint ep("summarizer", url);
            auto ids = ep.list_models();
            json arr = json::array();
            for (const auto& id : ids) {
                // Skip embedding models — they aren't useful for summarization.
                std::string lower = id;
                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                if (lower.find("embed") != std::string::npos) continue;
                if (lower.find("minilm") != std::string::npos) continue;
                arr.push_back(id);
            }
            res.set_content(json{{"models", arr}}.dump(), "application/json");
        }));

        // GET /embedding/status -> current vs desired identity + flags.
        svr.Get("/embedding/status", guarded([this](const UserInfo&, const httplib::Request&,
                                                    httplib::Response& res) {
            auto s = memory.embedding_status();
            res.set_content(json{
                {"current", {{"model", s.current_model}, {"vector_type", s.current_vtype},
                             {"dimensions", s.current_dims}, {"engine", s.current_engine}}},
                {"desired", {{"model", s.desired_model}, {"vector_type", s.desired_vtype},
                             {"dimensions", s.desired_dims}, {"engine", s.desired_engine}}},
                {"needs_update", s.needs_update},
                {"reembedding",  s.reembedding},
                {"degraded",     memory.embeddings_degraded()},
            }.dump(), "application/json");
        }));

        // POST /embedding/update -> run the staged re-embed now. Runs in a
        // detached thread so the request returns immediately; progress is
        // reflected via /embedding/status (reembedding flag) and SSE stats.
        svr.Post("/embedding/update", guarded([this](const UserInfo&, const httplib::Request&,
                                                     httplib::Response& res) {
            if (memory.is_reembedding()) {
                res.status = 409;
                res.set_content(R"({"error":"re-embedding already in progress"})", "application/json");
                return;
            }
            auto s = memory.embedding_status();
            if (!s.needs_update) {
                res.set_content(R"({"status":"up-to-date","reembedding":false})", "application/json");
                return;
            }
            std::thread([this]() {
                try { memory.update_embeddings(); }
                catch (const std::exception& e) {
                    Diskerror::Logger::critical(std::format("re-embed failed: {}", e.what()));
                }
            }).detach();
            res.set_content(R"({"status":"started","reembedding":true})", "application/json");
        }));

        // GET /restart/status -> bound (current) vs configured (desired) port
        // and whether a restart is needed to apply it. Same current-vs-desired
        // shape as embedding, but "current" is the port bound at startup and
        // "desired" is the live config port (no separate desired_port key).
        svr.Get("/restart/status", guarded([this](const UserInfo&, const httplib::Request&,
                                                  httplib::Response& res) {
            const bool tls = !config().tls_cert.empty() && !config().tls_key.empty();
            // "current" = the port bound at startup (Impl::port). "desired" =
            // the editable desired_port the dashboard writes (falls back to the
            // committed port when unset). needs_restart when they differ — the
            // startup rectify will adopt desired_port on the next restart.
            int desired = config().desired_port != 0 ? config().desired_port
                                                     : config().port;
            res.set_content(json{
                {"bound_port",      port},
                {"configured_port", desired},
                {"bound_bind",      host},
                {"configured_bind", config().bind_address},
                {"needs_restart",   (port != desired) ||
                                    (host != config().bind_address &&
                                     !(host == "127.0.0.1" && config().bind_address.empty()))},
                {"scheme",          tls ? "https" : "http"},
            }.dump(), "application/json");
        }));

        // POST /restart -> re-exec the daemon in place. Returns immediately
        // with the URL the dashboard should reconnect to (new port/scheme),
        // then the listeners tear down and main() re-execs. The dashboard
        // polls that URL and redirects the browser once it answers.
        svr.Post("/restart", guarded([this](const UserInfo&, const httplib::Request&,
                                            httplib::Response& res) {
            const bool tls = !config().tls_cert.empty() && !config().tls_key.empty();
            std::string h = config().bind_address.empty() ? "127.0.0.1" : config().bind_address;
            if (h == "0.0.0.0" || h == "::") h = "127.0.0.1";
            // Reconnect on the port the restart will bind: desired_port (which
            // the startup rectify adopts), falling back to the committed port.
            int new_port = config().desired_port != 0 ? config().desired_port
                                                       : config().port;
            std::string url = std::format("{}://{}:{}/dashboard",
                                          tls ? "https" : "http", h, new_port);
            res.set_content(json{{"status", "restarting"},
                                 {"reconnect_url", url},
                                 {"port", new_port}}.dump(), "application/json");
            Diskerror::Logger::info("POST /restart -> re-exec requested");
            // Trigger after the response flushes; the timer loop stops the
            // listeners and run() returns true so main() re-execs.
            g_restart_requested.store(true, std::memory_order_relaxed);
        }));

        // GET /events -> Server-Sent Events stream. Pushes a "stats" event
        // (server status + table sizes) and any new activity-log lines every
        // few seconds. One-directional; the dashboard subscribes with
        // EventSource. Config writes go through PUT /config, not this stream.
        svr.Get("/events", guarded([this](const UserInfo&, const httplib::Request&,
                                          httplib::Response& res) {
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            res.set_chunked_content_provider(
                "text/event-stream",
                [this](size_t /*offset*/, httplib::DataSink& sink) {
                    // Emit a stats frame.
                    std::string frame = "event: stats\ndata: " +
                                        build_stats_json().dump() + "\n\n";
                    if (!sink.write(frame.data(), frame.size())) return false;

                    // Emit any new activity-log lines as "activity" events.
                    for (auto& line : drain_activity_lines()) {
                        std::string a = "event: activity\ndata: " +
                                        json{{"line", line}}.dump() + "\n\n";
                        if (!sink.write(a.data(), a.size())) return false;
                    }

                    std::this_thread::sleep_for(std::chrono::seconds(2));
                    return true;   // keep the stream open
                });
        }));
    }

    // ---- Config/stats helpers ---------------------------------------------

    // JSON for a single config entry: current value + schema metadata so the
    // dashboard can render the right control without a second lookup.
    json config_entry_json(const lang::ConfigMeta& m, const std::string& value) {
        const char* type = "string";
        switch (m.type) {
            case lang::CfgType::Boolean: type = "boolean"; break;
            case lang::CfgType::Integer: type = "integer"; break;
            case lang::CfgType::Float:   type = "float";   break;
            case lang::CfgType::Enum:    type = "enum";    break;
            case lang::CfgType::String:  type = "string";  break;
            case lang::CfgType::Path:    type = "path";    break;
            case lang::CfgType::Text:    type = "text";    break;
        }
        const char* edit = "live";
        switch (m.edit) {
            case lang::CfgEdit::Live:            edit = "live";    break;
            case lang::CfgEdit::RestartRequired: edit = "restart"; break;
            case lang::CfgEdit::RebuildRequired: edit = "rebuild"; break;
            case lang::CfgEdit::Locked:          edit = "locked";  break;
        }
        json j = {
            {"key",     std::string(m.key)},
            {"section", std::string(m.section)},
            {"label",   std::string(m.pretty)},
            {"type",    type},
            {"edit",    edit},
            {"default", std::string(m.default_value)},
            {"help",    std::string(m.help)},
            {"value",   value},
        };
        if (m.type == lang::CfgType::Enum) {
            json opts = json::array();
            std::string_view o = m.options;
            size_t pos = 0;
            while (pos <= o.size()) {
                size_t c = o.find(',', pos);
                std::string_view tok = (c == std::string_view::npos)
                    ? o.substr(pos) : o.substr(pos, c - pos);
                if (!tok.empty()) opts.push_back(std::string(tok));
                if (c == std::string_view::npos) break;
                pos = c + 1;
            }
            j["options"] = opts;
        }
        return j;
    }

    // Full config: array of entries in schema (dashboard tab) order.
    json build_config_json() {
        json arr = json::array();
        const auto& cfg = config();
        for (const auto& m : lang::config_schema()) {
            auto v = get_config_value(cfg, m.key);
            arr.push_back(config_entry_json(m, v.value_or("")));
        }
        return json{{"config", arr}};
    }

    // Server status + table sizes for the top status pane.
    json build_stats_json() {
        json tables = json::object();
        try {
            for (auto& [name, n] : memory.backend()->table_row_counts())
                tables[name] = n;
        } catch (...) { /* backend unavailable — empty tables */ }
        return json{
            {"status",   "running"},
            {"version",  RAGGER_VERSION},
            {"memories", memory.count()},
            {"tables",   tables},
        };
    }

    // Return activity-log lines appended since the last drain. Best-effort:
    // tracks a byte offset into the live activity.log and reads the tail.
    std::vector<std::string> drain_activity_lines() {
        std::vector<std::string> out;
        const std::string path = config().resolved_log_file_path();
        std::error_code ec;
        auto size = std::filesystem::file_size(path, ec);
        if (ec) return out;
        if (activity_log_offset_ == 0 && size > 0) {
            // First read: start at EOF so we only stream NEW activity.
            activity_log_offset_ = size;
            return out;
        }
        if (size < activity_log_offset_) activity_log_offset_ = 0;  // rotated
        if (size == activity_log_offset_) return out;
        std::ifstream f(path, std::ios::binary);
        if (!f) return out;
        f.seekg(static_cast<std::streamoff>(activity_log_offset_));
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty()) out.push_back(line);
        }
        activity_log_offset_ = size;
        // Cap to the last ~20 lines so a burst doesn't flood the client.
        if (out.size() > 20) out.erase(out.begin(), out.end() - 20);
        return out;
    }
};


Server::Server(RaggerMemory& memory,
               const std::string& host,
               int port)
    : pImpl(std::make_unique<Impl>(memory, host, port))
{
}

Server::~Server() = default;

static bool is_port_available(const std::string& host, int port) {
    int sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return false;

    // Allow binding to TIME_WAIT sockets (matches Crow's SO_REUSEADDR)
    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

    int result = ::bind(sock, (struct sockaddr*)&addr, sizeof(addr));
    ::close(sock);
    return result == 0;
}

bool Server::run() {
    const auto& cfg = config();
    const bool want_unix = cfg.socket_enabled;
    const bool want_tcp  = cfg.tcp_enabled && !pImpl->host.empty();

    if (want_tcp && !is_port_available(pImpl->host, pImpl->port)) {
        Diskerror::Logger::critical(std::format(lang::ERR_PORT_IN_USE, pImpl->port));
        std::exit(1);
    }

    std::string addr;
    if (want_unix && want_tcp) {
        addr = cfg.resolved_socket_path() + " + " + pImpl->host + ":" + std::to_string(pImpl->port);
    } else if (want_unix) {
        addr = cfg.resolved_socket_path();
    } else {
        addr = pImpl->host + ":" + std::to_string(pImpl->port);
    }
    Diskerror::Logger::info(std::format(lang::MSG_SERVER_STARTING, addr));
    if (want_tcp) {
        Diskerror::Logger::info(std::format(lang::MSG_HEALTH_CHECK_TCP,
                                            pImpl->host + ":" + std::to_string(pImpl->port)));
    }
    if (want_unix) {
        Diskerror::Logger::info(std::format(lang::MSG_HEALTH_CHECK_UNIX, cfg.resolved_socket_path()));
    }

    // Write PID file (per-port)
    std::string port_str = std::to_string(pImpl->port);
    fs::create_directories("/tmp/ragger");
    pImpl->pid_file_ = "/tmp/ragger/server-" + port_str + ".pid";
    {
        std::ofstream pf(pImpl->pid_file_);
        if (pf) {
            pf << getpid();
            Diskerror::Logger::info(std::format(lang::MSG_PID_FILE, pImpl->pid_file_));
        }
    }

    // Install signal handlers
    struct sigaction sa{};
    sa.sa_flags = SA_RESTART;
    sigemptyset(&sa.sa_mask);

    sa.sa_handler = sigusr1_handler;
    sigaction(SIGUSR1, &sa, nullptr);

    sa.sa_handler = sighup_handler;
    sigaction(SIGHUP, &sa, nullptr);

    sa.sa_handler = sigterm_handler;
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGINT,  &sa, nullptr);

    // Start housekeeping timer (runs every 60s + on SIGUSR1)
    pImpl->start_housekeeping_timer();

    // Start the daemon-resident summarizer (L2 per-turn + L3 on idle pause).
    // Only the HTTP daemon runs this: it keeps the inference + embedder warm
    // and catches up any turns that were captured via MCP while the daemon
    // was down. See SummarizerService doc for the timestamp-linkage rule.
    pImpl->summarizer_ = std::make_unique<SummarizerService>(
        *pImpl->memory.backend(), pImpl->inference_.get());
    pImpl->summarizer_->start();

    // Launch each enabled listener on its own thread. listen() blocks until
    // the corresponding Server::stop() is called, so the main thread blocks
    // on whichever listener it runs last (kept on the main thread when only
    // one is enabled). When both run, the TCP listener is moved to a worker
    // and main blocks on the unix listener.
    std::thread tcp_thread;
    auto run_tcp = [this]() {
        Diskerror::Logger::info(std::format(ragger::lang::MSG_BIND_TCP, pImpl->host, std::to_string(pImpl->port)));
        bool ok = pImpl->tcp_svr->listen(pImpl->host, pImpl->port);
        if (!ok) {
            Diskerror::Logger::error(std::format(ragger::lang::ERR_LISTEN_TCP, std::to_string(errno)));
        }
    };
    auto run_unix = [this, &cfg]() {
        std::string sock_path = cfg.resolved_socket_path();
        fs::create_directories(fs::path(sock_path).parent_path());
        std::filesystem::remove(sock_path);  // clear stale socket
        pImpl->unix_svr.set_address_family(AF_UNIX);

        // Chmod the socket 0600 once it appears. httplib::listen blocks,
        // so do this from a short-lived helper thread.
        std::thread chmod_thread([sock_path]() {
            for (int i = 0; i < 50; ++i) {  // up to ~1s
                if (std::filesystem::exists(sock_path)) {
                    ::chmod(sock_path.c_str(), 0600);
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            Diskerror::Logger::info(std::format(lang::MSG_UNIX_CHMOD_TIMEOUT, sock_path));
        });
        chmod_thread.detach();

        Diskerror::Logger::info(std::format(lang::MSG_BIND_UNIX_SOCKET, sock_path));
        // NOTE: port must be non-zero even for AF_UNIX. httplib's bind_internal
        // only calls getsockname() when port==0, and that path only handles
        // AF_INET/AF_INET6 — for AF_UNIX it returns UnsupportedAddressFamily
        // AFTER a successful bind, causing listen() to fail silently.
        bool ok = pImpl->unix_svr.listen(sock_path, 80);  // port value ignored for AF_UNIX
        if (!ok) {
            Diskerror::Logger::error(std::format(ragger::lang::ERR_LISTEN_UNIX, sock_path, std::to_string(errno)));
        }
    };

    if (want_unix && want_tcp) {
        tcp_thread = std::thread(run_tcp);
        run_unix();
    } else if (want_unix) {
        run_unix();
    } else {
        run_tcp();
    }

    // If TCP was on its own thread, join it before returning. The listeners
    // were already stopped above (timer loop on shutdown/restart, or
    // Server::stop from a signal) — calling stop() again here races httplib's
    // shutdown (is_running_ still true while svr_sock_ is already invalid) and
    // trips an assert, so we only join.
    if (tcp_thread.joinable()) {
        tcp_thread.join();
    }

    // Cleanup
    if (pImpl->summarizer_) pImpl->summarizer_->stop();
    pImpl->stop_housekeeping_timer();
    if (!pImpl->pid_file_.empty()) std::remove(pImpl->pid_file_.c_str());

    return g_restart_requested.load(std::memory_order_relaxed);
}

bool Server::restart_requested() {
    return g_restart_requested.load(std::memory_order_relaxed);
}

void Server::stop() {
    if (pImpl->summarizer_) pImpl->summarizer_->stop();
    pImpl->stop_housekeeping_timer();
    if (pImpl->unix_svr.is_running()) pImpl->unix_svr.stop();
    if (pImpl->tcp_svr && pImpl->tcp_svr->is_running()) pImpl->tcp_svr->stop();
}

} // namespace ragger
