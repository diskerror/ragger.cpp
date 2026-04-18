/**
 * HTTP server for Ragger Memory implementation
 */

#include "ragger/server.h"
#include "ragger/memory.h"
#include "ragger/sqlite_backend.h"
#include "ragger/lang.h"
#include "ragger/logs.h"
#include "ragger/auth.h"
#include "ragger/config.h"
#include "ragger/inference.h"
#include "ragger/chat_sessions.h"
#include "nlohmann_json.hpp"

#include "httplib.h"

#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <iostream>
#include <optional>
#include <pwd.h>
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

static void sigusr1_handler(int) {
    g_housekeeping_requested.store(true, std::memory_order_relaxed);
}

static void sighup_handler(int) {
    g_config_reload_requested.store(true, std::memory_order_relaxed);
}

struct Server::Impl {
    httplib::Server svr;
    RaggerMemory&   memory;
    std::string     host;
    int             port;
    std::string     server_token_;
    std::optional<UserInfo> default_user_;
    std::unique_ptr<InferenceClient> inference_;
    ChatSessionManager session_mgr_;

    // Per-user memory cache (username → RaggerMemory)
    std::unordered_map<std::string, std::unique_ptr<RaggerMemory>> user_memories_;
    std::mutex user_memories_mutex_;

    // Housekeeping timer
    std::atomic<bool> timer_running_{false};
    std::thread timer_thread_;
    std::string pid_file_;

    Impl(RaggerMemory& mem, const std::string& h, int p)
        : memory(mem), host(h), port(p)
    {
        bootstrap_auth();
        init_inference();
        setup_routes();
        warmup();
    }

    void warmup() {
        // Pre-load embedding cache so first request isn't slow
        try {
            auto result = memory.search("warmup", 1, 0.0f);
            log_info("Warmup: embedding cache loaded (" 
                     + std::to_string(memory.count()) + " memories)");
        } catch (const std::exception& e) {
            log_info("Warmup: " + std::string(e.what()));
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
                log_info("Preloaded model: " + model_name);
            } else {
                log_info("Model preload skipped: " + err);
            }
        }).detach();
    }

    /// Run one housekeeping pass: summarize idle sessions + purge old conversations.
    void run_housekeeping() {
        const auto& cfg = config();
        int pause_minutes = cfg.chat_pause_minutes;
        float max_age_hours = cfg.cleanup_max_age_hours;

        // 1. Expire idle chat sessions + summarize via inference
        auto expired = session_mgr_.cleanup_expired(pause_minutes);
        int sessions_expired = (int)expired.size();

        // Summarize expired sessions in background threads
        for (auto& es : expired) {
            if (!inference_ || es.turns.empty()) continue;

            // Capture by value for the detached thread
            auto turns = es.turns;
            auto username = es.username;
            auto sid_short = es.session_id.substr(0, 8);
            auto* inf = inference_.get();

            // Resolve per-user memory
            RaggerMemory* user_mem = nullptr;
            {
                std::lock_guard<std::mutex> lock(user_memories_mutex_);
                auto it = user_memories_.find(username);
                if (it != user_memories_.end() && it->second) {
                    user_mem = it->second.get();
                }
            }
            if (!user_mem) user_mem = &memory;

            auto* mem_ptr = user_mem;
            std::thread([turns, username, sid_short, inf, mem_ptr]() {
                try {
                    // Build conversation text
                    std::string conv;
                    for (auto& [user_text, asst_text] : turns) {
                        conv += "**User:** " + user_text + "\n\n";
                        conv += "**Assistant:** " + asst_text + "\n\n";
                    }

                    std::vector<Message> messages = {
                        {"system", "Summarize this conversation into a concise memory entry. "
                         "Extract key facts, decisions, and action items. Be brief but complete."},
                        {"user", conv}
                    };

                    std::string summary = inf->chat(messages);

                    if (!summary.empty()) {
                        json meta = {
                            {"collection", "memory"},
                            {"category", "session-summary"},
                            {"source", "chat-session-" + sid_short}
                        };
                        mem_ptr->store(summary, meta);
                        log_info("Summarized session " + sid_short +
                                 " (" + std::to_string(turns.size()) + " turns)");
                    }
                } catch (const std::exception& e) {
                    log_error("Session summarization failed: " + std::string(e.what()));
                }
            }).detach();
        }

        // 2. Purge old conversation entries from all known user DBs
        int conversations_cleaned = 0;
        if (max_age_hours > 0) {
            auto now = std::chrono::system_clock::now();
            auto cutoff = now - std::chrono::duration_cast<std::chrono::system_clock::duration>(
                std::chrono::duration<double, std::ratio<3600>>(max_age_hours));
            auto cutoff_t = std::chrono::system_clock::to_time_t(cutoff);
            char cutoff_str[32];
            std::strftime(cutoff_str, sizeof(cutoff_str), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&cutoff_t));

            // Single-user mode: clean up the main memory's DB
            try {
                ragger::SqliteBackend temp_backend(memory.backend()->db_path());
                int deleted = temp_backend.cleanup_old_conversations(max_age_hours);
                conversations_cleaned += deleted;
                if (deleted > 0) {
                    log_info("Cleaned " + std::to_string(deleted)
                           + " expired conversations from main DB");
                }
            } catch (const std::exception& e) {
                log_error(std::string("Cleanup failed for main DB: ") + e.what());
            }
        }

        if (sessions_expired > 0 || conversations_cleaned > 0) {
            log_info("Housekeeping: " + std::to_string(sessions_expired) + " sessions expired, "
                   + std::to_string(conversations_cleaned) + " conversations cleaned");
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
        log_info("Housekeeping owner for user '" + username + "'");
        return true;
    }

    /// Start background timer for periodic housekeeping.
    /// Housekeeping only acts on users whose locks we hold.
    void start_housekeeping_timer() {
        int interval = config().housekeeping_interval;
        if (interval == 0) {
            log_info("Housekeeping: disabled (interval = 0)");
            return;
        }
        timer_running_ = true;
        timer_thread_ = std::thread([this, interval]() {
            while (timer_running_) {
                for (int i = 0; i < interval && timer_running_; ++i) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    // Check for signal-triggered housekeeping
                    if (g_housekeeping_requested.exchange(false)) {
                        log_info("Housekeeping triggered by signal");
                        run_housekeeping();
                    }
                    // Check for config reload (SIGHUP)
                    if (g_config_reload_requested.exchange(false)) {
                        int n = reload_config();
                        if (n > 0) {
                            log_info("Config reloaded: " + std::to_string(n) + " value(s) changed");
                            // Re-initialize inference client if endpoints changed
                            try {
                                inference_ = std::make_unique<InferenceClient>(
                                    InferenceClient::from_config(config()));
                                log_info("Inference client reloaded");
                            } catch (const std::exception& e) {
                                log_error("Inference client reload failed: " + std::string(e.what()));
                            }
                        } else {
                            log_info("Config reloaded: no changes");
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
        if (!client._endpoints.empty()) {
            inference_ = std::make_unique<InferenceClient>(std::move(client));
            log_info("Inference: enabled (" + std::to_string(inference_->_endpoints.size()) + " endpoint(s))");
        }
        
        // LM Proxy pass-through (if configured)
        if (!cfg.lm_proxy_url.empty()) {
            if (inference_) {
                inference_->set_lm_proxy_url(cfg.lm_proxy_url);
                log_info("LM proxy: enabled (" + cfg.lm_proxy_url + ")");
            } else {
                log_error("LM proxy configured but no inference endpoint available");
            }
        }
    }

    void bootstrap_auth() {
        // Single-user mode: ensure token exists
        server_token_ = ensure_token();
        
        if (!server_token_.empty()) {
            std::string token_hash = hash_token(server_token_);
            auto user = memory.backend()->get_user_by_token_hash(token_hash);
            
            if (!user) {
                // Auto-create the single user
                std::string username = "default";
                struct passwd* pw = getpwuid(getuid());
                if (pw) username = pw->pw_name;
                
                int user_id = memory.backend()->create_user(username, token_hash);
                user = UserInfo{user_id, username, token_hash, ""};
                log_info("Created user: " + username + " (id=" + std::to_string(user_id) + ")");
            }
            default_user_ = user;
        }
        log_info("Single-user mode initialized");
    }

    // --- Web sessions (password login, DB-backed) ---
    static constexpr int WEB_SESSION_TTL = 86400; // 24 hours

    std::optional<UserInfo> _check_web_session(const std::string& token) {
        return memory.backend()->get_web_session(token);
    }

    // --- Web root resolution ---
    std::string resolve_web_root() {
        const auto& cfg = config();
        if (!cfg.web_root.empty()) {
            auto p = expand_path(cfg.web_root);
            if (fs::is_directory(p)) return p;
        }
        // Fall back to common locations
        for (const auto& dir : {
            std::string("/var/ragger/www"),
            std::string("/usr/local/share/ragger/www"),
            std::string("web")
        }) {
            if (fs::is_directory(dir)) return dir;
        }
        return "";
    }

    static std::string mime_type(const std::string& path) {
        auto ext = fs::path(path).extension().string();
        if (ext == ".html") return "text/html; charset=utf-8";
        if (ext == ".css")  return "text/css; charset=utf-8";
        if (ext == ".js")   return "application/javascript; charset=utf-8";
        if (ext == ".json") return "application/json";
        if (ext == ".png")  return "image/png";
        if (ext == ".svg")  return "image/svg+xml";
        if (ext == ".ico")  return "image/x-icon";
        return "application/octet-stream";
    }

    std::optional<UserInfo> _check_auth(const httplib::Request& req) {
        const auto& cfg = config();
        
        // Extract Authorization header
        auto auth_header = req.get_header_value("Authorization");
        
        // Check cookie for web session token
        if (auth_header.empty()) {
            auto cookie = req.get_header_value("Cookie");
            std::string cookie_token;
            auto pos = cookie.find("ragger_token=");
            if (pos != std::string::npos) {
                auto start = pos + 13;
                auto end = cookie.find(';', start);
                cookie_token = cookie.substr(start, end == std::string::npos ? end : end - start);
            }
            if (!cookie_token.empty()) {
                auto ws = _check_web_session(cookie_token);
                if (ws) return ws;
                // Also try as bearer token
                auto user = memory.backend()->get_user_by_token_hash(hash_token(cookie_token));
                if (user) return user;
            }
            return std::nullopt;
        }

        // Parse "Bearer <token>"
        const std::string bearer_prefix = "Bearer ";
        if (auth_header.substr(0, bearer_prefix.size()) != bearer_prefix) {
            return std::nullopt;
        }
        std::string token = auth_header.substr(bearer_prefix.size());

        // Check web sessions first
        auto ws = _check_web_session(token);
        if (ws) return ws;

        // Hash and lookup in database
        std::string token_hash = hash_token(token);
        auto user = memory.backend()->get_user_by_token_hash(token_hash);
        if (user) {
            // Check if token rotation is needed (async, after this request)
            _check_token_rotation(*user);
            return user;
        }

        // Fallback: direct comparison with server token
        if (token == server_token_ && default_user_) {
            return default_user_;
        }

        return std::nullopt;
    }

    void _check_token_rotation(const UserInfo& user) {
        const auto& cfg = config();
        if (cfg.token_rotation_minutes <= 0) return;  // Rotation disabled

        auto rotated_at_opt = memory.backend()->get_user_token_rotated_at(user.username);

        // Get current time as ISO timestamp
        auto now = std::chrono::system_clock::now();
        auto now_tt = std::chrono::system_clock::to_time_t(now);
        std::tm now_gm{};
        gmtime_r(&now_tt, &now_gm);
        char now_buf[32];
        std::strftime(now_buf, sizeof(now_buf), "%Y-%m-%dT%H:%M:%SZ", &now_gm);
        std::string now_str(now_buf);

        bool needs_rotation = false;
        if (!rotated_at_opt) {
            // Never rotated — initialize with current time
            memory.backend()->update_user_token_rotated_at(user.username, now_str);
            return;
        }
        
        // Parse rotated_at timestamp
        std::string rotated_at = *rotated_at_opt;
        std::tm rotated_tm{};
        strptime(rotated_at.c_str(), "%Y-%m-%dT%H:%M:%SZ", &rotated_tm);
        auto rotated_tp = std::chrono::system_clock::from_time_t(timegm(&rotated_tm));
        
        auto age_minutes = std::chrono::duration_cast<std::chrono::minutes>(now - rotated_tp).count();
        
        // Grace window: skip rotation if rotated within last 60 seconds
        if (age_minutes < 1) return;
        
        if (age_minutes >= cfg.token_rotation_minutes) {
            needs_rotation = true;
        }
        
        if (needs_rotation) {
            // Rotate in background thread to not block this request
            std::thread([this, username = user.username, now_str]() {
                try {
                    auto [new_token, new_hash] = rotate_token_for_user(username);
                    memory.backend()->update_user_token(username, new_hash);
                    memory.backend()->update_user_token_rotated_at(username, now_str);
                    log_info("Rotated token for user: " + std::string(username));
                } catch (const std::exception& e) {
                    log_error(std::string("Token rotation failed for ") + username + ": " + e.what());
                }
            }).detach();
        }
    }

    /// Get per-user memory (or fallback to common).
    /// In single-user mode, always returns the main memory instance.
    RaggerMemory& _get_memory(const std::string& /*username*/) {
        // Single-user mode: always return the main memory
        return memory;
    }

    void setup_lm_proxy_routes();

    void setup_routes() {
        // GET /health
        svr.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
            json response = {
                {"status", "ok"},
                {"version", RAGGER_VERSION},
                {"commit", RAGGER_COMMIT},
                {"built", RAGGER_BUILD_DATE},
                {"memories", memory.count()}
            };
            log_http("GET /health 200");
            res.set_content(response.dump(), "application/json");
        });

        // GET /count
        svr.Get("/count", [this](const httplib::Request& req, httplib::Response& res) {
            auto user = _check_auth(req);
            if (!user) {
                log_http("GET /count 401");
                res.status = 401;
                res.set_content("Unauthorized", "text/plain");
                return;
            }
            auto& mem = _get_memory(user->username);
            json response = {
                {"count", mem.count()}
            };
            log_http("GET /count 200");
            res.set_content(response.dump(), "application/json");
        });

        // POST /store
        svr.Post("/store", [this](const httplib::Request& req, httplib::Response& res) {
            // Auth check
            auto user = _check_auth(req);
            if (!user) {
                log_http("POST /store 401");
                res.status = 401;
                res.set_content("Unauthorized", "text/plain");
                return;
            }

            try {
                auto body = json::parse(req.body);
                
                std::string text = body.value("text", "");
                json metadata = body.value("metadata", json::object());

                if (text.empty()) {
                    log_http("POST /store 400");
                    res.status = 400;
                    res.set_content("Missing 'text' field", "text/plain");
                    return;
                }

                // Ensure required metadata defaults
                if (!metadata.contains("collection") || metadata["collection"].get<std::string>().empty()) {
                    metadata["collection"] = "memory";
                }
                if (!metadata.contains("source") || metadata["source"].get<std::string>().empty()) {
                    metadata["source"] = user->username;
                }

                bool common = body.value("common", false);
                auto& mem = _get_memory(user->username);
                std::string id = mem.store(text, metadata, common);

                json response = {
                    {"id", id},
                    {"status", "stored"}
                };
                log_http("POST /store 200");
                res.set_content(response.dump(), "application/json");

            } catch (const json::exception& e) {
                log_http("POST /store 400");
                res.status = 400;
                res.set_content(std::string("JSON error: ") + e.what(), "text/plain");
            } catch (const std::exception& e) {
                log_http("POST /store 500");
                log_error(std::string("POST /store failed: ") + e.what());
                res.status = 500;
                res.set_content(std::string("Error: ") + e.what(), "text/plain");
            }
        });

        // POST /search
        svr.Post("/search", [this](const httplib::Request& req, httplib::Response& res) {
            auto user = _check_auth(req);
            if (!user) {
                log_http("POST /search 401");
                res.status = 401; res.set_content("Unauthorized", "text/plain"); return;
            }
            try {
                auto body = json::parse(req.body);
                std::string query = body.value("query", "");
                int limit = body.value("limit", 5);
                float min_score = body.value("min_score", 0.0f);
                std::vector<std::string> collections =
                    body.value("collections", std::vector<std::string>{});
                if (query.empty()) {
                    log_http("POST /search 400");
                    res.status = 400; res.set_content("Missing 'query' field", "text/plain"); return;
                }
                auto start_time = std::chrono::high_resolution_clock::now();
                auto& mem = _get_memory(user->username);
                SearchResponse search_response = mem.search(query, limit, min_score, collections);
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                json results = json::array();
                for (const auto& r : search_response.results) {
                    results.push_back({{"id", r.id}, {"text", r.text}, {"score", r.score},
                                       {"metadata", r.metadata}, {"timestamp", r.timestamp}});
                }
                json timing = search_response.timing;
                timing["total_ms"] = duration.count();
                json response = {{"results", results}, {"timing", timing}};
                std::ostringstream ql;
                ql << "query=\"" << query << "\" results=" << search_response.results.size()
                   << " time=" << duration.count() << "ms";
                log_query(ql.str());
                log_http("POST /search 200");
                res.set_content(response.dump(), "application/json");
            } catch (const json::exception& e) {
                log_http("POST /search 400");
                res.status = 400; res.set_content(std::string("JSON error: ") + e.what(), "text/plain");
            } catch (const std::exception& e) {
                log_http("POST /search 500");
                log_error(std::string("POST /search failed: ") + e.what());
                res.status = 500; res.set_content(std::string("Error: ") + e.what(), "text/plain");
            }
        });

        // DELETE /memory/:id
        svr.Delete(R"(/memory/(\d+))", [this](const httplib::Request& req, httplib::Response& res) {
            auto user = _check_auth(req);
            if (!user) {
                log_http("DELETE /memory 401");
                res.status = 401; res.set_content("Unauthorized", "text/plain"); return;
            }
            try {
                int id = std::stoi(req.matches[1]);
                auto& mem = _get_memory(user->username);
                bool deleted = mem.delete_memory(id);
                if (deleted) {
                    json response = {{"id", id}, {"status", "deleted"}};
                    log_http("DELETE /memory 200");
                    res.set_content(response.dump(), "application/json");
                } else {
                    log_http("DELETE /memory 404");
                    res.status = 404; res.set_content("Memory not found", "text/plain");
                }
            } catch (const std::exception& e) {
                log_http("DELETE /memory 500");
                log_error(std::string("DELETE /memory failed: ") + e.what());
                res.status = 500; res.set_content(std::string("Error: ") + e.what(), "text/plain");
            }
        });

        // POST /delete_batch
        svr.Post("/delete_batch", [this](const httplib::Request& req, httplib::Response& res) {
            auto user = _check_auth(req);
            if (!user) {
                log_http("POST /delete_batch 401");
                res.status = 401; res.set_content("Unauthorized", "text/plain"); return;
            }
            try {
                auto body = json::parse(req.body);
                if (!body.contains("ids") || !body["ids"].is_array()) {
                    log_http("POST /delete_batch 400");
                    res.status = 400; res.set_content("Missing or invalid 'ids' field", "text/plain"); return;
                }
                std::vector<int> ids = body["ids"].get<std::vector<int>>();
                auto& mem = _get_memory(user->username);
                int deleted = mem.delete_batch(ids);
                json response = {{"deleted", deleted}};
                log_http("POST /delete_batch 200");
                res.set_content(response.dump(), "application/json");
            } catch (const json::exception& e) {
                log_http("POST /delete_batch 400");
                res.status = 400; res.set_content(std::string("JSON error: ") + e.what(), "text/plain");
            } catch (const std::exception& e) {
                log_http("POST /delete_batch 500");
                log_error(std::string("POST /delete_batch failed: ") + e.what());
                res.status = 500; res.set_content(std::string("Error: ") + e.what(), "text/plain");
            }
        });

        // POST /search_by_metadata
        svr.Post("/search_by_metadata", [this](const httplib::Request& req, httplib::Response& res) {
            auto user = _check_auth(req);
            if (!user) {
                log_http("POST /search_by_metadata 401");
                res.status = 401; res.set_content("Unauthorized", "text/plain"); return;
            }
            try {
                auto body = json::parse(req.body);
                if (!body.contains("metadata") || !body["metadata"].is_object()) {
                    log_http("POST /search_by_metadata 400");
                    res.status = 400; res.set_content("Missing or invalid 'metadata' field", "text/plain"); return;
                }
                json metadata_filter = body["metadata"];
                int limit = body.value("limit", 0);
                std::string after = body.value("after", "");
                std::string before = body.value("before", "");
                auto& mem = _get_memory(user->username);
                auto results = mem.search_by_metadata(metadata_filter, limit, after, before);
                json results_json = json::array();
                for (const auto& r : results) {
                    results_json.push_back({{"id", r.id}, {"text", r.text},
                                            {"metadata", r.metadata}, {"timestamp", r.timestamp}});
                }
                json response = {{"results", results_json}, {"count", results.size()}};
                log_http("POST /search_by_metadata 200");
                res.set_content(response.dump(), "application/json");
            } catch (const json::exception& e) {
                log_http("POST /search_by_metadata 400");
                res.status = 400; res.set_content(std::string("JSON error: ") + e.what(), "text/plain");
            } catch (const std::exception& e) {
                log_http("POST /search_by_metadata 500");
                log_error(std::string("POST /search_by_metadata failed: ") + e.what());
                res.status = 500; res.set_content(std::string("Error: ") + e.what(), "text/plain");
            }
        });

        // PUT /user/model — set preferred model
        // PUT /user/model
        svr.Put("/user/model", [this](const httplib::Request& req, httplib::Response& res) {
            auto user = _check_auth(req);
            if (!user) { res.status = 401; res.set_content("Unauthorized", "text/plain"); return; }
            try {
                auto body = json::parse(req.body);
                std::string model = body.value("model", "");
                if (model.empty()) { res.status = 400; res.set_content("Missing 'model' field", "text/plain"); return; }
                std::string resolved = config().resolve_model(model);
                memory.backend()->update_user_preferred_model(user->username, resolved);
                preload_local_model(resolved);
                json response = {{"status", "updated"}, {"model", resolved}};
                log_http("PUT /user/model 200");
                res.set_content(response.dump(), "application/json");
            } catch (const json::exception& e) {
                res.status = 400; res.set_content(std::string("JSON error: ") + e.what(), "text/plain");
            } catch (const std::exception& e) {
                log_error(std::string("PUT /user/model failed: ") + e.what());
                res.status = 500; res.set_content(std::string("Error: ") + e.what(), "text/plain");
            }
        });

        // GET /user/model
        svr.Get("/user/model", [this](const httplib::Request& req, httplib::Response& res) {
            auto user = _check_auth(req);
            if (!user) { res.status = 401; res.set_content("Unauthorized", "text/plain"); return; }
            try {
                auto model_opt = memory.backend()->get_user_preferred_model(user->username);
                json response = model_opt ? json{{"model", *model_opt}} : json{{"model", nullptr}};
                log_http("GET /user/model 200");
                res.set_content(response.dump(), "application/json");
            } catch (const std::exception& e) {
                log_error(std::string("GET /user/model failed: ") + e.what());
                res.status = 500; res.set_content(std::string("Error: ") + e.what(), "text/plain");
            }
        });

        // DELETE /user/model
        svr.Delete("/user/model", [this](const httplib::Request& req, httplib::Response& res) {
            auto user = _check_auth(req);
            if (!user) { res.status = 401; res.set_content("Unauthorized", "text/plain"); return; }
            try {
                memory.backend()->update_user_preferred_model(user->username, "");
                log_http("DELETE /user/model 200");
                res.set_content(R"({"status":"cleared"})", "application/json");
            } catch (const std::exception& e) {
                log_error(std::string("DELETE /user/model failed: ") + e.what());
                res.status = 500; res.set_content(std::string("Error: ") + e.what(), "text/plain");
            }
        });

        // GET /user/token
        svr.Get("/user/token", [this](const httplib::Request& req, httplib::Response& res) {
            auto user = _check_auth(req);
            if (!user) { res.status = 401; res.set_content("Unauthorized", "text/plain"); return; }
            try {
                struct passwd* pw = getpwnam(user->username.c_str());
                if (!pw) { res.status = 404; res.set_content(R"({"error":"system user not found"})", "application/json"); return; }
                std::string token_file = std::string(pw->pw_dir) + "/.ragger/token";
                std::ifstream f(token_file);
                if (!f) { res.status = 404; res.set_content(R"({"error":"no token file"})", "application/json"); return; }
                std::string token;
                std::getline(f, token);
                size_t s = token.find_first_not_of(" \t\r\n");
                size_t e = token.find_last_not_of(" \t\r\n");
                if (s != std::string::npos) token = token.substr(s, e - s + 1);
                json response = {{"token", token}, {"username", user->username}};
                log_http("GET /user/token 200");
                res.set_content(response.dump(), "application/json");
            } catch (const std::exception& ex) {
                res.status = 500; res.set_content(std::string("Error: ") + ex.what(), "text/plain");
            }
        });

        // POST /user/rotate-token
        svr.Post("/user/rotate-token", [this](const httplib::Request& req, httplib::Response& res) {
            auto user = _check_auth(req);
            if (!user) { res.status = 401; res.set_content("Unauthorized", "text/plain"); return; }
            try {
                struct passwd* pw = getpwnam(user->username.c_str());
                if (!pw) { res.status = 404; res.set_content(R"({"error":"system user not found"})", "application/json"); return; }
                std::string new_token = ragger::generate_random_token();
                std::string token_path = std::string(pw->pw_dir) + "/.ragger/token";
                std::ofstream tf(token_path, std::ios::trunc);
                if (!tf) { res.status = 500; res.set_content(R"({"error":"cannot write token file"})", "application/json"); return; }
                tf << new_token << "\n";
                tf.close();
                std::string new_hash = ragger::hash_token(new_token);
                memory.backend()->update_user_token(user->username, new_hash);
                json response = {{"token", new_token}, {"username", user->username}, {"status", "rotated"}};
                log_http("POST /user/rotate-token 200");
                res.set_content(response.dump(), "application/json");
            } catch (const std::exception& ex) {
                log_error(std::string("rotate-token failed: ") + ex.what());
                res.status = 500; res.set_content(std::string("Error: ") + ex.what(), "text/plain");
            }
        });

        // POST /chat — memory-augmented chat with SSE streaming
        // POST /chat — memory-augmented chat with real SSE streaming
        svr.Post("/chat", [this](const httplib::Request& req, httplib::Response& res) {
            auto user = _check_auth(req);
            if (!user) {
                log_http("POST /chat 401");
                res.status = 401; res.set_content("Unauthorized", "text/plain"); return;
            }
            if (!inference_) {
                log_http("POST /chat 503");
                res.status = 503; res.set_content(R"({"error":"inference not configured"})", "application/json"); return;
            }

            try {
                auto body = json::parse(req.body);
                std::string message = body.value("message", "");
                if (message.empty()) {
                    res.status = 400; res.set_content(R"({"error":"message required"})", "application/json"); return;
                }

                std::string session_id = body.value("session_id", "");
                std::string request_model = body.value("model", "");
                auto& session = session_mgr_.get_or_create(session_id, user->username, memory.backend());

                // Search memory for context
                std::string memory_context;
                try {
                    int max_results = config().chat_max_memory_results;
                    auto& mem = _get_memory(user->username);
                    auto search_result = mem.search(message, max_results, 0.3f);
                    for (const auto& r : search_result.results) {
                        if (!memory_context.empty()) memory_context += "\n\n---\n\n";
                        memory_context += r.text;
                    }
                } catch (const std::exception& e) {
                    log_error(std::string("Memory search failed for /chat: ") + e.what());
                }

                // Resolve model
                auto preferred_model = memory.backend()->get_user_preferred_model(user->username);
                std::string use_model = preferred_model.value_or("");
                if (use_model.empty()) use_model = request_model;
                if (use_model.empty()) use_model = inference_->model;
                use_model = config().resolve_model(use_model);

                // Ensure model loaded
                auto load_err = inference_->ensure_model_loaded(use_model);
                if (!load_err.empty()) {
                    res.set_header("Content-Type", "text/event-stream");
                    res.set_header("Cache-Control", "no-cache");
                    std::string sse = "data: " + json{{"error", load_err}}.dump() + "\n\n"
                                    + "data: " + json{{"done", true}}.dump() + "\n\n";
                    res.set_content(sse, "text/event-stream");
                    log_http("POST /chat 200 (model load error)");
                    return;
                }

                // Build messages
                std::string system_prompt = ChatSessionManager::load_workspace_files();
                session.add_user_message(message);
                auto full_messages = session.build_messages(system_prompt, memory_context);

                // Capture variables for the chunked provider
                auto* inf = inference_.get();
                auto msg_copy = full_messages;
                auto model_copy = use_model;
                auto username = user->username;
                auto sid = session.session_id;
                auto msg_text = message;
                auto* backend_ptr = memory.backend();

                // Use chunked content provider for real SSE streaming
                res.set_header("Cache-Control", "no-cache");
                res.set_header("Connection", "keep-alive");
                res.set_header("X-Session-Id", sid);

                // We need to collect the full response for session/turn storage
                auto response_text = std::make_shared<std::string>();
                auto token_queue = std::make_shared<std::vector<std::string>>();
                auto done = std::make_shared<std::atomic<bool>>(false);
                auto mtx = std::make_shared<std::mutex>();
                auto cv = std::make_shared<std::condition_variable>();

                // Stream inference in a background thread, push tokens to queue
                std::thread([inf, msg_copy, model_copy, response_text, token_queue, done, mtx, cv, sid]() {
                    try {
                        inf->chat_stream(msg_copy, [&](const std::string& token) {
                            *response_text += token;
                            {
                                std::lock_guard<std::mutex> lock(*mtx);
                                token_queue->push_back(token);
                            }
                            cv->notify_one();
                        }, model_copy);
                    } catch (const std::exception& e) {
                        std::lock_guard<std::mutex> lock(*mtx);
                        token_queue->push_back(""); // sentinel for error
                    }
                    done->store(true);
                    cv->notify_one();
                }).detach();

                res.set_chunked_content_provider("text/event-stream",
                    [token_queue, done, mtx, cv, response_text, sid,
                     this, username, msg_text, backend_ptr](size_t /*offset*/, httplib::DataSink& sink) -> bool {
                        while (true) {
                            std::vector<std::string> batch;
                            {
                                std::unique_lock<std::mutex> lock(*mtx);
                                cv->wait_for(lock, std::chrono::milliseconds(50),
                                    [&] { return !token_queue->empty() || done->load(); });
                                batch.swap(*token_queue);
                            }

                            // Send any queued tokens
                            for (const auto& token : batch) {
                                if (token.empty() && done->load()) continue; // error sentinel
                                json event = {{"token", token}};
                                std::string sse = "data: " + event.dump() + "\n\n";
                                if (!sink.write(sse.data(), sse.size())) return false;
                            }

                            // If done and queue drained, send done event and finish
                            if (done->load()) {
                                // Drain any remaining
                                std::vector<std::string> remaining;
                                {
                                    std::lock_guard<std::mutex> lock(*mtx);
                                    remaining.swap(*token_queue);
                                }
                                for (const auto& token : remaining) {
                                    if (token.empty()) continue;
                                    json event = {{"token", token}};
                                    std::string sse = "data: " + event.dump() + "\n\n";
                                    if (!sink.write(sse.data(), sse.size())) return false;
                                }

                                // Done event
                                json done_event = {{"done", true}, {"session_id", sid}};
                                std::string sse = "data: " + done_event.dump() + "\n\n";
                                sink.write(sse.data(), sse.size());
                                sink.done();

                                // Post-stream: update session and store turn
                                if (!response_text->empty()) {
                                    auto& updated_session = session_mgr_.get_or_create(sid, username);
                                    updated_session.add_assistant_message(*response_text);
                                    
                                    // Save session to database for persistence
                                    try {
                                        json messages_array = json::array();
                                        for (const auto& msg : updated_session.messages) {
                                            messages_array.push_back({
                                                {"role", msg.role},
                                                {"content", msg.content}
                                            });
                                        }
                                        backend_ptr->save_chat_session(sid, username, messages_array.dump());
                                    } catch (const std::exception& e) {
                                        log_error(std::string("Session save failed: ") + e.what());
                                    }
                                    
                                    const auto& cfg = config();
                                    if (cfg.chat_store_turns != "false") {
                                        try {
                                            json turn_meta = {
                                                {"collection", "conversation"},
                                                {"category", "chat-turn"},
                                                {"source", "chat-http-" + username}
                                            };
                                            auto& mem = _get_memory(username);
                                            mem.store("User: " + msg_text + "\n\nAssistant: " + *response_text,
                                                      turn_meta);
                                        } catch (const std::exception& e) {
                                            log_error(std::string("Turn storage failed: ") + e.what());
                                        }
                                    }
                                }
                                log_http("POST /chat 200");
                                return false; // done streaming
                            }
                        }
                    }
                );

            } catch (const json::exception& e) {
                log_http("POST /chat 400");
                res.status = 400; res.set_content(std::string("JSON error: ") + e.what(), "text/plain");
            } catch (const std::exception& e) {
                log_http("POST /chat 500");
                log_error(std::string("POST /chat failed: ") + e.what());
                res.status = 500; res.set_content(std::string("Error: ") + e.what(), "text/plain");
            }
        });

        // POST /auth/login
        svr.Post("/auth/login", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                auto body = json::parse(req.body);
                std::string username = body.value("username", "");
                std::string password = body.value("password", "");
                if (username.empty() || password.empty()) {
                    res.status = 400; res.set_content(R"({"error":"username and password required"})", "application/json"); return;
                }
                auto user = memory.backend()->get_user_by_username(username);
                if (!user) { res.status = 401; res.set_content(R"({"error":"invalid credentials"})", "application/json"); return; }
                auto stored_hash = memory.backend()->get_user_password(username);
                if (!stored_hash) { res.status = 401; res.set_content(R"({"error":"no password set — use 'ragger passwd' first"})", "application/json"); return; }
                if (!verify_password(password, *stored_hash)) { res.status = 401; res.set_content(R"({"error":"invalid credentials"})", "application/json"); return; }
                std::string session_token = generate_random_token(32);
                memory.backend()->create_web_session(
                    session_token, username, user->id, WEB_SESSION_TTL);
                json result = {{"token", session_token}, {"username", username}, {"expires_in", WEB_SESSION_TTL}};
                log_http("POST /auth/login 200 (" + username + ")");
                res.set_content(result.dump(), "application/json");
            } catch (const std::exception& e) {
                log_error(std::string("Login error: ") + e.what());
                res.status = 500; res.set_content(R"({"error":"login failed"})", "application/json");
            }
        });

        // Static file serving — use httplib's built-in mount if web root exists
        // We set this up as a catch-all handler for unmatched GET requests
        svr.set_mount_point("/", resolve_web_root());

        // LM Proxy: OpenAI-compatible pass-through routes (if configured)
        if (inference_ && !inference_->lm_proxy_url().empty()) {
            setup_lm_proxy_routes();
        }
    }
};

// -----------------------------------------------------------------------
// LM Proxy: OpenAI-compatible pass-through routes
// -----------------------------------------------------------------------
void Server::Impl::setup_lm_proxy_routes() {
    // Helper: forward a POST to upstream, propagate status, sanitize errors
    auto forward_post = [this](const std::string& path,
                               const httplib::Request& req,
                               httplib::Response& res) {
        try {
            // Reject streaming requests — buffered proxy can't stream SSE
            auto body_json = json::parse(req.body, nullptr, false);
            if (!body_json.is_discarded() &&
                body_json.contains("stream") && body_json["stream"] == true) {
                res.status = 400;
                res.set_content(
                    R"({"error":"streaming not supported via proxy — connect directly to the inference service"})",
                    "application/json");
                log_http("POST " + path + " 400 (streaming rejected)");
                return;
            }

            auto resp = inference_->proxy_request(path, "POST", req.body);
            res.status = static_cast<int>(resp.status_code);
            log_http("POST " + path + " " + std::to_string(resp.status_code));
            res.set_content(resp.body, "application/json");
        } catch (const std::exception& e) {
            log_error("LM proxy " + path + " failed: " + e.what());
            res.status = 502;
            res.set_content(R"({"error":"upstream unavailable"})", "application/json");
        }
    };

    // GET /v1/models - list available models from upstream
    svr.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
        try {
            auto models = inference_->proxy_list_models();
            json response = {{"object", "list"}, {"data", json::array()}};
            for (const auto& model : models) {
                response["data"].push_back({{"id", model}, {"object", "model"}, {"owned_by", "lm-proxy"}});
            }
            log_http("GET /v1/models 200");
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            log_error(std::string("LM proxy /v1/models failed: ") + e.what());
            res.status = 502;
            res.set_content(R"({"error":"upstream unavailable"})", "application/json");
        }
    });

    // POST /v1/chat/completions - forward to upstream
    svr.Post("/v1/chat/completions", [this, forward_post](const httplib::Request& req, httplib::Response& res) {
        forward_post("/v1/chat/completions", req, res);
    });

    // POST /v1/completions - forward to upstream
    svr.Post("/v1/completions", [this, forward_post](const httplib::Request& req, httplib::Response& res) {
        forward_post("/v1/completions", req, res);
    });
}

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

void Server::run() {
    if (!is_port_available(pImpl->host, pImpl->port)) {
        log_error(std::string(lang::ERR_PORT_IN_USE_1) + std::to_string(pImpl->port) + lang::ERR_PORT_IN_USE_2);
        std::exit(1);
    }

    std::string addr = pImpl->host + ":" + std::to_string(pImpl->port);
    log_info(std::string(lang::MSG_SERVER_STARTING) + addr);
    log_info("  Health check: curl http://" + addr + "/health");
    
    const auto& cfg = config();

    // TLS support — if certs configured, create SSLServer instead
    // Note: httplib::Server is already created in Impl. For TLS we'd need
    // httplib::SSLServer. For now, TLS is handled via reverse proxy (Caddy/nginx).
    // TODO: To support native TLS, conditionally create SSLServer in Impl constructor.
    if (!cfg.tls_cert.empty() && !cfg.tls_key.empty()) {
        log_info("TLS config present — native TLS not yet supported with httplib. Use reverse proxy.");
    }

    // Write PID file (per-port)
    std::string port_str = std::to_string(pImpl->port);
    fs::create_directories("/tmp/ragger");
    pImpl->pid_file_ = "/tmp/ragger/server-" + port_str + ".pid";
    {
        std::ofstream pf(pImpl->pid_file_);
        if (pf) {
            pf << getpid();
            log_info("PID file: " + pImpl->pid_file_);
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

    // Start housekeeping timer (runs every 60s + on SIGUSR1)
    pImpl->start_housekeeping_timer();

    pImpl->svr.listen(pImpl->host, pImpl->port);

    // Cleanup
    pImpl->stop_housekeeping_timer();
    if (!pImpl->pid_file_.empty()) std::remove(pImpl->pid_file_.c_str());
}

void Server::stop() {
    pImpl->stop_housekeeping_timer();
    pImpl->svr.stop();
}

} // namespace ragger
