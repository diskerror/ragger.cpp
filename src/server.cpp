/**
 * HTTP server for Ragger Memory implementation
 */

#include "ragger/server.h"
#include "ragger/memory.h"
#include "ragger/lang.h"
#include "ragger/logs.h"
#include "ragger/auth.h"
#include "ragger/config.h"
#include "ragger/inference.h"
#include "ragger/chat_sessions.h"
#include "nlohmann_json.hpp"
#include "crow_all.h"

#include <chrono>
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
#include <sqlite3.h>
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
    crow::SimpleApp app;
    RaggerMemory&   memory;
    std::string     host;
    int             port;
    std::string     server_token_;
    std::optional<SqliteBackend::UserInfo> default_user_;
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

        // 1. Expire idle chat sessions
        auto expired = session_mgr_.cleanup_expired(pause_minutes);
        int sessions_expired = (int)expired.size();

        // TODO: summarize expired session turns via inference (like Python)

        // 2. Purge old conversation entries from all known user DBs
        int conversations_cleaned = 0;
        if (max_age_hours > 0) {
            auto now = std::chrono::system_clock::now();
            auto cutoff = now - std::chrono::duration_cast<std::chrono::system_clock::duration>(
                std::chrono::duration<double, std::ratio<3600>>(max_age_hours));
            auto cutoff_t = std::chrono::system_clock::to_time_t(cutoff);
            char cutoff_str[32];
            std::strftime(cutoff_str, sizeof(cutoff_str), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&cutoff_t));

            // Collect user DB paths only for users we hold housekeeping locks for
            std::vector<std::string> db_paths;
            {
                std::lock_guard<std::mutex> hk_lock(housekeeping_locks_mutex_);
                std::lock_guard<std::mutex> mem_lock(user_memories_mutex_);
                for (auto& [username, mem] : user_memories_) {
                    if (mem && mem->user_backend() && housekeeping_locks_.count(username)) {
                        db_paths.push_back(mem->user_backend()->db_path());
                    }
                }
            }

            for (const auto& db_path : db_paths) {
                try {
                    sqlite3* db = nullptr;
                    if (sqlite3_open(db_path.c_str(), &db) != SQLITE_OK) continue;
                    std::string sql = "DELETE FROM memories WHERE collection = 'conversation' AND timestamp < ?";
                    sqlite3_stmt* stmt = nullptr;
                    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
                        sqlite3_bind_text(stmt, 1, cutoff_str, -1, SQLITE_STATIC);
                        sqlite3_step(stmt);
                        int deleted = sqlite3_changes(db);
                        conversations_cleaned += deleted;
                        if (deleted > 0) {
                            log_info("Cleaned " + std::to_string(deleted)
                                   + " expired conversations from " + db_path);
                        }
                    }
                    sqlite3_finalize(stmt);
                    sqlite3_close(db);
                } catch (const std::exception& e) {
                    log_error("Cleanup failed for " + db_path + ": " + e.what());
                }
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
    }

    void bootstrap_auth() {
        const auto& cfg = config();
        
        if (cfg.single_user) {
            // Single-user mode: ensure token exists, create default user if needed
            server_token_ = ensure_token();
            
            if (!server_token_.empty()) {
                std::string token_hash = hash_token(server_token_);
                auto user = memory.backend()->get_user_by_token_hash(token_hash);
                
                if (!user) {
                    // Auto-create the single user
                    std::string username = "default";
                    // Try to use actual username
                    struct passwd* pw = getpwuid(getuid());
                    if (pw) username = pw->pw_name;
                    
                    int user_id = memory.backend()->create_user(username, token_hash, true);
                    user = SqliteBackend::UserInfo{user_id, username, true, token_hash, ""};
                    log_info("Created user: " + username + " (id=" + std::to_string(user_id) + ")");
                }
                default_user_ = user;
            }
        } else {
            // Multi-user mode: don't create tokens or default users.
            // Users are provisioned via install.sh / add-user.
            // Auth is validated per-request against the common DB.
            log_info("Multi-user mode: auth via provisioned user tokens");
        }
    }

    // --- Web sessions (password login) ---
    struct WebSession {
        std::string username;
        int user_id;
        bool is_admin;
        std::chrono::steady_clock::time_point expires;
    };
    std::unordered_map<std::string, WebSession> web_sessions_;
    std::mutex web_sessions_mutex_;
    static constexpr int WEB_SESSION_TTL = 86400; // 24 hours

    std::optional<SqliteBackend::UserInfo> _check_web_session(const std::string& token) {
        std::lock_guard<std::mutex> lock(web_sessions_mutex_);
        auto it = web_sessions_.find(token);
        if (it == web_sessions_.end()) return std::nullopt;
        if (std::chrono::steady_clock::now() > it->second.expires) {
            web_sessions_.erase(it);
            return std::nullopt;
        }
        return SqliteBackend::UserInfo{
            it->second.user_id, it->second.username,
            it->second.is_admin, "", ""
        };
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

    std::optional<SqliteBackend::UserInfo> _check_auth(const crow::request& req) {
        const auto& cfg = config();
        
        // Extract Authorization header
        auto auth_header = req.get_header_value("Authorization");
        
        // If single-user mode with no token configured, auth is disabled
        if (cfg.single_user && server_token_.empty()) {
            return SqliteBackend::UserInfo{0, "anonymous", false, "", ""};
        }
        
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

        // Hash and lookup in database (works for both modes)
        std::string token_hash = hash_token(token);
        auto user = memory.backend()->get_user_by_token_hash(token_hash);
        if (user) {
            // Check if token rotation is needed (async, after this request)
            _check_token_rotation(*user);
            return user;
        }

        // Fallback: direct comparison with server token (single-user only)
        if (cfg.single_user && token == server_token_ && default_user_) {
            return default_user_;
        }

        return std::nullopt;
    }

    void _check_token_rotation(const SqliteBackend::UserInfo& user) {
        const auto& cfg = config();
        if (cfg.token_rotation_minutes <= 0) return;  // Rotation disabled

        auto* backend = memory.backend();
        auto rotated_at_opt = backend->get_user_token_rotated_at(user.username);
        
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
            backend->update_user_token_rotated_at(user.username, now_str);
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
                    auto* backend = memory.backend();
                    backend->update_user_token(username, new_hash);
                    backend->update_user_token_rotated_at(username, now_str);
                    log_info("Rotated token for user: " + std::string(username));
                } catch (const std::exception& e) {
                    log_error(std::string("Token rotation failed for ") + username + ": " + e.what());
                }
            }).detach();
        }
    }

    /// Get per-user memory (or fallback to common).
    /// In single-user mode, always returns the main memory instance.
    RaggerMemory& _get_memory(const std::string& username) {
        if (config().single_user) return memory;

        std::lock_guard<std::mutex> lock(user_memories_mutex_);
        auto it = user_memories_.find(username);
        if (it != user_memories_.end()) {
            return it->second ? *it->second : memory;
        }

        // Try to open user's private DB
        auto user_mem = memory.for_user(username);
        if (!user_mem) {
            // Cache the miss so we don't retry every request
            user_memories_[username] = nullptr;
            return memory;
        }

        // Try to acquire housekeeping lock for this user (only if housekeeping enabled)
        if (config().housekeeping_interval > 0) {
            acquire_user_housekeeping_lock(username);
        }

        auto& ref = *user_mem;
        user_memories_[username] = std::move(user_mem);
        return ref;
    }

    void setup_routes() {
        // GET /health
        CROW_ROUTE(app, "/health")
        ([this]() {
            json response = {
                {"status", "ok"},
                {"version", RAGGER_VERSION},
                {"commit", RAGGER_COMMIT},
                {"built", RAGGER_BUILD_DATE},
                {"memories", memory.count()}
            };
            log_http("GET /health 200");
            return crow::response(response.dump());
        });

        // GET /count
        CROW_ROUTE(app, "/count")
        ([this](const crow::request& req) {
            auto user = _check_auth(req);
            if (!user) {
                log_http("GET /count 401");
                return crow::response(401, "Unauthorized");
            }
            auto& mem = _get_memory(user->username);
            json response = {
                {"count", mem.count()}
            };
            if (mem.is_multi_db()) {
                response["user"] = mem.user_backend()->count();
                response["common"] = mem.backend()->count();
            }
            log_http("GET /count 200");
            return crow::response(response.dump());
        });

        // POST /store
        CROW_ROUTE(app, "/store").methods(crow::HTTPMethod::POST)
        ([this](const crow::request& req) {
            // Auth check
            auto user = _check_auth(req);
            if (!user) {
                log_http("POST /store 401");
                return crow::response(401, "Unauthorized");
            }

            try {
                auto body = json::parse(req.body);
                
                std::string text = body.value("text", "");
                json metadata = body.value("metadata", json::object());

                if (text.empty()) {
                    log_http("POST /store 400");
                    return crow::response(400, "Missing 'text' field");
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
                return crow::response(response.dump());

            } catch (const json::exception& e) {
                log_http("POST /store 400");
                return crow::response(400, std::string("JSON error: ") + e.what());
            } catch (const std::exception& e) {
                log_http("POST /store 500");
                log_error(std::string("POST /store failed: ") + e.what());
                return crow::response(500, std::string("Error: ") + e.what());
            }
        });

        // POST /search
        CROW_ROUTE(app, "/search").methods(crow::HTTPMethod::POST)
        ([this](const crow::request& req) {
            auto user = _check_auth(req);
            if (!user) {
                log_http("POST /search 401");
                return crow::response(401, "Unauthorized");
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
                    return crow::response(400, "Missing 'query' field");
                }

                auto start_time = std::chrono::high_resolution_clock::now();
                
                auto& mem = _get_memory(user->username);
                SearchResponse search_response = mem.search(
                    query, limit, min_score, collections
                );

                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_time - start_time
                );

                // Build results array
                json results = json::array();
                for (const auto& result : search_response.results) {
                    json result_obj = {
                        {"id", result.id},
                        {"text", result.text},
                        {"score", result.score},
                        {"metadata", result.metadata},
                        {"timestamp", result.timestamp}
                    };
                    results.push_back(result_obj);
                }

                // Add timing from backend plus total request time
                json timing = search_response.timing;
                timing["total_ms"] = duration.count();

                json response = {
                    {"results", results},
                    {"timing", timing}
                };

                // Log query
                std::ostringstream query_log_msg;
                query_log_msg << "query=\"" << query << "\" "
                             << "results=" << search_response.results.size() << " "
                             << "time=" << duration.count() << "ms";
                log_query(query_log_msg.str());

                log_http("POST /search 200");
                return crow::response(response.dump());

            } catch (const json::exception& e) {
                log_http("POST /search 400");
                return crow::response(400, std::string("JSON error: ") + e.what());
            } catch (const std::exception& e) {
                log_http("POST /search 500");
                log_error(std::string("POST /search failed: ") + e.what());
                return crow::response(500, std::string("Error: ") + e.what());
            }
        });

        // DELETE /memory/<id>
        CROW_ROUTE(app, "/memory/<int>").methods(crow::HTTPMethod::DELETE)
        ([this](const crow::request& req, int id) {
            // Auth check
            auto user = _check_auth(req);
            if (!user) {
                log_http("DELETE /memory/<int> 401");
                return crow::response(401, "Unauthorized");
            }

            try {
                auto& mem = _get_memory(user->username);
                bool deleted = mem.delete_memory(id);
                
                if (deleted) {
                    json response = {
                        {"id", id},
                        {"status", "deleted"}
                    };
                    log_http("DELETE /memory/<int> 200");
                    return crow::response(200, response.dump());
                } else {
                    log_http("DELETE /memory/<int> 404");
                    return crow::response(404, "Memory not found");
                }
            } catch (const std::exception& e) {
                log_http("DELETE /memory/<int> 500");
                log_error(std::string("DELETE /memory/<int> failed: ") + e.what());
                return crow::response(500, std::string("Error: ") + e.what());
            }
        });

        // POST /delete_batch
        CROW_ROUTE(app, "/delete_batch").methods(crow::HTTPMethod::POST)
        ([this](const crow::request& req) {
            // Auth check
            auto user = _check_auth(req);
            if (!user) {
                log_http("POST /delete_batch 401");
                return crow::response(401, "Unauthorized");
            }

            try {
                auto body = json::parse(req.body);
                
                if (!body.contains("ids") || !body["ids"].is_array()) {
                    log_http("POST /delete_batch 400");
                    return crow::response(400, "Missing or invalid 'ids' field");
                }

                std::vector<int> ids = body["ids"].get<std::vector<int>>();
                auto& mem = _get_memory(user->username);
                int deleted = mem.delete_batch(ids);

                json response = {
                    {"deleted", deleted}
                };
                log_http("POST /delete_batch 200");
                return crow::response(response.dump());

            } catch (const json::exception& e) {
                log_http("POST /delete_batch 400");
                return crow::response(400, std::string("JSON error: ") + e.what());
            } catch (const std::exception& e) {
                log_http("POST /delete_batch 500");
                log_error(std::string("POST /delete_batch failed: ") + e.what());
                return crow::response(500, std::string("Error: ") + e.what());
            }
        });

        // POST /search_by_metadata
        CROW_ROUTE(app, "/search_by_metadata").methods(crow::HTTPMethod::POST)
        ([this](const crow::request& req) {
            auto user = _check_auth(req);
            if (!user) {
                log_http("POST /search_by_metadata 401");
                return crow::response(401, "Unauthorized");
            }
            try {
                auto body = json::parse(req.body);
                
                if (!body.contains("metadata") || !body["metadata"].is_object()) {
                    log_http("POST /search_by_metadata 400");
                    return crow::response(400, "Missing or invalid 'metadata' field");
                }

                json metadata_filter = body["metadata"];
                int limit = body.value("limit", 0);

                auto& mem = _get_memory(user->username);
                auto results = mem.search_by_metadata(metadata_filter, limit);

                // Build results array
                json results_json = json::array();
                for (const auto& result : results) {
                    json result_obj = {
                        {"id", result.id},
                        {"text", result.text},
                        {"metadata", result.metadata},
                        {"timestamp", result.timestamp}
                    };
                    results_json.push_back(result_obj);
                }

                json response = {
                    {"results", results_json},
                    {"count", results.size()}
                };

                log_http("POST /search_by_metadata 200");
                return crow::response(response.dump());

            } catch (const json::exception& e) {
                log_http("POST /search_by_metadata 400");
                return crow::response(400, std::string("JSON error: ") + e.what());
            } catch (const std::exception& e) {
                log_http("POST /search_by_metadata 500");
                log_error(std::string("POST /search_by_metadata failed: ") + e.what());
                return crow::response(500, std::string("Error: ") + e.what());
            }
        });

        // PUT /user/model — set preferred model
        CROW_ROUTE(app, "/user/model").methods(crow::HTTPMethod::PUT)
        ([this](const crow::request& req) {
            auto user = _check_auth(req);
            if (!user) {
                log_http("PUT /user/model 401");
                return crow::response(401, "Unauthorized");
            }
            try {
                auto body = json::parse(req.body);
                std::string model = body.value("model", "");
                if (model.empty()) {
                    log_http("PUT /user/model 400");
                    return crow::response(400, "Missing 'model' field");
                }
                
                auto* backend = memory.backend();
                // Resolve alias before storing
                std::string resolved = config().resolve_model(model);
                backend->update_user_preferred_model(user->username, resolved);

                // Preload on local engines
                preload_local_model(resolved);
                
                json response = {
                    {"status", "updated"},
                    {"model", resolved}
                };
                log_http("PUT /user/model 200");
                return crow::response(200, response.dump());
                
            } catch (const json::exception& e) {
                log_http("PUT /user/model 400");
                return crow::response(400, std::string("JSON error: ") + e.what());
            } catch (const std::exception& e) {
                log_http("PUT /user/model 500");
                log_error(std::string("PUT /user/model failed: ") + e.what());
                return crow::response(500, std::string("Error: ") + e.what());
            }
        });

        // GET /user/model — get preferred model
        CROW_ROUTE(app, "/user/model").methods(crow::HTTPMethod::GET)
        ([this](const crow::request& req) {
            auto user = _check_auth(req);
            if (!user) {
                log_http("GET /user/model 401");
                return crow::response(401, "Unauthorized");
            }
            try {
                auto* backend = memory.backend();
                auto model_opt = backend->get_user_preferred_model(user->username);
                
                json response;
                if (model_opt) {
                    response = {
                        {"model", *model_opt}
                    };
                } else {
                    response = {
                        {"model", nullptr}
                    };
                }
                log_http("GET /user/model 200");
                return crow::response(200, response.dump());
                
            } catch (const std::exception& e) {
                log_http("GET /user/model 500");
                log_error(std::string("GET /user/model failed: ") + e.what());
                return crow::response(500, std::string("Error: ") + e.what());
            }
        });

        // DELETE /user/model — clear preferred model
        CROW_ROUTE(app, "/user/model").methods(crow::HTTPMethod::DELETE)
        ([this](const crow::request& req) {
            auto user = _check_auth(req);
            if (!user) {
                log_http("DELETE /user/model 401");
                return crow::response(401, "Unauthorized");
            }
            try {
                auto* backend = memory.backend();
                backend->update_user_preferred_model(user->username, "");
                
                json response = {
                    {"status", "cleared"}
                };
                log_http("DELETE /user/model 200");
                return crow::response(200, response.dump());
                
            } catch (const std::exception& e) {
                log_http("DELETE /user/model 500");
                log_error(std::string("DELETE /user/model failed: ") + e.what());
                return crow::response(500, std::string("Error: ") + e.what());
            }
        });

        // POST /chat — memory-augmented chat with SSE streaming
        CROW_ROUTE(app, "/chat").methods(crow::HTTPMethod::POST)
        ([this](const crow::request& req) -> crow::response {
            auto user = _check_auth(req);
            if (!user) {
                log_http("POST /chat 401");
                return crow::response(401, "Unauthorized");
            }

            if (!inference_) {
                log_http("POST /chat 503");
                return crow::response(503, "{\"error\": \"inference not configured\"}");
            }

            try {
                auto body = json::parse(req.body);
                std::string message = body.value("message", "");
                if (message.empty()) {
                    log_http("POST /chat 400");
                    return crow::response(400, "{\"error\": \"message required\"}");
                }

                std::string session_id = body.value("session_id", "");
                std::string request_model = body.value("model", "");

                // Get or create session
                auto& session = session_mgr_.get_or_create(session_id, user->username);

                // Search memory for context
                std::string memory_context;
                try {
                    const auto& cfg = config();
                    int max_results = cfg.chat_max_memory_results;
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
                auto* backend = memory.backend();
                auto preferred_model = backend->get_user_preferred_model(user->username);
                std::string use_model = preferred_model.value_or("");
                if (use_model.empty()) use_model = request_model;
                if (use_model.empty()) use_model = inference_->model;

                // Resolve alias
                use_model = config().resolve_model(use_model);

                // Ensure model is loaded (auto-load for local engines)
                auto load_err = inference_->ensure_model_loaded(use_model);
                if (!load_err.empty()) {
                    json err_event = {{"error", load_err}};
                    json done_event = {{"done", true}};
                    std::string sse = "data: " + err_event.dump() + "\n\n"
                                    + "data: " + done_event.dump() + "\n\n";
                    auto resp = crow::response(200, sse);
                    resp.set_header("Content-Type", "text/event-stream");
                    resp.set_header("Cache-Control", "no-cache");
                    resp.set_header("Connection", "close");
                    log_http("POST /chat 200 (model load error)");
                    return resp;
                }

                // Build messages with persona + memory + history
                std::string system_prompt = ChatSessionManager::load_workspace_files();
                session.add_user_message(message);
                auto full_messages = session.build_messages(system_prompt, memory_context);

                // Stream response from inference, collect SSE body
                std::string sse_body;
                std::string response_text;

                inference_->chat_stream(full_messages, [&](const std::string& token) {
                    response_text += token;
                    json event = {{"token", token}};
                    sse_body += "data: " + event.dump() + "\n\n";
                }, use_model);

                // Done event
                json done_event = {
                    {"done", true},
                    {"session_id", session.session_id}
                };
                sse_body += "data: " + done_event.dump() + "\n\n";

                // Update session
                if (!response_text.empty()) {
                    session.add_assistant_message(response_text);

                    // Store turn if configured
                    const auto& cfg = config();
                    if (cfg.chat_store_turns != "false") {
                        try {
                            json turn_meta = {
                                {"collection", "conversation"},
                                {"category", "chat-turn"},
                                {"source", "chat-http-" + user->username}
                            };
                            auto& mem = _get_memory(user->username);
                            mem.store(
                                "User: " + message + "\n\nAssistant: " + response_text,
                                turn_meta
                            );
                        } catch (const std::exception& e) {
                            log_error(std::string("Turn storage failed: ") + e.what());
                        }
                    }
                }

                // Cleanup expired sessions in background
                const auto& cfg = config();
                auto expired = session_mgr_.cleanup_expired(cfg.chat_pause_minutes);
                // TODO: summarize expired sessions in background

                crow::response res(200);
                res.set_header("Content-Type", "text/event-stream");
                res.set_header("Cache-Control", "no-cache");
                res.set_header("X-Session-Id", session.session_id);
                res.body = sse_body;
                log_http("POST /chat 200");
                return res;

            } catch (const json::exception& e) {
                log_http("POST /chat 400");
                return crow::response(400, std::string("JSON error: ") + e.what());
            } catch (const std::exception& e) {
                log_http("POST /chat 500");
                log_error(std::string("POST /chat failed: ") + e.what());
                return crow::response(500, std::string("Error: ") + e.what());
            }
        });

        // POST /register — register a user in the common DB
        CROW_ROUTE(app, "/register").methods(crow::HTTPMethod::POST)
        ([this](const crow::request& req) -> crow::response {
            try {
                auto body = json::parse(req.body);
                std::string username = body.value("username", "");
                if (username.empty()) {
                    log_http("POST /register 400");
                    return crow::response(400, "{\"error\": \"username required\"}");
                }

                // Extract bearer token
                auto auth_header = req.get_header_value("Authorization");
                const std::string bearer_prefix = "Bearer ";
                if (auth_header.size() <= bearer_prefix.size() ||
                    auth_header.substr(0, bearer_prefix.size()) != bearer_prefix) {
                    log_http("POST /register 401");
                    return crow::response(401, "{\"error\": \"bearer token required\"}");
                }
                std::string provided_token = auth_header.substr(bearer_prefix.size());

                // Verify: read token from user's file on filesystem
                struct passwd* pw = getpwnam(username.c_str());
                if (!pw) {
                    log_http("POST /register 400");
                    return crow::response(400,
                        "{\"error\": \"unknown system user: " + username + "\"}");
                }
                std::string user_token_file =
                    std::string(pw->pw_dir) + "/.ragger/token";
                if (!std::filesystem::exists(user_token_file)) {
                    log_http("POST /register 400");
                    return crow::response(400,
                        "{\"error\": \"no token file for " + username + "\"}");
                }
                std::ifstream f(user_token_file);
                std::string file_token;
                std::getline(f, file_token);
                // trim
                size_t s = file_token.find_first_not_of(" \t\r\n");
                size_t e = file_token.find_last_not_of(" \t\r\n");
                if (s != std::string::npos)
                    file_token = file_token.substr(s, e - s + 1);

                if (provided_token != file_token) {
                    log_http("POST /register 403");
                    return crow::response(403,
                        "{\"error\": \"token does not match user's token file\"}");
                }

                // Register in DB
                std::string hashed = hash_token(provided_token);
                auto* backend = memory.backend();
                auto existing = backend->get_user_by_username(username);
                json result;
                if (existing) {
                    if (existing->token_hash != hashed) {
                        backend->update_user_token(username, hashed);
                    }
                    result = {{"status", "exists"}, {"user_id", existing->id},
                              {"username", username}};
                } else {
                    bool is_admin = body.value("is_admin", false);
                    int user_id = backend->create_user(username, hashed, is_admin);
                    result = {{"status", "created"}, {"user_id", user_id},
                              {"username", username}};
                }
                log_http("POST /register 200");
                return crow::response(200, result.dump());
            } catch (const std::exception& e) {
                log_http("POST /register 500");
                log_error(std::string("POST /register failed: ") + e.what());
                return crow::response(500, std::string("Error: ") + e.what());
            }
        });

        // POST /auth/login — password authentication → session token
        CROW_ROUTE(app, "/auth/login").methods(crow::HTTPMethod::POST)
        ([this](const crow::request& req) -> crow::response {
            try {
                auto body = json::parse(req.body);
                std::string username = body.value("username", "");
                std::string password = body.value("password", "");

                if (username.empty() || password.empty()) {
                    return crow::response(400, R"({"error":"username and password required"})");
                }

                auto user = memory.backend()->get_user_by_username(username);
                if (!user) {
                    return crow::response(401, R"({"error":"invalid credentials"})");
                }

                auto stored_hash = memory.backend()->get_user_password(username);
                if (!stored_hash) {
                    return crow::response(401, R"({"error":"no password set — use 'ragger passwd' first"})");
                }

                if (!verify_password(password, *stored_hash)) {
                    return crow::response(401, R"({"error":"invalid credentials"})");
                }

                // Generate session token
                std::string session_token = generate_random_token(32);
                {
                    std::lock_guard<std::mutex> lock(web_sessions_mutex_);
                    web_sessions_[session_token] = WebSession{
                        username, user->id, user->is_admin,
                        std::chrono::steady_clock::now() + std::chrono::seconds(WEB_SESSION_TTL)
                    };
                }

                json result = {
                    {"token", session_token},
                    {"username", username},
                    {"expires_in", WEB_SESSION_TTL}
                };
                log_http("POST /auth/login 200 (" + username + ")");
                return crow::response(200, result.dump());
            } catch (const std::exception& e) {
                log_error(std::string("Login error: ") + e.what());
                return crow::response(500, R"({"error":"login failed"})");
            }
        });

        // Static file serving (web UI) — no auth required
        CROW_ROUTE(app, "/<path>")
        ([this](const crow::request& req, const std::string& path) -> crow::response {
            std::string web_root = resolve_web_root();
            if (web_root.empty()) {
                return crow::response(404, "Not found");
            }

            // Map / to index.html
            std::string safe_path = path.empty() ? "index.html" : path;

            // Prevent directory traversal
            if (safe_path.find("..") != std::string::npos) {
                return crow::response(403, "Forbidden");
            }

            fs::path file_path = fs::path(web_root) / safe_path;
            if (!fs::is_regular_file(file_path)) {
                return crow::response(404, "Not found");
            }

            std::ifstream file(file_path, std::ios::binary);
            if (!file) {
                return crow::response(500, "Cannot read file");
            }

            std::ostringstream ss;
            ss << file.rdbuf();
            auto resp = crow::response(200, ss.str());
            resp.set_header("Content-Type", mime_type(file_path.string()));
            return resp;
        });

        // Root path
        CROW_ROUTE(app, "/")
        ([this](const crow::request& req) -> crow::response {
            std::string web_root = resolve_web_root();
            if (web_root.empty()) {
                return crow::response(200, R"({"status":"ok","message":"Ragger API"})");
            }

            fs::path file_path = fs::path(web_root) / "index.html";
            if (!fs::is_regular_file(file_path)) {
                return crow::response(200, R"({"status":"ok","message":"Ragger API"})");
            }

            std::ifstream file(file_path, std::ios::binary);
            std::ostringstream ss;
            ss << file.rdbuf();
            auto resp = crow::response(200, ss.str());
            resp.set_header("Content-Type", "text/html; charset=utf-8");
            return resp;
        });
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

void Server::run() {
    if (!is_port_available(pImpl->host, pImpl->port)) {
        log_error(std::string(lang::ERR_PORT_IN_USE_1) + std::to_string(pImpl->port) + lang::ERR_PORT_IN_USE_2);
        std::exit(1);
    }

    std::string addr = pImpl->host + ":" + std::to_string(pImpl->port);
    log_info(std::string(lang::MSG_SERVER_STARTING) + addr);
    log_info("  Health check: curl http://" + addr + "/health");
    
    const auto& cfg = config();

    auto& a = pImpl->app;
    a.loglevel(crow::LogLevel::Warning)
     .bindaddr(pImpl->host)
     .port(pImpl->port)
     .multithreaded();

    if (!cfg.server_name.empty()) {
        a.server_name(cfg.server_name);
    }

    // TLS support
    if (!cfg.tls_cert.empty() && !cfg.tls_key.empty()) {
        auto cert_path = expand_path(cfg.tls_cert);
        auto key_path = expand_path(cfg.tls_key);
        if (fs::exists(cert_path) && fs::exists(key_path)) {
            a.ssl_file(cert_path, key_path);
            log_info("TLS enabled: " + cert_path);
        } else {
            log_error("TLS certificates not found — starting without encryption");
            if (!fs::exists(cert_path)) log_error("  Missing: " + cert_path);
            if (!fs::exists(key_path))  log_error("  Missing: " + key_path);
        }
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

    a.run();

    // Cleanup
    pImpl->stop_housekeeping_timer();
    if (!pImpl->pid_file_.empty()) std::remove(pImpl->pid_file_.c_str());
}

void Server::stop() {
    pImpl->stop_housekeeping_timer();
    pImpl->app.stop();
}

} // namespace ragger
