/**
 * HTTP server for Ragger Memory implementation
 */

#include "ragger/server.h"
#include "ragger/memory.h"
#include "ragger/lang.h"
#include "ragger/logs.h"
#include "ragger/auth.h"
#include "ragger/config.h"
#include "nlohmann_json.hpp"
#include "crow_all.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <pwd.h>
#include <sstream>
#include <thread>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

namespace ragger {

using json = nlohmann::json;

struct Server::Impl {
    crow::SimpleApp app;
    RaggerMemory&   memory;
    std::string     host;
    int             port;
    std::string     server_token_;
    std::optional<SqliteBackend::UserInfo> default_user_;

    Impl(RaggerMemory& mem, const std::string& h, int p)
        : memory(mem), host(h), port(p)
    {
        bootstrap_auth();
        setup_routes();
    }

    void bootstrap_auth() {
        // Ensure token exists (create if needed)
        server_token_ = ensure_token();
        
        if (!server_token_.empty()) {
            // Hash token and check if user exists
            std::string token_hash = hash_token(server_token_);
            auto user = memory.backend()->get_user_by_token_hash(token_hash);
            
            if (!user) {
                // Create default user
                std::string username = "default";
                int user_id = memory.backend()->create_user(username, token_hash, true);
                user = SqliteBackend::UserInfo{user_id, username, true, token_hash, ""};
                std::cerr << "Created default user: " << username << " (id=" << user_id << ")\n";
            }
            default_user_ = user;
        }
    }

    std::optional<SqliteBackend::UserInfo> _check_auth(const crow::request& req) {
        // If no token configured, auth is disabled
        if (server_token_.empty()) {
            return SqliteBackend::UserInfo{0, "anonymous", false, "", ""};
        }

        // Extract Authorization header
        auto auth_header = req.get_header_value("Authorization");
        if (auth_header.empty()) {
            return std::nullopt;
        }

        // Parse "Bearer <token>"
        const std::string bearer_prefix = "Bearer ";
        if (auth_header.substr(0, bearer_prefix.size()) != bearer_prefix) {
            return std::nullopt;
        }
        std::string token = auth_header.substr(bearer_prefix.size());

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
                    std::cerr << "Rotated token for user: " << username << std::endl;
                } catch (const std::exception& e) {
                    log_error(std::string("Token rotation failed for ") + username + ": " + e.what());
                }
            }).detach();
        }
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
            json response = {
                {"count", memory.count()}
            };
            if (memory.is_multi_db()) {
                response["user"] = memory.user_backend()->count();
                response["common"] = memory.backend()->count();
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

                bool common = body.value("common", false);
                std::string id = memory.store(text, metadata, common);

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
                
                SearchResponse search_response = memory.search(
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
                bool deleted = memory.delete_memory(id);
                
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
                int deleted = memory.delete_batch(ids);

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

                auto results = memory.search_by_metadata(metadata_filter, limit);

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
                backend->update_user_preferred_model(user->username, model);
                
                json response = {
                    {"status", "updated"},
                    {"model", model}
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
        std::cerr << lang::ERR_PORT_IN_USE_1 << pImpl->port
                  << lang::ERR_PORT_IN_USE_2 << std::endl;
        std::exit(1);
    }

    std::cout << lang::MSG_SERVER_STARTING 
              << pImpl->host << ":" << pImpl->port << std::endl;
    
    pImpl->app.loglevel(crow::LogLevel::Warning)
              .bindaddr(pImpl->host)
              .port(pImpl->port)
              .multithreaded()
              .run();
}

void Server::stop() {
    pImpl->app.stop();
}

} // namespace ragger
