/**
 * HTTP server for Ragger Memory implementation
 */

#include "ragger/server.h"
#include "ragger/memory.h"
#include "ragger/lang.h"
#include "ragger/logs.h"
#include "ragger/auth.h"
#include "nlohmann_json.hpp"
#include "crow_all.h"

#include <chrono>
#include <iostream>
#include <optional>
#include <sstream>
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
        // Load server token
        server_token_ = load_token();
        
        if (!server_token_.empty()) {
            // Hash token and check if user exists
            std::string token_hash = hash_token(server_token_);
            auto user = memory.backend()->get_user_by_token_hash(token_hash);
            
            if (!user) {
                // Create default user
                std::string username = "default";
                int user_id = memory.backend()->create_user(username, token_hash, true);
                user = SqliteBackend::UserInfo{user_id, username, true, token_hash};
                std::cerr << "Created default user: " << username << " (id=" << user_id << ")\n";
            }
            default_user_ = user;
        }
    }

    std::optional<SqliteBackend::UserInfo> _check_auth(const crow::request& req) {
        // If no token configured, auth is disabled
        if (server_token_.empty()) {
            return SqliteBackend::UserInfo{0, "anonymous", false, ""};
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
            return user;
        }

        // Fallback: direct comparison with server token
        if (token == server_token_ && default_user_) {
            return default_user_;
        }

        return std::nullopt;
    }

    void setup_routes() {
        // GET /health
        CROW_ROUTE(app, "/health")
        ([this]() {
            json response = {
                {"status", "ok"},
                {"version", RAGGER_VERSION},
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
    
    pImpl->app.bindaddr(pImpl->host)
              .port(pImpl->port)
              .multithreaded()
              .run();
}

void Server::stop() {
    pImpl->app.stop();
}

} // namespace ragger
