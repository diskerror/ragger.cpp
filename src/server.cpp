/**
 * HTTP server for Ragger Memory implementation
 */

#include "ragger/server.h"
#include "ragger/memory.h"
#include "nlohmann_json.hpp"
#include "crow_all.h"

#include <chrono>
#include <iostream>

namespace ragger {

using json = nlohmann::json;

struct Server::Impl {
    crow::SimpleApp app;
    RaggerMemory&   memory;
    std::string     host;
    int             port;

    Impl(RaggerMemory& mem, const std::string& h, int p)
        : memory(mem), host(h), port(p)
    {
        setup_routes();
    }

    void setup_routes() {
        // GET /health
        CROW_ROUTE(app, "/health")
        ([this]() {
            json response = {
                {"status", "ok"},
                {"memories", memory.count()}
            };
            return crow::response(response.dump());
        });

        // GET /count
        CROW_ROUTE(app, "/count")
        ([this]() {
            json response = {
                {"count", memory.count()}
            };
            return crow::response(response.dump());
        });

        // POST /store
        CROW_ROUTE(app, "/store").methods(crow::HTTPMethod::POST)
        ([this](const crow::request& req) {
            try {
                auto body = json::parse(req.body);
                
                std::string text = body.value("text", "");
                json metadata = body.value("metadata", json::object());

                if (text.empty()) {
                    return crow::response(400, "Missing 'text' field");
                }

                std::string id = memory.store(text, metadata);

                json response = {
                    {"id", id},
                    {"status", "stored"}
                };
                return crow::response(response.dump());

            } catch (const json::exception& e) {
                return crow::response(400, std::string("JSON error: ") + e.what());
            } catch (const std::exception& e) {
                return crow::response(500, std::string("Error: ") + e.what());
            }
        });

        // POST /search
        CROW_ROUTE(app, "/search").methods(crow::HTTPMethod::POST)
        ([this](const crow::request& req) {
            try {
                auto body = json::parse(req.body);

                std::string query = body.value("query", "");
                int limit = body.value("limit", 5);
                float min_score = body.value("min_score", 0.0f);
                std::vector<std::string> collections = 
                    body.value("collections", std::vector<std::string>{});

                if (query.empty()) {
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

                return crow::response(response.dump());

            } catch (const json::exception& e) {
                return crow::response(400, std::string("JSON error: ") + e.what());
            } catch (const std::exception& e) {
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

void Server::run() {
    std::cout << "Starting Ragger server on " 
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
