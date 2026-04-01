/**
 * HTTP client for Ragger Memory daemon
 *
 * Thin client that mirrors RaggerMemory interface but uses HTTP API.
 * Used by CLI when daemon is running — avoids loading the embedding model.
 */
#pragma once

#include <string>
#include <vector>
#include <optional>
#include "nlohmann_json.hpp"
#include "ragger/sqlite_backend.h"  // for SearchResult, SearchResponse

namespace ragger {

using json = nlohmann::json;

class RaggerClient {
public:
    RaggerClient(const std::string& host = "127.0.0.1", int port = 8432,
                 const std::string& token = "");
    
    /// Check if the daemon is running (GET /health succeeds)
    bool is_available() const;
    
    /// Store a memory via HTTP. Returns memory ID.
    std::string store(const std::string& text, json metadata = {});
    
    /// Search via HTTP.
    SearchResponse search(const std::string& query, int limit = 5,
                          float min_score = 0.0f,
                          std::vector<std::string> collections = {});
    
    /// Get count via HTTP.
    int count();
    
    /// Delete a memory via HTTP.
    bool delete_memory(int memory_id);
    
    /// Delete batch via HTTP.
    int delete_batch(const std::vector<int>& memory_ids);
    
    /// Search by metadata via HTTP.
    std::vector<SearchResult> search_by_metadata(const json& metadata_filter, int limit = 0);

    /// Register user via HTTP POST /register.
    json register_user(const std::string& username);

private:
    std::string host_;
    int port_;
    std::string token_;
    
    /// Make an HTTP request. Returns {status_code, response_body}.
    struct HttpResponse {
        int status;
        std::string body;
    };
    
    HttpResponse http_get(const std::string& path) const;
    HttpResponse http_post(const std::string& path, const std::string& body) const;
    HttpResponse http_delete(const std::string& path) const;
    HttpResponse http_request(const std::string& method, const std::string& path,
                              const std::string& body = "") const;
};

} // namespace ragger
