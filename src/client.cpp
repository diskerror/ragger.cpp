/**
 * HTTP client for Ragger Memory daemon
 */
#include "client.h"

#include "lang.h"
#include "util/http_client.h"

#include <format>
#include <stdexcept>
#include <string>
#include <vector>

namespace ragger {

// S28 (C m17): the json -> SearchResult loop was duplicated verbatim in
// search() and search_by_metadata(). One parser now; output identical
// (fields read are exactly those the old loops read). Not in client.h:
// internal to this TU plus the test's extern declaration (S28 pin choice —
// the public client path needs a live daemon, which tests must not assume).
std::vector<SearchResult> parse_search_results(const json& j) {
    std::vector<SearchResult> results;
    if (j.contains("results") && j["results"].is_array()) {
        for (const auto& item : j["results"]) {
            SearchResult sr;
            // ID might be string or int depending on server
            if (item.contains("id")) {
                if (item["id"].is_string()) {
                    sr.id = std::stoi(item["id"].get<std::string>());
                } else {
                    sr.id = item["id"].get<int>();
                }
            } else {
                sr.id = 0;
            }
            sr.text = item.value("text", "");
            sr.score = item.value("score", 0.0f);
            sr.metadata = item.value("metadata", json::object());
            sr.timestamp = item.value("timestamp", "");
            results.push_back(sr);
        }
    }
    return results;
}

RaggerClient::RaggerClient(const std::string& host, int port,
                           const std::string& token)
    : host_(host), port_(port), token_(token) {}

bool RaggerClient::is_available() const {
    try {
        auto resp = http_get("/health");
        if (resp.status != 200) return false;
        
        // Parse JSON and check status field
        auto j = json::parse(resp.body, nullptr, false);
        if (j.is_discarded()) return false;
        
        return j.contains("status") && j["status"] == "ok";
    } catch (...) {
        return false;
    }
}

std::string RaggerClient::store(const std::string& text, json metadata) {
    json payload;
    payload["text"] = text;
    if (!metadata.is_null() && !metadata.empty()) {
        payload["metadata"] = metadata;
    }
    
    auto resp = http_post("/store", payload.dump());
    if (resp.status < 200 || resp.status >= 300) {
        throw std::runtime_error(std::format(lang::ERR_CLIENT_STORE, resp.status));
    }
    
    auto j = json::parse(resp.body);
    return j.value("id", "");
}

SearchResponse RaggerClient::search(const std::string& query, int limit,
                                     float min_score,
                                     std::vector<std::string> collections) {
    json payload;
    payload["query"] = query;
    payload["limit"] = limit;
    payload["min_score"] = min_score;
    if (!collections.empty()) {
        payload["collections"] = collections;
    }
    
    auto resp = http_post("/search", payload.dump());
    if (resp.status < 200 || resp.status >= 300) {
        throw std::runtime_error(std::format(lang::ERR_CLIENT_SEARCH, resp.status));
    }
    
    auto j = json::parse(resp.body);
    SearchResponse result;
    result.results = parse_search_results(j);

    if (j.contains("timing")) {
        result.timing = j["timing"];
    }
    
    return result;
}

int RaggerClient::count() {
    auto resp = http_get("/count");
    if (resp.status < 200 || resp.status >= 300) {
        throw std::runtime_error(std::format(lang::ERR_CLIENT_COUNT, resp.status));
    }
    
    auto j = json::parse(resp.body);
    return j.value("count", 0);
}

std::vector<SearchResult> RaggerClient::search_by_metadata(const json& metadata_filter, int limit) {
    json payload;
    payload["metadata"] = metadata_filter;
    if (limit > 0) {
        payload["limit"] = limit;
    }
    
    auto resp = http_post("/search_by_metadata", payload.dump());
    if (resp.status < 200 || resp.status >= 300) {
        throw std::runtime_error(std::format(lang::ERR_CLIENT_SEARCH_META, resp.status));
    }
    
    auto j = json::parse(resp.body);
    return parse_search_results(j);
}

// HTTP implementation helpers

RaggerClient::HttpResponse RaggerClient::http_get(const std::string& path) const {
    return http_request("GET", path);
}

RaggerClient::HttpResponse RaggerClient::http_post(const std::string& path,
                                                     const std::string& body) const {
    return http_request("POST", path, body);
}

RaggerClient::HttpResponse RaggerClient::http_request(const std::string& method,
                                                        const std::string& path,
                                                        const std::string& body) const {
    std::string url = std::format("http://{}:{}{}", host_, port_, path);

    std::vector<std::string> headers{"Content-Type: application/json"};
    if (!token_.empty()) {
        headers.push_back(util::bearer_header(token_));
    }

    util::HttpClient http;
    auto resp = http.request(method, url, body, headers,
                              /*timeout_sec=*/5, /*connect_timeout_sec=*/5);

    if (!resp.ok()) {
        throw std::runtime_error(std::format(lang::ERR_CLIENT_CONNECT, resp.error));
    }

    HttpResponse result;
    result.status = static_cast<int>(resp.status);
    result.body = std::move(resp.body);
    return result;
}

} // namespace ragger
