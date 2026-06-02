/**
 * HTTP client for Ragger Memory daemon
 */
#include "ragger/client.h"

#include "ragger/lang.h"

#include <format>
#include <stdexcept>
#include <string>

#include <curl/curl.h>

namespace ragger {

namespace {
// libcurl write callback — append the response body into a std::string.
size_t write_to_string(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* out = static_cast<std::string*>(userdata);
    out->append(ptr, size * nmemb);
    return size * nmemb;
}
} // namespace

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
            result.results.push_back(sr);
        }
    }
    
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

bool RaggerClient::delete_memory(int memory_id) {
    auto resp = http_delete("/memory/" + std::to_string(memory_id));
    if (resp.status < 200 || resp.status >= 300) {
        return false;
    }
    
    auto j = json::parse(resp.body);
    return j.value("status", "") == "deleted";
}

int RaggerClient::delete_batch(const std::vector<int>& memory_ids) {
    json payload;
    payload["ids"] = memory_ids;
    
    auto resp = http_post("/delete_batch", payload.dump());
    if (resp.status < 200 || resp.status >= 300) {
        throw std::runtime_error(std::format(lang::ERR_CLIENT_DELETE_BATCH, resp.status));
    }
    
    auto j = json::parse(resp.body);
    return j.value("deleted", 0);
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

json RaggerClient::register_user(const std::string& username) {
    json payload;
    payload["username"] = username;
    // admin flag removed — sudo is the admin gate

    auto resp = http_post("/register", payload.dump());
    if (resp.status < 200 || resp.status >= 300) {
        throw std::runtime_error(std::format(lang::ERR_CLIENT_REGISTER, resp.status));
    }
    return json::parse(resp.body);
}

// HTTP implementation helpers

RaggerClient::HttpResponse RaggerClient::http_get(const std::string& path) const {
    return http_request("GET", path);
}

RaggerClient::HttpResponse RaggerClient::http_post(const std::string& path,
                                                     const std::string& body) const {
    return http_request("POST", path, body);
}

RaggerClient::HttpResponse RaggerClient::http_delete(const std::string& path) const {
    return http_request("DELETE", path);
}

RaggerClient::HttpResponse RaggerClient::http_request(const std::string& method,
                                                        const std::string& path,
                                                        const std::string& body) const {
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error(std::format(lang::ERR_CLIENT_SOCKET, "curl_easy_init failed"));
    }

    std::string url = std::format("http://{}:{}{}", host_, port_, path);
    std::string response_body;

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    std::string auth_header;
    if (!token_.empty()) {
        auth_header = "Authorization: Bearer " + token_;
        headers = curl_slist_append(headers, auth_header.c_str());
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, method.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    if (!body.empty()) {
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
    }
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_to_string);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);        // matches the old 5s socket timeout
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

    CURLcode rc = curl_easy_perform(curl);
    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (rc != CURLE_OK) {
        throw std::runtime_error(std::format(lang::ERR_CLIENT_CONNECT, curl_easy_strerror(rc)));
    }

    HttpResponse result;
    result.status = static_cast<int>(status);
    result.body = std::move(response_body);
    return result;
}

} // namespace ragger
