/**
 * HTTP client for Ragger Memory daemon
 */
#include "ragger/client.h"

#include <cstring>
#include <stdexcept>
#include <sstream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

namespace ragger {

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
        throw std::runtime_error("Store failed: HTTP " + std::to_string(resp.status));
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
        throw std::runtime_error("Search failed: HTTP " + std::to_string(resp.status));
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
        throw std::runtime_error("Count failed: HTTP " + std::to_string(resp.status));
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
        throw std::runtime_error("Delete batch failed: HTTP " + std::to_string(resp.status));
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
        throw std::runtime_error("Search by metadata failed: HTTP " + std::to_string(resp.status));
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
        throw std::runtime_error("Register failed: HTTP " + std::to_string(resp.status));
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
    // Create socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        throw std::runtime_error("Failed to create socket: " + std::string(strerror(errno)));
    }
    
    // Set timeout (5 seconds)
    struct timeval timeout;
    timeout.tv_sec = 5;
    timeout.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    
    // Connect
    struct sockaddr_in server_addr;
    std::memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port_);
    
    if (inet_pton(AF_INET, host_.c_str(), &server_addr.sin_addr) <= 0) {
        close(sock);
        throw std::runtime_error("Invalid address: " + host_);
    }
    
    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        close(sock);
        throw std::runtime_error("Connection failed: " + std::string(strerror(errno)));
    }
    
    // Build HTTP request
    std::ostringstream request;
    request << method << " " << path << " HTTP/1.1\r\n";
    request << "Host: " << host_ << ":" << port_ << "\r\n";
    request << "Content-Type: application/json\r\n";
    
    if (!token_.empty()) {
        request << "Authorization: Bearer " << token_ << "\r\n";
    }
    
    if (!body.empty()) {
        request << "Content-Length: " << body.length() << "\r\n";
    }
    
    request << "Connection: close\r\n";
    request << "\r\n";
    
    if (!body.empty()) {
        request << body;
    }
    
    // Send request
    std::string req_str = request.str();
    ssize_t sent = send(sock, req_str.c_str(), req_str.length(), 0);
    if (sent < 0) {
        close(sock);
        throw std::runtime_error("Failed to send request: " + std::string(strerror(errno)));
    }
    
    // Read response
    std::string response_data;
    char buffer[4096];
    ssize_t bytes_read;
    
    while ((bytes_read = recv(sock, buffer, sizeof(buffer), 0)) > 0) {
        response_data.append(buffer, bytes_read);
    }
    
    close(sock);
    
    if (bytes_read < 0) {
        throw std::runtime_error("Failed to read response: " + std::string(strerror(errno)));
    }
    
    // Parse response
    HttpResponse result;
    
    // Find status line
    size_t status_end = response_data.find("\r\n");
    if (status_end == std::string::npos) {
        throw std::runtime_error("Invalid HTTP response: no status line");
    }
    
    std::string status_line = response_data.substr(0, status_end);
    
    // Parse status code (format: "HTTP/1.1 200 OK")
    size_t first_space = status_line.find(' ');
    size_t second_space = status_line.find(' ', first_space + 1);
    
    if (first_space == std::string::npos || second_space == std::string::npos) {
        throw std::runtime_error("Invalid HTTP response: malformed status line");
    }
    
    std::string status_code_str = status_line.substr(first_space + 1, second_space - first_space - 1);
    result.status = std::stoi(status_code_str);
    
    // Find body (after \r\n\r\n)
    size_t body_start = response_data.find("\r\n\r\n");
    if (body_start != std::string::npos) {
        result.body = response_data.substr(body_start + 4);
    }
    
    return result;
}

} // namespace ragger
