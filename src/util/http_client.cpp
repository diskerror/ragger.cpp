#include "util/http_client.h"

#include <stdexcept>

namespace ragger::util {

namespace {
size_t write_to_string(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* out = static_cast<std::string*>(userdata);
    out->append(ptr, size * nmemb);
    return size * nmemb;
}
} // namespace

// See header: single source of the Bearer scheme literal. One line so the
// grep gate in PHASE1 Step 2 (no scheme prefix outside this function) holds.
std::string bearer_header(std::string_view key) { return "Authorization: Bearer " + std::string(key); }

HttpClient::HttpClient() {
    curl_ = curl_easy_init();
    if (!curl_) {
        throw std::runtime_error("curl_easy_init failed");
    }
}

HttpClient::~HttpClient() {
    if (curl_) curl_easy_cleanup(curl_);
}

HttpClient::Response HttpClient::request(const std::string& method,
                                          const std::string& url,
                                          const std::string& body,
                                          const std::vector<std::string>& headers,
                                          long timeout_sec,
                                          long connect_timeout_sec) {
    curl_easy_reset(curl_);

    Response resp;

    struct curl_slist* header_list = nullptr;
    for (const auto& h : headers) {
        header_list = curl_slist_append(header_list, h.c_str());
    }

    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl_, CURLOPT_NOSIGNAL, 1L);
    if (timeout_sec > 0) curl_easy_setopt(curl_, CURLOPT_TIMEOUT, timeout_sec);
    if (connect_timeout_sec > 0) curl_easy_setopt(curl_, CURLOPT_CONNECTTIMEOUT, connect_timeout_sec);
    if (header_list) curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, header_list);

    if (method == "GET") {
        // default method, nothing to set
    } else if (method == "POST") {
        curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, body.c_str());
        curl_easy_setopt(curl_, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
    } else {
        curl_easy_setopt(curl_, CURLOPT_CUSTOMREQUEST, method.c_str());
        if (!body.empty()) {
            curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, body.c_str());
            curl_easy_setopt(curl_, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
        }
    }

    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, write_to_string);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &resp.body);

    resp.result = curl_easy_perform(curl_);
    if (resp.result == CURLE_OK) {
        curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &resp.status);
    } else {
        resp.error = curl_easy_strerror(resp.result);
    }

    if (header_list) curl_slist_free_all(header_list);

    return resp;
}

} // namespace ragger::util
