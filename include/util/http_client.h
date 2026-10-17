/**
 * HttpClient — a thin RAII wrapper over a libcurl easy handle.
 *
 * Every inference/embedding/client call site used to hand-roll
 * curl_easy_init/setopt/perform/cleanup plus its own write-callback and
 * curl_slist header list — five to ten near-identical lines repeated at
 * every call site, each one a chance to leak the handle or the header list
 * on an early return. HttpClient owns the CURL* for its whole lifetime and
 * exposes a single synchronous request() (plus get()/post() convenience
 * wrappers) that takes a plain method/url/body/headers and returns a
 * Response — no callback to write, no cleanup to remember.
 *
 *   util::HttpClient http;
 *   auto r = http.post(url, body.dump(),
 *                       {"Content-Type: application/json",
 *                        util::bearer_header(api_key)});
 *   if (!r.ok()) throw std::runtime_error(r.error);
 *   if (r.status != 200) ...
 *   auto json = nlohmann::json::parse(r.body);
 *
 * One HttpClient may be reused for multiple requests (each request() call
 * resets the handle), which is fine because curl connection reuse/DNS
 * caching benefits from a shared handle. It is not thread-safe — use one
 * instance per thread, same as a raw CURL* would require.
 */
#pragma once

#include <curl/curl.h>

#include <string>
#include <string_view>
#include <vector>

namespace ragger::util {

/// The one place the Bearer auth-header scheme literal is spelled. Commit
/// 340ab92 committed a scrubbed three-asterisk marker where the scheme word
/// belonged (a tool's output redaction pasted back as source), which made
/// every keyed request 401. Build auth headers with this, never a pasted
/// literal; scripts/check-source.sh fails the release gate on scrubbed
/// markers in the tree.
std::string bearer_header(std::string_view key);

class HttpClient {
public:
    struct Response {
        CURLcode result = CURLE_OK;
        long status = 0;
        std::string body;
        std::string error;   // curl_easy_strerror(result) when result != CURLE_OK

        bool ok() const { return result == CURLE_OK; }
    };

    HttpClient();
    ~HttpClient();

    HttpClient(const HttpClient&) = delete;
    HttpClient& operator=(const HttpClient&) = delete;
    HttpClient(HttpClient&& o) noexcept : curl_(o.curl_) { o.curl_ = nullptr; }

    /// Perform a synchronous HTTP request. `headers` are raw header lines
    /// ("Content-Type: application/json"); `body` is sent as-is (POSTFIELDS)
    /// when non-empty and the method isn't GET. `timeout_sec` bounds the
    /// whole transfer; `connect_timeout_sec` <= 0 leaves curl's default.
    Response request(const std::string& method,
                      const std::string& url,
                      const std::string& body = "",
                      const std::vector<std::string>& headers = {},
                      long timeout_sec = 30,
                      long connect_timeout_sec = 0);

    Response get(const std::string& url,
                 const std::vector<std::string>& headers = {},
                 long timeout_sec = 30) {
        return request("GET", url, "", headers, timeout_sec);
    }

    Response post(const std::string& url,
                  const std::string& body,
                  const std::vector<std::string>& headers = {},
                  long timeout_sec = 30) {
        return request("POST", url, body, headers, timeout_sec);
    }

private:
    CURL* curl_ = nullptr;
};

} // namespace ragger::util
