/**
 * test_http.cpp — HTTP route tests against a live daemon (S53, C §7 items 1/3).
 *
 * One daemon for the whole binary (see harness/http_fixture.h): it is started on
 * an ephemeral port inside the TestEnv scratch base and stopped + joined by the
 * fixture's destructor, so nothing is listening once the process exits. Status
 * codes below were PROBED against the running server, not copied from the
 * review: where the review's guess differed from the implementation the code's
 * value is asserted and the difference is noted inline.
 */
#include "harness/http_fixture.h"

#include "nlohmann_json.hpp"

#include <print>
#include <string>

using nlohmann::json;
using ragger_test::HttpFixture;

namespace {

/// POST JSON and return the parsed body; the status is checked by the caller.
/// Error responses are text/plain, not JSON: when the body does not parse as a
/// JSON object, `_status`/`_raw` are still set on a fresh object.
json post_json(HttpFixture& fx, const std::string& path, const json& body,
               bool with_auth = true) {
    auto r = fx.client.post(fx.url(path), body.dump(),
                            with_auth ? fx.auth()
                                      : std::vector<std::string>{"Content-Type: application/json"});
    CHECK(r.ok());
    json out = json::object();
    if (!r.body.empty()) {
        auto parsed = json::parse(r.body, nullptr, /*allow_exceptions=*/false);
        if (!parsed.is_discarded() && parsed.is_object()) parsed.swap(out);
    }
    out["_status"] = r.status;
    out["_raw"] = r.body;
    return out;
}

} // namespace

// GET /health is the unauthenticated readiness probe.
void t_health_no_token(HttpFixture& fx) {
    std::println("  t_health_no_token...");
    auto r = fx.client.get(fx.url("/health"));
    CHECK(r.ok());
    CHECK(r.status == 200);
    const auto j = json::parse(r.body);
    CHECK(j["status"] == "ok");
    std::println(" OK");
}

// Auth contract on loopback. The review expected 401 for a missing token, but
// the server pre-authenticates loopback and unix-socket requests as the
// default user (a656d4b, documented in docs/http-api.md "Requests arriving on
// the unix socket or from loopback ... are pre-authenticated"), so a missing
// token on 127.0.0.1 is a documented 200, not 401. The non-loopback 401 path
// is not exercisable from a single-host test without publishing the listener
// on a LAN interface; noted as a coverage gap in the step report. What IS
// reachable from loopback is the cross-origin write guard (C §1.8): a POST
// whose Origin authority differs from Host is refused 403 after auth.
void t_loopback_auth_contract(HttpFixture& fx) {
    std::println("  t_loopback_auth_contract...");
    auto r = fx.client.post(fx.url("/search"), R"({"query":"alpha"})",
                            {"Content-Type: application/json"});
    CHECK(r.ok());
    CHECK(r.status == 200);  // loopback pre-authenticated (see header comment)

    auto xorigin = fx.client.post(fx.url("/store"), R"({"text":"x-origin probe"})",
                                  {"Content-Type: application/json",
                                   "Origin: http://evil.example"});
    CHECK(xorigin.ok());
    CHECK(xorigin.status == 403);

    // Same-origin (Origin == Host) passes the guard.
    auto sorigin = fx.client.post(fx.url("/store"), R"({"text":"same-origin probe"})",
                                  {"Content-Type: application/json",
                                   "Origin: http://127.0.0.1:" + std::to_string(fx.port)});
    CHECK(sorigin.ok());
    CHECK(sorigin.status == 200);
    std::println(" OK");
}

// Store then search: status and parseable JSON are the contract here (the
// embedder may be absent, and text-only scoring depends on corpus idf).
void t_store_then_search(HttpFixture& fx) {
    std::println("  t_store_then_search...");
    auto stored = post_json(fx, "/store", {{"text", "alpha bravo storage probe"}});
    CHECK(stored["_status"] == 200);

    auto found = post_json(fx, "/search", {{"query", "alpha bravo storage probe"}});
    CHECK(found["_status"] == 200);
    CHECK(found.contains("results"));
    CHECK(found["results"].is_array());
    bool echoed = false;
    for (const auto& hit : found["results"])
        if (hit.contains("text") && hit["text"].get<std::string>().find("alpha bravo") != std::string::npos)
            echoed = true;
    std::println(" OK (results={}, text_found={})", found["results"].size(), echoed);
}

// An empty text is a client error, not a stored row.
void t_store_empty_text(HttpFixture& fx) {
    std::println("  t_store_empty_text...");
    auto r = post_json(fx, "/store", {{"text", ""}});
    CHECK(r["_status"] == 400);
    std::println(" OK");
}

// S27: a mistyped metadata value is rejected with a message that names the
// expectation, not coerced.
void t_store_metadata_type_error(HttpFixture& fx) {
    std::println("  t_store_metadata_type_error...");
    auto r = post_json(fx, "/store",
                       {{"text", "typed metadata probe"},
                        {"metadata", {{"collection", 5}}}});
    CHECK(r["_status"] == 400);
    CHECK(r["_raw"].get<std::string>().find("must be a string") != std::string::npos);
    std::println(" OK");
}

// /memory/search validates its argument types too: limit must be an integer.
void t_memory_search_bad_limit(HttpFixture& fx) {
    std::println("  t_memory_search_bad_limit...");
    auto r = post_json(fx, "/memory/search", {{"query", "x"}, {"limit", "10"}});
    CHECK(r["_status"] == 400);
    std::println(" OK");
}

// A decision stored through the generic surface goes through the SAME
// validation as POST /decision: a status outside the vocabulary is rejected.
void t_decision_validation_parity(HttpFixture& fx) {
    std::println("  t_decision_validation_parity...");
    auto good = post_json(fx, "/store",
                          {{"text", "decision via store surface"},
                           {"metadata", {{"category", "decision"},
                                         {"trigger", "when probing parity"}}}});
    CHECK(good["_status"] == 200);
    // The generic surface routes by category and reports the stored kind.
    CHECK(good["kind"] == "decision");

    // The row is visible through the agent search surface as knowledge with
    // kind=decision (probed shape: id "knowledge:<id>", extra.kind).
    auto found = post_json(fx, "/memory/search", {{"query", "decision via store surface"}});
    CHECK(found["_status"] == 200);
    bool seen = false;
    const std::string want_id = "knowledge:" + good["id"].get<std::string>();
    for (const auto& hit : found.value("results", json::array()))
        if (hit.value("id", "") == want_id && hit.contains("extra") &&
            hit["extra"].value("kind", "") == "decision")
            seen = true;
    CHECK(seen);

    auto bad = post_json(fx, "/store",
                         {{"text", "decision with bogus status"},
                          {"metadata", {{"category", "decision"},
                                        {"trigger", "when probing status"},
                                        {"status", "not-a-status"}}}});
    CHECK(bad["_status"] == 400);

    auto direct = post_json(fx, "/decision",
                            {{"text", "decision direct with bogus status"},
                             {"trigger", "when probing status"},
                             {"status", "not-a-status"}});
    CHECK(direct["_status"] == 400);
    std::println(" OK");
}

// An unrouted path is a 404, not a 500 or an auth failure.
void t_unknown_route(HttpFixture& fx) {
    std::println("  t_unknown_route...");
    auto r = fx.client.post(fx.url("/no-such-route"), "{}", fx.auth());
    CHECK(r.ok());
    CHECK(r.status == 404);
    std::println(" OK");
}

int main() {
    HttpFixture fx("http");
    CHECK(fx.ready);
    std::println("Running HTTP route tests against 127.0.0.1:{}:", fx.port);

    t_health_no_token(fx);
    t_loopback_auth_contract(fx);
    t_store_then_search(fx);
    t_store_empty_text(fx);
    t_store_metadata_type_error(fx);
    t_memory_search_bad_limit(fx);
    t_decision_validation_parity(fx);
    t_unknown_route(fx);

    std::println("test_http: all passed");
    return 0;
}
