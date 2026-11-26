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

#include "mcp.h"
#include "config_access.h"
#include "util/time.h"
#include "nlohmann_json.hpp"

#include <print>
#include <algorithm>
#include <filesystem>
#include <iterator>
#include <set>
#include <sqlite3.h>
#include <string>
#include <string_view>

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

/// Send a hand-built request over a fresh TCP connection and return the HTTP
/// status code. curl always synthesizes a Host header, so the "no Host header"
/// failure path (X7/C4) can only be exercised with a raw socket.
int raw_http_status(int port, const std::string& request) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    CHECK(fd >= 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return -1;
    }
    size_t off = 0;
    while (off < request.size()) {
        ssize_t n = ::send(fd, request.data() + off, request.size() - off, 0);
        if (n <= 0) { ::close(fd); return -1; }
        off += static_cast<size_t>(n);
    }
    std::string resp;
    char buf[2048];
    while (resp.find('\n') == std::string::npos) {
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        resp.append(buf, static_cast<size_t>(n));
    }
    ::close(fd);
    if (resp.rfind("HTTP/1.1 ", 0) != 0 && resp.rfind("HTTP/1.0 ", 0) != 0)
        return -1;
    auto sp = resp.find(' ', 5);
    if (sp == std::string::npos) return -1;
    return std::atoi(resp.c_str() + sp + 1);
}

} // namespace

// Minimal read/write access to the fixture DB for row-level assertions
// (LR S59). Same process, same WAL — committed daemon writes are visible.
struct RawDb {
    sqlite3* h = nullptr;
    explicit RawDb(const std::string& path) {
        CHECK(sqlite3_open(path.c_str(), &h) == SQLITE_OK);
    }
    ~RawDb() { if (h) sqlite3_close(h); }
    RawDb(const RawDb&) = delete;
    RawDb& operator=(const RawDb&) = delete;
    void exec(const std::string& sql) {
        char* err = nullptr;
        const int rc = sqlite3_exec(h, sql.c_str(), nullptr, nullptr, &err);
        if (rc != SQLITE_OK) {
            std::println(stderr, "RawDb exec failed: {} ({})", sql, err ? err : "?");
            CHECK(false);
        }
    }
    long long scalar(const std::string& sql) {
        sqlite3_stmt* st = nullptr;
        CHECK(sqlite3_prepare_v2(h, sql.c_str(), -1, &st, nullptr) == SQLITE_OK);
        long long v = -999;
        if (sqlite3_step(st) == SQLITE_ROW) v = sqlite3_column_int64(st, 0);
        sqlite3_finalize(st);
        return v;
    }
};

// X25 pin (GREEN before AND after the split work): the default (non-split)
// POST /memory/search response has exactly the top-level key `results`, and
// every hit has exactly the keys {id, layer, score, matched_by, created,
// snippet, extra}. This proves the default path is untouched by split mode.
void x25_test_default_response_shape_pinned(HttpFixture& fx) {
    std::println("  x25_test_default_response_shape_pinned...");
    auto stored = post_json(fx, "/store",
                            {{"text", "x25 shape pin note about flux capacitors"}});
    CHECK(stored["_status"] == 200);
    auto r = post_json(fx, "/memory/search", {{"query", "flux capacitor"}});
    CHECK(r["_status"] == 200);
    std::set<std::string> top;
    for (auto it = r.begin(); it != r.end(); ++it) {
        if (it.key() == "_status" || it.key() == "_raw") continue;  // helper-added
        top.insert(it.key());
    }
    CHECK(top == std::set<std::string>{"results"});
    CHECK(r["results"].is_array());
    const std::set<std::string> want{"id", "layer", "score", "matched_by",
                                     "created", "snippet", "extra"};
    for (const auto& h : r["results"]) {
        std::set<std::string> keys;
        for (auto it = h.begin(); it != h.end(); ++it) keys.insert(it.key());
        CHECK(keys == want);
    }
    std::println(" OK");
}

// id fields differ per path (plain note: string; insight: int) — normalize.
std::string id_of(const json& j) {
    return j["id"].is_string() ? j["id"].get<std::string>() : j["id"].dump();
}

// X25 failure paths (each named in the step): the parser messages arrive
// prefixed "ERROR: " by guarded(); an unknown layer with split must still be
// a 400 (not swallowed by the vector run's embedder catch).
void x25_test_split_validation(HttpFixture& fx) {
    std::println("  x25_test_split_validation...");
    auto r = post_json(fx, "/memory/search", {{"query", "x"}, {"split", "yes"}});
    CHECK(r["_status"] == 400);
    CHECK(r["_raw"] == "ERROR: split must be a boolean");

    r = post_json(fx, "/memory/search",
                  {{"query", "x"}, {"split", true}, {"mode", "semantic"}});
    CHECK(r["_status"] == 400);
    CHECK(r["_raw"] == "ERROR: split cannot be combined with mode");

    r = post_json(fx, "/memory/search",
                  {{"query", "x"}, {"split", true}, {"min_vector_score", -1}});
    CHECK(r["_status"] == 400);
    CHECK(r["_raw"] == "ERROR: min_vector_score must be >= 0");

    r = post_json(fx, "/memory/search", {{"query", "x"}, {"min_vector_score", 0.3}});
    CHECK(r["_status"] == 400);
    CHECK(r["_raw"] == "ERROR: min_vector_score/min_fts_score require split");

    r = post_json(fx, "/memory/search", {{"query", ""}, {"split", true}});
    CHECK(r["_status"] == 400);
    CHECK(r["_raw"] == "ERROR: query is required");

    json bad_layer = {{"query", "x"}, {"split", true}};
    bad_layer["layers"] = json::array({"nonsense"});
    r = post_json(fx, "/memory/search", bad_layer);
    CHECK(r["_status"] == 400);
    CHECK(r["_raw"].get<std::string>().find("unknown layer") != std::string::npos);
    std::println(" OK");
}

// X25: split-mode properties on a fixture of 6 knowledge notes (two sharing
// the rare token x25zorkmid). Scores come from the real embedder, so every
// assertion is a property, never a constant. Skips (not silently passes)
// when the embedding model is absent — same rule as test_sqlite_backend.
void x25_test_split_properties(HttpFixture& fx) {
    std::println("  x25_test_split_properties...");
    const auto& md = fx.env.model_dir();
    if (!std::filesystem::exists(md + "/model.onnx") &&
        !std::filesystem::exists(md + "/onnx/model.onnx")) {
        std::println(" SKIPPED (no embedding model)");
        return;
    }

    const auto handles = [](const json& arr) {
        std::set<std::string> s;
        for (const auto& h : arr) s.insert(h["id"].get<std::string>());
        return s;
    };
    const auto uni = [](const std::set<std::string>& a, const std::set<std::string>& b) {
        std::set<std::string> r = a;
        r.insert(b.begin(), b.end());
        return r;
    };
    const auto inter = [](const std::set<std::string>& a, const std::set<std::string>& b) {
        std::set<std::string> r;
        std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                              std::inserter(r, r.begin()));
        return r;
    };
    // Score-descending with ties by handle ascending (the specified order).
    const auto ordered_by = [](const json& arr, const char* key) {
        for (size_t i = 1; i < arr.size(); ++i) {
            const double a = arr[i - 1][key].get<double>(), b = arr[i][key].get<double>();
            CHECK(a >= b);
            if (a == b)
                CHECK(arr[i - 1]["id"].get<std::string>() < arr[i]["id"].get<std::string>());
        }
    };

    // Fixture: 2 notes carry the rare token, 4 do not.
    std::string rare1, rare2;
    for (const char* text : {
             "The x25zorkmid coin was minted in copper and pays the gear tax",
             "x25zorkmid settlements are due every third Tuesday at the harbor house",
             "Hydraulic pump maintenance schedule for the north gallery",
             "The regatta committee postponed the sail race because of fog",
             "Sourdough starter needs feeding twice a day in warm weather",
             "Fiber optic splicing requires a fusion arc and cleaved glass"}) {
        auto s = post_json(fx, "/store", {{"text", text}});
        CHECK(s["_status"] == 200);
        if (std::string_view(text).find("x25zorkmid") != std::string_view::npos)
            (rare1.empty() ? rare1 : rare2) = "knowledge:" + id_of(s);
    }
    CHECK(!rare1.empty() && !rare2.empty());

    const std::string q = "x25zorkmid copper gear settlement";
    const int limit = 3;
    const auto sp = post_json(fx, "/memory/search",
                              {{"query", q}, {"split", true}, {"limit", limit}});
    CHECK(sp["_status"] == 200);

    // Output shape (exact): four top-level keys, three groups, seven hit keys.
    std::set<std::string> top;
    for (auto it = sp.begin(); it != sp.end(); ++it) {
        if (it.key() == "_status" || it.key() == "_raw") continue;  // helper-added
        top.insert(it.key());
    }
    CHECK(top == (std::set<std::string>{"groups", "vector_available",
                                        "min_vector_score", "min_fts_score"}));
    CHECK(sp["vector_available"] == true);
    CHECK(sp["min_vector_score"] == 0);
    CHECK(sp["min_fts_score"] == 0);
    const auto& g = sp["groups"];
    CHECK(g.size() == 3 && g.contains("both") && g.contains("vector_only") &&
          g.contains("fts_only"));
    const std::set<std::string> hit_keys{"id", "layer", "created", "snippet",
                                         "extra", "vector_score", "fts_score"};
    for (const char* name : {"both", "vector_only", "fts_only"})
        for (const auto& h : g[name]) {
            std::set<std::string> keys;
            for (auto it = h.begin(); it != h.end(); ++it) keys.insert(it.key());
            CHECK(keys == hit_keys);
        }

    const auto hb = handles(g["both"]), hv = handles(g["vector_only"]),
               hf = handles(g["fts_only"]);
    // (1) disjoint: no handle appears in two lists.
    CHECK(inter(hb, hv).empty() && inter(hb, hf).empty() && inter(hv, hf).empty());
    // (2) null rule: both -> two non-null scores; vector_only -> fts_score
    // null + non-null vector_score; fts_only the reverse.
    for (const auto& h : g["both"])
        CHECK(!h["vector_score"].is_null() && !h["fts_score"].is_null());
    for (const auto& h : g["vector_only"])
        CHECK(!h["vector_score"].is_null() && h["fts_score"].is_null());
    for (const auto& h : g["fts_only"])
        CHECK(h["vector_score"].is_null() && !h["fts_score"].is_null());
    // (3) union rule: both ∪ vector_only == the semantic run at limit;
    // both ∪ fts_only == the keyword run at limit (compared as sets).
    const auto sem = post_json(fx, "/memory/search",
                               {{"query", q}, {"mode", "semantic"}, {"limit", limit}});
    CHECK(sem["_status"] == 200);
    const auto kw = post_json(fx, "/memory/search",
                              {{"query", q}, {"mode", "keyword"}, {"limit", limit}});
    CHECK(kw["_status"] == 200);
    CHECK(uni(hb, hv) == handles(sem["results"]));
    CHECK(uni(hb, hf) == handles(kw["results"]));
    // (5) size: each side at most limit.
    CHECK(uni(hb, hv).size() <= static_cast<size_t>(limit));
    CHECK(uni(hb, hf).size() <= static_cast<size_t>(limit));
    // (7) order: both/vector_only by vector_score desc, fts_only by
    // fts_score desc (ties by handle ascending).
    ordered_by(g["both"], "vector_score");
    ordered_by(g["vector_only"], "vector_score");
    ordered_by(g["fts_only"], "fts_score");

    // (4) minimums: a vector minimum above every cosine empties both and
    // vector_only; an unreachable FTS minimum empties fts_only and both
    // (both requires presence in BOTH lists).
    const auto sp_v = post_json(fx, "/memory/search",
                                {{"query", q}, {"split", true}, {"limit", limit},
                                 {"min_vector_score", 0.99999}});
    CHECK(sp_v["_status"] == 200);
    CHECK(sp_v["groups"]["both"].empty());
    CHECK(sp_v["groups"]["vector_only"].empty());
    CHECK(sp_v["min_vector_score"].get<double>() > 0.9999);  // echoed (float path)
    const auto sp_f = post_json(fx, "/memory/search",
                                {{"query", q}, {"split", true}, {"limit", limit},
                                 {"min_fts_score", 1e9}});
    CHECK(sp_f["_status"] == 200);
    CHECK(sp_f["groups"]["fts_only"].empty());
    CHECK(sp_f["groups"]["both"].empty());

    // (6) rare token: a query that is exactly the rare token must put both
    // notes carrying it in fts_only or both. Observed: the FTS run finds
    // both (exact token match), and the semantic run also ranks them top-2,
    // so they land in `both`; the contract accepts either side.
    const auto sp_rare = post_json(fx, "/memory/search",
                                   {{"query", "x25zorkmid"}, {"split", true},
                                    {"limit", limit}});
    CHECK(sp_rare["_status"] == 200);
    const auto rare_side = uni(handles(sp_rare["groups"]["fts_only"]),
                               handles(sp_rare["groups"]["both"]));
    CHECK(rare_side.count(rare1) == 1);
    CHECK(rare_side.count(rare2) == 1);
    const auto where = [&](const std::string& id) {
        return handles(sp_rare["groups"]["both"]).count(id) ? "both" : "fts_only";
    };
    std::println("      rare notes landed: {} -> {}, {} -> {}", rare1, where(rare1),
                 rare2, where(rare2));
    std::println(" OK");
}

// Saving the summarizer model through PUT /config takes effect at once:
// /health reports the model from the live inference client, not from the
// config, so it changes only if the client was rebuilt.
void t_summarizer_model_live(HttpFixture& fx) {
    std::println("  t_summarizer_model_live...");
    auto put = [&](const char* key, const char* value) {
        const std::string body = json{{"value", value}}.dump();
        return fx.client.request("PUT", fx.url(std::string("/config/") + key), body, fx.auth());
    };
    auto health_model = [&] {
        auto r = fx.client.get(fx.url("/health"));
        return json::parse(r.body).value("summarizer_model", std::string("?"));
    };
    CHECK(put("summarizer_api_url", "http://127.0.0.1:9/v1").status == 200);
    CHECK(put("summarizer_model", "lab-model-a").status == 200);
    CHECK(health_model() == "lab-model-a");
    CHECK(put("summarizer_model", "lab-model-b").status == 200);
    CHECK(health_model() == "lab-model-b");
    // Restore the fixture's defaults for the tests that follow.
    CHECK(put("summarizer_model", "").status == 200);
    CHECK(put("summarizer_api_url", "").status == 200);
    std::println(" OK");
}

// GET /health is the unauthenticated readiness probe.
void t_health_no_token(HttpFixture& fx) {
    std::println("  t_health_no_token...");
    auto r = fx.client.get(fx.url("/health"));
    CHECK(r.ok());
    CHECK(r.status == 200);
    const auto j = json::parse(r.body);
    CHECK(j["status"] == "ok");
    // Capture/summarizer readiness flags (capture_summarizer_flags).
    CHECK(j.contains("capture_turns") && j["capture_turns"].is_boolean());
    CHECK(j.contains("summarizer_model") && j["summarizer_model"].is_string());
    CHECK(j.contains("summarizer_engine") && j["summarizer_engine"].is_string());
    CHECK(j.contains("summarizer_reachable") && j["summarizer_reachable"].is_boolean());
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

// X7 (C4): DNS-rebinding guard. A page on evil.example whose DNS resolves to
// 127.0.0.1 reaches the daemon over loopback. Before this fix every loopback
// peer was pre-authenticated regardless of its Host header, so such a page
// could call guarded routes with no token. Now a loopback request is
// pre-authenticated only when its Host header also names loopback; a foreign
// Host must present a token (401, not 403 — a token holder still works).
void t_dns_rebinding_host_guard(HttpFixture& fx) {
    std::println("  t_dns_rebinding_host_guard...");
    const std::string loopback_host = "127.0.0.1:" + std::to_string(fx.port);

    // Loopback Host, no token → pre-authenticated 200 (local curl/CLI/plugin
    // keep working unchanged).
    auto ok = fx.client.get(fx.url("/count"), {"Host: " + loopback_host});
    CHECK(ok.ok());
    CHECK(ok.status == 200);

    // Foreign Host over loopback, no token → NOT pre-authenticated → 401.
    auto rebinding = fx.client.get(fx.url("/count"), {"Host: evil.example"});
    CHECK(rebinding.ok());
    CHECK(rebinding.status == 401);

    // Foreign Host over loopback WITH a valid bearer token → 200.
    auto tokened = fx.client.get(
        fx.url("/count"),
        {"Host: evil.example", ragger::util::bearer_header(fx.token)});
    CHECK(tokened.ok());
    CHECK(tokened.status == 200);

    // Failure path: no Host header at all → treated as non-loopback → 401.
    // curl always sends a Host, so this needs a raw socket.
    const std::string no_host =
        "GET /count HTTP/1.1\r\nConnection: close\r\n\r\n";
    CHECK(raw_http_status(fx.port, no_host) == 401);

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

// X19: the generic surface no longer stores decisions at all.
// `/store category=decision` is a 400 that names the decision verb and says a
// trigger is required, and writes nothing; the decision verb keeps its own
// validation (a status outside the vocabulary is still a 400).
void t_decision_validation_parity(HttpFixture& fx) {
    std::println("  t_decision_validation_parity...");
    const std::string db_path = fx.env.base() + "/http_test.db";

    auto rejected = post_json(fx, "/store",
                              {{"text", "decision via store surface"},
                               {"metadata", {{"category", "decision"},
                                             {"trigger", "when probing parity"}}}});
    CHECK(rejected["_status"] == 400);
    const std::string rmsg = rejected["_raw"].get<std::string>();
    CHECK(rmsg.rfind("ERROR: ", 0) == 0);
    CHECK(rmsg.find("trigger") != std::string::npos);
    CHECK(rmsg.find("memory_decision") != std::string::npos);

    // Nothing was written: the knowledge count is unchanged.
    {
        RawDb db(db_path);
        const long long before = db.scalar("SELECT COUNT(*) FROM knowledge");
        auto again = post_json(fx, "/store",
                               {{"text", "decision via store surface, second try"},
                                {"metadata", {{"category", "decision"}}}});
        CHECK(again["_status"] == 400);
        CHECK(db.scalar("SELECT COUNT(*) FROM knowledge") == before);
    }

    // The decision verb is the way: a trigger writes kind='decision' and the
    // row is visible through the agent search surface (probed shape: id
    // "knowledge:<id>", extra.kind).
    auto good = post_json(fx, "/decision",
                          {{"text", "decision via the decision verb"},
                           {"trigger", "when probing parity"}});
    CHECK(good["_status"] == 200);
    CHECK(good["kind"] == "decision");

    auto found = post_json(fx, "/memory/search", {{"query", "decision via the decision verb"}});
    CHECK(found["_status"] == 200);
    bool seen = false;
    const std::string want_id = "knowledge:" + id_of(good);
    for (const auto& hit : found.value("results", json::array()))
        if (hit.value("id", "") == want_id && hit.contains("extra") &&
            hit["extra"].value("kind", "") == "decision")
            seen = true;
    CHECK(seen);

    auto bad = post_json(fx, "/decision",
                         {{"text", "decision direct with bogus status"},
                          {"trigger", "when probing status"},
                          {"status", "not-a-status"}});
    CHECK(bad["_status"] == 400);
    std::println(" OK");
}

// LR S70 (R4): `store` is ONE function across transports. For the same
// arguments the HTTP body and the MCP payload must agree (ids differ per
// row; that is the only field allowed to). Envelopes stay per-surface (D-4):
// HTTP 400 "ERROR: <msg>" vs MCP isError with the bare message.
void t_store_http_mcp_parity(HttpFixture& fx) {
    std::println("  t_store_http_mcp_parity...");
    auto mcp_call = [&](const json& args) {
        return ragger::mcp_tool_call(*fx.mem, {{"name", "store"}, {"arguments", args}});
    };
    auto mcp_payload = [&](const json& args) {
        auto r = mcp_call(args);
        CHECK(r.contains("content"));
        CHECK(!r.value("isError", false));
        return json::parse(r["content"][0]["text"].get<std::string>());
    };

    // note form: same payload shape and kind on both surfaces
    const json args{{"text", "lr70 parity note"}};
    const auto h = post_json(fx, "/store", args);
    CHECK(h["_status"] == 200);
    const auto m = mcp_payload(args);
    CHECK(m["status"] == "stored");
    CHECK(m["kind"] == "note");
    CHECK(h["kind"] == m["kind"]);
    CHECK(!id_of(h).empty() && !id_of(m).empty());

    // X19: category=decision is rejected on BOTH surfaces with the
    // same message (envelope differs per surface: HTTP "ERROR: <msg>" text,
    // MCP isError with the bare message).
    const json dargs{{"text", "lr70 parity decision"},
                     {"metadata", {{"category", "decision"}}}};
    const auto dh = post_json(fx, "/store", dargs);
    CHECK(dh["_status"] == 400);
    const auto dr = mcp_call(dargs);
    CHECK(dr.value("isError", false));
    const std::string dmsg = dh["_raw"].get<std::string>();
    CHECK(dmsg.rfind("ERROR: ", 0) == 0);
    CHECK(dr["content"][0]["text"].get<std::string>() == dmsg.substr(7));

    // bad status: rejected with the SAME message (envelope differs)
    const json bargs{{"text", "lr70 parity bad status"},
                     {"metadata", {{"status", "not-a-status"}}}};
    const auto bh = post_json(fx, "/store", bargs);
    CHECK(bh["_status"] == 400);
    const auto br = mcp_call(bargs);
    CHECK(br.value("isError", false));
    const std::string hmsg = bh["_raw"].get<std::string>();
    CHECK(hmsg.rfind("ERROR: ", 0) == 0);
    CHECK(br["content"][0]["text"].get<std::string>() == hmsg.substr(7));

    // empty text: one message now (was: HTTP "Missing 'text' field" /
    // MCP "Error: text parameter required" — intended to disappear)
    const auto eh = post_json(fx, "/store", json{{"text", ""}});
    CHECK(eh["_status"] == 400);
    CHECK(eh["_raw"] == "ERROR: text is required");
    const auto er = mcp_call(json{{"text", ""}});
    CHECK(er.value("isError", false));
    CHECK(er["content"][0]["text"] == "text is required");

    // metadata.collection type validation now reaches MCP too (was HTTP-only)
    const auto cr = mcp_call(json{{"text", "lr70 typed meta"},
                                  {"metadata", {{"collection", 5}}}});
    CHECK(cr.value("isError", false));
    CHECK(cr["content"][0]["text"].get<std::string>().find("must be a string") != std::string::npos);
    std::println(" OK");
}

// LR S71 (R4): read verbs moved into agent_api — one body per verb. The
// differential pins (scratch capture, pre/post move) proved byte-identity;
// this test pins the two UNIFIED behaviors: empty query is now rejected on
// every surface with one message (was: MCP-only check on memory_search,
// HTTP-only "Missing 'query' field" on /search, MCP "Error: query parameter
// required").
void t_read_verbs_parity_s71(HttpFixture& fx) {
    std::println("  t_read_verbs_parity_s71...");
    auto mcp_err = [&](const char* tool, const json& args) {
        auto r = ragger::mcp_tool_call(*fx.mem, {{"name", tool}, {"arguments", args}});
        CHECK(r.value("isError", false));
        return r["content"][0]["text"].get<std::string>();
    };

    const auto ms = post_json(fx, "/memory/search", {{"query", ""}});
    CHECK(ms["_status"] == 400);
    CHECK(ms["_raw"] == "ERROR: query is required");
    CHECK(mcp_err("memory_search", {{"query", ""}}) == "query is required");

    const auto s = post_json(fx, "/search", {{"query", ""}});
    CHECK(s["_status"] == 400);
    CHECK(s["_raw"] == "ERROR: query is required");
    CHECK(mcp_err("search", {{"query", ""}}) == "query is required");

    // happy paths still agree between surfaces (shape, not values — scores
    // differ slightly between the daemon and standalone embedder paths)
    const auto mh = post_json(fx, "/memory/search", {{"query", "lr70 parity note"}});
    CHECK(mh["_status"] == 200);
    auto mr = ragger::mcp_tool_call(*fx.mem, {{"name", "memory_search"},
                                              {"arguments", {{"query", "lr70 parity note"}}}});
    CHECK(!mr.value("isError", false));
    const auto mp = json::parse(mr["content"][0]["text"].get<std::string>());
    CHECK(mp.contains("results"));
    CHECK(mp["results"].size() == mh["results"].size());
    if (!mp["results"].empty()) {
        const auto& a = mp["results"][0];
        for (const char* k : {"id", "layer", "score", "matched_by", "created", "snippet", "extra"})
            CHECK(a.contains(k));
    }
    std::println(" OK");
}

// LR S72 (R4): the write verbs, turn and session are ONE function across
// transports — the permanent R4 regression. For the same arguments the HTTP
// body and the MCP payload must agree (ids differ per row; source
// provenance differs per surface by design). Error messages agree too;
// only the envelope differs (D-4).
void t_write_verbs_parity_s72(HttpFixture& fx) {
    std::println("  t_write_verbs_parity_s72...");
    auto mcp_call = [&](const char* tool, const json& args) {
        auto r = ragger::mcp_tool_call(*fx.mem, {{"name", tool}, {"arguments", args}});
        return std::make_pair(r.value("isError", false),
                              r["content"][0]["text"].get<std::string>());
    };
    // MCP error text == HTTP body minus the "ERROR: " prefix
    auto err_parity = [&](const char* tool, const json& args, const json& http_resp) {
        CHECK(http_resp["_status"] == 400);
        const auto [is_err, text] = mcp_call(tool, args);
        CHECK(is_err);
        const std::string raw = http_resp["_raw"].get<std::string>();
        CHECK(raw.rfind("ERROR: ", 0) == 0);
        CHECK(text == raw.substr(7));
    };

    // ---- memory_note (plain) --------------------------------------------
    const json note_args{{"text", "lr72 note parity"}, {"tags", "lr72"}};
    const auto nh = post_json(fx, "/memory/note", note_args);
    CHECK(nh["_status"] == 200);
    const auto [nerr, ntext] = mcp_call("memory_note", note_args);
    CHECK(!nerr);
    const auto nm = json::parse(ntext);
    CHECK(nm["status"] == nh["status"]);
    CHECK(nm.contains("id") && nh.contains("id"));
    CHECK(nm.contains("possible_supersedes") == nh.contains("possible_supersedes"));
    CHECK(!nm.contains("kind") && !nh.contains("kind")); // plain path shape unchanged

    // ---- memory_note (insight) — seed an episode to point at (LR S80) ----
    const std::string db_path = fx.env.base() + "/http_test.db";
    const int turn = fx.mem->store_turn("lr72 q", "lr72 a", "", false, "lr72-sess");
    CHECK(turn > 0);
    long long epid = -1;
    {
        RawDb db(db_path);
        db.exec("INSERT INTO episodes (text, session_id, span_start, span_end) "
                "SELECT 'lr72 evidence episode', session_id, created_at, created_at "
                "FROM turns WHERE turn_id = " + std::to_string(turn));
        epid = db.scalar("SELECT episode_id FROM episodes WHERE text = 'lr72 evidence episode'");
    }
    CHECK(epid > 0);
    const json ins_args{{"text", "lr72 parity insight"},
                        {"evidence", json::array({epid})},
                        {"trigger", "when probing s72 parity"}};
    const auto ih = post_json(fx, "/memory/note", ins_args);
    CHECK(ih["_status"] == 200);
    CHECK(ih["kind"] == "insight");
    const auto [ierr, itext] = mcp_call("memory_note", ins_args);
    CHECK(!ierr);
    const auto im = json::parse(itext);
    CHECK(im["kind"] == "insight");
    CHECK(im.contains("superseded") == ih.contains("superseded"));
    // trigger on a plain note: same message everywhere (D-3 rule)
    err_parity("memory_note", {{"text", "lr72 bad trigger"}, {"trigger", "nope"}},
               post_json(fx, "/memory/note", {{"text", "lr72 bad trigger"}, {"trigger", "nope"}}));

    // ---- memory_decision --------------------------------------------------
    const json dec_args{{"text", "lr72 parity decision"},
                        {"trigger", "when probing s72 decision parity"}};
    const auto dh = post_json(fx, "/decision", dec_args);
    CHECK(dh["_status"] == 200);
    CHECK(dh["kind"] == "decision");
    const auto [derr, dtext] = mcp_call("memory_decision", dec_args);
    CHECK(!derr);
    const auto dm = json::parse(dtext);
    CHECK(dm["kind"] == "decision");
    CHECK(dm["status"] == dh["status"]);
    CHECK(dm.contains("superseded") == dh.contains("superseded"));
    CHECK(dm.contains("possible_supersedes") == dh.contains("possible_supersedes"));
    err_parity("memory_decision", {{"text", "lr72 decision no trigger"}},
               post_json(fx, "/decision", {{"text", "lr72 decision no trigger"}}));

    // ---- LR S85: evidence by episode in every transport ------------------
    // Mixed forms in ONE array (bare int + "episode:<id>" handle) store the
    // same rows on both surfaces; a turn_summary handle is a 400 naming
    // `episode` on both.
    long long epid2 = -1;
    {
        RawDb db(db_path);
        db.exec("INSERT INTO episodes (text, session_id, span_start, span_end) "
                "SELECT 'lr72 evidence episode 2', session_id, created_at + 3600, created_at + 3600 "
                "FROM turns WHERE turn_id = " + std::to_string(turn));
        epid2 = db.scalar("SELECT episode_id FROM episodes WHERE text = 'lr72 evidence episode 2'");
    }
    CHECK(epid2 > 0);
    const json mix_args{{"text", "lr85 mixed evidence"},
                        {"trigger", "when probing s85"},
                        {"evidence", json::array({epid, std::string("episode:") + std::to_string(epid2)})}};
    const auto mh_resp = post_json(fx, "/decision", mix_args);
    CHECK(mh_resp["_status"] == 200);
    {
        RawDb db(db_path);
        CHECK(db.scalar("SELECT COUNT(*) FROM knowledge_episode_evidence WHERE knowledge_id = "
                        + id_of(mh_resp)) == 2);
    }
    const auto [s85_err, s85_text] = mcp_call("memory_decision", mix_args);
    CHECK(!s85_err);
    {
        RawDb db(db_path);
        const std::string mid = id_of(json::parse(s85_text));
        CHECK(db.scalar("SELECT COUNT(*) FROM knowledge_episode_evidence WHERE knowledge_id = "
                        + mid) == 2);
    }
    err_parity("memory_decision",
               {{"text", "lr85 bad evidence"}, {"trigger", "x"},
                {"evidence", json::array({std::string("turn_summary:5")})}},
               post_json(fx, "/decision", {{"text", "lr85 bad evidence"}, {"trigger", "x"},
                                           {"evidence", json::array({std::string("turn_summary:5")})}}));
    {
        const auto [berr, btext] = mcp_call("memory_decision",
            {{"text", "lr85 bad evidence 2"}, {"trigger", "x"},
             {"evidence", json::array({std::string("turn_summary:5")})}});
        CHECK(berr);
        CHECK(btext.find("episode") != std::string::npos);
    }

    // ---- memory_update (tags on each surface's own note; text rejected) --
    const auto uh = post_json(fx, "/memory/update",
                              {{"id", "knowledge:" + id_of(nh)}, {"tags", "lr72,updated"}});
    CHECK(uh["_status"] == 200);
    const auto [uerr, utext] = mcp_call("memory_update",
                                        {{"id", "knowledge:" + nm["id"].get<std::string>()},
                                         {"tags", "lr72,updated"}});
    CHECK(!uerr);
    const auto um = json::parse(utext);
    // agent_update answers the updated row: same key set, same tags
    CHECK(um.size() + 2 == uh.size());   // uh carries _status/_raw bookkeeping
    CHECK(um.value("tags", "") == uh.value("tags", ""));
    err_parity("memory_update", {{"id", "knowledge:1"}, {"text", "no in-place edit"}},
               post_json(fx, "/memory/update", {{"id", "knowledge:1"}, {"text", "no in-place edit"}}));

    // ---- memory_review (LR S88: HTTP only; MCP no longer lists it) -------
    const auto rh = post_json(fx, "/memory/review", {{"action", "list"}});
    CHECK(rh["_status"] == 200);
    {
        const auto [rerr, rtext] = mcp_call("memory_review", {{"action", "list"}});
        CHECK(rerr);   // removed from tools/list AND dispatch (D7, D-2)
        CHECK(rtext.find("unknown tool") != std::string::npos
              || rtext.find("Unknown tool") != std::string::npos);
    }

    // ---- turn (capture_turn) ----------------------------------------------
    const json targs{{"user", "lr72 turn user"}, {"assistant", "lr72 turn assistant"},
                     {"session_id", "lr72-sess-turn"}};
    const auto th = post_json(fx, "/turn", targs);
    CHECK(th["_status"] == 200);
    CHECK(th["status"] == "captured");   // capture_turns defaults on
    const auto [cerr, ctext] = mcp_call("capture_turn", targs);
    CHECK(!cerr);
    const auto cm = json::parse(ctext);
    CHECK(cm["status"] == th["status"]);
    CHECK(cm.contains("turn_id"));
    // missing user: unified message + channel (D16: was a JSON error body on
    // HTTP, 'Error: user parameter required' on MCP)
    const auto mu = post_json(fx, "/turn", json{{"assistant", "x"}});
    CHECK(mu["_status"] == 400);
    CHECK(mu["_raw"] == "ERROR: user is required");
    const auto [merr, mtext] = mcp_call("capture_turn", json{{"assistant", "x"}});
    CHECK(merr);
    CHECK(mtext == "user is required");

    // ---- session (build_context) ------------------------------------------
    auto sr = fx.client.get(fx.url("/session/lr72-sess"), fx.auth());
    CHECK(sr.ok());
    const auto sh = json::parse(sr.body);
    const auto [berr, btext] = mcp_call("build_context", {{"session_id", "lr72-sess"}});
    CHECK(!berr);
    CHECK(json::parse(btext) == sh);    // fixture leaves build_context off: {status:"disabled"}
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

// LR S58: the legacy hard-delete surface is gone — DELETE /memory/<id> and
// POST /delete_batch are unknown routes now (404 like any other), not 200/400.
void t_delete_routes_gone(HttpFixture& fx) {
    std::println("  t_delete_routes_gone...");
    auto d = fx.client.request("DELETE", fx.url("/memory/1"), "", fx.auth());
    CHECK(d.ok());
    CHECK(d.status == 404);
    auto b = fx.client.post(fx.url("/delete_batch"), R"({"ids":[1]})", fx.auth());
    CHECK(b.ok());
    CHECK(b.status == 404);
    std::println(" OK");
}

// X8 (M4): GET /user/token is removed — it handed the bearer token to any
// pre-authenticated caller and read ~/.ragger/token (ignoring --ragger-base).
// It is an unknown route now (404 like any other), not 200/404-with-token.
void t_user_token_route_gone(HttpFixture& fx) {
    std::println("  t_user_token_route_gone...");
    auto r = fx.client.get(fx.url("/user/token"), fx.auth());
    CHECK(r.ok());
    CHECK(r.status == 404);
    std::println(" OK");
}

// LR S59: memory_note accepts trigger (insights only) and session.
void t_memory_note_trigger_session(HttpFixture& fx) {
    std::println("  t_memory_note_trigger_session...");
    const std::string db_path = fx.env.base() + "/http_test.db";

    // Seed a session + turn through the shared backend, then an episode raw
    // (the fixture runs no summarizer). LR S80: evidence points at episodes.
    const int turn = fx.mem->store_turn("lr59 q", "lr59 a", "", false, "lr59-sess");
    CHECK(turn > 0);
    long long epid = -1;
    {
        RawDb db(db_path);
        db.exec("INSERT INTO episodes (text, session_id, span_start, span_end) "
                "SELECT 'lr59 evidence episode', session_id, created_at, created_at "
                "FROM turns WHERE turn_id = " + std::to_string(turn));
        epid = db.scalar("SELECT episode_id FROM episodes WHERE text = 'lr59 evidence episode'");
    }
    CHECK(epid > 0);

    // Insight with trigger -> stored with trigger; memory_search echoes it.
    auto ins = post_json(fx, "/memory/note",
                         {{"text", "lr59 synthesized insight"},
                          {"evidence", json::array({epid})},
                          {"trigger", "when probing note triggers"}});
    CHECK(ins["_status"] == 200);
    CHECK(ins["kind"] == "insight");
    {
        RawDb db(db_path);
        CHECK(db.scalar("SELECT COUNT(*) FROM knowledge WHERE knowledge_id = " + id_of(ins) +
                        " AND trigger = 'when probing note triggers'") == 1);
    }
    auto found = post_json(fx, "/memory/search", {{"query", "lr59 synthesized insight"}});
    CHECK(found["_status"] == 200);
    bool saw_trigger = false;
    for (const auto& hit : found.value("results", json::array()))
        if (hit.contains("extra") &&
            hit["extra"].value("trigger", "") == "when probing note triggers")
            saw_trigger = true;
    CHECK(saw_trigger);

    // Plain note with session -> knowledge.session_id set.
    auto note = post_json(fx, "/memory/note",
                          {{"text", "lr59 note with session"}, {"session", "lr59-sess"}});
    CHECK(note["_status"] == 200);
    {
        RawDb db(db_path);
        CHECK(db.scalar("SELECT session_id IS NOT NULL FROM knowledge WHERE knowledge_id = "
                        + id_of(note)) == 1);
    }

    // Trigger on a plain note -> 400 naming the rule.
    auto bad = post_json(fx, "/memory/note",
                         {{"text", "lr59 note with trigger"}, {"trigger", "nope"}});
    CHECK(bad["_status"] == 400);
    CHECK(bad["_raw"].get<std::string>().find("only valid on an insight") != std::string::npos);

    // Unknown session GUID -> stored with NULL session, no error.
    auto unk = post_json(fx, "/memory/note",
                         {{"text", "lr59 note unknown session"},
                          {"session", "lr59-no-such-guid"}});
    CHECK(unk["_status"] == 200);
    {
        RawDb db(db_path);
        CHECK(db.scalar("SELECT session_id IS NULL FROM knowledge WHERE knowledge_id = "
                        + id_of(unk)) == 1);
    }
    std::println(" OK");
}

// LR S76 (D-7): document_search is one function across transports, and the
// document layer is rejected on both with the same pointer message.
void t_document_search_parity_s76(HttpFixture& fx) {
    std::println("  t_document_search_parity_s76...");
    ragger::DocumentChunk d;
    d.text = "The lr76 harbor ledger lists otter sightings.";
    d.title = "lr76 ledger";
    d.path = "/docs/lr76-ledger.md";
    d.chunk_index = 1;
    CHECK(fx.mem->store_document(d) > 0);

    const auto dh = post_json(fx, "/memory/documents",
                              {{"query", "harbor ledger otter"}});
    CHECK(dh["_status"] == 200);
    CHECK(!dh["results"].empty());
    {
        bool found = false;
        for (const auto& h : dh["results"])
            if (h.value("layer", "") == "document" &&
                h["extra"].value("source_path", "") == "/docs/lr76-ledger.md")
                found = true;
        CHECK(found);
    }

    auto mr = ragger::mcp_tool_call(*fx.mem, {{"name", "document_search"},
                                              {"arguments", {{"query", "harbor ledger otter"}}}});
    CHECK(!mr.value("isError", false));
    const auto mp = json::parse(mr["content"][0]["text"].get<std::string>());
    CHECK(mp["results"].size() == dh["results"].size());

    // source filter narrows on both surfaces.
    const auto dn = post_json(fx, "/memory/documents",
                              {{"query", "harbor ledger otter"}, {"source", "no-such"}});
    CHECK(dn["_status"] == 200);
    CHECK(dn["results"].empty());

    // Missing query: same message, both surfaces (D-4 mapping).
    const auto bad = post_json(fx, "/memory/documents", {{"query", ""}});
    CHECK(bad["_status"] == 400);
    CHECK(bad["_raw"] == "ERROR: query is required");

    // LR S77: unknown mode -> 400 with the same message on both surfaces.
    const auto bm = post_json(fx, "/memory/documents",
                              {{"query", "otter"}, {"mode", "astrology"}});
    CHECK(bm["_status"] == 400);
    CHECK(bm["_raw"].get<std::string>().find("hybrid, semantic, keyword") != std::string::npos);
    auto br = ragger::mcp_tool_call(*fx.mem, {{"name", "document_search"},
                                              {"arguments", {{"query", "otter"},
                                                             {"mode", "astrology"}}}});
    CHECK(br.value("isError", false));
    CHECK(br["content"][0]["text"].get<std::string>().find("hybrid, semantic, keyword")
              != std::string::npos);

    // The document layer is rejected with the pointer on both surfaces.
    const auto ml = post_json(fx, "/memory/search",
                              {{"query", "otter"}, {"layers", json::array({"document"})}});
    CHECK(ml["_status"] == 400);
    CHECK(ml["_raw"].get<std::string>().find("document_search") != std::string::npos);
    auto er = ragger::mcp_tool_call(*fx.mem, {{"name", "memory_search"},
                                              {"arguments", {{"query", "otter"},
                                                             {"layers", json::array({"document"})}}}});
    CHECK(er.value("isError", false));
    CHECK(er["content"][0]["text"].get<std::string>().find("document_search") != std::string::npos);
    std::println(" OK");
}

/// LR S92 (D4): the dashboard carries the Maintenance card — every element id
/// the job runner drives, one button per S91 job, the Dry-run checkbox, and
/// the route paths the buttons post to. No browser in CTest: the shipped HTML
/// is compiled into the binary, so fetching it asserts the wiring exists.
static void t_dashboard_maintenance_s92(HttpFixture& fx) {
    const auto r = fx.client.get(fx.url("/dashboard"), fx.auth());
    CHECK(r.status == 200);
    const std::string html = r.body;
    CHECK(!html.empty());
    for (const char* needle : {"maintenance-card", "maint-btn-reindex",
                               "maint-btn-turnsummary", "maint-btn-triggers",
                               "maint-btn-stranded", "maint-btn-sessions",
                               "maint-dryrun", "maint-progress",
                               "maint-cancel", "maint-result",
                               "/jobs/", "/jobs/cancel"}) {
        CHECK(html.find(needle) != std::string::npos);
    }
    // No embedding job button (rule 17: Reid runs re-embed himself).
    CHECK(html.find("maint-btn-embed") == std::string::npos);
}

/// LR S95 (P12): GET /tools is the MCP tools/list array over HTTP — same
/// embedded tools/*.json, same gates. Equality is exact (same json values),
/// and flipping build_context adds exactly that tool on both surfaces.
static void t_tools_parity_s95(HttpFixture& fx) {
    const auto r = fx.client.get(fx.url("/tools"), fx.auth());
    CHECK(r.status == 200);
    const auto body = nlohmann::json::parse(r.body);
    CHECK(body == ragger::mcp_tools_list(ragger::config().capture_turns,
                                         ragger::config().build_context));
    const auto has = [](const nlohmann::json& j, const char* name) {
        for (const auto& t : j.at("tools"))
            if (t.at("name").get<std::string>() == name) return true;
        return false;
    };
    CHECK(has(body, "store"));
    CHECK(!has(body, "build_context"));  // default off (default-settings.txt)
    CHECK(ragger::set_config_persisted("build_context", "true").has_value());
    const auto body2 =
        nlohmann::json::parse(fx.client.get(fx.url("/tools"), fx.auth()).body);
    CHECK(has(body2, "build_context"));
    CHECK(body2 == ragger::mcp_tools_list(ragger::config().capture_turns, true));
    (void)ragger::set_config_persisted("build_context", "");  // restore default
}

// X22 test 5: /turn after a same-exchange merge must enqueue the
// turn's CURRENT assistant_text (the whole reply), not just the fragment of
// the request that merged it. No inference is configured in this fixture, so
// the summarizer's LR S62 floor copies the enqueued source verbatim: the turn summary
// row that reappears after the merge shows exactly what was enqueued.
void x22_test_turn_merge_enqueues_whole_reply(HttpFixture& fx) {
    std::println("  x22_test_turn_merge_enqueues_whole_reply...");
    const std::string guid = "x22-http-merge";
    RawDb db((fx.env.base() + "/http_test.db").c_str());
    auto turn_summary_for = [&](int turn_id) -> std::string {
        sqlite3_stmt* st = nullptr;
        CHECK(sqlite3_prepare_v2(db.h,
            "SELECT text FROM turn_summaries WHERE turn_id = ?", -1, &st, nullptr) == SQLITE_OK);
        sqlite3_bind_int(st, 1, turn_id);
        std::string t;
        if (sqlite3_step(st) == SQLITE_ROW) {
            const unsigned char* c = sqlite3_column_text(st, 0);
            t = c ? reinterpret_cast<const char*>(c) : "";
        }
        sqlite3_finalize(st);
        return t;
    };

    // Capture 1; the floor copy lands.
    const auto r1 = post_json(fx, "/turn",
        {{"user", "x22 whole reply question"}, {"assistant", "first half alpha"},
         {"session_id", guid}});
    CHECK(r1["_status"] == 200);
    CHECK(r1["status"] == "captured");
    const int turn_id = r1["turn_id"].get<int>();
    std::string ts_text;
    for (int i = 0; i < 200 && ts_text.empty(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        ts_text = turn_summary_for(turn_id);
    }
    CHECK(ts_text == "x22 whole reply question | first half alpha");

    // Drop the turn-summary row, then merge a second reply into the SAME turn. The
    // live enqueue must carry the merged text — the floor copy proves it.
    db.exec("DELETE FROM turn_summaries WHERE turn_id = " + std::to_string(turn_id));
    const auto r2 = post_json(fx, "/turn",
        {{"user", "x22 whole reply question"}, {"assistant", "second half beta"},
         {"session_id", guid}});
    CHECK(r2["_status"] == 200);
    CHECK(r2["turn_id"] == turn_id);   // merged into the same row
    ts_text.clear();
    for (int i = 0; i < 200 && ts_text.empty(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        ts_text = turn_summary_for(turn_id);
    }
    CHECK(ts_text.find("first half alpha") != std::string::npos);    // not just the fragment
    CHECK(ts_text.find("second half beta") != std::string::npos);
    std::println(" ok");
}

// X32 pin: POST /turn WITHOUT started_at stores turns.started_on NULL (the
// capture hook sends it only after the plugin update; NULL keeps today's
// gap behaviour).
void x32_pin_turn_default_started_on_null(HttpFixture& fx) {
    std::println("  x32_pin_turn_default_started_on_null...");
    RawDb db((fx.env.base() + "/http_test.db").c_str());
    const auto r = post_json(fx, "/turn",
        {{"user", "x32 pin no started_at"}, {"assistant", "x32 pin reply"},
         {"session_id", "x32-pin-default-null"}});
    CHECK(r["_status"] == 200);
    CHECK(r["status"] == "captured");
    sqlite3_stmt* st = nullptr;
    CHECK(sqlite3_prepare_v2(db.h,
        "SELECT started_on FROM turns WHERE turn_id = ?", -1, &st, nullptr) == SQLITE_OK);
    sqlite3_bind_int(st, 1, r["turn_id"].get<int>());
    CHECK(sqlite3_step(st) == SQLITE_ROW);
    CHECK(sqlite3_column_type(st, 0) == SQLITE_NULL);
    sqlite3_finalize(st);
    std::println(" ok");
}

// X32: POST /turn started_at — valid epoch, valid db-format timestamp string,
// and each validation error (400 + exact message).
void x32_test_turn_started_at(HttpFixture& fx) {
    std::println("  x32_test_turn_started_at...");
    RawDb db((fx.env.base() + "/http_test.db").c_str());
    auto started_of = [&](int turn_id) -> long long {
        return db.scalar("SELECT COALESCE(started_on, -1) FROM turns WHERE turn_id = "
                         + std::to_string(turn_id));
    };

    // Valid JSON integer epoch (in the past): stored verbatim.
    const auto now = ragger::db_epoch();
    const auto r1 = post_json(fx, "/turn",
        {{"user", "x32 started epoch"}, {"assistant", "reply one"},
         {"session_id", "x32-started-at"}, {"started_at", now - 60}});
    CHECK(r1["_status"] == 200);
    CHECK(started_of(r1["turn_id"].get<int>()) == now - 60);

    // Valid db-format timestamp string (what source_timestamp accepts),
    // round-tripped through the shared parser.
    const std::string ts = ragger::db_timestamp(static_cast<std::time_t>(now - 120));
    const auto r2 = post_json(fx, "/turn",
        {{"user", "x32 started string"}, {"assistant", "reply two"},
         {"session_id", "x32-started-at"}, {"started_at", ts}});
    CHECK(r2["_status"] == 200);
    CHECK(started_of(r2["turn_id"].get<int>()) == now - 120);

    // null = not given -> NULL (same as absent).
    const auto r3 = post_json(fx, "/turn",
        {{"user", "x32 started null"}, {"assistant", "reply three"},
         {"session_id", "x32-started-at"}, {"started_at", nullptr}});
    CHECK(r3["_status"] == 200);
    CHECK(started_of(r3["turn_id"].get<int>()) == -1);   // COALESCE sentinel

    // Wrong JSON type (float) -> the type error.
    const auto e1 = post_json(fx, "/turn",
        {{"user", "x32 started bad type"}, {"assistant", "x"},
         {"session_id", "x32-started-at"}, {"started_at", 12.5}});
    CHECK(e1["_status"] == 400);
    CHECK(e1["_raw"] == "ERROR: started_at must be an integer epoch or a timestamp string");

    // Unparseable string -> the same type error.
    const auto e2 = post_json(fx, "/turn",
        {{"user", "x32 started bad string"}, {"assistant", "x"},
         {"session_id", "x32-started-at"}, {"started_at", "yesterday-ish"}});
    CHECK(e2["_status"] == 400);
    CHECK(e2["_raw"] == "ERROR: started_at must be an integer epoch or a timestamp string");

    // More than a day ahead -> future error (checked before after-turn-time).
    const auto e3 = post_json(fx, "/turn",
        {{"user", "x32 started far future"}, {"assistant", "x"},
         {"session_id", "x32-started-at"}, {"started_at", now + 2 * 86400}});
    CHECK(e3["_status"] == 400);
    CHECK(e3["_raw"] == "ERROR: started_at is in the future");

    // Slightly ahead of now (the turn's created_at) -> after-turn-time error.
    const auto e4 = post_json(fx, "/turn",
        {{"user", "x32 started near future"}, {"assistant", "x"},
         {"session_id", "x32-started-at"}, {"started_at", now + 60}});
    CHECK(e4["_status"] == 400);
    CHECK(e4["_raw"] == "ERROR: started_at is after the turn time");

    // MCP parity: same validation through the capture_turn tool.
    const auto mr = ragger::mcp_tool_call(*fx.mem,
        {{"name", "capture_turn"},
         {"arguments", {{"user", "x32 started mcp"}, {"assistant", "x"},
                        {"session_id", "x32-started-at"},
                        {"started_at", now + 60}}}});
    CHECK(mr.value("isError", false));
    CHECK(mr["content"][0]["text"].get<std::string>() == "started_at is after the turn time");
    std::println(" ok");
}

// POST /turn ended_at: stored as created_at (spool replay); started_at is
// validated against it; bad values are 400s.
void t_turn_ended_at(HttpFixture& fx) {
    std::println("  t_turn_ended_at...");
    RawDb db((fx.env.base() + "/http_test.db").c_str());
    const auto now = ragger::db_epoch();
    const auto r1 = post_json(fx, "/turn",
        {{"user", "ended_at replay one"}, {"assistant", "reply"},
         {"session_id", "ended-at"}, {"started_at", now - 7200}, {"ended_at", now - 3600}});
    CHECK(r1.value("status", "") == "captured");
    const int id = r1["turn_id"].get<int>();
    CHECK(db.scalar("SELECT created_at FROM turns WHERE turn_id = " + std::to_string(id)) == now - 3600);
    CHECK(db.scalar("SELECT started_on FROM turns WHERE turn_id = " + std::to_string(id)) == now - 7200);

    const auto e1 = post_json(fx, "/turn",
        {{"user", "ended_at bad order"}, {"assistant", "x"}, {"session_id", "ended-at"},
         {"started_at", now - 60}, {"ended_at", now - 3600}});
    CHECK(e1["_raw"] == "ERROR: started_at is after the turn time");
    const auto e2 = post_json(fx, "/turn",
        {{"user", "ended_at future"}, {"assistant", "x"}, {"session_id", "ended-at"},
         {"ended_at", now + 2 * 86400}});
    CHECK(e2["_raw"] == "ERROR: ended_at is in the future");
    const auto e3 = post_json(fx, "/turn",
        {{"user", "ended_at type"}, {"assistant", "x"}, {"session_id", "ended-at"},
         {"ended_at", 1.5}});
    CHECK(e3["_raw"] == "ERROR: ended_at must be an integer epoch or a timestamp string");
    std::println(" ok");
}

int main() {
    HttpFixture fx("http");
    CHECK(fx.ready);
    std::println("Running HTTP route tests against 127.0.0.1:{}:", fx.port);

    t_health_no_token(fx);
    t_summarizer_model_live(fx);
    t_loopback_auth_contract(fx);
    t_dns_rebinding_host_guard(fx);
    t_store_then_search(fx);
    t_store_empty_text(fx);
    t_store_metadata_type_error(fx);
    t_memory_search_bad_limit(fx);
    x25_test_default_response_shape_pinned(fx);
    x25_test_split_validation(fx);
    x25_test_split_properties(fx);
    t_decision_validation_parity(fx);
    t_store_http_mcp_parity(fx);
    t_read_verbs_parity_s71(fx);
    t_write_verbs_parity_s72(fx);
    t_unknown_route(fx);
    t_delete_routes_gone(fx);
    t_user_token_route_gone(fx);
    t_memory_note_trigger_session(fx);
    t_document_search_parity_s76(fx);
    t_dashboard_maintenance_s92(fx);
    t_tools_parity_s95(fx);
    x22_test_turn_merge_enqueues_whole_reply(fx);
    x32_pin_turn_default_started_on_null(fx);
    x32_test_turn_started_at(fx);
    t_turn_ended_at(fx);

    std::println("test_http: all passed");
    return 0;
}
