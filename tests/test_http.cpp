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
#include <sqlite3.h>
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

// id fields differ per path (plain note: string; insight: int) — normalize.
std::string id_of(const json& j) {
    return j["id"].is_string() ? j["id"].get<std::string>() : j["id"].dump();
}

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

// LR S59: memory_note accepts trigger (insights only) and session.
void t_memory_note_trigger_session(HttpFixture& fx) {
    std::println("  t_memory_note_trigger_session...");
    const std::string db_path = fx.env.base() + "/http_test.db";

    // Seed a session + turn through the shared backend, then its L2 row raw
    // (the fixture runs no summarizer).
    const int turn = fx.mem->store_turn("lr59 q", "lr59 a", "", false, "lr59-sess");
    CHECK(turn > 0);
    long long tsid = -1;
    {
        RawDb db(db_path);
        db.exec("INSERT INTO turn_summaries (text, turn_id, session_id, turn_datetime) "
                "SELECT 'lr59 evidence summary', " + std::to_string(turn) +
                ", session_id, created_at FROM turns WHERE turn_id = " + std::to_string(turn));
        tsid = db.scalar("SELECT turn_summary_id FROM turn_summaries WHERE turn_id = "
                         + std::to_string(turn));
    }
    CHECK(tsid > 0);

    // Insight with trigger -> stored with trigger; memory_search echoes it.
    auto ins = post_json(fx, "/memory/note",
                         {{"text", "lr59 synthesized insight"},
                          {"evidence", json::array({tsid})},
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
    t_delete_routes_gone(fx);
    t_memory_note_trigger_session(fx);

    std::println("test_http: all passed");
    return 0;
}
