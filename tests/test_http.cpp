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

    // decision form: category routes to kind='decision' on both surfaces
    const json dargs{{"text", "lr70 parity decision"},
                     {"metadata", {{"category", "decision"}}}};
    const auto dh = post_json(fx, "/store", dargs);
    CHECK(dh["_status"] == 200);
    CHECK(dh["kind"] == "decision");
    CHECK(mcp_payload(dargs)["kind"] == "decision");

    // bad status: rejected with the SAME message (envelope differs)
    const json bargs{{"text", "lr70 parity bad status"},
                     {"metadata", {{"category", "decision"}, {"status", "not-a-status"}}}};
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

    // ---- memory_note (insight) — seed an L2 row to point at -------------
    const std::string db_path = fx.env.base() + "/http_test.db";
    const int turn = fx.mem->store_turn("lr72 q", "lr72 a", "", false, "lr72-sess");
    CHECK(turn > 0);
    long long tsid = -1;
    {
        RawDb db(db_path);
        db.exec("INSERT INTO turn_summaries (text, turn_id, session_id, turn_datetime) "
                "SELECT 'lr72 evidence summary', " + std::to_string(turn) +
                ", session_id, created_at FROM turns WHERE turn_id = " + std::to_string(turn));
        tsid = db.scalar("SELECT turn_summary_id FROM turn_summaries WHERE turn_id = "
                         + std::to_string(turn));
    }
    CHECK(tsid > 0);
    const json ins_args{{"text", "lr72 parity insight"},
                        {"evidence", json::array({tsid})},
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

    // ---- memory_review ----------------------------------------------------
    const auto rh = post_json(fx, "/memory/review", {{"action", "list"}});
    CHECK(rh["_status"] == 200);
    const auto [rerr, rtext] = mcp_call("memory_review", {{"action", "list"}});
    CHECK(!rerr);
    auto rh_clean = rh;
    rh_clean.erase("_status");
    rh_clean.erase("_raw");
    CHECK(json::parse(rtext) == rh_clean);   // empty queue: byte-equal payloads
    err_parity("memory_review", {{"action", "explode"}},
               post_json(fx, "/memory/review", {{"action", "explode"}}));

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
    t_store_http_mcp_parity(fx);
    t_read_verbs_parity_s71(fx);
    t_write_verbs_parity_s72(fx);
    t_unknown_route(fx);
    t_delete_routes_gone(fx);
    t_memory_note_trigger_session(fx);
    t_document_search_parity_s76(fx);

    std::println("test_http: all passed");
    return 0;
}
