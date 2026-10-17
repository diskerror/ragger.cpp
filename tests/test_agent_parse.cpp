/**
 * Agent API request-parsing tests (review 2B).
 *
 * The parse_* functions are the single JSON->struct translation shared by
 * the HTTP surface (server.cpp) and the MCP surface (mcp.cpp); these tests
 * pin the documented defaults, the tags array/string duality, the
 * absent-vs-null distinction for optional fields, and the project/cwd
 * inference rules. Semantic validation is deliberately NOT tested here --
 * that lives in the backend tests (test_sqlite_backend.cpp).
 */
#include "agent_parse.h"
#include "check.h"
#include "config.h"
#include "mcp.h"
#include "test_env.h"
#include "memory.h"
#include "util/parse.h"

#include <filesystem>
#include <iostream>
#include <print>

namespace fs = std::filesystem;
using nlohmann::json;

static const std::string TEMP_DB = "/tmp/ragger_test_agent_parse.db";

static void cleanup() {
    fs::remove(TEMP_DB);
    fs::remove(TEMP_DB + "-wal");
    fs::remove(TEMP_DB + "-shm");
}

// ---- pure parsers (no backend needed) ---------------------------------

void test_tags_to_comma_string() {
    std::println("  test_tags_to_comma_string...");
    CHECK(ragger::tags_to_comma_string(json::object()) == "");
    CHECK(ragger::tags_to_comma_string(json{{"tags", json::array()}}) == "");
    CHECK(ragger::tags_to_comma_string(
              json{{"tags", {"a", "b", "c"}}}) == "a,b,c");
    CHECK(ragger::tags_to_comma_string(
              json{{"tags", "a,b"}}) == "a,b");
    // Other types are ignored (not an error at parse level; the backend
    // never sees them).
    CHECK(ragger::tags_to_comma_string(json{{"tags", 42}}) == "");
    // Custom key.
    CHECK(ragger::tags_to_comma_string(json{{"labels", {"x"}}}, "labels") == "x");
    std::println(" OK");
}

void test_supersedes_json() {
    std::println("  test_supersedes_json...");
    auto empty = ragger::supersedes_json({});
    CHECK(empty.is_array() && empty.empty());
    auto j = ragger::supersedes_json({{7, 0.5f}, {9, 0.25f}});
    CHECK(j.size() == 2);
    CHECK(j[0]["id"] == 7);
    CHECK(j[1]["id"] == 9);
    std::println(" OK");
}

void test_parse_agent_get_query() {
    std::println("  test_parse_agent_get_query...");
    auto q = ragger::parse_agent_get_query(json::object());
    CHECK(q.ids.empty());
    CHECK(q.expand == "none");
    CHECK(q.session == "");
    CHECK(q.after_turn == 0 && q.before_turn == 0);
    CHECK(q.limit == 20);

    auto p = ragger::parse_agent_get_query(json{
        {"ids", {"decision:1", "turn:2"}},
        {"expand", "sources"},
        {"session", "guid-abc"},
        {"after_turn", 5},
        {"before_turn", 9},
        {"limit", 3}});
    CHECK(p.ids.size() == 2 && p.ids[0] == "decision:1");
    CHECK(p.expand == "sources");
    CHECK(p.session == "guid-abc");
    CHECK(p.after_turn == 5 && p.before_turn == 9);
    CHECK(p.limit == 3);
    std::println(" OK");
}

void test_parse_update_request() {
    std::println("  test_parse_update_request...");
    auto u = ragger::parse_update_request(json{{"id", "decision:1"}});
    CHECK(u.id == "decision:1");
    CHECK(!u.status.has_value() && !u.tags.has_value() &&
          !u.project.has_value() && !u.text.has_value());

    // null means absent (stays nullopt), a string sets the value.
    auto n = ragger::parse_update_request(json{
        {"id", "summary:2"},
        {"status", nullptr},
        {"tags", "flange"},
        {"project", ""}});
    CHECK(!n.status.has_value());
    CHECK(n.tags.has_value() && *n.tags == "flange");
    CHECK(n.project.has_value() && n.project->empty());   // "" = clear

    // text is forwarded when present (agent_update rejects it with the
    // canonical message; both surfaces must behave the same).
    auto t = ragger::parse_update_request(json{
        {"id", "decision:1"}, {"text", "rewrite"}});
    CHECK(t.text.has_value() && *t.text == "rewrite");
    std::println(" OK");
}

// ---- parsers with project/cwd inference --------------------------------

void test_search_and_timeline_parsing(ragger::RaggerMemory& mem) {
    std::println("  test_search_and_timeline_parsing...");

    // Defaults.
    auto q = ragger::parse_agent_search_query(json{{"query", "gears"}}, mem);
    CHECK(q.query == "gears");
    CHECK(q.layers.empty());
    CHECK(q.status == "active");
    CHECK(q.limit == 10 && q.max_tokens == 0);
    CHECK(q.project == "");

    // Explicit fields.
    auto p = ragger::parse_agent_search_query(json{
        {"query", "torque"},
        {"layers", {"decision", "insight"}},
        {"since", "30d"}, {"until", "2026-01-01"},
        {"status", "any"},
        {"session", "g1"}, {"exclude_session", "g2"},
        {"limit", 5}, {"max_tokens", 800},
        {"project", "ParseProj"}}, mem);
    CHECK(p.layers.size() == 2 && p.layers[1] == "insight");
    CHECK(p.since == "30d" && p.until == "2026-01-01");
    CHECK(p.status == "any");
    CHECK(p.session == "g1" && p.exclude_session == "g2");
    CHECK(p.limit == 5 && p.max_tokens == 800);
    CHECK(p.project == "ParseProj");   // explicit wins over absent cwd

    // cwd inference: matching path fills the filter...
    auto c = ragger::parse_agent_search_query(
        json{{"query", "x"}, {"cwd", "/tmp/ragger_parse_proj/src"}}, mem);
    CHECK(c.project == "/tmp/ragger_parse_proj/src");

    // ...a non-matching cwd leaves it empty (not an error).
    auto m = ragger::parse_agent_search_query(
        json{{"query", "x"}, {"cwd", "/nope/nothing"}}, mem);
    CHECK(m.project == "");

    // explicit project wins over a matching cwd.
    auto w = ragger::parse_agent_search_query(json{
        {"query", "x"}, {"project", "ParseProj"},
        {"cwd", "/tmp/ragger_parse_proj/src"}}, mem);
    CHECK(w.project == "ParseProj");

    // Timeline: defaults + cwd inference.
    auto tl = ragger::parse_timeline_query(json::object(), mem);
    CHECK(tl.granularity == "episode");
    CHECK(tl.limit == 20);
    CHECK(tl.cursor == "" && tl.project == "");
    auto tc = ragger::parse_timeline_query(
        json{{"cwd", "/tmp/ragger_parse_proj"}}, mem);
    CHECK(tc.project == "/tmp/ragger_parse_proj");
    std::println(" OK");
}

void test_decision_input_parsing(ragger::RaggerMemory& mem) {
    std::println("  test_decision_input_parsing...");

    auto in = ragger::parse_decision_input(json{{"text", "Always X."}}, mem);
    CHECK(in.text == "Always X.");
    CHECK(in.status == "current");
    CHECK(!in.supersedes.has_value());
    CHECK(in.evidence.empty());
    CHECK(in.project == "");

    // tags: array form and string form produce the same comma string.
    auto ta = ragger::parse_decision_input(
        json{{"text", "t"}, {"tags", {"a", "b"}}}, mem);
    auto ts = ragger::parse_decision_input(
        json{{"text", "t"}, {"tags", "a,b"}}, mem);
    CHECK(ta.tags == "a,b" && ts.tags == "a,b");

    // supersedes null stays absent; an int sets it.
    auto sn = ragger::parse_decision_input(
        json{{"text", "t"}, {"supersedes", nullptr}}, mem);
    CHECK(!sn.supersedes.has_value());
    auto sv = ragger::parse_decision_input(
        json{{"text", "t"}, {"supersedes", 12}}, mem);
    CHECK(sv.supersedes.has_value() && *sv.supersedes == 12);

    // evidence + session + cwd inference.
    auto ev = ragger::parse_decision_input(json{
        {"text", "t"}, {"evidence", {3, 5}},
        {"session", "guid-9"},
        {"cwd", "/tmp/ragger_parse_proj/sub"}}, mem);
    CHECK(ev.evidence.size() == 2 && ev.evidence[1] == 5);
    CHECK(ev.session_guid == "guid-9");
    CHECK(ev.project == "/tmp/ragger_parse_proj/sub");

    // note-insight branch.
    auto ni = ragger::parse_note_insight(
        json{{"evidence", {1}}, {"tags", {"n1", "n2"}}}, "note text", "P");
    CHECK(ni.text == "note text");
    CHECK(ni.project == "P");
    CHECK(ni.evidence.size() == 1);
    CHECK(ni.tags == "n1,n2");
    std::println(" OK");
}

// PHASE2 Step 4: strict positive-int parser (C §1.10). Rejects the forms
// std::stoi wrongly accepts ("0", "007", "-1", "1x", " 1") and overflow.
void test_parse_positive_int() {
    std::println("  test_parse_positive_int...");
    int out = -999;
    for (const char* ok : {"1", "9", "10", "2147483647"}) {
        out = -999;
        CHECK(ragger::util::parse_positive_int(ok, out));
        CHECK(out == std::stoi(ok));
    }
    for (const char* bad : {"", "0", "007", "-1", "+1", "1x", "x1",
                            " 1", "1 ", "2147483648", "99999999999999999999"}) {
        out = -4242;
        CHECK(!ragger::util::parse_positive_int(bad, out));
        CHECK(out == -4242);  // out untouched on failure
    }
    std::println(" OK");
}

// PHASE2 Step 4: a mistyped MCP argument ({"limit":"10"}) must come back as
// an isError tool result, not throw (C §1.12).
void test_mcp_tool_call_bad_arg_type(ragger::RaggerMemory& mem) {
    std::println("  test_mcp_tool_call_bad_arg_type...");
    json params = {{"name", "memory_search"},
                   {"arguments", {{"query", "hello"}, {"limit", "10"}}}};
    bool threw = false;
    json r;
    try {
        r = ragger::mcp_tool_call(mem, params);
    } catch (...) {
        threw = true;
    }
    CHECK(!threw);
    CHECK(r.value("isError", false) == true);
    std::println(" OK");
}

int main() {
    TestEnv env("agent_parse");  // must be first: isolates from ~/.ragger
    std::println("Running agent-parse tests:");
    cleanup();

    test_tags_to_comma_string();
    test_supersedes_json();
    test_parse_agent_get_query();
    test_parse_update_request();
    test_parse_positive_int();

    {
        ragger::RaggerMemory mem(TEMP_DB);
        mem.create_project("ParseProj", "/tmp/ragger_parse_proj", "");
        test_search_and_timeline_parsing(mem);
        test_decision_input_parsing(mem);
        test_mcp_tool_call_bad_arg_type(mem);
        mem.close();
    }

    cleanup();
    std::println("test_agent_parse: all checks passed");
    return 0;
}
