/**
 * Agent API request-parsing tests (review 2B).
 *
 * The parse_* functions are the single JSON->struct translation shared by
 * the HTTP surface (server.cpp) and the MCP surface (mcp.cpp); these tests
 * pin the documented defaults, the tags array/string duality, the
 * absent-vs-null distinction for optional fields. "project" is a tag now
 * (PHASE4) — the parsers no longer resolve or infer anything.
 * Semantic validation is deliberately NOT tested here --
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
#include <set>
#include <sstream>

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
    CHECK(!u.status.has_value() && !u.tags.has_value() && !u.text.has_value());

    // null means absent (stays nullopt), a string sets the value. A
    // leftover "project" field is simply ignored (unknown field).
    auto n = ragger::parse_update_request(json{
        {"id", "summary:2"},
        {"status", nullptr},
        {"tags", "flange"},
        {"project", "gone"}});
    CHECK(!n.status.has_value());
    CHECK(n.tags.has_value() && *n.tags == "flange");

    // text is forwarded when present (agent_update rejects it with the
    // canonical message; both surfaces must behave the same).
    auto t = ragger::parse_update_request(json{
        {"id", "decision:1"}, {"text", "rewrite"}});
    CHECK(t.text.has_value() && *t.text == "rewrite");
    std::println(" OK");
}

// ---- pure query/input parsers ------------------------------------------

void test_search_and_timeline_parsing() {
    std::println("  test_search_and_timeline_parsing...");

    // Defaults.
    auto q = ragger::parse_agent_search_query(json{{"query", "gears"}});
    CHECK(q.query == "gears");
    CHECK(q.layers.empty());
    CHECK(q.status == "active");
    CHECK(q.limit == 10 && q.max_tokens == 0);

    // Explicit fields. A leftover "project"/"cwd" is ignored (PHASE4).
    auto p = ragger::parse_agent_search_query(json{
        {"query", "torque"},
        {"layers", {"decision", "insight"}},
        {"since", "30d"}, {"until", "2026-01-01"},
        {"status", "any"},
        {"session", "g1"}, {"exclude_session", "g2"},
        {"limit", 5}, {"max_tokens", 800},
        {"project", "ParseProj"}, {"cwd", "/tmp/x"}});
    CHECK(p.layers.size() == 2 && p.layers[1] == "insight");
    CHECK(p.since == "30d" && p.until == "2026-01-01");
    CHECK(p.status == "any");
    CHECK(p.session == "g1" && p.exclude_session == "g2");
    CHECK(p.limit == 5 && p.max_tokens == 800);

    // Timeline: defaults.
    auto tl = ragger::parse_timeline_query(json::object());
    CHECK(tl.granularity == "episode");
    CHECK(tl.limit == 20);
    CHECK(tl.cursor == "");
    std::println(" OK");
}

void test_decision_input_parsing() {
    std::println("  test_decision_input_parsing...");

    auto in = ragger::parse_decision_input(json{{"text", "Always X."}});
    CHECK(in.text == "Always X.");
    CHECK(in.status == "current");
    CHECK(!in.supersedes.has_value());
    CHECK(in.evidence.empty());

    // tags: array form and string form produce the same comma string.
    auto ta = ragger::parse_decision_input(
        json{{"text", "t"}, {"tags", {"a", "b"}}});
    auto ts = ragger::parse_decision_input(
        json{{"text", "t"}, {"tags", "a,b"}});
    CHECK(ta.tags == "a,b" && ts.tags == "a,b");

    // supersedes null stays absent; an int sets it.
    auto sn = ragger::parse_decision_input(
        json{{"text", "t"}, {"supersedes", nullptr}});
    CHECK(!sn.supersedes.has_value());
    auto sv = ragger::parse_decision_input(
        json{{"text", "t"}, {"supersedes", 12}});
    CHECK(sv.supersedes.has_value() && *sv.supersedes == 12);

    // evidence + session.
    auto ev = ragger::parse_decision_input(json{
        {"text", "t"}, {"evidence", {3, 5}},
        {"session", "guid-9"}});
    CHECK(ev.evidence.size() == 2 && ev.evidence[1] == 5);
    CHECK(ev.session_guid == "guid-9");

    // LR S80 (D2): evidence accepts bare ints and "episode:N" handles;
    // turn_summary:N / turn:N handles are rejected pointing at episode ids.
    auto eh = ragger::parse_decision_input(json{
        {"text", "t"}, {"evidence", {std::string("episode:7"), 9}}});
    CHECK(eh.evidence.size() == 2 && eh.evidence[0] == 7 && eh.evidence[1] == 9);
    for (const char* h : {"turn_summary:7", "turn:7", "nonsense"}) {
        bool et = false;
        try { ragger::parse_decision_input(json{{"text", "t"}, {"evidence", {std::string(h)}}}); }
        catch (const std::invalid_argument& e) {
            et = std::string(e.what()).find("episode") != std::string::npos;
        }
        CHECK(et);
    }

    // note-insight branch.
    auto ni = ragger::parse_note_insight(
        json{{"evidence", {1}}, {"tags", {"n1", "n2"}}}, "note text");
    CHECK(ni.text == "note text");
    CHECK(ni.evidence.size() == 1);
    CHECK(ni.tags == "n1,n2");

    // LR S59: the insight branch also carries trigger + session.
    auto ni2 = ragger::parse_note_insight(
        json{{"evidence", {1}}, {"trigger", "when probing"}, {"session", "guid-3"}},
        "note text");
    CHECK(ni2.trigger == "when probing");
    CHECK(ni2.session_guid == "guid-3");

    // LR S59 (D-3): trigger without evidence is a client error; with
    // evidence or absent it passes.
    bool threw = false;
    try { ragger::validate_note_request(json{{"text", "t"}, {"trigger", "x"}}); }
    catch (const std::invalid_argument& e) {
        threw = true;
        CHECK(std::string(e.what()).find("only valid on an insight") != std::string::npos);
    }
    CHECK(threw);
    threw = false;
    try { ragger::validate_note_request(json{{"text", "t"}, {"trigger", "x"},
                                             {"evidence", {1}}}); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(!threw);
    threw = false;
    try { ragger::validate_note_request(json{{"text", "t"}}); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(!threw);
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

// CR S6: an unknown get_config key must say "Unknown config key: <key>", not
// reuse the unknown-tool message; an unknown tool keeps the old message.
static std::string p10_mcp_error_text(ragger::RaggerMemory& mem,
                                      const std::string& tool,
                                      const json& arguments) {
    json params = {{"name", tool}, {"arguments", arguments}};
    auto r = ragger::mcp_tool_call(mem, params);
    CHECK(r.value("isError", false) == true);
    return r["content"][0]["text"].get<std::string>();
}

void p10_test_mcp_unknown_config_key(ragger::RaggerMemory& mem) {
    std::println("  p10_test_mcp_unknown_config_key...");
    CHECK(p10_mcp_error_text(mem, "get_config", {{"key", "nope"}})
          == "Unknown config key: nope");
    CHECK(p10_mcp_error_text(mem, "no_such_tool", json::object())
          == "Unknown tool: no_such_tool");
    std::println(" OK");
}

// S54 (C §7 item 2): the tools/list contract. mcp_tools_list takes the two
// gating flags directly, so all four combinations are testable without
// touching persisted settings (the old tools_list() read config() itself).
void p12_mcp_tools_list_contract() {
    std::println("  p12_mcp_tools_list_contract...");
    const auto find_tool = [](const json& tools, const char* name) -> json {
        for (const auto& t : tools)
            if (t.value("name", "") == name) return t;
        return json();
    };
    for (const bool cap : {false, true}) {
        for (const bool ctx : {false, true}) {
            const auto result = ragger::mcp_tools_list(cap, ctx);
            CHECK(result.contains("tools"));
            const auto tools = result["tools"];
            CHECK(tools.is_array());
            // Every tool: unique name, non-empty description, object schema.
            std::set<std::string> names;
            for (const auto& t : tools) {
                CHECK(t.contains("name"));
                CHECK(names.insert(t["name"].get<std::string>()).second);
                CHECK(!t.value("description", "").empty());
                CHECK(t.contains("inputSchema"));
                CHECK(t["inputSchema"].value("type", "") == "object");
            }
            // Gating: the two hook tools appear exactly when their flag is on.
            CHECK(find_tool(tools, "capture_turn").is_object() == cap);
            CHECK(find_tool(tools, "build_context").is_object() == ctx);
            // LR S88 (D7, D-2): memory_review left the agent tool lists —
            // curation is user-driven (HTTP route + CLI remain).
            CHECK(find_tool(tools, "memory_review").is_null());
        }
    }

    // S6 vocabulary: memory_search's layer list must not offer a bare
    // `summary` layer (turn_summary is a different, legitimate name — so the
    // check parses the enumeration instead of substring-matching "summary").
    const auto tools = ragger::mcp_tools_list(true, true)["tools"];
    const auto ms = find_tool(tools, "memory_search");
    CHECK(ms.is_object());
    const std::string ms_desc = ms.value("description", "");
    const auto open = ms_desc.find("narrows by layer (");
    CHECK(open != std::string::npos);
    const auto close = ms_desc.find(')', open);
    CHECK(close != std::string::npos);
    bool saw_turn_summary = false;
    std::istringstream layer_list(ms_desc.substr(open + 18, close - open - 18));
    std::string layer;
    while (std::getline(layer_list, layer, ',')) {
        const auto b = layer.find_first_not_of(" \t");
        const auto e = layer.find_last_not_of(" \t");
        layer = b == std::string::npos ? "" : layer.substr(b, e - b + 1);
        CHECK(layer != "summary");
        CHECK(layer != "decisions");
        CHECK(layer != "insights");
        if (layer == "turn_summary") saw_turn_summary = true;
    }
    CHECK(saw_turn_summary);

    // The search tool must not hard-code a numeric default in its schema
    // text (the limit follows the default_search_limit setting).
    const auto search = find_tool(tools, "search");
    CHECK(search.is_object());
    CHECK(search.dump().find("default: 5") == std::string::npos);
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
    test_search_and_timeline_parsing();
    test_decision_input_parsing();

    {
        ragger::RaggerMemory mem(TEMP_DB);
        test_mcp_tool_call_bad_arg_type(mem);
        p10_test_mcp_unknown_config_key(mem);
        p12_mcp_tools_list_contract();
        mem.close();
    }

    cleanup();
    std::println("test_agent_parse: all checks passed");
    return 0;
}
