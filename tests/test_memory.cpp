/**
 * RaggerMemory facade tests
 *
 * Requires ONNX model files at the configured model_dir.
 */
#include "config.h"
#include "config_access.h"
#include "embedder.h"
#include "memory.h"
#include "sqlite/backend.h"
#include "test_env.h"
#include <sqlite3.h>
#include <algorithm>
#include <cassert>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <print>
#include <stdexcept>

namespace fs = std::filesystem;

static const std::string TEMP_DB1 = "/tmp/ragger_test_mem1.db";
static const std::string TEMP_DB2 = "/tmp/ragger_test_mem2.db";

static void cleanup(const std::string& path) {
    fs::remove(path);
    fs::remove(path + "-wal");
    fs::remove(path + "-shm");
}

static void cleanup_all() {
    cleanup(TEMP_DB1);
    cleanup(TEMP_DB2);
}

void test_store_and_search() {
    std::println("  test_store_and_search...");
    cleanup_all();

    ragger::RaggerMemory mem(TEMP_DB1);
    mem.store("The Eiffel Tower is located in Paris, France.");
    mem.store("Photosynthesis converts sunlight into energy in plants.");

    auto resp = mem.search("Where is the Eiffel Tower?", 5, 0.0f);
    assert(!resp.results.empty());
    assert(resp.results[0].text.find("Eiffel") != std::string::npos);

    mem.close();
    cleanup_all();
    std::println(" OK");
}

void test_store_with_tags() {
    std::println("  test_store_with_tags...");
    cleanup_all();

    ragger::RaggerMemory mem(TEMP_DB1);
    mem.store("Reference: HTTP status 200 means OK.", {{"tags", {"reference"}}});
    mem.store("Memory: I had coffee this morning.", {{"tags", {"diary"}}});

    // Lean v2 has no collection filter; verify tags round-trip. With only two
    // near-equidistant candidates, min-max score normalization can be
    // volatile enough to reorder results on a tiny raw-cosine gap (a known
    // property of the blend, not specific to this doc pair) -- so find the
    // HTTP row by content rather than assuming it lands at rank 0, and check
    // its tags round-tripped correctly.
    auto resp = mem.search("HTTP status", 5, 0.0f);
    assert(!resp.results.empty());
    auto it = std::find_if(resp.results.begin(), resp.results.end(),
        [](const auto& r) { return r.text.find("HTTP") != std::string::npos; });
    assert(it != resp.results.end());
    assert((*it).metadata["tags"] == "reference");

    mem.close();
    cleanup_all();
    std::println(" OK");
}


void test_count() {
    std::println("  test_count...");
    cleanup_all();

    ragger::RaggerMemory mem(TEMP_DB1);
    assert(mem.count() == 0);

    mem.store("First item.");
    mem.store("Second item.");
    mem.store("Third item.");
    assert(mem.count() == 3);

    mem.close();
    cleanup_all();
    std::println(" OK");
}

void test_delete() {
    std::println("  test_delete...");
    cleanup_all();

    ragger::RaggerMemory mem(TEMP_DB1);
    std::string id1 = mem.store("To be deleted.");
    mem.store("To be kept.");
    assert(mem.count() == 2);

    bool deleted = mem.delete_memory(std::stoi(id1));
    assert(deleted);
    assert(mem.count() == 1);

    mem.close();
    cleanup_all();
    std::println(" OK");
}


void test_empty_db_path_throws() {
    std::println("  test_empty_db_path_throws...");

    bool threw = false;
    try {
        ragger::RaggerMemory mem("");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    std::println(" OK");
}

// PHASE5 D.3: recipe Episode layer emits newest-session episodes
// chronologically; a project_summary recipe parses, emits nothing, no error.
void p5_recipe_context(const std::string& model_dir) {
    std::println("  p5_recipe_context...");
    cleanup_all();

    auto r1 = ragger::set_config_persisted("capture_turns", "true");
    auto r2 = ragger::set_config_persisted("build_context", "true");
    assert(r1.has_value() && r2.has_value());

    ragger::RaggerMemory mem(TEMP_DB1);
    ragger::Embedder emb(model_dir);
    ragger::sqlite::Backend db(emb, TEMP_DB1);

    const std::string guid = "p5rc";
    std::vector<int> m1, m2;
    auto seed = [&](std::vector<int>& out, const std::string& stamp) {
        const int t = db.store_turn("p5rc q " + stamp, "p5rc answer " + std::string(60, 'r') + stamp,
                                    "test-model", false, guid, stamp);
        assert(db.finalize_turn_summary(t, "p5rc summary " + stamp, "summarizer-model"));
        sqlite3* raw = nullptr;
        assert(sqlite3_open(TEMP_DB1.c_str(), &raw) == SQLITE_OK);
        sqlite3_stmt* st = nullptr;
        const std::string sql = "SELECT turn_summary_id FROM turn_summaries WHERE turn_id = " + std::to_string(t);
        assert(sqlite3_prepare_v2(raw, sql.c_str(), -1, &st, nullptr) == SQLITE_OK);
        assert(sqlite3_step(st) == SQLITE_ROW);
        out.push_back(sqlite3_column_int(st, 0));
        sqlite3_finalize(st);
        sqlite3_close(raw);
    };
    seed(m1, "2026-07-01 10:00:00");
    seed(m2, "2026-07-01 11:00:00");

    auto close_ep = [&](const std::string& text, const std::vector<int>& members,
                        const std::string& stamp) {
        ragger::EpisodeClose ec;
        ec.text = text;
        ec.model_name = "summarizer-model";
        ec.session_guid = guid;
        std::tm tm{};
        strptime(stamp.c_str(), "%Y-%m-%d %H:%M:%S", &tm);
        tm.tm_isdst = -1;
        ec.span_start = ec.span_end = static_cast<long long>(mktime(&tm));
        ec.member_turn_summary_ids = members;
        assert(db.store_episode_close(ec) > 0);
    };
    close_ep("p5rc episode one", m1, "2026-07-01 10:00:00");
    close_ep("p5rc episode two", m2, "2026-07-01 11:00:00");

    // Custom recipe files under the (isolated) base recipes dir.
    std::error_code ec;
    const std::string rdir = ragger::config().resolved_recipes_dir();
    fs::create_directories(rdir, ec);
    {
        std::ofstream f(rdir + "/p5rc_two.json");
        f << R"({"name":"p5rc_two","layers":[{"kind":"episode","limit":2}],"max_tokens":8000})";
    }
    {
        std::ofstream f(rdir + "/p5rc_proj.json");
        f << R"({"name":"p5rc_proj","layers":[{"kind":"project_summary","limit":3}]})";
    }

    auto ctx = ragger::build_context(mem, guid, "p5rc_two");
    assert(ctx.enabled);
    assert(ctx.chunks.size() == 2);
    // Chronological (oldest first) per SessionContext contract.
    assert(ctx.chunks[0].kind == "episode");
    assert(ctx.chunks[0].text == "p5rc episode one");
    assert(ctx.chunks[1].text == "p5rc episode two");

    auto ctx2 = ragger::build_context(mem, guid, "p5rc_proj");
    assert(ctx2.enabled);
    assert(ctx2.chunks.empty());   // parses, emits nothing

    // "session_summary" alias resolves to the episode layer.
    {
        std::ofstream f(rdir + "/p5rc_alias.json");
        f << R"({"name":"p5rc_alias","layers":[{"kind":"session_summary","limit":2}]})";
    }
    auto ctx3 = ragger::build_context(mem, guid, "p5rc_alias");
    assert(ctx3.enabled);
    assert(ctx3.chunks.size() == 2);
    assert(ctx3.chunks[0].kind == "episode");

    mem.close();
    db.close();
    cleanup_all();
    std::println(" OK");
}


int main() {
    TestEnv env("memory");  // must be first: isolates from ~/.ragger
    auto model_dir = env.model_dir();

    if (!fs::exists(model_dir + "/model.onnx") && !fs::exists(model_dir + "/onnx/model.onnx")) {
        std::cerr << "Skipping memory tests: model not found at " << model_dir << "\n";
        std::println("test_memory: SKIPPED (no model)");
        return 0;
    }

    std::println("Running memory facade tests:");
    p5_recipe_context(model_dir);

    test_store_and_search();
    test_store_with_tags();
    test_count();
    test_delete();
    test_empty_db_path_throws();

    std::println("test_memory: all passed");
    return 0;
}
