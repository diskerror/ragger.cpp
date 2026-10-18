#include "episode_boundaries.h"
#include "backoff.h"
#include "summarizer.h"
#include "summarizer_service.h"
#include "inference.h"
#include "embedder.h"
#include "sqlite/backend.h"
#include "test_env.h"

#include "check.h"
#include <chrono>
#include <memory>
#include <print>
#include <string>
#include <thread>
#include <sqlite3.h>

using ragger::is_system_injected_turn;
using ragger::summarize_transcript;
using ragger::strip_thinking;

// System-injected user turns: should be detected (summarizer skips them).
void test_detects_system_markers() {
    std::print("  test_detects_system_markers...");

    CHECK(is_system_injected_turn(
        "[System note: Your previous turn was interrupted before you could "
        "process the last tool result(s)."));
    CHECK(is_system_injected_turn(
        "[Note: model was just switched from claude-sonnet-4-6 to "
        "claude-opus-4-8 via Anthropic. Adjust your self-identification.]"));
    CHECK(is_system_injected_turn(
        "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted."));
    CHECK(is_system_injected_turn(
        "[IMPORTANT: Background process proc_b623911a9a5d completed (exit code 1)."));

    // Leading whitespace must not defeat detection.
    CHECK(is_system_injected_turn("   \n[System note: foo"));

    std::println(" ok");
}

// Genuine human messages: must NOT be treated as system-injected.
void test_ignores_human_turns() {
    std::print("  test_ignores_human_turns...");

    CHECK(!is_system_injected_turn("Next item: storing system messages."));
    CHECK(!is_system_injected_turn("stop. leaving it blank is appropriate."));
    CHECK(!is_system_injected_turn(""));
    // A bracketed but non-system opener is left alone.
    CHECK(!is_system_injected_turn("[note] check this out"));
    // OUT-OF-BAND wraps a real human message — intentionally NOT a marker.
    CHECK(!is_system_injected_turn(
        "[OUT-OF-BAND USER MESSAGE] actually, do it the other way"));

    std::println(" ok");
}

// Assistant-only turns (system-injected user side blanked upstream): the
// trivial-turn shortcut must return the assistant text cleanly, never a
// degenerate "User: " prefix or empty-prompt artifact. This path returns
// before any inference call, so an empty-endpoint client is safe.
void test_assistant_only_trivial() {
    std::print("  test_assistant_only_trivial...");

    ragger::InferenceClient inf({}, "", 4096);  // no endpoints — never called

    // Short assistant-only turn (<120 chars) → verbatim assistant text.
    auto s = summarize_transcript(inf, {{ std::string{}, "Stopped." }});
    CHECK(s == "Stopped.");

    // Empty user + empty assistant → empty (nothing to summarize).
    auto e = summarize_transcript(inf, {{ std::string{}, std::string{} }});
    CHECK(e.empty());

    // Normal short pair still joins user | assistant.
    auto p = summarize_transcript(inf, {{ "hi", "hello there" }});
    CHECK(p == "hi | hello there");

    std::println(" ok");
}

// Stripping: capture-time removal of leading system annotations.
void test_strip_system_prefix() {
    std::print("  test_strip_system_prefix...");
    using ragger::strip_system_injected_prefix;

    // The reported case: system note + the real one-word message.
    CHECK(strip_system_injected_prefix(
        "[System note: Your previous turn was interrupted before you could "
        "process the last tool result(s). The conversation history contains "
        "tool outputs you haven't responded to yet.]\n\nstop") == "stop");

    // Note with nothing after it → empty.
    CHECK(strip_system_injected_prefix(
        "[System note: interrupted, please continue.]\n\n").empty());

    // Model-switch note + real message.
    CHECK(strip_system_injected_prefix(
        "[Note: model was just switched from a to b via X.]\n\nturn 154")
        == "turn 154");

    // Stacked notes strip in sequence.
    CHECK(strip_system_injected_prefix(
        "[Note: model was just switched from a to b via X.]\n"
        "[System note: interrupted.]\n\ngo") == "go");

    // Leading whitespace before the marker is tolerated.
    CHECK(strip_system_injected_prefix("  \n[System note: x]\n\nhi") == "hi");

    // A normal human message is returned untouched (trimmed).
    CHECK(strip_system_injected_prefix("just a normal message")
        == "just a normal message");

    // Brackets inside the real message are preserved (only a LEADING marker
    // block is stripped).
    CHECK(strip_system_injected_prefix("use arr[0] not arr[1]")
        == "use arr[0] not arr[1]");

    // Unterminated note → nothing trustworthy follows → empty.
    CHECK(strip_system_injected_prefix(
        "[System note: truncated with no close").empty());

    std::println(" ok");
}

void test_strip_thinking() {
    std::print("  test_strip_thinking...");

    // Basic <think>…</think> block stripped
    CHECK(strip_thinking("<think>some reasoning</think>The actual answer.")
        == "The actual answer.");

    // Unterminated <think> — drop to end
    CHECK(strip_thinking("<think>reasoning never closed").empty());

    // Multiple blocks
    CHECK(strip_thinking("<think>a</think>Answer<think>b</think> here.")
        == "Answer here.");

    // Bare "Thinking Process:\n…\n\n" preamble
    CHECK(strip_thinking("Thinking Process:\n\n1. Do X\n2. Do Y\n\nThe real summary.")
        == "The real summary.");

    // Leading whitespace before preamble
    CHECK(strip_thinking("\n\nThinking Process:\nstuff\n\nActual answer here.")
        == "Actual answer here.");

    // Normal summary untouched
    CHECK(strip_thinking("Reid deployed the daemon successfully.")
        == "Reid deployed the daemon successfully.");

    // Empty input
    CHECK(strip_thinking("").empty());

    std::println(" ok");
}

// PHASE1 Step 6: the summarizer resolves the inference client through a
// provider ONCE PER JOB — a job enqueued after the provider starts handing
// out a different client must observe the new one (C §1.3). Hermetic: the
// clients have no endpoints, and the turns are trivial (<80 chars), so the
// L2 path only marks them summarized with the client's memory_model — no
// network, no model.
void test_summarizer_provider_per_job(const std::string& base) {
    std::println("  test_summarizer_provider_per_job...");

    namespace fs = std::filesystem;
    const std::string dir = base + "/sumprov";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string db_path = dir + "/memories.db";

    ragger::Embedder emb;  // default: disabled — no embedding needed here
    ragger::sqlite::Backend backend(emb, db_path);

    int calls = 0;
    auto provider = [&calls]() -> std::shared_ptr<ragger::InferenceClient> {
        ++calls;
        auto c = std::make_shared<ragger::InferenceClient>(
            std::vector<ragger::Endpoint>{}, "", 4096);  // no endpoints
        c->memory_model = "prov-model-" + std::to_string(calls);
        return c;
    };

    ragger::SummarizerService svc(backend, provider);
    svc.start();

    // Store turns AFTER start so the startup catch-up scan (empty DB)
    // cannot race them; enqueue explicitly.
    int t1 = backend.store_turn("ping one", "reply one is long enough ok");
    int t2 = backend.store_turn("ping two", "reply two is long enough ok");
    CHECK(t1 > 0 && t2 > 0);
    svc.enqueue_turn(t1, "ping one", "reply one is long enough ok", "", "", "");
    svc.enqueue_turn(t2, "ping two", "reply two is long enough ok", "", "", "");

    auto wait_for = [&](int turn_id) {
        for (int i = 0; i < 200 && !backend.turn_summary_exists(turn_id); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        CHECK(backend.turn_summary_exists(turn_id));
    };
    wait_for(t1);
    wait_for(t2);
    svc.stop();

    // Read back the summary model names recorded per turn.
    sqlite3* db = nullptr;
    CHECK(sqlite3_open(db_path.c_str(), &db) == SQLITE_OK);
    auto model_for = [&](int turn_id) -> std::string {
        sqlite3_stmt* st = nullptr;
        sqlite3_prepare_v2(db,
            "SELECT m.name FROM turn_summaries ts "
            "JOIN models m ON m.model_id = ts.summary_model_id "
            "WHERE ts.turn_id = ?", -1, &st, nullptr);
        sqlite3_bind_int(st, 1, turn_id);
        std::string name;
        if (sqlite3_step(st) == SQLITE_ROW)
            name = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
        sqlite3_finalize(st);
        return name;
    };
    const std::string m1 = model_for(t1);
    const std::string m2 = model_for(t2);
    sqlite3_close(db);

    // Both jobs got a provider-issued client, and each job saw its own
    // (distinct) client — a cached raw pointer could not do that.
    CHECK(m1.rfind("prov-model-", 0) == 0);
    CHECK(m2.rfind("prov-model-", 0) == 0);
    CHECK(m1 != m2);
    std::println(" OK (jobs saw {} and {})", m1, m2);
}

// PHASE6 2.4 + 3: a live-enqueued trivial turn is consumed by the worker
// and marked WITHOUT any inference server (provider -> nullptr): marker row
// exists, model is the "(skipped)" pseudo-model, and a second enqueue
// creates no second row.
void p6_test_live_enqueue_skipped(const std::string& base) {
    std::println("  p6_test_live_enqueue_skipped...");

    namespace fs = std::filesystem;
    const std::string dir = base + "/p6live";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string db_path = dir + "/memories.db";

    ragger::Embedder emb;  // disabled — trivial path needs no embedding
    ragger::sqlite::Backend backend(emb, db_path);

    ragger::SummarizerService svc(backend, [] { return std::shared_ptr<ragger::InferenceClient>(nullptr); });
    svc.start();

    // Trivial: combined < 80 chars.
    int t = backend.store_turn("hi", "ok");
    CHECK(t > 0);
    svc.enqueue_turn(t, "hi", "ok", "", "", "");

    bool marked = false;
    for (int i = 0; i < 100 && !backend.turn_summary_exists(t); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    marked = backend.turn_summary_exists(t);
    CHECK(marked);   // consumed WITHOUT inference (pre-Phase 6 this never happened)

    svc.enqueue_turn(t, "hi", "ok", "", "", "");   // duplicate: must no-op
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    svc.stop();

    sqlite3* db = nullptr;
    CHECK(sqlite3_open(db_path.c_str(), &db) == SQLITE_OK);
    auto scalar = [&](const std::string& sql) -> long long {
        sqlite3_stmt* st = nullptr;
        sqlite3_prepare_v2(db, sql.c_str(), -1, &st, nullptr);
        long long v = (sqlite3_step(st) == SQLITE_ROW) ? sqlite3_column_int64(st, 0) : -1;
        sqlite3_finalize(st);
        return v;
    };
    CHECK(scalar("SELECT COUNT(*) FROM turn_summaries WHERE turn_id = " +
                 std::to_string(t) + " AND text = ''") == 1);
    CHECK(scalar("SELECT COUNT(*) FROM turn_summaries ts JOIN models m "
                 "ON m.model_id = ts.summary_model_id WHERE m.name = '(skipped)'") == 1);
    sqlite3_close(db);
    backend.close();
    std::println(" OK");
}

// PHASE6 4: Backoff contract (pure, injected time).
void p6_test_backoff() {
    std::println("  p6_test_backoff...");
    using namespace std::chrono;
    const auto t0 = steady_clock::time_point(seconds(1000));

    ragger::Backoff b(seconds(300), seconds(21600));
    CHECK(b.due(t0));                       // initially due
    b.failed(t0);
    CHECK(!b.due(t0 + seconds(299)));
    CHECK(!b.due(t0 + seconds(299) + milliseconds(999)));
    CHECK(b.due(t0 + seconds(300)));        // base delay
    b.failed(t0 + seconds(300));            // second failure doubles (10 min)
    CHECK(!b.due(t0 + seconds(899)));
    CHECK(b.due(t0 + seconds(900)));        // next retry = t0+300 + 600
    for (int i = 0; i < 20; ++i)           // many failures cap at 6 h
        b.failed(t0 + seconds(600));
    CHECK(!b.due(t0 + hours(5)));
    CHECK(b.due(t0 + hours(6) + seconds(600)));
    b.succeeded();
    CHECK(b.due(t0));                       // always due again
    std::println(" OK");
}

// PHASE6 5: concurrent enqueue_catch_up calls cannot double-queue.
void p6_test_single_scan(const std::string& base) {
    std::println("  p6_test_single_scan...");

    namespace fs = std::filesystem;
    const std::string dir = base + "/p6scan";
    fs::remove_all(dir);
    fs::create_directories(dir);

    ragger::Embedder emb;
    ragger::sqlite::Backend backend(emb, dir + "/memories.db");

    for (int i = 0; i < 3; ++i)
        CHECK(backend.store_turn("scan q" + std::to_string(i),
                                 "scan answer " + std::string(60, 's') + std::to_string(i)) > 0);

    // No start(): the worker would drain the queue and race the assertion.
    ragger::SummarizerService svc(backend, [] { return std::shared_ptr<ragger::InferenceClient>(nullptr); });

    auto hammer = [&] { for (int i = 0; i < 50; ++i) svc.enqueue_catch_up(); };
    std::thread a(hammer), b(hammer);
    a.join(); b.join();

    // 3 unsummarized turns -> at most 3 L2 jobs, never 6+.
    CHECK(svc.queue_size() <= 3);
    CHECK(svc.queue_size() == 3);
    backend.close();
    std::println(" OK");
}

// PHASE6 7: find_episode_boundaries — table from the spec. v(c) = (c,
// sqrt(1-c^2)) against (1,0) gives cosine c; both embeddings equal -> avg = c.
void p6_test_episode_boundaries() {
    std::println("  p6_test_episode_boundaries...");
    using ragger::EpisodeBoundaryParams;
    using ragger::EpisodeCandidateTurn;

    auto cand = [](float cos_val, long long epoch) {
        EpisodeCandidateTurn c;
        const float s = std::sqrt(1.0f - cos_val * cos_val);
        c.turn_embedding    = {cos_val, s};
        c.summary_embedding = {cos_val, s};
        c.epoch             = epoch;
        return c;
    };
    const EpisodeBoundaryParams p{0.5f, 0.8f, 0.1f, 15.0f};
    auto cut = [&](float c, long long gap_min) {
        auto v = ragger::find_episode_boundaries(
            {cand(1.0f, 0), cand(c, gap_min * 60)}, p);
        return v.size() == 2;   // boundary at index 1?
    };

    CHECK(!cut(0.95f, 0));
    CHECK(!cut(0.95f, 600));    // capped threshold 0.8
    CHECK(!cut(0.65f, 0));
    CHECK(!cut(0.65f, 14));     // int(14/15) = 0 -> threshold stays 0.5
    CHECK( cut(0.65f, 30));     // 0.5 + 0.1*2 = 0.7 >= 0.65
    CHECK( cut(0.65f, 600));    // capped 0.8 >= 0.65
    CHECK( cut(0.10f, 0));

    // Degenerate inputs.
    CHECK(ragger::find_episode_boundaries({}, p).empty());
    auto one = ragger::find_episode_boundaries({cand(0.9f, 0)}, p);
    CHECK(one.size() == 1 && one[0] == 0);

    // step_minutes = 0 disables time scaling.
    EpisodeBoundaryParams no_scale{0.5f, 0.8f, 0.1f, 0.0f};
    auto v = ragger::find_episode_boundaries({cand(1.0f, 0), cand(0.65f, 600 * 60)}, no_scale);
    CHECK(v.size() == 1);

    // Three candidates -> ascending index list starting at 0. (Consecutive
    // pairs must actually be dissimilar: v(c) vs v(c) is cosine 1, so use
    // alternating unit vectors pointing opposite ways, cos = -1.)
    auto flip = [](long long epoch, bool pos) {
        EpisodeCandidateTurn c;
        c.turn_embedding    = {pos ? 1.0f : -1.0f, 0.0f};
        c.summary_embedding = {pos ? 1.0f : -1.0f, 0.0f};
        c.epoch             = epoch;
        return c;
    };
    auto three = ragger::find_episode_boundaries(
        {flip(0, true), flip(60, false), flip(120, true)}, p);
    CHECK(three.size() == 3);
    CHECK(three[0] == 0 && three[1] == 1 && three[2] == 2);
    std::println(" OK");
}


int main() {
    TestEnv env("summarizer");
    std::println("test_summarizer:");
    test_detects_system_markers();
    test_ignores_human_turns();
    test_assistant_only_trivial();
    test_strip_system_prefix();
    test_strip_thinking();
    test_summarizer_provider_per_job(env.base());
    p6_test_backoff();
    p6_test_episode_boundaries();
    p6_test_live_enqueue_skipped(env.base());
    p6_test_single_scan(env.base());
    std::println("All summarizer tests passed.");
    return 0;
}
