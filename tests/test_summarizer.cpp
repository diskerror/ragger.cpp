#include "summarizer.h"
#include "summarizer_service.h"
#include "inference.h"
#include "embedder.h"
#include "sqlite/backend.h"
#include "test_env.h"

#include <cassert>
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

    assert(is_system_injected_turn(
        "[System note: Your previous turn was interrupted before you could "
        "process the last tool result(s)."));
    assert(is_system_injected_turn(
        "[Note: model was just switched from claude-sonnet-4-6 to "
        "claude-opus-4-8 via Anthropic. Adjust your self-identification.]"));
    assert(is_system_injected_turn(
        "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted."));
    assert(is_system_injected_turn(
        "[IMPORTANT: Background process proc_b623911a9a5d completed (exit code 1)."));

    // Leading whitespace must not defeat detection.
    assert(is_system_injected_turn("   \n[System note: foo"));

    std::println(" ok");
}

// Genuine human messages: must NOT be treated as system-injected.
void test_ignores_human_turns() {
    std::print("  test_ignores_human_turns...");

    assert(!is_system_injected_turn("Next item: storing system messages."));
    assert(!is_system_injected_turn("stop. leaving it blank is appropriate."));
    assert(!is_system_injected_turn(""));
    // A bracketed but non-system opener is left alone.
    assert(!is_system_injected_turn("[note] check this out"));
    // OUT-OF-BAND wraps a real human message — intentionally NOT a marker.
    assert(!is_system_injected_turn(
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
    assert(s == "Stopped.");

    // Empty user + empty assistant → empty (nothing to summarize).
    auto e = summarize_transcript(inf, {{ std::string{}, std::string{} }});
    assert(e.empty());

    // Normal short pair still joins user | assistant.
    auto p = summarize_transcript(inf, {{ "hi", "hello there" }});
    assert(p == "hi | hello there");

    std::println(" ok");
}

// Stripping: capture-time removal of leading system annotations.
void test_strip_system_prefix() {
    std::print("  test_strip_system_prefix...");
    using ragger::strip_system_injected_prefix;

    // The reported case: system note + the real one-word message.
    assert(strip_system_injected_prefix(
        "[System note: Your previous turn was interrupted before you could "
        "process the last tool result(s). The conversation history contains "
        "tool outputs you haven't responded to yet.]\n\nstop") == "stop");

    // Note with nothing after it → empty.
    assert(strip_system_injected_prefix(
        "[System note: interrupted, please continue.]\n\n").empty());

    // Model-switch note + real message.
    assert(strip_system_injected_prefix(
        "[Note: model was just switched from a to b via X.]\n\nturn 154")
        == "turn 154");

    // Stacked notes strip in sequence.
    assert(strip_system_injected_prefix(
        "[Note: model was just switched from a to b via X.]\n"
        "[System note: interrupted.]\n\ngo") == "go");

    // Leading whitespace before the marker is tolerated.
    assert(strip_system_injected_prefix("  \n[System note: x]\n\nhi") == "hi");

    // A normal human message is returned untouched (trimmed).
    assert(strip_system_injected_prefix("just a normal message")
        == "just a normal message");

    // Brackets inside the real message are preserved (only a LEADING marker
    // block is stripped).
    assert(strip_system_injected_prefix("use arr[0] not arr[1]")
        == "use arr[0] not arr[1]");

    // Unterminated note → nothing trustworthy follows → empty.
    assert(strip_system_injected_prefix(
        "[System note: truncated with no close").empty());

    std::println(" ok");
}

void test_strip_thinking() {
    std::print("  test_strip_thinking...");

    // Basic <think>…</think> block stripped
    assert(strip_thinking("<think>some reasoning</think>The actual answer.")
        == "The actual answer.");

    // Unterminated <think> — drop to end
    assert(strip_thinking("<think>reasoning never closed").empty());

    // Multiple blocks
    assert(strip_thinking("<think>a</think>Answer<think>b</think> here.")
        == "Answer here.");

    // Bare "Thinking Process:\n…\n\n" preamble
    assert(strip_thinking("Thinking Process:\n\n1. Do X\n2. Do Y\n\nThe real summary.")
        == "The real summary.");

    // Leading whitespace before preamble
    assert(strip_thinking("\n\nThinking Process:\nstuff\n\nActual answer here.")
        == "Actual answer here.");

    // Normal summary untouched
    assert(strip_thinking("Reid deployed the daemon successfully.")
        == "Reid deployed the daemon successfully.");

    // Empty input
    assert(strip_thinking("").empty());

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

int main() {
    TestEnv env("summarizer");
    std::println("test_summarizer:");
    test_detects_system_markers();
    test_ignores_human_turns();
    test_assistant_only_trivial();
    test_strip_system_prefix();
    test_strip_thinking();
    test_summarizer_provider_per_job(env.base());
    std::println("All summarizer tests passed.");
    return 0;
}
