#include "episode_boundaries.h"
#include "backoff.h"
#include "summarizer.h"
#include "summarizer_service.h"
#include "inference.h"
#include "embedder.h"
#include "sqlite/backend.h"
#include "test_env.h"

#include "check.h"
#include "util/utf8.h"
#include "nlohmann_json.hpp"

#include <atomic>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <print>
#include <string>
#include <thread>
#include <sqlite3.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

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

// S14 (C m15): Endpoint::is_local() parses the host instead of matching
// substrings anywhere in the URL. Empty or schemeless URLs return false and
// must never throw.
void p10_endpoint_is_local() {
    std::println("  p10_endpoint_is_local...");
    auto local = [](const char* url) {
        return ragger::Endpoint("t", url).is_local();
    };
    CHECK(local("http://localhost:1234/v1"));
    CHECK(local("http://127.0.0.1:8080"));
    CHECK(local("http://192.168.1.5:1234/v1"));
    CHECK(local("http://10.0.0.7/v1"));
    CHECK(local("http://0.0.0.0:1234"));
    CHECK(local("http://[::1]:1234/v1"));
    CHECK(!local("https://api.openai.com/v1"));
    CHECK(!local("https://host110.example.com/v1"));
    CHECK(!local("https://10.example.com/v1"));
    CHECK(!local("https://api.example.com:8010/v1"));
    CHECK(!local("http://beelink:1234/v1"));
    CHECK(!local("http://user@api.example.com/v1"));
    CHECK(!local(""));
    std::println(" ok");
}

namespace {

// Minimal OpenAI-style stub: counts chat calls, replies with fixed content.
struct StubChat {
    int ls = -1;
    int port = 0;
    std::atomic<bool> running{true};
    std::atomic<int> calls{0};
    std::string reply;
    mutable std::mutex bmu;   // read from system_of() on a const stub
    std::vector<std::string> bodies;
    std::thread srv;
    explicit StubChat(std::string reply_text) : reply(std::move(reply_text)) {
        ls = ::socket(AF_INET, SOCK_STREAM, 0);
        CHECK(ls >= 0);
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        a.sin_port = 0;
        CHECK(::bind(ls, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0);
        CHECK(::listen(ls, 8) == 0);
        socklen_t len = sizeof a;
        ::getsockname(ls, reinterpret_cast<sockaddr*>(&a), &len);
        port = ntohs(a.sin_port);
        srv = std::thread([this] {
            while (running.load()) {
                int c = ::accept(ls, nullptr, nullptr);
                if (c < 0) break;
                std::string req;
                char buf[8192];
                size_t hdr_end = std::string::npos;
                long long cl = -1;
                while (true) {
                    ssize_t n = ::recv(c, buf, sizeof buf, 0);
                    if (n <= 0) break;
                    req.append(buf, static_cast<size_t>(n));
                    if (hdr_end == std::string::npos) {
                        hdr_end = req.find("\r\n\r\n");
                        if (hdr_end != std::string::npos) {
                            std::string head = req.substr(0, hdr_end);
                            for (auto& ch : head) ch = static_cast<char>(::tolower(ch));
                            auto p = head.find("content-length:");
                            if (p != std::string::npos)
                                cl = std::stoll(head.substr(p + 15));
                        }
                    }
                    if (cl >= 0 && req.size() >= hdr_end + 4 + static_cast<size_t>(cl))
                        break;
                }
                if (cl >= 0) {
                    std::string body = req.substr(hdr_end + 4, static_cast<size_t>(cl));
                    if (body.find("\"messages\"") != std::string::npos) {
                        ++calls;
                        std::lock_guard<std::mutex> lk(bmu);
                        bodies.push_back(std::move(body));
                    }
                }
                const std::string body =
                    R"({"choices":[{"message":{"content":")" + reply + R"("}}]})";
                const std::string resp =
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    "Content-Length: " + std::to_string(body.size()) +
                    "\r\nConnection: close\r\n\r\n" + body;
                ::send(c, resp.data(), resp.size(), 0);
                ::close(c);
            }
        });
    }
    ~StubChat() {
        running.store(false);
        ::close(ls);
        if (srv.joinable()) srv.join();
    }
    std::shared_ptr<ragger::InferenceClient> client() const {
        auto c = std::make_shared<ragger::InferenceClient>(
            std::vector<ragger::Endpoint>{ragger::Endpoint(
                "stub", "http://127.0.0.1:" + std::to_string(port) + "/v1")},
            "stub-model", 64);
        c->memory_model = "stub-model";
        c->timeout_sec = 5;
        return c;
    }
    // System-message content of the Nth captured chat request.
    std::string system_of(size_t idx) const {
        std::lock_guard<std::mutex> lk(bmu);
        if (idx >= bodies.size()) return {};
        nlohmann::json j = nlohmann::json::parse(bodies[idx]);
        for (auto& m : j["messages"])
            if (m.value("role", "") == "system")
                return m["content"].get<std::string>();
        return {};
    }
};

} // namespace


// PHASE1 Step 6: the summarizer resolves the inference client through a
// provider ONCE PER JOB — a job enqueued after the provider starts handing
// out a different client must observe the new one (C §1.3). LR S63: the
// turns used to be trivial and rode the "(skipped)" marker path; that skip
// is gone, so the turns are now long enough to reach the model path (a stub
// endpoint answers) and each row records its job's client memory_model.
void test_summarizer_provider_per_job(const std::string& base) {
    std::println("  test_summarizer_provider_per_job...");

    namespace fs = std::filesystem;
    const std::string dir = base + "/sumprov";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string db_path = dir + "/memories.db";

    ragger::Embedder emb;  // default: disabled — no embedding needed here
    ragger::sqlite::Backend backend(emb, db_path);

    StubChat stub("SUMMARY");
    int calls = 0;
    auto provider = [&calls, &stub]() -> std::shared_ptr<ragger::InferenceClient> {
        ++calls;
        auto c = stub.client();
        c->memory_model = "prov-model-" + std::to_string(calls);
        return c;
    };

    ragger::SummarizerService svc(backend, provider);
    svc.start();

    // Store turns AFTER start so the startup catch-up scan (empty DB)
    // cannot race them; enqueue explicitly. 20+20 words and >120 combined
    // chars: above the S62 floor AND past summarize_transcript's verbatim
    // shortcut, so the model path actually runs.
    std::string u, a;
    for (int i = 0; i < 20; ++i) u += "ques" + std::to_string(i) + " ";
    for (int i = 0; i < 20; ++i) a += "resp" + std::to_string(i) + " ";
    int t1 = backend.store_turn(u, a);
    int t2 = backend.store_turn(u + "x", a);
    CHECK(t1 > 0 && t2 > 0);
    svc.enqueue_turn(t1, u, a, "", "", "");
    svc.enqueue_turn(t2, u + "x", a, "", "", "");

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

// LR S63 (replaces p6_test_live_enqueue_skipped): the trivial-turn skip is
// gone. A one-word exchange with NO inference server (provider -> nullptr)
// now goes through the S62 floor and lands a REAL turn_summaries row:
// text = the source, embedding/terms like any L2, summary_model_id = the
// turn's model (never the "(skipped)" pseudo-model, never text=''). The turn
// still leaves the unsummarized queue (the PHASE6 3 invariant, now carried
// by the floor), and the row is a normal row: an episode candidate.
void lr63_test_pleasantries_real_rows(const std::string& base, const std::string& model_dir) {
    std::println("  lr63_test_pleasantries_real_rows...");

    namespace fs = std::filesystem;
    const std::string dir = base + "/lr63pleasant";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string db_path = dir + "/memories.db";

    // Real embedder when the model is present (the episode-candidate
    // predicate requires non-NULL embeddings); disabled otherwise and that
    // one assertion is skipped (never silently).
    const bool have_model = fs::exists(model_dir + "/model.onnx") ||
                            fs::exists(model_dir + "/onnx/model.onnx");
    std::unique_ptr<ragger::Embedder> emb = have_model
        ? std::make_unique<ragger::Embedder>(model_dir)
        : std::make_unique<ragger::Embedder>();
    ragger::sqlite::Backend backend(*emb, db_path);

    ragger::SummarizerService svc(backend, [] { return std::shared_ptr<ragger::InferenceClient>(nullptr); });
    svc.start();

    const std::string guid = "lr63-session";
    int t = backend.store_turn("hi", "ok", "", false, guid);
    CHECK(t > 0);
    svc.enqueue_turn(t, "hi", "ok", "turn-model-z", guid, "");

    bool marked = false;
    for (int i = 0; i < 100 && !backend.turn_summary_exists(t); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    marked = backend.turn_summary_exists(t);
    CHECK(marked);   // consumed WITHOUT inference (floor needs no endpoint)

    svc.enqueue_turn(t, "hi", "ok", "turn-model-z", guid, "");   // dup: no-op
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
    // A REAL row, not a marker: text is the source, model is the turn's.
    CHECK(scalar("SELECT COUNT(*) FROM turn_summaries WHERE turn_id = " +
                 std::to_string(t) + " AND text = 'hi | ok'") == 1);
    CHECK(scalar("SELECT COUNT(*) FROM turn_summaries WHERE turn_id = " +
                 std::to_string(t) + " AND text = ''") == 0);
    CHECK(scalar("SELECT COUNT(*) FROM turn_summaries ts JOIN models m "
                 "ON m.model_id = ts.summary_model_id WHERE m.name = '(skipped)'") == 0);
    CHECK(scalar("SELECT COUNT(*) FROM turn_summaries ts JOIN models m "
                 "ON m.model_id = ts.summary_model_id WHERE ts.turn_id = " +
                 std::to_string(t) + " AND m.name = 'turn-model-z'") == 1);
    sqlite3_close(db);

    // Leaves the unsummarized queue (PHASE6 3 invariant).
    auto un = backend.unsummarized_turns(0);
    for (const auto& u : un) CHECK(u.turn_id != t);

    // A normal row: episode candidate (the candidate query excludes '' rows
    // and requires embeddings — needs the real model, see have_model above).
    if (have_model) {
        auto cands = backend.episode_candidate_turns(guid, "");
        CHECK(cands.size() == 1);
        CHECK(cands[0].summary_text == "hi | ok");
    } else {
        std::println("    (episode-candidate check skipped: no model)");
    }
    backend.close();
    std::println(" OK");
}

// S25 (C M10): "not configured" is not "unreachable". With no inference
// endpoint, non-trivial turns must NOT count toward max_turn_failures and
// must never be stamped 'bad' — they stay unsummarized until an endpoint
// exists. Pre-fix, repeating the job max_turn_failures times wrote 'bad'
// rows (then the back-off reset deleted them, forever).
void p10_summarizer_no_endpoint(const std::string& base) {
    std::println("  p10_summarizer_no_endpoint...");

    namespace fs = std::filesystem;
    const std::string dir = base + "/p10noep";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string db_path = dir + "/memories.db";

    ragger::Embedder emb;  // disabled — no embedding needed here
    ragger::sqlite::Backend backend(emb, db_path);

    ragger::SummarizerService svc(
        backend, [] { return std::shared_ptr<ragger::InferenceClient>(nullptr); });
    svc.start();

    // Non-trivial turns (combined >= 80 chars, assistant >= 20).
    const std::string u = "a user question that is definitely long enough to be real content";
    const std::string a = "and an assistant reply also long enough to survive the trivial filter";
    int t1 = backend.store_turn(u, a);
    int t2 = backend.store_turn(u + " two", a);
    int t3 = backend.store_turn(u + " three", a);
    CHECK(t1 > 0 && t2 > 0 && t3 > 0);
    // 4 rounds > max_turn_failures (3): the old path stamped 'bad' here.
    for (int round = 0; round < 4; ++round) {
        svc.enqueue_turn(t1, u, a, "", "", "");
        svc.enqueue_turn(t2, u + " two", a, "", "", "");
        svc.enqueue_turn(t3, u + " three", a, "", "", "");
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
    svc.stop();

    sqlite3* db = nullptr;
    CHECK(sqlite3_open(db_path.c_str(), &db) == SQLITE_OK);
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM turn_summaries", -1, &st, nullptr);
    long long rows = (sqlite3_step(st) == SQLITE_ROW) ? sqlite3_column_int64(st, 0) : -1;
    sqlite3_finalize(st);
    sqlite3_close(db);
    CHECK(rows == 0);   // no 'bad' stamps, no markers — nothing written

    // All three turns are still unsummarized (failure counter untouched).
    auto un = backend.unsummarized_turns(0);
    CHECK(un.size() == 3);
    backend.close();
    std::println(" ok");
}

// S26 (C m12): pin the EXACT system-prompt text produced by the shared
// budget formula for both public summarize entry points. The transcript
// expectation was captured pre-refactor; the TEXTS expectation is the
// LR S64 episode template (its own prompt, {N} substituted, same budget
// fill). Extraction of SummaryBudget/make_budget must keep the arithmetic.
void p10_summarizer_budget_prompt() {
    std::println("  p10_summarizer_budget_prompt...");

    // --- canned expectations ------------------------------------------------
    static const std::string EXPECTED_TRANSCRIPT_PROMPT =
        "Summarize this conversation into a concise memory entry. Extract key "
        "facts, decisions, questions asked, topics discussed. Write in third "
        "person past tense. Target about 44 characters; never exceed 89. If "
        "the exchange is trivial (a single command like /exit, a one-line "
        "greeting), respond with a short phrase, not a full sentence.";
    static const std::string EXPECTED_TEXTS_PROMPT =
        "You are given 2 consecutive summaries of one conversation span. "
        "Write a single standalone summary of the whole span that keeps every "
        "decision, configuration choice, technical fact, name, and date. Do "
        "not invent anything that is not present in the input. Do not use "
        "headers, do not wrap the text in quotation marks, and refer to the "
        "parties by role (the user, the assistant), never by pronoun. The "
        "summary must never be longer than the input. Target about 40 "
        "characters; never exceed 58.";

    // --- fake OpenAI endpoint: captures request bodies, replies SUMMARY ---
    int ls = ::socket(AF_INET, SOCK_STREAM, 0);
    CHECK(ls >= 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = 0;
    CHECK(::bind(ls, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0);
    CHECK(::listen(ls, 8) == 0);
    socklen_t len = sizeof a;
    ::getsockname(ls, reinterpret_cast<sockaddr*>(&a), &len);
    const int port = ntohs(a.sin_port);

    std::atomic<bool> running{true};
    std::mutex cmu;
    std::vector<std::string> bodies;
    std::thread srv([&] {
        const std::string resp_body =
            R"({"choices":[{"message":{"content":"SUMMARY"}}]})";
        const std::string resp =
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            "Content-Length: " + std::to_string(resp_body.size()) +
            "\r\nConnection: close\r\n\r\n" + resp_body;
        while (running.load()) {
            int c = ::accept(ls, nullptr, nullptr);
            if (c < 0) break;
            std::string req;
            char buf[8192];
            size_t hdr_end = std::string::npos;
            long long cl = -1;
            while (true) {
                ssize_t n = ::recv(c, buf, sizeof buf, 0);
                if (n <= 0) break;
                req.append(buf, static_cast<size_t>(n));
                if (hdr_end == std::string::npos) {
                    hdr_end = req.find("\r\n\r\n");
                    if (hdr_end != std::string::npos) {
                        std::string head = req.substr(0, hdr_end);
                        // lowercase scan for content-length
                        for (auto& ch : head) ch = static_cast<char>(::tolower(ch));
                        auto p = head.find("content-length:");
                        if (p != std::string::npos)
                            cl = std::stoll(head.substr(p + 15));
                    }
                }
                if (cl >= 0 && req.size() >= hdr_end + 4 + static_cast<size_t>(cl))
                    break;
            }
            if (cl >= 0) {
                std::string body = req.substr(hdr_end + 4, static_cast<size_t>(cl));
                if (body.find("\"messages\"") != std::string::npos) {
                    std::lock_guard<std::mutex> lk(cmu);
                    bodies.push_back(std::move(body));
                }
            }
            ::send(c, resp.data(), resp.size(), 0);
            ::close(c);
        }
    });

    // Deterministic config: built-in default prompt, default percentages.
    CHECK(ragger::update_config([](ragger::Config& c) {
        c.summarizer_prompt = "";
        c.summarizer_target_pct = 30;
        c.summarizer_max_pct = 60;
        return true; }));

    ragger::InferenceClient inf(
        {ragger::Endpoint("fake", "http://127.0.0.1:" + std::to_string(port) + "/v1")},
        "m", 64);
    inf.memory_model = "m";
    inf.timeout_sec = 5;

    const std::string u = "What changed in the schema migration?";
    const std::string as =
        "The migration adds the terms table, drops FTS5 shadows, and "
        "rebuilds the text index from scratch on first start.";
    const auto out1 = summarize_transcript(inf, {{u, as}});
    CHECK(out1 == "SUMMARY");

    const std::vector<std::string> texts = {
        "First turn summary about the Beelink deployment.",
        "Second turn summary about llama.cpp Vulkan builds."};
    const auto out2 = ragger::summarize_texts(inf, texts);
    CHECK(out2 == "SUMMARY");

    running.store(false);
    ::close(ls);
    srv.join();

    // Extract the system message content of the last request per call.
    auto system_of = [&](size_t idx) -> std::string {
        std::lock_guard<std::mutex> lk(cmu);
        CHECK(bodies.size() >= 2);
        nlohmann::json j = nlohmann::json::parse(bodies[idx]);
        for (auto& m : j["messages"])
            if (m.value("role", "") == "system")
                return m["content"].get<std::string>();
        return {};
    };
    const std::string p1 = system_of(0);
    const std::string p2 = system_of(1);
    CHECK(p1 == EXPECTED_TRANSCRIPT_PROMPT);
    CHECK(p2 == EXPECTED_TEXTS_PROMPT);
    std::println(" ok");
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


// PHASE9A: chat() must give up on an endpoint that accepts and never answers.
void p9a_test_chat_times_out() {
    std::println("  p9a_test_chat_times_out...");
    int ls = ::socket(AF_INET, SOCK_STREAM, 0);
    CHECK(ls >= 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = 0;                                   // kernel picks a free port
    CHECK(::bind(ls, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0);
    CHECK(::listen(ls, 1) == 0);
    socklen_t len = sizeof a;
    ::getsockname(ls, reinterpret_cast<sockaddr*>(&a), &len);
    const int port = ntohs(a.sin_port);

    // Accept, then hold the connection open without a byte of reply.
    std::thread srv([&] {
        int c = ::accept(ls, nullptr, nullptr);
        if (c >= 0) { std::this_thread::sleep_for(std::chrono::seconds(4)); ::close(c); }
    });

    ragger::InferenceClient inf(
        {ragger::Endpoint("stall", "http://127.0.0.1:" + std::to_string(port) + "/v1")}, "m", 64);
    inf.timeout_sec = 1;

    const auto t0 = std::chrono::steady_clock::now();
    bool threw = false;
    std::string what;
    try { inf.chat({{"user", "hi"}}); }
    catch (const std::exception& e) { threw = true; what = e.what(); }
    const auto secs = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - t0).count();

    CHECK(threw);
    CHECK(secs < 4);                                  // our timeout fired, not the server's hang-up
    CHECK(what.find("imeout") != std::string::npos);  // curl: "Timeout was reached"
    srv.join();
    ::close(ls);
    std::println(" ok");
}

// PHASE9B step 3c: utf8_safe_prefix never cuts mid-sequence.
void p9b_test_utf8_safe_prefix() {
    std::print("  p9b_test_utf8_safe_prefix...\n");
    using ragger::utf8_safe_prefix;
    CHECK(utf8_safe_prefix("abc", 10) == "abc");
    CHECK(utf8_safe_prefix("abc", 0) == "");
    CHECK(utf8_safe_prefix("h\xc3\xa9llo", 2) == "h");
    CHECK(utf8_safe_prefix("h\xc3\xa9llo", 3) == "h\xc3\xa9");
    CHECK(utf8_safe_prefix("\xe2\x82\xacx", 2) == "");
    CHECK(utf8_safe_prefix("\xe2\x82\xacx", 3) == "\xe2\x82\xac");
    CHECK(utf8_safe_prefix("\xe2\x82\xacx", 4) == "\xe2\x82\xacx");
    CHECK(utf8_safe_prefix("\xf0\x9f\x98\x80", 3) == "");
    CHECK(utf8_safe_prefix("\xf0\x9f\x98\x80", 4) == "\xf0\x9f\x98\x80");
    // Invalid input: still a prefix, no crash, no underflow.
    const std::string bad = "\x80\x80\x80";
    const auto r = utf8_safe_prefix(bad, 1);
    CHECK(r.size() <= 1);
    CHECK(bad.rfind(r, 0) == 0);

    // Property: every cut of a mixed string is valid UTF-8 (or empty).
    const std::string mixed = "a" "\xc3\xa9" "\xe2\x82\xac" "\xf0\x9f\x98\x80" "b" "\xc3\xa9" "c";
    auto valid_utf8 = [](const std::string& s) {
        size_t i = 0;
        while (i < s.size()) {
            const unsigned char c = s[i];
            size_t need;
            if (c < 0x80) need = 1;
            else if ((c & 0xE0) == 0xC0) need = 2;
            else if ((c & 0xF0) == 0xE0) need = 3;
            else if ((c & 0xF8) == 0xF0) need = 4;
            else return false;
            if (i + need > s.size()) return false;
            for (size_t k = 1; k < need; ++k)
                if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) return false;
            i += need;
        }
        return true;
    };
    for (size_t m = 0; m <= mixed.size() + 1; ++m) {
        const auto cut = utf8_safe_prefix(mixed, m);
        CHECK(cut.size() <= m);
        CHECK(mixed.rfind(cut, 0) == 0);
        CHECK(valid_utf8(cut));
    }
    std::println(" OK");
}

// PHASE9B step 3a: a prompt that vformat rejects must behave like an
// inference failure (warn + return ""), never escape the function.
void p9b_test_bad_prompt_no_escape(const std::string& base) {
    std::print("  p9b_test_bad_prompt_no_escape...\n");
    // A malformed prompt is published straight into the live Config: the
    // sanctioned doors (validate_config_value, step 3b) now reject it, which
    // is the point — but a row that predates that guard can still be live,
    // and the summarizer must survive it regardless.
    (void)base;
    ragger::Embedder emb;  // disabled; embedding not needed on this path
    {
        CHECK(ragger::update_config([](ragger::Config& c) { c.summarizer_prompt = "{"; return true; }));
        ragger::InferenceClient inf({}, "", 4096);  // no endpoints
        const std::vector<std::pair<std::string, std::string>> turns = {
            {"question about harpsichord concertos",
             std::string(120, 'x') + " answer about baroque music"},
        };
        bool threw = false;
        try {
            const auto out = summarize_transcript(inf, turns);
            CHECK(out.empty());
        } catch (...) { threw = true; }
        CHECK(!threw);
        threw = false;
        try {
            const auto out = ragger::summarize_texts(inf, {"some long text ", std::string(120, 'y')});
            CHECK(out.empty());
        } catch (...) { threw = true; }
        CHECK(!threw);

        // A valid prompt still reaches the (failing) inference path.
        CHECK(ragger::update_config([](ragger::Config& c) {
            c.summarizer_prompt = "summarize in {} chars, max {}"; return true; }));
        threw = false;
        try {
            const auto out = summarize_transcript(inf, turns);
            CHECK(out.empty());   // inference failure -> ""
        } catch (...) { threw = true; }
        CHECK(!threw);
    }
    std::println(" OK");
}

// X10 (M13): a malformed `summarizer_prompt` must behave EXACTLY like an
// inference failure, which is what build_system_prompt's comment promises.
// That means BOTH halves: the failure value AND no request at all — at HEAD
// the empty prompt was dropped from the message list and the request went out
// with no system message, so a result was accepted (and stored) instead of the
// caller's failure bookkeeping running.
void x10_test_bad_prompt_sends_no_request() {
    std::println("  x10_test_bad_prompt_sends_no_request...");
    StubChat stub("SUMMARY");
    auto inf = stub.client();

    // Long enough to clear summarize_transcript's 120-char trivial shortcut
    // (that path returns before any prompt is built, by design).
    const std::string u = "question about the schema migration and its terms table";
    const std::string a = std::string(120, 'x') + " answer about baroque music";
    const std::vector<std::pair<std::string, std::string>> turns = {{u, a}};

    auto set_prompt = [](const std::string& p) {
        CHECK(ragger::update_config([&](ragger::Config& c) {
            c.summarizer_prompt = p; return true; }));
    };

    // Unbalanced brace, and an out-of-range argument index: both make
    // std::vformat throw, so the prompt cannot be built.
    for (const std::string& bad : {std::string("{"), std::string("{9}")}) {
        set_prompt(bad);
        const int before = stub.calls.load();
        bool threw = false;
        std::string out;
        try { out = summarize_transcript(*inf, turns); }
        catch (...) { threw = true; }
        CHECK(!threw);                            // must not escape (PHASE9B 3a)
        CHECK(out.empty());                       // the inference-failure value
        CHECK(stub.calls.load() == before);       // ...and NO request issued
    }

    // Control with a valid prompt: exactly one request, system message present.
    set_prompt("summarize in {} chars, max {}");
    const int t_before = stub.calls.load();
    const auto ok = summarize_transcript(*inf, turns);
    CHECK(ok == "SUMMARY");
    CHECK(stub.calls.load() == t_before + 1);
    CHECK(!stub.system_of(static_cast<size_t>(t_before)).empty());

    // summarize_texts carries its OWN built-in episode prompt (LR S64), which
    // config cannot malform: pin that the valid prompt still sends exactly one
    // request with a system message (the empty-prompt guard is the same two
    // lines in that function; proven by the negative control in the step
    // report, since no config value can reach it).
    const int e_before = stub.calls.load();
    const auto ok2 = ragger::summarize_texts(*inf, {
        "First summary about the Beelink deployment and its Vulkan builds.",
        "Second summary about llama.cpp server flags and ctx sizes."});
    CHECK(ok2 == "SUMMARY");
    CHECK(stub.calls.load() == e_before + 1);
    CHECK(!stub.system_of(static_cast<size_t>(e_before)).empty());

    set_prompt("");   // built-in default for whatever runs after this
    std::println(" ok");
}

// ---------------------------------------------------------------------------
// LR S62: shorter-text rule.
// ---------------------------------------------------------------------------


// LR S62: the pure shorter-text rule (step's expected-value table).
void lr62_test_choose_summary() {
    std::println("  lr62_test_choose_summary...");
    using ragger::choose_summary;
    using ragger::count_words;

    const std::string src5 = "one two three four five";
    CHECK(count_words("  padded   words \n") == 2);
    CHECK(count_words("") == 0);

    auto c = choose_summary(src5, "a b c d e f");     // 5 / 6 -> source
    CHECK(c.copied && c.text == src5);
    c = choose_summary(src5, "a b c d e");            // 5 / 5 tie -> source
    CHECK(c.copied && c.text == src5);

    std::string big, small;
    for (int i = 0; i < 100; ++i) big += "w" + std::to_string(i) + " ";
    for (int i = 0; i < 40; ++i)  small += "s" + std::to_string(i) + " ";
    c = choose_summary(big, small);                   // 100 / 40 -> summary
    CHECK(!c.copied && c.text == small);

    // One-token source: compare Unicode code points, not words.
    const std::string jp = "日本語のテキスト";          // 8 code points, no whitespace
    CHECK(count_words(jp) == 1);
    c = choose_summary(jp, "日本語");                  // 3 < 8 -> summary
    CHECK(!c.copied && c.text == "日本語");
    c = choose_summary(jp, "日本語のテキストです");      // 9 >= 8 -> source
    CHECK(c.copied && c.text == jp);

    // Empty summary: not a valid summary — caller treats it as failure.
    c = choose_summary(src5, "");
    CHECK(!c.copied && c.text.empty());
    std::println(" ok");
}

// LR S62: the floor. A source at/below summarizer_min_words (default 20)
// is copied with NO model call; one word over calls the model once. The
// copy records the TURN's model (addendum 3), the model summary records the
// summarizer's model.
void lr62_test_floor_copies_without_model(const std::string& base) {
    std::println("  lr62_test_floor_copies_without_model...");

    namespace fs = std::filesystem;
    const std::string dir = base + "/lr62floor";
    fs::remove_all(dir);
    fs::create_directories(dir);
    ragger::Embedder emb;   // disabled: rows land with NULL embeddings
    ragger::sqlite::Backend backend(emb, dir + "/memories.db");

    StubChat stub("SUMMARY");
    ragger::SummarizerService svc(backend, [&stub] { return stub.client(); });
    svc.start();

    // 9 + 10 words + the join's "|" token = exactly 20 -> floor.
    const std::string u = "one two three four five six seven eight nine";
    const std::string a20 = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen";
    const std::string a21 = a20 + " twenty";
    int t20 = backend.store_turn(u, a20);
    int t21 = backend.store_turn(u, a21);
    CHECK(t20 > 0 && t21 > 0);
    svc.enqueue_turn(t20, u, a20, "turn-model-x", "", "");
    for (int i = 0; i < 100 && !backend.turn_summary_exists(t20); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(backend.turn_summary_exists(t20));
    CHECK(stub.calls.load() == 0);   // floor: NO model call

    svc.enqueue_turn(t21, u, a21, "turn-model-x", "", "");
    for (int i = 0; i < 100 && !backend.turn_summary_exists(t21); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(backend.turn_summary_exists(t21));
    CHECK(stub.calls.load() == 1);   // 21 words: the model IS called
    svc.stop();

    sqlite3* db = nullptr;
    CHECK(sqlite3_open((dir + "/memories.db").c_str(), &db) == SQLITE_OK);
    auto row = [&](int turn_id) -> std::pair<std::string, std::string> {
        sqlite3_stmt* st = nullptr;
        sqlite3_prepare_v2(db,
            "SELECT ts.text, COALESCE(m.name,'') FROM turn_summaries ts "
            "LEFT JOIN models m ON m.model_id = ts.summary_model_id "
            "WHERE ts.turn_id = ?", -1, &st, nullptr);
        sqlite3_bind_int(st, 1, turn_id);
        std::pair<std::string, std::string> r;
        if (sqlite3_step(st) == SQLITE_ROW) {
            r.first  = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
            r.second = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
        }
        sqlite3_finalize(st);
        return r;
    };
    const auto r20 = row(t20);
    CHECK(r20.first == u + " | " + a20);       // the source, verbatim
    CHECK(r20.second == "turn-model-x");       // copy provenance: the TURN's model
    const auto r21 = row(t21);
    CHECK(r21.first == "SUMMARY");             // model's shorter summary won
    CHECK(r21.second == "stub-model");         // model provenance: the summarizer
    sqlite3_close(db);
    backend.close();
    std::println(" ok");
}

// LR S62 (addendum 3): when the model's summary measures LONGER than the
// source, the source is copied and the row records the turn's model.
void lr62_test_copy_provenance(const std::string& base) {
    std::println("  lr62_test_copy_provenance...");

    namespace fs = std::filesystem;
    const std::string dir = base + "/lr62prov";
    fs::remove_all(dir);
    fs::create_directories(dir);
    ragger::Embedder emb;
    ragger::sqlite::Backend backend(emb, dir + "/memories.db");

    // Stub replies with 30 words — longer than the 25-word source.
    std::string long_reply;
    for (int i = 0; i < 30; ++i) long_reply += "r" + std::to_string(i) + " ";
    StubChat stub(long_reply);
    ragger::SummarizerService svc(backend, [&stub] { return stub.client(); });
    svc.start();

    // 12 + 12 + "|" = 25 words: above the floor, so the model runs once.
    // Tokens are long enough that the (still present, removed in S63)
    // trivial-turn filter does not fire.
    std::string u, a;
    for (int i = 0; i < 12; ++i) u += "user" + std::to_string(i) + " ";
    for (int i = 0; i < 12; ++i) a += "asst" + std::to_string(i) + " ";
    int t = backend.store_turn(u, a);
    CHECK(t > 0);
    svc.enqueue_turn(t, u, a, "turn-model-y", "", "");
    for (int i = 0; i < 100 && !backend.turn_summary_exists(t); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(backend.turn_summary_exists(t));
    CHECK(stub.calls.load() == 1);
    svc.stop();

    sqlite3* db = nullptr;
    CHECK(sqlite3_open((dir + "/memories.db").c_str(), &db) == SQLITE_OK);
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(db,
        "SELECT ts.text, COALESCE(m.name,'') FROM turn_summaries ts "
        "LEFT JOIN models m ON m.model_id = ts.summary_model_id "
        "WHERE ts.turn_id = ?", -1, &st, nullptr);
    sqlite3_bind_int(st, 1, t);
    std::string text, model;
    if (sqlite3_step(st) == SQLITE_ROW) {
        text  = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
        model = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
    }
    sqlite3_finalize(st);
    sqlite3_close(db);
    CHECK(text == u + " | " + a);       // bloated summary rejected: source kept
    CHECK(model == "turn-model-y");     // copy provenance: the TURN's model
    backend.close();
    std::println(" ok");
}

// LR S64: summarize_texts gets its OWN episode prompt (the L2 prompt
// reused verbatim over concatenated summaries produced narrative junk).
// The prompt must: name the input as N consecutive summaries of one span
// (N substituted), demand a standalone summary keeping decisions /
// configuration / technical facts / names / dates, forbid invention,
// headers, wrapping quotes, demand role references, forbid longer-than-
// input, and carry the two filled size slots. summarize_transcript keeps
// the L2 prompt (pinned by p10_summarizer_budget_prompt).
void lr64_test_episode_prompt() {
    std::println("  lr64_test_episode_prompt...");
    StubChat stub("SUMMARY");
    auto inf = stub.client();

    const std::vector<std::string> texts = {
        "First summary about the Beelink deployment and its Vulkan builds.",
        "Second summary about llama.cpp server flags and ctx sizes.",
        "Third summary about the release checklist for version nudge."};
    const auto out = ragger::summarize_texts(*inf, texts);
    CHECK(out == "SUMMARY");
    CHECK(stub.calls.load() == 1);

    const std::string p = stub.system_of(0);
    // Not the L2 prompt anymore.
    CHECK(p.find("Summarize this conversation into a concise memory entry") ==
          std::string::npos);
    // N substituted: three inputs -> "3 consecutive summaries".
    CHECK(p.find("3 consecutive summaries") != std::string::npos);
    // The required clauses.
    CHECK(p.find("standalone") != std::string::npos);
    CHECK(p.find("decision") != std::string::npos);
    CHECK(p.find("configuration") != std::string::npos);
    CHECK(p.find("technical fact") != std::string::npos);
    CHECK(p.find("name") != std::string::npos);
    CHECK(p.find("date") != std::string::npos);
    CHECK(p.find("invent") != std::string::npos);
    CHECK(p.find("header") != std::string::npos);
    CHECK(p.find("quotation mark") != std::string::npos);
    CHECK(p.find("role") != std::string::npos);
    CHECK(p.find("longer than the input") != std::string::npos);
    // The two size slots are filled (budget: 183 chars -> target 54, cap 109).
    CHECK(p.find("Target about 54 characters") != std::string::npos);
    CHECK(p.find("never exceed 109") != std::string::npos);
    // No unfilled placeholder survived.
    CHECK(p.find('{') == std::string::npos);
    CHECK(p.find('}') == std::string::npos);
    std::println(" ok");
}

// LR S65 (D-5 check): the poison-abandon path stays. With an endpoint that
// is configured but UNREACHABLE (connection refused — "unreachable", not
// "not configured"), a long turn failing max_turn_failures (3) times gets
// EXACTLY ONE turn_summaries row: text = '' and model = 'bad'. It leaves
// the unsummarized queue; reset_abandoned_turn_summaries removes it and
// re-queues the turn.
void lr65_test_abandon_marker(const std::string& base) {
    std::println("  lr65_test_abandon_marker...");

    namespace fs = std::filesystem;
    const std::string dir = base + "/lr65abandon";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string db_path = dir + "/memories.db";

    ragger::Embedder emb;  // disabled — no row with text is written here
    ragger::sqlite::Backend backend(emb, db_path);

    // A port that was bound and closed: connections are refused.
    int probe = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = 0;
    CHECK(::bind(probe, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0);
    socklen_t len = sizeof a;
    ::getsockname(probe, reinterpret_cast<sockaddr*>(&a), &len);
    const int dead_port = ntohs(a.sin_port);
    ::close(probe);

    ragger::SummarizerService svc(backend, [dead_port] {
        auto c = std::make_shared<ragger::InferenceClient>(
            std::vector<ragger::Endpoint>{ragger::Endpoint(
                "dead", "http://127.0.0.1:" + std::to_string(dead_port) + "/v1")},
            "dead-model", 64);
        c->memory_model = "dead-model";
        c->timeout_sec = 2;
        return c;
    });
    svc.start();

    // Long turn: above the S62 floor and the trivial thresholds (the old
    // filter would have skipped it before the failure counter existed).
    std::string u, a2;
    for (int i = 0; i < 20; ++i) u += "ques" + std::to_string(i) + " ";
    for (int i = 0; i < 20; ++i) a2 += "resp" + std::to_string(i) + " ";
    int t = backend.store_turn(u, a2);
    CHECK(t > 0);

    // max_turn_failures default is 3: the 3rd failure abandons.
    for (int round = 0; round < 4; ++round) {
        svc.enqueue_turn(t, u, a2, "", "", "");
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
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
    // EXACTLY one row, empty text, model 'bad'.
    CHECK(scalar("SELECT COUNT(*) FROM turn_summaries WHERE turn_id = " +
                 std::to_string(t)) == 1);
    CHECK(scalar("SELECT COUNT(*) FROM turn_summaries ts JOIN models m "
                 "ON m.model_id = ts.summary_model_id WHERE ts.turn_id = " +
                 std::to_string(t) + " AND ts.text = '' AND m.name = 'bad'") == 1);
    sqlite3_close(db);

    // Gone after reset, and back in the unsummarized candidate set.
    CHECK(backend.reset_abandoned_turn_summaries(0) == 1);
    CHECK(!backend.turn_summary_exists(t));
    bool found = false;
    for (auto& p : backend.unsummarized_turns(0))
        if (p.turn_id == t) found = true;
    CHECK(found);
    backend.close();
    std::println(" ok");
}

// X22 (Q2/L2): a merged turn's OPEN L2 (episode_id NULL) is refreshed in
// place by the catch-up tick. Pin first (RED at HEAD: no `stale` column, no
// L2Refresh job kind, no refresh path in enqueue_catch_up).
//
// Shape: short first exchange -> LR S62 floor copy (no model call, so the
// old text carries a word the refreshed text will not). A long continuation
// merges (stale = 1, text unchanged); enqueue_catch_up + the worker drain
// must rewrite the L2 from a source containing BOTH reply parts, clear the
// counter, re-embed, and rewrite the text index — while turn_summary_id,
// turn_datetime and summarized_on stay exactly as they were.
void x22_test_l2_refresh_after_merge(const std::string& base, const std::string& model_dir) {
    std::println("  x22_test_l2_refresh_after_merge...");

    namespace fs = std::filesystem;
    const std::string dir = base + "/x22refresh";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string db_path = dir + "/memories.db";

    // Real embedder when the model is present (the embedding-differs check
    // needs one); disabled otherwise and that ONE assertion is skipped
    // loudly, never silently.
    const bool have_model = fs::exists(model_dir + "/model.onnx") ||
                            fs::exists(model_dir + "/onnx/model.onnx");
    std::unique_ptr<ragger::Embedder> emb = have_model
        ? std::make_unique<ragger::Embedder>(model_dir)
        : std::make_unique<ragger::Embedder>();
    ragger::sqlite::Backend backend(*emb, db_path);

    StubChat stub("X22STUBSUMMARY");
    ragger::SummarizerService svc(backend, [&stub] { return stub.client(); });
    svc.start();

    sqlite3* db = nullptr;
    CHECK(sqlite3_open(db_path.c_str(), &db) == SQLITE_OK);
    auto scalar = [&](const std::string& sql) -> long long {
        sqlite3_stmt* st = nullptr;
        CHECK(sqlite3_prepare_v2(db, sql.c_str(), -1, &st, nullptr) == SQLITE_OK);
        long long v = -999;
        if (sqlite3_step(st) == SQLITE_ROW) v = sqlite3_column_int64(st, 0);
        sqlite3_finalize(st);
        return v;
    };
    auto text = [&](const std::string& sql) -> std::string {
        sqlite3_stmt* st = nullptr;
        CHECK(sqlite3_prepare_v2(db, sql.c_str(), -1, &st, nullptr) == SQLITE_OK);
        std::string v;
        if (sqlite3_step(st) == SQLITE_ROW) {
            const unsigned char* c = sqlite3_column_text(st, 0);
            v = c ? reinterpret_cast<const char*>(c) : "";
        }
        sqlite3_finalize(st);
        return v;
    };

    const std::string guid = "x22-refresh";
    const std::string u  = "x22 refresh question alphaonly";
    const std::string a1 = "x22 answer one betaonly";
    const int t = backend.store_turn(u, a1, "turn-model-x", false, guid);
    CHECK(t > 0);
    svc.enqueue_turn(t, u, a1, "turn-model-x", guid, "");
    for (int i = 0; i < 100 && !backend.turn_summary_exists(t); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(backend.turn_summary_exists(t));
    CHECK(stub.calls.load() == 0);   // floor: NO model call

    const std::string w = " FROM turn_summaries WHERE turn_summary_id = " + std::to_string(
        scalar("SELECT turn_summary_id FROM turn_summaries WHERE turn_id = " + std::to_string(t)));
    const int ts_id = static_cast<int>(scalar(
        "SELECT turn_summary_id FROM turn_summaries WHERE turn_id = " + std::to_string(t)));
    const long long dt0  = scalar("SELECT turn_datetime" + w);
    const long long sum0 = scalar("SELECT summarized_on" + w);
    CHECK(text("SELECT text" + w) == u + " | " + a1);
    CHECK(scalar("SELECT stale" + w) == 0);
    std::vector<char> emb0;
    {
        sqlite3_stmt* st = nullptr;
        CHECK(sqlite3_prepare_v2(db, ("SELECT embedding" + w).c_str(), -1, &st, nullptr) == SQLITE_OK);
        if (sqlite3_step(st) == SQLITE_ROW) {
            const void* b = sqlite3_column_blob(st, 0);
            const int n = sqlite3_column_bytes(st, 0);
            if (b && n > 0) emb0.assign(static_cast<const char*>(b),
                                static_cast<const char*>(b) + n);
        }
        sqlite3_finalize(st);
    }

    // Merge a long continuation: stale -> 1, the L2 text stays as it was.
    std::string a2;
    for (int i = 0; i < 25; ++i) a2 += "cont" + std::to_string(i) + " ";
    backend.store_turn(u, a2, "turn-model-x", false, guid);
    CHECK(scalar("SELECT stale" + w) == 1);
    CHECK(text("SELECT text" + w) == u + " | " + a1);

    // Two merges accumulate before one refresh clears them (spec test 2).
    backend.store_turn(u, "cont25 tailword zulu", "turn-model-x", false, guid);
    CHECK(scalar("SELECT stale" + w) == 2);

    // Tick + drain.
    svc.enqueue_catch_up();
    for (int i = 0; i < 200 && scalar("SELECT stale" + w) != 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(scalar("SELECT stale" + w) == 0);
    svc.stop();

    // The refresh called the model exactly once, on a source holding BOTH
    // reply parts, and the fake summary replaced the floor copy.
    CHECK(stub.calls.load() == 1);
    {
        std::lock_guard<std::mutex> lk(stub.bmu);
        CHECK(stub.bodies.size() == 1);
        CHECK(stub.bodies[0].find("betaonly") != std::string::npos);   // part 1
        CHECK(stub.bodies[0].find("cont0") != std::string::npos);      // part 2
        CHECK(stub.bodies[0].find("tailword") != std::string::npos);   // part 3
    }
    CHECK(text("SELECT text" + w) == "X22STUBSUMMARY");
    CHECK(text("SELECT m.name FROM turn_summaries ts "
               "JOIN models m ON m.model_id = ts.summary_model_id"
               " WHERE turn_summary_id = " + std::to_string(ts_id)) == "stub-model");
    CHECK(scalar("SELECT turn_datetime" + w) == dt0);
    CHECK(scalar("SELECT summarized_on" + w) == sum0);
    CHECK(scalar("SELECT episode_id IS NULL" + w) == 1);

    // Embedding re-computed (differs from the floor copy's blob).
    if (have_model) {
        std::vector<char> emb1;
        sqlite3_stmt* st = nullptr;
        CHECK(sqlite3_prepare_v2(db, ("SELECT embedding" + w).c_str(), -1, &st, nullptr) == SQLITE_OK);
        if (sqlite3_step(st) == SQLITE_ROW) {
            const void* b = sqlite3_column_blob(st, 0);
            const int n = sqlite3_column_bytes(st, 0);
            if (b && n > 0) emb1.assign(static_cast<const char*>(b),
                                static_cast<const char*>(b) + n);
        }
        sqlite3_finalize(st);
        CHECK(!emb1.empty());
        CHECK(emb1 != emb0);
    } else {
        std::println("    (embedding-differs check skipped: no embedding model)");
    }
    sqlite3_close(db);

    // Text index rewritten: a word only the NEW text has is found on the
    // turn_summary corpus; a word only the OLD text has is not.
    bool new_hit = false, old_hit = false;
    for (const auto& r : backend.search_text_only("X22STUBSUMMARY", 20).results)
        if (r.metadata.value("source", "") == "turn_summary" && r.id == ts_id) new_hit = true;
    for (const auto& r : backend.search_text_only("alphaonly", 20).results)
        if (r.metadata.value("source", "") == "turn_summary" && r.id == ts_id) old_hit = true;
    CHECK(new_hit);
    CHECK(!old_hit);
    backend.close();
    std::println(" ok");
}

// X22 (Q2/L2): an L2 row that already belongs to an episode is flagged by a
// merge but NOT refreshed by the X22 tick — refreshing it means rebuilding
// the episode summary, which is X23's job.
void x22_test_l2_refresh_skips_in_episode(const std::string& base) {
    std::println("  x22_test_l2_refresh_skips_in_episode...");

    namespace fs = std::filesystem;
    const std::string dir = base + "/x22episode";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string db_path = dir + "/memories.db";

    ragger::Embedder emb;   // disabled: no embedding needed here
    ragger::sqlite::Backend backend(emb, db_path);

    StubChat stub("X22SHOULDRUN");
    ragger::SummarizerService svc(backend, [&stub] { return stub.client(); });
    svc.start();

    const std::string guid = "x22-episode";
    const std::string u  = "x22 episode question";
    const std::string a1 = "x22 episode answer one";
    const int t = backend.store_turn(u, a1, "turn-model-x", false, guid);
    CHECK(t > 0);
    CHECK(backend.finalize_turn_summary(t, "Episode member summary", "summarizer-model"));

    // Close an episode over the member (span = the turn's own minute).
    sqlite3* db = nullptr;
    CHECK(sqlite3_open(db_path.c_str(), &db) == SQLITE_OK);
    auto scalar = [&](const std::string& sql) -> long long {
        sqlite3_stmt* st = nullptr;
        CHECK(sqlite3_prepare_v2(db, sql.c_str(), -1, &st, nullptr) == SQLITE_OK);
        long long v = -999;
        if (sqlite3_step(st) == SQLITE_ROW) v = sqlite3_column_int64(st, 0);
        sqlite3_finalize(st);
        return v;
    };
    auto text = [&](const std::string& sql) -> std::string {
        sqlite3_stmt* st = nullptr;
        CHECK(sqlite3_prepare_v2(db, sql.c_str(), -1, &st, nullptr) == SQLITE_OK);
        std::string v;
        if (sqlite3_step(st) == SQLITE_ROW) {
            const unsigned char* c = sqlite3_column_text(st, 0);
            v = c ? reinterpret_cast<const char*>(c) : "";
        }
        sqlite3_finalize(st);
        return v;
    };
    const int ts_id = static_cast<int>(scalar(
        "SELECT turn_summary_id FROM turn_summaries WHERE turn_id = " + std::to_string(t)));
    const long long when = scalar("SELECT turn_datetime FROM turn_summaries WHERE turn_summary_id = "
                                  + std::to_string(ts_id));
    sqlite3_close(db);

    ragger::EpisodeClose ec;
    ec.text = "Episode over x22";
    ec.model_name = "summarizer-model";
    ec.session_guid = guid;
    ec.span_start = when - 30;
    ec.span_end   = when + 30;
    ec.member_turn_summary_ids = {ts_id};
    CHECK(backend.store_episode_close(ec) > 0);

    // Merge: the in-episode row is flagged, and the X22 tick must leave it.
    backend.store_turn(u, "x22 episode answer two continuation tail", "turn-model-x", false, guid);
    db = nullptr;
    CHECK(sqlite3_open(db_path.c_str(), &db) == SQLITE_OK);
    const std::string w = " FROM turn_summaries WHERE turn_summary_id = " + std::to_string(ts_id);
    CHECK(scalar("SELECT stale" + w) == 1);
    CHECK(scalar("SELECT episode_id IS NOT NULL" + w) == 1);

    svc.enqueue_catch_up();
    std::this_thread::sleep_for(std::chrono::milliseconds(400));   // drain window
    CHECK(scalar("SELECT stale" + w) == 1);                        // NOT refreshed
    CHECK(text("SELECT text" + w) == "Episode member summary");
    CHECK(stub.calls.load() == 0);                                 // no model call
    sqlite3_close(db);
    svc.stop();
    backend.close();
    std::println(" ok");
}

// X22 (Q2/L2): with inference down the refresh writes nothing — the row
// stays stale (retried next tick), the text is untouched, and nothing is
// counted toward poison-abandon (no 'bad' marker appears).
void x22_test_l2_refresh_inference_down(const std::string& base) {
    std::println("  x22_test_l2_refresh_inference_down...");

    namespace fs = std::filesystem;
    const std::string dir = base + "/x22down";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string db_path = dir + "/memories.db";

    ragger::Embedder emb;   // disabled
    ragger::sqlite::Backend backend(emb, db_path);

    ragger::SummarizerService svc(backend, [] { return std::shared_ptr<ragger::InferenceClient>(nullptr); });
    svc.start();

    const std::string guid = "x22-down";
    const std::string u  = "x22 down question";
    const std::string a1 = "x22 down answer one";
    const int t = backend.store_turn(u, a1, "turn-model-x", false, guid);
    CHECK(t > 0);
    CHECK(backend.finalize_turn_summary(t, "Down member summary", "summarizer-model"));

    backend.store_turn(u, "x22 down answer two continuation", "turn-model-x", false, guid);

    sqlite3* db = nullptr;
    CHECK(sqlite3_open(db_path.c_str(), &db) == SQLITE_OK);
    auto scalar = [&](const std::string& sql) -> long long {
        sqlite3_stmt* st = nullptr;
        CHECK(sqlite3_prepare_v2(db, sql.c_str(), -1, &st, nullptr) == SQLITE_OK);
        long long v = -999;
        if (sqlite3_step(st) == SQLITE_ROW) v = sqlite3_column_int64(st, 0);
        sqlite3_finalize(st);
        return v;
    };
    auto text = [&](const std::string& sql) -> std::string {
        sqlite3_stmt* st = nullptr;
        CHECK(sqlite3_prepare_v2(db, sql.c_str(), -1, &st, nullptr) == SQLITE_OK);
        std::string v;
        if (sqlite3_step(st) == SQLITE_ROW) {
            const unsigned char* c = sqlite3_column_text(st, 0);
            v = c ? reinterpret_cast<const char*>(c) : "";
        }
        sqlite3_finalize(st);
        return v;
    };
    const std::string w = " FROM turn_summaries WHERE turn_summary_id = " + std::to_string(scalar(
        "SELECT turn_summary_id FROM turn_summaries WHERE turn_id = " + std::to_string(t)));
    CHECK(scalar("SELECT stale" + w) == 1);

    // Several ticks: still stale, text unchanged, no poison marker anywhere.
    for (int i = 0; i < 4; ++i) {
        svc.enqueue_catch_up();
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
    CHECK(scalar("SELECT stale" + w) == 1);
    CHECK(text("SELECT text" + w) == "Down member summary");
    CHECK(scalar("SELECT COUNT(*) FROM turn_summaries ts JOIN models m "
                 "ON m.model_id = ts.summary_model_id WHERE m.name = 'bad'") == 0);
    sqlite3_close(db);
    svc.stop();
    backend.close();
    std::println(" ok");
}

int main() {
    TestEnv env("summarizer");
    std::println("test_summarizer:");
    test_detects_system_markers();
    test_ignores_human_turns();
    test_assistant_only_trivial();
    test_strip_system_prefix();
    test_strip_thinking();
    p10_endpoint_is_local();
    test_summarizer_provider_per_job(env.base());
    p6_test_backoff();
    p6_test_episode_boundaries();
    lr63_test_pleasantries_real_rows(env.base(), env.model_dir());
    p10_summarizer_no_endpoint(env.base());
    p10_summarizer_budget_prompt();
    lr62_test_choose_summary();
    lr62_test_floor_copies_without_model(env.base());
    lr62_test_copy_provenance(env.base());
    lr64_test_episode_prompt();
    lr65_test_abandon_marker(env.base());
    p6_test_single_scan(env.base());
    x22_test_l2_refresh_after_merge(env.base(), env.model_dir());
    x22_test_l2_refresh_skips_in_episode(env.base());
    x22_test_l2_refresh_inference_down(env.base());
    p9b_test_utf8_safe_prefix();
    p9b_test_bad_prompt_no_escape(env.base());
    x10_test_bad_prompt_sends_no_request();
    p9a_test_chat_times_out();
    std::println("All summarizer tests passed.");
    return 0;
}