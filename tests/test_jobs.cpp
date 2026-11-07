/**
 * test_jobs.cpp — LR S91: the daemon's maintenance jobs runner (D4, D-6).
 *
 * One daemon (harness/http_fixture.h) with its inference endpoint pointed at
 * a local stub BEFORE the Server is constructed (the fixture's after_env
 * hook), so the backfill jobs can run. The stub has three modes: reject
 * (HTTP 500 — the default, so the daemon's summarizer catch-up fails exactly
 * like it does with no endpoint configured), fast (immediate valid reply),
 * and slow (sleeps, the latch that keeps a job `running` long enough to
 * probe GET /jobs, the 409, and cancel).
 *
 * Pinned here (step contract): unknown job -> 400; dry-run jobs report
 * counts and write nothing; single-flight 409; cancel between batches leaves
 * a consistent DB and the rerun completes the rest; a rerun after completion
 * selects 0; a failing job ends `failed` with the message; the SSE /events
 * stream carries `job` frames with growing `done`; daemon shutdown while a
 * job runs cancels and joins (fixture destruction returns; the DB is
 * consistent).
 */
#include "harness/http_fixture.h"

#include "inference.h"
#include "nlohmann_json.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <print>
#include <sqlite3.h>
#include <string>
#include <thread>
#include <vector>

using nlohmann::json;
using ragger_test::HttpFixture;

namespace {

// ---------------------------------------------------------------------------
// stub inference endpoint (bf_stub shape + latch modes)
// ---------------------------------------------------------------------------

struct job_stub {
    int ls = -1;
    int port = 0;
    std::atomic<bool> running{true};
    std::atomic<int> calls{0};
    // 0 = reject (500), 1 = fast 200, 2 = slow 200 (sleep slow_ms first)
    std::atomic<int> mode{0};
    std::atomic<int> slow_ms{1500};
    std::string reply = "when asked about the thing";
    std::thread srv;

    job_stub() {
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
                bool is_chat = false;
                if (cl >= 0) {
                    const std::string body = req.substr(hdr_end + 4, static_cast<size_t>(cl));
                    if (body.find("\"messages\"") != std::string::npos) {
                        ++calls;
                        is_chat = true;
                    }
                }
                if (!is_chat || mode.load() == 0) {
                    const std::string out = R"({"error":"stub refuses"})";
                    const std::string resp =
                        "HTTP/1.1 500 Internal Server Error\r\n"
                        "Content-Type: application/json\r\nContent-Length: " +
                        std::to_string(out.size()) +
                        "\r\nConnection: close\r\n\r\n" + out;
                    ::send(c, resp.data(), resp.size(), 0);
                    ::close(c);
                    continue;
                }
                if (mode.load() == 2) {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(slow_ms.load()));
                }
                const std::string out =
                    R"({"choices":[{"message":{"content":")" + reply + R"("}}]})";
                const std::string resp =
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    "Content-Length: " + std::to_string(out.size()) +
                    "\r\nConnection: close\r\n\r\n" + out;
                ::send(c, resp.data(), resp.size(), 0);
                ::close(c);
            }
        });
    }
    ~job_stub() {
        running.store(false);
        ::close(ls);
        if (srv.joinable()) srv.join();
    }
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

template <typename F>
bool poll_until(F fn, int timeout_ms, int step_ms = 100) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (fn()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(step_ms));
    }
    return false;
}

json get_jobs(HttpFixture& fx) {
    auto r = fx.client.get(fx.url("/jobs"), fx.auth());
    CHECK(r.ok());
    CHECK(r.status == 200);
    return json::parse(r.body);
}

/// The finished job's state from GET /jobs ("" when there is none yet).
std::string last_state(HttpFixture& fx) {
    auto j = get_jobs(fx);
    return j["last"].is_null() ? "" : j["last"]["state"].get<std::string>();
}

/// Wait for the job `name` to finish in state `state` (matches the name so a
/// second job never passes on the first job's `last`).
bool wait_last(HttpFixture& fx, const std::string& name,
               const std::string& state, int timeout_ms = 10000) {
    return poll_until([&] {
        auto j = get_jobs(fx);
        return !j["last"].is_null() &&
               j["last"]["name"].get<std::string>() == name &&
               j["last"]["state"].get<std::string>() == state;
    }, timeout_ms);
}

/// The finished job's message ("" when there is none yet).
std::string last_message(HttpFixture& fx) {
    auto j = get_jobs(fx);
    return j["last"].is_null() ? "" : j["last"]["message"].get<std::string>();
}

long post_job(HttpFixture& fx, const std::string& name, bool dry_run = false) {
    json body = json{{"dry_run", dry_run}};
    auto r = fx.client.post(fx.url("/jobs/" + name), body.dump(), fx.auth());
    return r.status;
}

/// Seed a trigger-less decision. `/decision` (store_knowledge) requires a
/// trigger; `/store category=decision` is the legacy path that produces
/// exactly the trigger-less rows the back-fill targets.
long seed_decision(HttpFixture& fx, const std::string& text) {
    auto r = fx.client.post(fx.url("/store"),
                            json{{"text", text},
                                 {"metadata", {{"category", "decision"}}}}.dump(),
                            fx.auth());
    CHECK(r.ok());
    CHECK(r.status == 200);
    auto parsed = json::parse(r.body);
    if (parsed.contains("id") && parsed["id"].is_number())
        return parsed["id"].get<long>();
    if (parsed.contains("knowledge_id"))
        return parsed["knowledge_id"].get<long>();
    CHECK(!parsed.is_null());
    return -1;
}

int triggers_set(sqlite3* db) {
    sqlite3_stmt* st = nullptr;
    CHECK(sqlite3_prepare_v2(
        db, "SELECT COUNT(*) FROM knowledge WHERE kind IN ('decision','insight')"
            " AND trigger IS NOT NULL AND trigger != ''",
        -1, &st, nullptr) == SQLITE_OK);
    int n = -1;
    if (sqlite3_step(st) == SQLITE_ROW) n = sqlite3_column_int(st, 0);
    sqlite3_finalize(st);
    return n;
}

// ---------------------------------------------------------------------------
// SSE reader: collects `event: job` frames from /events on its own client.
// ---------------------------------------------------------------------------

struct sse_reader {
    std::thread th;
    std::atomic<bool> stop{false};
    std::mutex mu;
    std::vector<json> job_frames;   // parsed data of every `job` event

    void start(const std::string& url, const std::vector<std::string>& headers) {
        th = std::thread([this, url, headers] {
            ragger::util::HttpClient http;
            std::string buf;
            http.stream(url, headers,
                        [this, &buf](std::string_view chunk) {
                            if (stop.load()) return false;
                            buf.append(chunk);
                            // SSE frames are separated by a blank line.
                            size_t pos;
                            while ((pos = buf.find("\n\n")) != std::string::npos) {
                                const std::string frame = buf.substr(0, pos);
                                buf.erase(0, pos + 2);
                                const std::string ev = "event: job\ndata: ";
                                if (frame.rfind(ev, 0) == 0) {
                                    auto j = json::parse(frame.substr(ev.size()),
                                                         nullptr, false);
                                    if (!j.is_discarded()) {
                                        std::lock_guard<std::mutex> lk(mu);
                                        job_frames.push_back(std::move(j));
                                    }
                                }
                            }
                            return true;
                        },
                        120);
        });
    }
    void stop_and_join() {
        stop.store(true);
        if (th.joinable()) th.join();
    }
    size_t count() {
        std::lock_guard<std::mutex> lk(mu);
        return job_frames.size();
    }
    int max_done() {
        std::lock_guard<std::mutex> lk(mu);
        int m = 0;
        for (const auto& j : job_frames) {
            const auto& job = !j["current"].is_null() ? j["current"] : j["last"];
            if (job.is_null()) continue;
            m = std::max(m, job["done"].get<int>());
        }
        return m;
    }
};

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

void t_jobs_unknown(HttpFixture& fx) {
    CHECK(post_job(fx, "bogus") == 400);
    // re-embed is deliberately NOT a job (rule 17: Reid runs it).
    CHECK(post_job(fx, "re-embed") == 400);
    // No job ever started: nothing current, nothing last.
    auto j = get_jobs(fx);
    CHECK(j["current"].is_null());
    CHECK(j["last"].is_null());
}

void t_jobs_dry_runs(HttpFixture& fx) {
    CHECK(post_job(fx, "reindex", true) == 202);
    CHECK(wait_last(fx, "reindex", "done"));
    CHECK(last_message(fx).find("would reindex") != std::string::npos);

    CHECK(post_job(fx, "housekeeping", true) == 202);
    CHECK(wait_last(fx, "housekeeping", "done"));
    CHECK(last_message(fx).find("would purge") != std::string::npos);
}

void t_jobs_cancel_and_rerun(HttpFixture& fx, job_stub& stub, sqlite3* db,
                             sse_reader& sse) {
    seed_decision(fx, "The deployment target for the widget service is the beelink box.");
    seed_decision(fx, "Backups of the memories database run weekly to the external drive.");
    const int before = triggers_set(db);

    stub.mode.store(2);            // slow: each model call sleeps
    stub.slow_ms.store(1200);
    CHECK(post_job(fx, "backfill:triggers") == 202);

    // Running: observable via GET /jobs.
    if (!poll_until([&] {
            auto j = get_jobs(fx);
            return !j["current"].is_null() &&
                   j["current"]["state"].get<std::string>() != "queued";
        }, 3000)) {
        std::println(stderr, "DIAG jobs after 3s: {}", get_jobs(fx).dump());
        CHECK(false);
    }

    // Single-flight: a second start is refused while one runs.
    CHECK(post_job(fx, "backfill:triggers") == 409);

    // Cancel between batches: the first row completes (it was already in
    // flight), the second is never started.
    {
        auto r = fx.client.post(fx.url("/jobs/cancel"), "{}", fx.auth());
        CHECK(r.status == 200);
    }
    CHECK(wait_last(fx, "backfill:triggers", "cancelled"));

    const auto msg = last_message(fx);
    if (msg.find("1 applied") == std::string::npos) {
        std::println(stderr, "DIAG cancel msg: {} (stub calls: {})", msg,
                     stub.calls.load());
        CHECK(false);
    }
    // Consistent partial run: exactly one of the two rows gained a trigger.
    CHECK(triggers_set(db) == before + 1);

    // Rerun finishes the rest, and a rerun after completion selects 0.
    stub.mode.store(1);
    CHECK(post_job(fx, "backfill:triggers") == 202);
    CHECK(wait_last(fx, "backfill:triggers", "done"));
    CHECK(last_message(fx).find("1 applied") != std::string::npos);
    CHECK(triggers_set(db) == before + 2);

    CHECK(post_job(fx, "backfill:triggers") == 202);
    CHECK(wait_last(fx, "backfill:triggers", "done"));
    CHECK(last_message(fx).find("0 candidate") != std::string::npos);

    // The SSE stream carried job frames with growing `done` throughout.
    // The /events provider polls every 2 s, so wait for its frames rather
    // than assuming they landed mid-test.
    CHECK(poll_until([&] { return sse.count() >= 1 && sse.max_done() >= 1; },
                     15000));
}

void t_jobs_failed(HttpFixture& fx, job_stub& stub) {
    seed_decision(fx, "The summarizer prompt keeps decisions, names and dates from the span.");
    stub.mode.store(0);            // every model call fails
    CHECK(post_job(fx, "backfill:triggers") == 202);
    CHECK(wait_last(fx, "backfill:triggers", "failed"));
    CHECK(last_message(fx).find("1 failed") != std::string::npos);
}

} // namespace

int main() {
    using namespace ragger_test;

    // The stub exists before the daemon so its port can go into the config
    // the Server reads at startup (init_inference).
    job_stub stub;

    auto fx = std::make_unique<HttpFixture>("jobs", [&stub] {
        // Inference endpoint -> the stub. Until a test arms the stub it
        // answers 500, so the daemon's summarizer behaves exactly as it
        // does today with an unreachable endpoint (no new writes race the
        // tests).
        bool ok = ragger::update_config([&stub](ragger::Config& c) {
            c.inference_api_url =
                "http://127.0.0.1:" + std::to_string(stub.port) + "/v1";
            c.inference_model = "stub-model";
            // The scratch base inherits the developer's live
            // inference_endpoints (TestEnv captures them before the
            // override): drop them so EVERY model — including the
            // summarizer model — resolves to the stub, not localhost:1234.
            c.inference_endpoints.clear();
            c.summarizer_api_url.clear();
            c.summarizer_model = "stub-model";
            return true;
        });
        CHECK(ok);
        // Silence the housekeeping timer: its catch-up must never race the
        // job tests.
        auto r = ragger::set_config_persisted("housekeeping_interval", "0");
        CHECK(r.has_value());
    });
    CHECK(fx->ready);
    std::println("Running job-runner tests against 127.0.0.1:{}:", fx->port);

    sqlite3* db = nullptr;
    CHECK(sqlite3_open((fx->env.base() + "/http_test.db").c_str(), &db) == SQLITE_OK);

    sse_reader sse;
    sse.start(fx->url("/events"), {ragger::util::bearer_header(fx->token)});

    t_jobs_unknown(*fx);
    t_jobs_dry_runs(*fx);
    t_jobs_cancel_and_rerun(*fx, stub, db, sse);
    t_jobs_failed(*fx, stub);

    sse.stop_and_join();
    sqlite3_close(db);

    // Daemon shutdown while a job runs (LAST: destroys the fixture). The
    // worker is blocked in a slow model call; stop() -> run() returns ->
    // stop_background_threads cancels and joins it. Two pins:
    //  * the destructor returning at all = the worker was joined (a hung
    //    worker would hang this process and fail the CTest timeout);
    //  * the elapsed time < two slow calls = the cancel was observed at the
    //    batch boundary — the second candidate was never started (the DB
    //    consistency of a cancelled backfill is pinned by the cancel test).
    seed_decision(*fx, "Reindex rebuilds the text index for every registered text table.");
    stub.mode.store(2);
    stub.slow_ms.store(2500);
    CHECK(post_job(*fx, "backfill:triggers") == 202);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));  // running
    const auto t0 = std::chrono::steady_clock::now();
    fx.reset();   // stop + join (waits out the in-flight call, then cancels)
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
    CHECK(elapsed < 4500);   // one slow call (2500) + slack, not two (5000)

    std::println("test_jobs: all passed");
    return 0;
}
