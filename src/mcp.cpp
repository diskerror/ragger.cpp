/**
 * MCP (Model Context Protocol) server implementation.
 *
 * JSON-RPC 2.0 over stdin/stdout.  Tools exposed:
 *   store  — write a memory
 *   search — semantic + BM25 hybrid search
 *
 * Housekeeping: if no ragger HTTP server PID file is found, a
 * background thread runs periodic cleanup so the database doesn't
 * grow unbounded when the user only uses MCP mode.
 */

#include "ragger/mcp.h"
#include "ragger/config.h"
#include "ragger/lang.h"
#include "ragger/logs.h"
#include "ragger/memory.h"
#include "ragger/sqlite_backend.h"
#include "nlohmann_json.hpp"

#include <chrono>
#include <csignal>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <print>
#include <pwd.h>
#include <thread>
#include <unistd.h>

namespace ragger {

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

/// Returns true if a ragger HTTP server PID file exists and the process
/// is still alive, meaning it owns housekeeping for the database.
static bool http_server_running() {
    namespace fs = std::filesystem;
    const std::string dir = "/tmp/ragger";
    try {
        for (const auto& entry : fs::directory_iterator(dir)) {
            auto name = entry.path().filename().string();
            if (name.rfind("server-", 0) == 0 &&
                name.substr(name.size() - 4) == ".pid") {
                std::ifstream pf(entry.path());
                pid_t pid = 0;
                if (pf >> pid && pid > 0 && kill(pid, 0) == 0)
                    return true;
            }
        }
    } catch (...) {}
    return false;
}

/// MCP tool definitions returned for tools/list.
static nlohmann::json tools_list() {
    return {{"tools", nlohmann::json::array({
        {
            {"name", "store"},
            {"description", "Store a memory for later semantic retrieval."},
            {"inputSchema", {
                {"type", "object"},
                {"properties", {
                    {"text",     {{"type", "string"},
                                  {"description", "The text content to store."}}},
                    {"metadata", {{"type", "object"},
                                  {"description", "Optional metadata (category, tags, source, collection, etc.)."}}}
                }},
                {"required", nlohmann::json::array({"text"})}
            }}
        },
        {
            {"name", "search"},
            {"description", "Search stored memories by semantic similarity."},
            {"inputSchema", {
                {"type", "object"},
                {"properties", {
                    {"query",       {{"type", "string"},
                                    {"description", "The search query."}}},
                    {"limit",       {{"type", "integer"},
                                    {"description", "Maximum number of results (default: 5)."}}},
                    {"min_score",   {{"type", "number"},
                                    {"description", "Minimum similarity score 0–1 (default: 0.0)."}}},
                    {"collections", {{"type", "array"},
                                    {"items", {{"type", "string"}}},
                                    {"description", "Filter by collection names."}}}
                }},
                {"required", nlohmann::json::array({"query"})}
            }}
        }
    })}};
}

/// Dispatch a tools/call request and return the MCP result object.
static nlohmann::json tool_call(RaggerMemory& memory,
                                const nlohmann::json& params) {
    const auto tool_name = params.value("name", "");
    const auto arguments = params.value("arguments", nlohmann::json::object());

    auto text_result = [](const std::string& text) {
        return nlohmann::json{
            {"content", nlohmann::json::array({{{"type", "text"}, {"text", text}}})}
        };
    };
    auto error_result = [](const std::string& msg) {
        return nlohmann::json{
            {"content", nlohmann::json::array({{{"type", "text"}, {"text", msg}}})},
            {"isError", true}
        };
    };

    if (tool_name == "store") {
        const auto text = arguments.value("text", "");
        if (text.empty())
            return error_result(ragger::lang::ERR_MCP_TEXT_REQUIRED);
        const auto metadata = arguments.value("metadata", nlohmann::json::object());
        const auto id = memory.store(text, metadata);
        return text_result(nlohmann::json({{"id", id}, {"status", "stored"}}).dump());

    } else if (tool_name == "search") {
        const auto query = arguments.value("query", "");
        if (query.empty())
            return error_result(ragger::lang::ERR_MCP_QUERY_REQUIRED);
        const int   limit     = arguments.value("limit", 5);
        const float min_score = arguments.value("min_score", 0.0f);
        const auto  colls     = arguments.value("collections",
                                                std::vector<std::string>{});
        const auto  response  = memory.search(query, limit, min_score, colls);

        nlohmann::json arr = nlohmann::json::array();
        for (const auto& r : response.results) {
            arr.push_back({
                {"id",        r.id},
                {"text",      r.text},
                {"score",     r.score},
                {"metadata",  r.metadata},
                {"timestamp", r.timestamp}
            });
        }
        return text_result(arr.dump());

    } else {
        return error_result("Unknown tool: " + tool_name);
    }
}

/// Background thread: periodically remove expired memories when no
/// HTTP server is doing it.
static void housekeeping_thread(RaggerMemory& memory,
                                const std::string& /*username*/) {
    const auto& cfg = config();
    const float max_age_hours = cfg.cleanup_max_age_hours;
    const int   interval      = cfg.housekeeping_interval;
    if (interval == 0) return;

    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(interval));
        if (http_server_running()) continue;  // hand off to HTTP server
        if (max_age_hours <= 0) continue;

        try {
            SqliteBackend tmp(memory.backend()->db_path());
            const int deleted = tmp.cleanup_old_conversations(max_age_hours);
            if (deleted > 0) {
                log_info("MCP housekeeping: cleaned "
                         + std::to_string(deleted) + " expired conversations");
            }
        } catch (...) {}
    }
}

// -----------------------------------------------------------------------
// Public entry point
// -----------------------------------------------------------------------

void run_mcp(RaggerMemory& memory) {
    // Spawn housekeeping only when this is the sole Ragger process.
    if (!http_server_running()) {
        struct passwd* pw = getpwuid(getuid());
        const std::string username = pw ? pw->pw_name : "default";
        std::thread(housekeeping_thread,
                    std::ref(memory), username).detach();
    }

    // JSON-RPC 2.0 read loop
    auto send = [](const nlohmann::json& msg) {
        std::println("{}", msg.dump());
    };

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        if (line[0] == '{') {
            try {
                auto request = nlohmann::json::parse(line);
                const auto method = request.value("method", "");
                const auto params = request.value("params",
                                                  nlohmann::json::object());

                // Notifications have no "id" — no response required.
                if (!request.contains("id")) continue;
                const auto req_id = request["id"];

                nlohmann::json result;

                if (method == "initialize") {
                    result = {
                        {"protocolVersion", "2024-11-05"},
                        {"capabilities",    {{"tools", nlohmann::json::object()}}},
                        {"serverInfo",      {{"name",    "ragger-memory"},
                                             {"version", RAGGER_VERSION}}}
                    };

                } else if (method == "tools/list") {
                    result = tools_list();

                } else if (method == "tools/call") {
                    result = tool_call(memory, params);

                } else {
                    send({{"jsonrpc", "2.0"}, {"id", req_id},
                          {"error", {{"code",    -32601},
                                     {"message", "Method not found: " + method}}}});
                    continue;
                }

                send({{"jsonrpc", "2.0"}, {"id", req_id}, {"result", result}});

            } catch (const std::exception& e) {
                send({{"jsonrpc", "2.0"},
                      {"id",      nullptr},
                      {"error",   {{"code", -32603}, {"message", e.what()}}}});
            }

        } else {
            // Plain text → interactive search shortcut
            try {
                const auto response = memory.search(line);
                if (response.results.empty()) {
                    std::println("{}", ragger::lang::MSG_MCP_NO_RESULTS);
                } else {
                    for (size_t i = 0; i < response.results.size(); ++i) {
                        const auto& r      = response.results[i];
                        const auto  source = r.metadata.value("source", "");
                        const auto  coll   = r.metadata.value("collection", "");
                        std::print("{}. [score: {:.3f}]", i + 1, r.score);
                        if (!source.empty()) std::print(" ({})", source);
                        if (!coll.empty())   std::print(" [{}]", coll);
                        std::println("");
                        std::print("   {}", r.text.substr(0, 200));
                        if (r.text.size() > 200) std::print("...");
                        std::println("\n");
                    }
                }
            } catch (const std::exception& e) {
                std::println("Error: {}", e.what());
            }
        }
    }
}

} // namespace ragger
