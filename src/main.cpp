/**
 * ragger — C++ port of Ragger Memory
 *
 * Verb-style CLI: ragger <verb> [options] [args]
 * No verb or 'help' prints usage.
 */
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <pwd.h>
#include <regex>
#include <set>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include <signal.h>
#include <iomanip>
#include <curl/curl.h>
#include <thread>
#include <chrono>

#include "ProgramOptions.h"
#include "ragger/auth.h"
#include "ragger/chat.h"
#include "ragger/client.h"
#include "ragger/config.h"
#include "ragger/import.h"
#include "ragger/inference.h"
#include "ragger/lang.h"
#include "ragger/logs.h"
#include "ragger/memory.h"
#include "ragger/server.h"
#include "ragger/llama_manager.h"
#include "ragger/embedder.h"
#include "ragger/sqlite_backend.h"
#include "nlohmann_json.hpp"

using namespace ragger::lang;
namespace fs = std::filesystem;

// -----------------------------------------------------------------------
// Import: uses ragger::chunk_markdown from import.h
// -----------------------------------------------------------------------
static void do_import(ragger::RaggerMemory& memory,
                      const std::string& filepath,
                      const std::string& collection,
                      int min_chunk_size) {
    if (!fs::exists(filepath)) {
        throw std::runtime_error("File not found: " + filepath);
    }

    std::ifstream file(filepath);
    std::string text((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());

    auto chunks = ragger::chunk_markdown(text, min_chunk_size);

    auto filename = fs::path(filepath).filename().string();
    std::cout << "Importing " << chunks.size() << " chunks from " << filename << "...\n";

    for (size_t i = 0; i < chunks.size(); ++i) {
        nlohmann::json meta = {
            {"source", filepath},
            {"chunk", (int)(i + 1)},
            {"total_chunks", (int)chunks.size()}
        };
        if (!collection.empty()) meta["collection"] = collection;
        if (!chunks[i].section.empty()) meta["section"] = chunks[i].section;

        auto id = memory.store(chunks[i].text, meta);
        std::cout << "  Chunk " << (i + 1) << "/" << chunks.size() << ": " << id << "\n";
    }
    std::cout << "✓ Imported " << chunks.size() << " chunks\n";
}

// -----------------------------------------------------------------------
// Export: reassemble chunks into files with heading deduplication
// -----------------------------------------------------------------------

/// Split a chunk's text into heading lines and body text.
/// During import, the full heading chain is prepended to each chunk.
static std::pair<std::vector<std::string>, std::string>
split_heading_body(const std::string& text) {
    std::vector<std::string> headings;
    std::istringstream stream(text);
    std::string line;
    std::vector<std::string> all_lines;

    while (std::getline(stream, line)) {
        all_lines.push_back(line);
    }

    size_t i = 0;
    while (i < all_lines.size()) {
        // Trim whitespace for matching
        std::string trimmed = all_lines[i];
        auto start = trimmed.find_first_not_of(" \t");
        if (start != std::string::npos) trimmed = trimmed.substr(start);
        auto end = trimmed.find_last_not_of(" \t\r\n");
        if (end != std::string::npos) trimmed = trimmed.substr(0, end + 1);

        // Check if line starts with # followed by space
        if (!trimmed.empty() && trimmed[0] == '#') {
            auto space_pos = trimmed.find(' ');
            bool is_heading = space_pos != std::string::npos;
            if (is_heading) {
                // Verify all chars before space are #
                bool all_hash = true;
                for (size_t j = 0; j < space_pos; j++) {
                    if (trimmed[j] != '#') { all_hash = false; break; }
                }
                if (all_hash && space_pos <= 6) {
                    headings.push_back(trimmed);
                    i++;
                    // Skip blank lines between headings
                    while (i < all_lines.size()) {
                        std::string t = all_lines[i];
                        auto s = t.find_first_not_of(" \t\r\n");
                        if (s == std::string::npos) { i++; continue; }
                        break;
                    }
                    continue;
                }
            }
        }
        break;
    }

    // Build body from remaining lines
    std::string body;
    for (size_t j = i; j < all_lines.size(); j++) {
        if (!body.empty()) body += '\n';
        body += all_lines[j];
    }
    // Trim trailing whitespace
    auto last = body.find_last_not_of(" \t\r\n");
    if (last != std::string::npos) body = body.substr(0, last + 1);

    return {headings, body};
}

/// Check if a line is a markdown heading (# through ######)
static bool is_heading_line(const std::string& line) {
    auto start = line.find_first_not_of(" \t");
    if (start == std::string::npos) return false;
    std::string trimmed = line.substr(start);
    if (trimmed.empty() || trimmed[0] != '#') return false;
    auto space = trimmed.find(' ');
    if (space == std::string::npos || space > 6) return false;
    for (size_t i = 0; i < space; i++) {
        if (trimmed[i] != '#') return false;
    }
    return true;
}

static std::string trim_line(const std::string& s) {
    auto start = s.find_first_not_of(" \t");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

static void do_export_docs(ragger::RaggerMemory& memory,
                           const std::string& collection,
                           const std::string& dest_dir) {
    fs::create_directories(dest_dir);

    auto all = memory.load_all(collection);
    if (all.empty()) {
        std::cout << "No documents found in collection '" << collection << "'\n";
        return;
    }

    // Group by source filename
    std::map<std::string, std::vector<ragger::SearchResult*>> files;
    for (auto& r : all) {
        std::string source = r.metadata.value("source", "unknown.md");
        files[source].push_back(&r);
    }

    std::cout << "Exporting " << files.size() << " documents from '" << collection << "'...\n";

    for (auto& [source, chunks] : files) {
        std::set<std::string> seen_headings;
        std::vector<std::string> output_parts;

        for (auto* chunk : chunks) {
            auto [headings, body] = split_heading_body(chunk->text);

            // Emit only new headings
            std::vector<std::string> new_headings;
            for (const auto& h : headings) {
                if (seen_headings.find(h) == seen_headings.end()) {
                    seen_headings.insert(h);
                    new_headings.push_back(h);
                }
            }

            if (!new_headings.empty()) {
                std::string heading_block;
                for (const auto& h : new_headings) {
                    if (!heading_block.empty()) heading_block += "\n\n";
                    heading_block += h;
                }
                output_parts.push_back(heading_block);
            }

            if (!body.empty()) {
                // Filter duplicate headings from body
                std::istringstream bs(body);
                std::string bline;
                std::vector<std::string> filtered;
                while (std::getline(bs, bline)) {
                    std::string trimmed = trim_line(bline);
                    if (is_heading_line(trimmed) &&
                        seen_headings.find(trimmed) != seen_headings.end()) {
                        continue;  // skip duplicate heading in body
                    }
                    if (is_heading_line(trimmed)) {
                        seen_headings.insert(trimmed);
                    }
                    filtered.push_back(bline);
                }
                std::string filtered_body;
                for (const auto& fl : filtered) {
                    if (!filtered_body.empty()) filtered_body += '\n';
                    filtered_body += fl;
                }
                // Trim
                auto last = filtered_body.find_last_not_of(" \t\r\n");
                if (last != std::string::npos) {
                    filtered_body = filtered_body.substr(0, last + 1);
                }
                if (!filtered_body.empty()) {
                    output_parts.push_back(filtered_body);
                }
            }
        }

        // Join with double newlines
        std::string content;
        for (const auto& part : output_parts) {
            if (!content.empty()) content += "\n\n";
            content += part;
        }
        content += "\n";

        // Collapse triple+ newlines
        std::string collapsed;
        int newline_count = 0;
        for (char c : content) {
            if (c == '\n') {
                newline_count++;
                if (newline_count <= 2) collapsed += c;
            } else {
                newline_count = 0;
                collapsed += c;
            }
        }

        auto out_path = fs::path(dest_dir) / fs::path(source).filename();
        std::ofstream f(out_path);
        f << collapsed;
        std::cout << "  " << fs::path(source).filename().string()
                  << " (" << chunks.size() << " chunks)\n";
    }
    std::cout << "✓ Exported " << files.size() << " documents to " << dest_dir << "\n";
}

static void do_export_memories(ragger::RaggerMemory& memory,
                               const std::string& dest_dir,
                               const std::string& group_by) {
    fs::create_directories(dest_dir);

    auto all = memory.load_all("memory");
    if (all.empty()) {
        std::cout << "No memories to export\n";
        return;
    }

    // Group entries
    std::map<std::string, std::vector<ragger::SearchResult*>> groups;
    for (auto& r : all) {
        std::string key;
        if (group_by == "date") {
            key = r.timestamp.substr(0, 10);  // YYYY-MM-DD
        } else if (group_by == "category") {
            key = r.metadata.value("category", "uncategorized");
        } else if (group_by == "collection") {
            key = r.metadata.value("collection", "memory");
        } else {
            key = "all";
        }
        groups[key].push_back(&r);
    }

    std::cout << "Exporting " << all.size() << " memories ("
              << groups.size() << " groups by " << group_by << ")...\n";

    for (auto& [key, entries] : groups) {
        auto out_path = fs::path(dest_dir) / (key + ".md");
        std::ofstream f(out_path);
        f << "# Memories — " << key << "\n\n";

        for (auto* r : entries) {
            std::string header;
            if (!r->timestamp.empty()) {
                header = r->timestamp.substr(0, 19);
            }
            auto cat = r->metadata.value("category", "");
            if (!cat.empty()) {
                if (!header.empty()) header += " | ";
                header += "**" + cat + "**";
            }
            f << "### " << header << "\n\n" << r->text << "\n\n---\n\n";
        }

        std::cout << "  " << key << ".md (" << entries.size() << " entries)\n";
    }
    std::cout << "✓ Exported " << all.size() << " memories to " << dest_dir << "\n";
}

static void do_export_all(ragger::RaggerMemory& memory,
                          const std::string& dest_dir,
                          const std::string& group_by) {
    auto colls = memory.collections();
    for (auto& col : colls) {
        if (col == "memory") {
            do_export_memories(memory, (fs::path(dest_dir) / "memories").string(), group_by);
        } else {
            do_export_docs(memory, col, (fs::path(dest_dir) / col).string());
        }
    }
}

// -----------------------------------------------------------------------
// Chat: simple REPL with memory context injection
// -----------------------------------------------------------------------
static void do_chat(const std::string& db_path, const std::string& model_dir) {
    const auto& cfg = ragger::config();

    // Build inference client from config
    ragger::InferenceClient inference = ragger::InferenceClient::from_config(cfg);

    if (inference._endpoints.empty()) {
        std::cout << "Error: no inference endpoints configured.\n";
        std::cout << "Add to ragger.ini:\n\n";
        std::cout << "  [inference]\n";
        std::cout << "  api_url = http://localhost:1234/v1\n";
        std::cout << "  api_key = lmstudio-local\n\n";
        std::cout << "Or for multiple endpoints:\n\n";
        std::cout << "  [inference.local]\n";
        std::cout << "  api_url = http://localhost:1234/v1\n";
        std::cout << "  api_key = lmstudio-local\n";
        std::cout << "  models = qwen/*, llama/*\n";
        return;
    }

    // Load memory for context search
    ragger::RaggerMemory memory(db_path, model_dir);

    // Create and run chat session
    ragger::Chat chat(memory, inference, cfg.inference_model);
    chat.run();
}

// -----------------------------------------------------------------------
// MCP: JSON-RPC server over stdin/stdout
// -----------------------------------------------------------------------
static void do_mcp(ragger::RaggerMemory& memory) {
    auto send_response = [](const nlohmann::json& response) {
        std::cout << response.dump() << std::endl;
    };

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        if (line[0] == '{') {
            try {
                auto request = nlohmann::json::parse(line);
                auto method = request.value("method", "");
                auto params = request.value("params", nlohmann::json::object());
                auto req_id = request.value("id", nlohmann::json());

                nlohmann::json result;

                if (method == "memory_store") {
                    auto text = params.value("text", "");
                    auto metadata = params.value("metadata", nlohmann::json::object());
                    if (text.empty()) throw std::runtime_error("text parameter required");
                    auto id = memory.store(text, metadata);
                    result = {{"id", id}, {"status", "stored"}};

                } else if (method == "memory_search") {
                    auto query = params.value("query", "");
                    int limit = params.value("limit", 5);
                    float min_score = params.value("min_score", 0.0f);
                    auto collections = params.value("collections", std::vector<std::string>{});
                    if (query.empty()) throw std::runtime_error("query parameter required");
                    auto response = memory.search(query, limit, min_score, collections);
                    nlohmann::json results_arr = nlohmann::json::array();
                    for (auto& r : response.results) {
                        results_arr.push_back({
                            {"id", r.id}, {"text", r.text}, {"score", r.score},
                            {"metadata", r.metadata}, {"timestamp", r.timestamp}
                        });
                    }
                    result = {{"results", results_arr}};

                } else {
                    throw std::runtime_error("Unknown method: " + method);
                }

                send_response({
                    {"jsonrpc", "2.0"}, {"id", req_id}, {"result", result}
                });

            } catch (const std::exception& e) {
                send_response({
                    {"jsonrpc", "2.0"},
                    {"id", nullptr},
                    {"error", {{"code", -32603}, {"message", e.what()}}}
                });
            }
        } else {
            // Plain text → search shortcut
            try {
                auto response = memory.search(line);
                if (response.results.empty()) {
                    std::cout << "No results found." << std::endl;
                } else {
                    for (size_t i = 0; i < response.results.size(); ++i) {
                        auto& r = response.results[i];
                        auto source = r.metadata.value("source", "");
                        auto collection = r.metadata.value("collection", "");
                        std::cout << (i + 1) << ". [score: "
                                  << std::fixed << std::setprecision(3) << r.score << "]";
                        if (!source.empty()) std::cout << " (" << source << ")";
                        if (!collection.empty()) std::cout << " [" << collection << "]";
                        std::cout << "\n   " << r.text.substr(0, 200);
                        if (r.text.size() > 200) std::cout << "...";
                        std::cout << "\n\n";
                    }
                }
            } catch (const std::exception& e) {
                std::cout << "Error: " << e.what() << std::endl;
            }
        }
    }
}

// -----------------------------------------------------------------------
// User provisioning
// -----------------------------------------------------------------------

/// Provision a user: create ~/.ragger/ and token file.
/// Returns {token, created}. If token already exists, returns {existing, false}.
static std::pair<std::string, bool> provision_user(
        const std::string& username,
        const std::string& home_override = "") {
    std::string home_dir = home_override;
    if (home_dir.empty()) {
        struct passwd* pw = getpwnam(username.c_str());
        if (!pw) throw std::runtime_error("User not found: " + username);
        home_dir = pw->pw_dir;
    }

    std::string ragger_dir = home_dir + "/.ragger";
    std::string tok_path = ragger_dir + "/token";

    // Check existing
    if (fs::exists(tok_path)) {
        std::ifstream f(tok_path);
        std::string token;
        std::getline(f, token);
        // trim
        size_t s = token.find_first_not_of(" \t\r\n");
        size_t e = token.find_last_not_of(" \t\r\n");
        if (s != std::string::npos) {
            token = token.substr(s, e - s + 1);
            if (!token.empty()) return {token, false};
        }
    }

    // Create directory and token
    fs::create_directories(ragger_dir);
    std::string token = ragger::generate_token();
    {
        std::ofstream f(tok_path);
        f << token << "\n";
    }
    chmod(tok_path.c_str(), 0640);

    // Set ownership if running as root
    if (getuid() == 0) {
        struct passwd* pw = getpwnam(username.c_str());
        if (pw) {
            chown(ragger_dir.c_str(), pw->pw_uid, pw->pw_gid);
            chown(tok_path.c_str(), pw->pw_uid, pw->pw_gid);
        }
    }

    return {token, true};
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main(int argc, char** argv) {
    Diskerror::ProgramOptions opts(CLI_DESCRIPTION);
    opts.add_options()
        ("help,h", CLI_HELP)
        ("version,V", CLI_VERSION)
        ("config", Diskerror::po::value<std::string>()->default_value(""), CLI_CONFIG_FILE)
        ("host", Diskerror::po::value<std::string>(), CLI_HOST)
        ("port,p", Diskerror::po::value<int>(), CLI_PORT)
        ("db", Diskerror::po::value<std::string>(), CLI_DB)
        ("model-dir", Diskerror::po::value<std::string>(), CLI_MODEL_DIR)
        ("collection", Diskerror::po::value<std::string>()->default_value(""), "Collection name")
        ("min-chunk-size", Diskerror::po::value<int>(), "Min chunk size for import")
        ("group-by", Diskerror::po::value<std::string>()->default_value("date"), "Grouping for export (date|category|collection)")
        ("admin", "Grant admin privileges (for add-user)")
        ("yes,y", "Skip confirmation prompts (for scripting)")
        ("no-llama", "Don't start llama-server subprocess")
    ;
    opts.add_hidden_options()
        ("command", Diskerror::po::value<std::string>()->default_value("help"), CLI_COMMAND)
        ("args", Diskerror::po::value<std::vector<std::string>>(), CLI_ARGS)
    ;
    opts.add_positional("command", 1);
    opts.add_positional("args", -1);

    try {
        opts.run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    auto command = opts["command"].as<std::string>();

    if (opts.count("help") || command == "help") {
        std::cout << "ragger " << RAGGER_VERSION << "\n\n";
        std::cout << "Usage: ragger <command> [options] [args]\n\n";
        std::cout << "Commands:\n";
        std::cout << "  serve              Start the HTTP server (default: 127.0.0.1:8432)\n";
        std::cout << "  search <query>     Search memories by meaning\n";
        std::cout << "  store <text>       Store a new memory\n";
        std::cout << "  count              Show number of stored memories\n";
        std::cout << "  import <file...>   Import files (paragraph-aware chunking)\n";
        std::cout << "  export <mode> <dir>  Export docs|memories|all to directory\n";
        std::cout << "  chat               Interactive chat with memory context\n";
        std::cout << "  mcp                Start MCP server (JSON-RPC over stdin/stdout)\n";
        std::cout << "  add-self           Provision yourself (create token)\n";
        std::cout << "  add-user <name>    Provision a user (requires sudo)\n";
        std::cout << "  add-all            Provision all users (requires sudo)\n";
        std::cout << "  rebuild-bm25       Rebuild the BM25 keyword index\n";
        std::cout << "  rebuild-embeddings Rebuild embeddings for all memories\n";
        std::cout << "  model <list|download|remove>  Manage GGUF models\n";
        std::cout << "  llama <start|stop|status>  Manage llama-server subprocess\n";
        std::cout << "  help               Show this help\n";
        std::cout << "  version            Show version\n";
        std::cout << "\nOptions:\n";
        std::cout << opts.to_string() << "\n";
        return 0;
    }

    if (opts.count("version") || command == "version") {
        std::cout << "ragger " << RAGGER_VERSION << "\n"
                  << "commit " << RAGGER_COMMIT << "\n"
                  << "built  " << RAGGER_BUILD_DATE << "\n";
        return 0;
    }

    // Load config file
    try {
        ragger::init_config(opts["config"].as<std::string>());
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    const auto& cfg = ragger::config();

    // CLI overrides
    std::string host       = opts.count("host")      ? opts["host"].as<std::string>()      : cfg.host;
    int         port       = opts.count("port")      ? opts["port"].as<int>()               : cfg.port;
    std::string db_path    = opts.count("db")         ? opts["db"].as<std::string>()         : "";
    std::string model_dir  = opts.count("model-dir")  ? opts["model-dir"].as<std::string>()  : "";
    std::string collection = opts["collection"].as<std::string>();
    int min_chunk_size     = opts.count("min-chunk-size")
                             ? opts["min-chunk-size"].as<int>()
                             : cfg.minimum_chunk_size;
    std::string group_by   = opts["group-by"].as<std::string>();

    try {
        if (command == "serve") {
            ragger::setup_logging(false, true);
            const auto& cfg = ragger::config();
            std::unique_ptr<ragger::RaggerMemory> mem_ptr;
            if (!cfg.single_user) {
                auto common_path = cfg.resolved_common_db_path();
                auto user_path = cfg.resolved_db_path();
                mem_ptr = std::make_unique<ragger::RaggerMemory>(
                    common_path, model_dir, user_path);
                std::cout << "Multi-user mode: common=" << common_path
                          << ", user=" << user_path << "\n";
            } else {
                mem_ptr = std::make_unique<ragger::RaggerMemory>(db_path, model_dir);
            }
            auto& memory = *mem_ptr;
            char buf[128];
            std::snprintf(buf, sizeof(buf), MSG_LOADED_MEMORIES, memory.count());
            std::cout << buf << "\n";

            // Start llama-server subprocess if enabled
            std::unique_ptr<ragger::LlamaManager> llama;
            bool run_llama = cfg.llama_enabled && !opts.count("no-llama");
            if (run_llama) {
                llama = std::make_unique<ragger::LlamaManager>();
                if (!llama->start()) {
                    std::cerr << "Warning: llama-server failed to start, continuing without it\n";
                    llama.reset();
                }
            }

            ragger::Server server(memory, host, port);
            server.run();

            // Clean up llama-server on exit
            if (llama) llama->stop();

        } else if (command == "chat") {
            ragger::setup_logging(false, false);
            do_chat(db_path, model_dir);

        } else if (command == "search") {
            ragger::setup_logging(false, false);
            auto args = opts.getParams("args");
            if (args.empty()) {
                std::cerr << CLI_USAGE_SEARCH << "\n";
                return 1;
            }
            std::string query;
            for (size_t i = 0; i < args.size(); ++i) {
                if (i > 0) query += " ";
                query += args[i];
            }

            std::vector<std::string> colls;
            if (!collection.empty()) colls.push_back(collection);

            // Try daemon first (thin client — no model loading)
            auto token = ragger::load_token();
            ragger::RaggerClient client(cfg.host, cfg.port, token);
            ragger::SearchResponse response;

            if (client.is_available()) {
                response = client.search(query, cfg.default_search_limit,
                                        cfg.default_min_score, colls);
            } else {
                // Fall back to direct DB access
                ragger::RaggerMemory memory(db_path, model_dir);
                response = memory.search(query, cfg.default_search_limit,
                                        cfg.default_min_score, colls);
            }

            nlohmann::json output = nlohmann::json::array();
            for (const auto& r : response.results) {
                output.push_back({
                    {"id", r.id}, {"score", r.score},
                    {"text", r.text}, {"metadata", r.metadata}
                });
            }
            std::cout << output.dump(2) << "\n";

        } else if (command == "store") {
            ragger::setup_logging(false, false);
            auto args = opts.getParams("args");
            if (args.empty()) {
                std::cerr << CLI_USAGE_STORE << "\n";
                return 1;
            }
            std::string text;
            for (size_t i = 0; i < args.size(); ++i) {
                if (i > 0) text += " ";
                text += args[i];
            }

            nlohmann::json meta = {};
            if (!collection.empty()) meta["collection"] = collection;

            // Try daemon first (thin client — no model loading)
            auto token = ragger::load_token();
            ragger::RaggerClient client(cfg.host, cfg.port, token);
            std::string id;

            if (client.is_available()) {
                id = client.store(text, meta);
            } else {
                // Fall back to direct DB access
                ragger::RaggerMemory memory(db_path, model_dir);
                id = memory.store(text, meta);
            }
            std::cout << MSG_STORED_WITH_ID << id << "\n";

        } else if (command == "count") {
            ragger::setup_logging(false, false);
            
            // Try daemon first (thin client — no model loading)
            auto token = ragger::load_token();
            ragger::RaggerClient client(cfg.host, cfg.port, token);
            int count;

            if (client.is_available()) {
                count = client.count();
            } else {
                // Fall back to direct DB access
                ragger::RaggerMemory memory(db_path, model_dir);
                count = memory.count();
            }
            std::cout << count << "\n";

        } else if (command == "import") {
            ragger::setup_logging(false, false);
            auto args = opts.getParams("args");
            if (args.empty()) {
                std::cerr << "Usage: ragger import <file> [file...] [--collection name]\n";
                return 1;
            }
            ragger::RaggerMemory memory(db_path, model_dir);
            for (auto& filepath : args) {
                do_import(memory, filepath, collection, min_chunk_size);
            }

        } else if (command == "export") {
            ragger::setup_logging(false, false);
            auto args = opts.getParams("args");
            if (args.size() < 2) {
                std::cerr << "Usage: ragger export <docs|memories|all> <dest-dir> [--collection name]\n";
                return 1;
            }
            std::string mode = args[0];
            std::string dest = args[1];

            ragger::RaggerMemory memory(db_path, model_dir);
            if (mode == "docs") {
                if (collection.empty()) {
                    std::cerr << "Error: --collection required for docs export\n";
                    return 1;
                }
                do_export_docs(memory, collection, dest);
            } else if (mode == "memories") {
                do_export_memories(memory, dest, group_by);
            } else if (mode == "all") {
                do_export_all(memory, dest, group_by);
            } else {
                std::cerr << "Unknown export mode: " << mode << "\n";
                return 1;
            }

        } else if (command == "mcp") {
            ragger::setup_logging(false, false);
            ragger::RaggerMemory memory(db_path, model_dir);
            do_mcp(memory);

        } else if (command == "rebuild-bm25") {
            ragger::setup_logging(false, false);
            ragger::RaggerMemory memory(db_path, model_dir);
            int count = memory.rebuild_bm25();
            std::cout << "✓ BM25 index rebuilt: " << count << " documents\n";

        } else if (command == "rebuild-embeddings") {
            ragger::setup_logging(false, false);
            
            // Get count first (before loading full memory)
            ragger::RaggerMemory memory_temp(db_path, model_dir);
            int total_count = memory_temp.count();
            memory_temp.close();
            
            // Warning + confirmation prompt
            std::cout << "This will re-embed all " << total_count 
                      << " memories. The server should be stopped first.\n";
            
            bool proceed = false;
            if (opts.count("yes")) {
                proceed = true;
            } else {
                std::cout << "Continue? [y/N] ";
                std::string answer;
                std::getline(std::cin, answer);
                proceed = (answer == "y" || answer == "Y");
            }
            
            if (!proceed) {
                std::cout << "Aborted.\n";
                return 0;
            }
            
            // Backup the database file
            std::string actual_db_path = db_path.empty() ? cfg.resolved_db_path() : db_path;
            std::string backup_path = actual_db_path + ".bak";
            try {
                fs::copy_file(actual_db_path, backup_path, 
                             fs::copy_options::overwrite_existing);
                std::cout << "Database backed up to: " << backup_path << "\n";
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to create backup: " << e.what() << "\n";
            }
            
            // Rebuild embeddings
            ragger::RaggerMemory memory(db_path, model_dir);
            int count = memory.rebuild_embeddings();
            std::cout << "✓ Embeddings rebuilt: " << count << " documents\n";

        } else if (command == "add-self") {
            ragger::setup_logging(false, false);
            // getpwuid is more reliable than getlogin in non-TTY contexts
            struct passwd* self_pw = getpwuid(getuid());
            char* login = self_pw ? self_pw->pw_name : nullptr;
            if (!login) {
                std::cerr << "Error: cannot determine username\n";
                return 1;
            }
            std::string username(login);
            auto [token, created] = provision_user(username);
            if (created)
                std::cout << "✓ Created ~/.ragger/token for " << username << "\n";
            else
                std::cout << "Token already exists for " << username << "\n";
            // Register via daemon HTTP API
            try {
                ragger::RaggerClient client(host, port, token);
                if (client.is_available()) {
                    auto result = client.register_user(username);
                    if (result.contains("user_id"))
                        std::cout << "✓ Registered in database (user_id: "
                                  << result["user_id"] << ")\n";
                } else {
                    std::cout << "Server not running. Will auto-register on first authenticated request.\n";
                }
            } catch (const std::exception& e) {
                std::cout << "Server will auto-register on first authenticated request.\n";
            }

        } else if (command == "add-user") {
            ragger::setup_logging(false, false);
            auto args = opts.getParams("args");
            if (args.empty()) {
                std::cerr << "Usage: ragger add-user <username> [--admin]\n";
                return 1;
            }
            std::string username = args[0];
            bool is_admin = opts.count("admin") > 0;
            try {
                auto [token, created] = provision_user(username);
                if (created)
                    std::cout << "✓ Created token for " << username << "\n";
                else
                    std::cout << "Token already exists for " << username << "\n";
                // Register via daemon
                ragger::RaggerClient client(host, port, token);
                if (client.is_available()) {
                    auto result = client.register_user(username, is_admin);
                    if (result.contains("user_id"))
                        std::cout << "✓ Registered in database (user_id: "
                                  << result["user_id"] << ")\n";
                } else {
                    std::cout << "Server not running. Will auto-register on first authenticated request.\n";
                }
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << "\n";
                return 1;
            }

        } else if (command == "add-all") {
            ragger::setup_logging(false, false);
            if (getuid() != 0) {
                std::cerr << "Error: add-all requires sudo\n";
                return 1;
            }
            // Non-login shells — service accounts use these
            static const std::set<std::string> nologin_shells = {
                "/usr/bin/false", "/bin/false",
                "/sbin/nologin", "/usr/sbin/nologin",
            };
            static const std::set<std::string> skip_users = {
                "root", "nobody", "nfsnobody",
            };
            static const std::set<std::string> skip_homes = {
                "/", "/var", "/var/empty", "/dev/null", "/nonexistent",
            };
            int count = 0;
            setpwent();
            while (struct passwd* pw = getpwent()) {
                // Skip users with non-login shells
                if (pw->pw_shell && nologin_shells.count(pw->pw_shell)) continue;
                // Skip users without home directories
                if (!fs::is_directory(pw->pw_dir)) continue;
                // Skip root, nobody, etc.
                if (skip_users.count(pw->pw_name)) continue;
                // Skip system home directories
                if (skip_homes.count(pw->pw_dir)) continue;

                std::string uname(pw->pw_name);
                try {
                    auto [token, created] = provision_user(uname, pw->pw_dir);
                    std::string status = created ? "created" : "exists";
                    // Register via daemon
                    try {
                        ragger::RaggerClient client(host, port, token);
                        if (client.is_available()) {
                            client.register_user(uname);
                            status += ", registered";
                        } else {
                            status += ", pending registration";
                        }
                    } catch (...) {
                        status += ", pending registration";
                    }
                    std::cout << "  " << uname << ": " << status << "\n";
                    ++count;
                } catch (const std::exception& e) {
                    std::cout << "  " << uname << ": error (" << e.what() << ")\n";
                }
            }
            endpwent();
            std::cout << "✓ Processed " << count << " users\n";

        } else if (command == "model") {
            ragger::setup_logging(false, false);
            const auto& cfg = ragger::config();

            // Resolve model_dir
            std::string model_dir = cfg.llama_model_dir.empty()
#ifdef __APPLE__
                ? ragger::expand_path("~/Library/Caches/llama.cpp")
#else
                ? ragger::expand_path("~/.cache/llama.cpp")
#endif
                : ragger::expand_path(cfg.llama_model_dir);

            std::string subcmd = (argc > 2) ? argv[2] : "list";

            if (subcmd == "list") {
                std::cout << "Model directory: " << model_dir << "\n\n";

                // List .gguf files
                std::vector<std::pair<std::string, uintmax_t>> models;
                if (fs::is_directory(model_dir)) {
                    for (const auto& entry : fs::directory_iterator(model_dir)) {
                        if (entry.path().extension() == ".gguf") {
                            models.emplace_back(entry.path().filename().string(),
                                              entry.file_size());
                        }
                    }
                }

                if (models.empty()) {
                    std::cout << "No models found.\n";
                } else {
                    std::sort(models.begin(), models.end());
                    for (const auto& [name, size] : models) {
                        double gb = static_cast<double>(size) / (1024.0 * 1024.0 * 1024.0);
                        std::cout << "  " << name << "  ("
                                  << std::fixed << std::setprecision(1) << gb << " GB)\n";
                    }
                    std::cout << "\n" << models.size() << " model(s)\n";
                }

                // Show aliases
                if (!cfg.model_aliases.empty()) {
                    std::cout << "\nAliases:\n";
                    for (const auto& [alias, full] : cfg.model_aliases) {
                        std::cout << "  " << alias << " → " << full << "\n";
                    }
                }

            } else if (subcmd == "download") {
                // Check admin: root always allowed, otherwise check is_admin in DB
                if (getuid() != 0) {
                    std::string username = [] {
                        struct passwd* pw = getpwuid(getuid());
                        return pw ? std::string(pw->pw_name) : "unknown";
                    }();
                    try {
                        auto common_path = cfg.single_user
                            ? cfg.resolved_db_path()
                            : cfg.resolved_common_db_path();
                        ragger::Embedder embedder(cfg.resolved_model_dir());
                        ragger::SqliteBackend db(embedder, common_path);
                        auto user = db.get_user_by_username(username);
                        if (user && !user->is_admin) {
                            std::cerr << "Error: model download requires admin privileges\n";
                            return 1;
                        }
                        // !user = not registered yet, allow (first-run)
                    } catch (...) {
                        // No DB or no users table — allow (first-run scenario)
                    }
                }
                if (argc < 4) {
                    std::cout << "Usage: ragger model download <repo>[:quant]\n";
                    std::cout << "  e.g.: ragger model download Qwen/Qwen3-8B-GGUF:Q4_K_M\n";
                    return 1;
                }
                std::string repo = argv[3];

                // Use llama-server's --hf-repo to download, then immediately exit
                // llama-server downloads to ~/.cache/llama.cpp/ automatically
                std::string binary = cfg.llama_binary;
                if (binary == "llama-server") {
                    // Try common locations
                    if (fs::exists("/opt/local/bin/llama-server"))
                        binary = "/opt/local/bin/llama-server";
                    else if (fs::exists("/usr/local/bin/llama-server"))
                        binary = "/usr/local/bin/llama-server";
                }

                std::cout << "Downloading " << repo << " ...\n";
                std::cout << "(Press Ctrl+C to abort — download is resumable)\n\n";

                // Build command: llama-server --hf-repo <repo> -ngl 0 --port 0
                // Port 0 = don't actually serve. We just want the download.
                // Unfortunately llama-server doesn't have a download-only mode,
                // so we run it briefly and kill after model loads.
                std::vector<std::string> dl_args = {
                    binary, "--hf-repo", repo,
                    "-ngl", "0",
                    "--host", "127.0.0.1",
                    "--port", "19999"  // unlikely to conflict
                };

                std::vector<char*> dl_argv;
                for (auto& a : dl_args) dl_argv.push_back(a.data());
                dl_argv.push_back(nullptr);

                pid_t pid = fork();
                if (pid == 0) {
                    execvp(dl_argv[0], dl_argv.data());
                    _exit(127);
                }
                if (pid < 0) {
                    std::cerr << "Error: fork failed\n";
                    return 1;
                }

                // Wait for download to complete (model loads = download done)
                // Check health on port 19999
                auto start_time = std::chrono::steady_clock::now();
                bool downloaded = false;
                while (true) {
                    // Check if child died
                    int status;
                    pid_t result = waitpid(pid, &status, WNOHANG);
                    if (result == pid) {
                        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
                            downloaded = true;
                        }
                        break;
                    }

                    // Check if server is ready (= model downloaded and loaded)
                    CURL* curl = curl_easy_init();
                    if (curl) {
                        curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:19999/health");
                        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 1L);
                        curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
                        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
                        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,
                            +[](char*, size_t s, size_t n, void*) -> size_t { return s * n; });
                        CURLcode res = curl_easy_perform(curl);
                        long code = 0;
                        if (res == CURLE_OK)
                            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
                        curl_easy_cleanup(curl);
                        if (code == 200) {
                            downloaded = true;
                            kill(pid, SIGTERM);
                            waitpid(pid, &status, 0);
                            break;
                        }
                    }

                    std::this_thread::sleep_for(std::chrono::seconds(2));
                }

                if (downloaded) {
                    std::cout << "\n✓ Download complete. Model saved to " << model_dir << "\n";
                    std::cout << "  Add an alias in [models] section of ragger.ini to use it.\n";
                } else {
                    std::cerr << "\nDownload may have failed. Check output above.\n";
                    return 1;
                }

            } else if (subcmd == "remove") {
                if (getuid() != 0) {
                    std::string username = [] {
                        struct passwd* pw = getpwuid(getuid());
                        return pw ? std::string(pw->pw_name) : "unknown";
                    }();
                    try {
                        auto common_path = cfg.single_user
                            ? cfg.resolved_db_path()
                            : cfg.resolved_common_db_path();
                        ragger::Embedder embedder(cfg.resolved_model_dir());
                        ragger::SqliteBackend db(embedder, common_path);
                        auto user = db.get_user_by_username(username);
                        if (user && !user->is_admin) {
                            std::cerr << "Error: model remove requires admin privileges\n";
                            return 1;
                        }
                    } catch (...) {
                        // No DB or no users table — allow (first-run scenario)
                    }
                }
                if (argc < 4) {
                    std::cout << "Usage: ragger model remove <filename.gguf>\n";
                    return 1;
                }
                std::string filename = argv[3];

                // Resolve alias
                auto it = cfg.model_aliases.find(filename);
                if (it != cfg.model_aliases.end()) {
                    filename = it->second;
                }

                // Build full path
                fs::path model_path;
                if (filename.find('/') != std::string::npos) {
                    model_path = ragger::expand_path(filename);
                } else {
                    model_path = fs::path(model_dir) / filename;
                }

                if (!fs::exists(model_path)) {
                    std::cerr << "Not found: " << model_path << "\n";
                    return 1;
                }

                auto size = fs::file_size(model_path);
                double gb = static_cast<double>(size) / (1024.0 * 1024.0 * 1024.0);
                std::cout << "Remove " << model_path.filename().string()
                          << " (" << std::fixed << std::setprecision(1) << gb << " GB)? [y/N] ";
                std::string confirm;
                std::getline(std::cin, confirm);
                if (confirm == "y" || confirm == "Y") {
                    fs::remove(model_path);
                    // Also remove .etag sidecar if present
                    auto etag = model_path;
                    etag += ".etag";
                    if (fs::exists(etag)) fs::remove(etag);
                    std::cout << "✓ Removed\n";
                } else {
                    std::cout << "Cancelled\n";
                }

            } else {
                std::cerr << "Usage: ragger model <list|download|remove>\n";
                return 1;
            }

        } else if (command == "llama") {
            ragger::setup_logging(false, false);
            const auto& cfg = ragger::config();

            if (!cfg.llama_enabled) {
                std::cerr << "Error: [llama] not enabled in config\n";
                return 1;
            }

            std::string subcmd = (argc > 2) ? argv[2] : "status";
            ragger::LlamaManager llama;

            if (subcmd == "start") {
                if (!llama.start()) return 1;
                pid_t pid = llama.release();  // Don't kill on exit
                std::cout << "llama-server running (pid " << pid << "). Use 'ragger llama stop' to stop.\n";
            } else if (subcmd == "stop") {
                // Find llama-server by port using lsof
                std::string cmd = "lsof -ti tcp:" + std::to_string(cfg.llama_port) + " 2>/dev/null";
                FILE* pipe = popen(cmd.c_str(), "r");
                if (pipe) {
                    char buf[64];
                    std::string pids;
                    while (fgets(buf, sizeof(buf), pipe)) pids += buf;
                    pclose(pipe);
                    if (pids.empty()) {
                        std::cout << "No process found on port " << cfg.llama_port << "\n";
                    } else {
                        std::istringstream iss(pids);
                        std::string pid_str;
                        while (std::getline(iss, pid_str)) {
                            pid_str.erase(pid_str.find_last_not_of(" \t\r\n") + 1);
                            if (!pid_str.empty()) {
                                pid_t pid = std::stoi(pid_str);
                                kill(pid, SIGTERM);
                                std::cout << "Sent SIGTERM to pid " << pid << "\n";
                            }
                        }
                    }
                }
            } else if (subcmd == "status") {
                auto s = llama.status();
                std::cout << "llama-server:\n";
                std::cout << "  enabled: " << (cfg.llama_enabled ? "true" : "false") << "\n";
                std::cout << "  model:   " << cfg.llama_model << "\n";
                std::cout << "  host:    " << cfg.llama_host << "\n";
                std::cout << "  port:    " << cfg.llama_port << "\n";
                std::cout << "  running: " << (s.running ? "yes" : "no") << "\n";
                if (s.running) {
                    std::cout << "  pid:     " << s.pid << "\n";
                }
            } else {
                std::cerr << "Usage: ragger llama <start|stop|status>\n";
                return 1;
            }

        } else {
            std::cerr << CLI_UNKNOWN_COMMAND << command << "\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
