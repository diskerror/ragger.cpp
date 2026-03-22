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
#include <regex>
#include <sstream>

#include "ProgramOptions.h"
#include "ragger/config.h"
#include "ragger/import.h"
#include "ragger/inference.h"
#include "ragger/lang.h"
#include "ragger/memory.h"
#include "ragger/server.h"
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
// Export: reassemble chunks into files or grouped markdown
// -----------------------------------------------------------------------
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
        std::string content;
        for (auto* chunk : chunks) {
            if (!content.empty()) content += "\n\n";
            content += chunk->text;
        }
        content += "\n";

        auto out_path = fs::path(dest_dir) / fs::path(source).filename();
        std::ofstream f(out_path);
        f << content;
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
static std::string load_workspace_files() {
    /**
     * Load workspace MD files for the system prompt.
     * 
     * SOUL.md: /var/ragger/SOUL.md (common persona) > ~/.ragger/SOUL.md (fallback)
     * USER.md, AGENTS.md, TOOLS.md: ~/.ragger/ only (per-user)
     * 
     * Returns combined text for injection into system prompt.
     */
    std::string user_dir = ragger::expand_path("~/.ragger");
    std::string common_dir = "/var/ragger";

    std::vector<std::string> sections;

    // SOUL.md: common first, user fallback
    std::string soul_path = common_dir + "/SOUL.md";
    if (!fs::exists(soul_path)) {
        soul_path = user_dir + "/SOUL.md";
    }
    if (fs::exists(soul_path)) {
        std::ifstream f(soul_path);
        std::string content((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
        // Trim trailing whitespace
        content.erase(content.find_last_not_of(" \t\r\n") + 1);
        if (!content.empty()) {
            sections.push_back(content);
        }
    }

    // Per-user files
    for (const auto& filename : {"USER.md", "AGENTS.md", "TOOLS.md"}) {
        std::string path = user_dir + "/" + filename;
        if (fs::exists(path)) {
            std::ifstream f(path);
            std::string content((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
            content.erase(content.find_last_not_of(" \t\r\n") + 1);
            if (!content.empty()) {
                sections.push_back(content);
            }
        }
    }

    // Join sections with separator
    std::string result;
    for (size_t i = 0; i < sections.size(); ++i) {
        if (i > 0) result += "\n\n---\n\n";
        result += sections[i];
    }
    return result;
}

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

    // Load workspace files for persona
    std::string workspace = load_workspace_files();

    // Conversation history — start with workspace as system prompt
    std::vector<ragger::Message> messages;
    if (!workspace.empty()) {
        messages.push_back({"system", workspace});
    }

    std::cout << "Ragger Chat (model: " << cfg.inference_model << ")\n";
    std::cout << "Type '/quit' or Ctrl+D to exit\n\n";

    std::string line;
    while (true) {
        std::cout << "You: " << std::flush;
        if (!std::getline(std::cin, line)) {
            std::cout << "\nGoodbye!\n";
            break;
        }

        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (line.empty()) continue;

        if (line == "/quit" || line == "/exit") {
            std::cout << "Goodbye!\n";
            break;
        }

        // Search memory for context
        std::vector<std::string> context_chunks;
        try {
            auto result = memory.search(line, 3, 0.3f);
            for (const auto& r : result.results) {
                context_chunks.push_back(r.text);
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: memory search failed: " << e.what() << "\n";
        }

        // Build message with context
        std::vector<ragger::Message> current_messages = messages;

        if (!context_chunks.empty()) {
            std::string context_text;
            for (size_t i = 0; i < context_chunks.size(); ++i) {
                if (i > 0) context_text += "\n\n---\n\n";
                context_text += context_chunks[i];
            }
            std::string memory_block = "\n\n## Relevant memories:\n\n" + context_text;

            // Append to existing system message or create one
            if (!current_messages.empty() && current_messages[0].role == "system") {
                current_messages[0].content += memory_block;
            } else {
                current_messages.insert(current_messages.begin(),
                                       {"system", memory_block});
            }
        }

        current_messages.push_back({"user", line});

        // Send to inference API (streaming)
        std::cout << "Assistant: " << std::flush;
        std::string response_text;

        try {
            inference.chat_stream(current_messages, [&](const std::string& token) {
                std::cout << token << std::flush;
                response_text += token;
            });
            std::cout << "\n";
        } catch (const std::exception& e) {
            std::cout << "\nError: " << e.what() << "\n";
            continue;
        }

        // Update conversation history
        messages.push_back({"user", line});
        messages.push_back({"assistant", response_text});

        std::cout << "\n";  // blank line between exchanges
    }
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
        std::cout << opts.to_string() << "\n";
        return 0;
    }

    if (opts.count("version") || command == "version") {
        std::cout << "ragger " << RAGGER_VERSION << "\n";
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
            ragger::RaggerMemory memory(db_path, model_dir);
            char buf[128];
            std::snprintf(buf, sizeof(buf), MSG_LOADED_MEMORIES, memory.count());
            std::cout << buf << "\n";

            ragger::Server server(memory, host, port);
            server.run();

        } else if (command == "chat") {
            do_chat(db_path, model_dir);

        } else if (command == "search") {
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

            ragger::RaggerMemory memory(db_path, model_dir);
            std::vector<std::string> colls;
            if (!collection.empty()) colls.push_back(collection);
            auto response = memory.search(query, cfg.default_search_limit,
                                           cfg.default_min_score, colls);

            nlohmann::json output = nlohmann::json::array();
            for (const auto& r : response.results) {
                output.push_back({
                    {"id", r.id}, {"score", r.score},
                    {"text", r.text}, {"metadata", r.metadata}
                });
            }
            std::cout << output.dump(2) << "\n";

        } else if (command == "store") {
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

            ragger::RaggerMemory memory(db_path, model_dir);
            auto id = memory.store(text, meta);
            std::cout << MSG_STORED_WITH_ID << id << "\n";

        } else if (command == "count") {
            ragger::RaggerMemory memory(db_path, model_dir);
            std::cout << memory.count() << "\n";

        } else if (command == "import") {
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
            ragger::RaggerMemory memory(db_path, model_dir);
            do_mcp(memory);

        } else if (command == "rebuild-bm25") {
            ragger::RaggerMemory memory(db_path, model_dir);
            int count = memory.rebuild_bm25();
            std::cout << "✓ BM25 index rebuilt: " << count << " documents\n";

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
