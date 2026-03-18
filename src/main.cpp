/**
 * raggerc — C++ port of Ragger Memory
 *
 * Verb-style CLI: raggerc <verb> [options] [args]
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
#include "ragger/lang.h"
#include "ragger/memory.h"
#include "ragger/server.h"
#include "nlohmann_json.hpp"

using namespace ragger::lang;
namespace fs = std::filesystem;

// -----------------------------------------------------------------------
// Import: paragraph-aware file chunking (mirrors Python import_file)
// -----------------------------------------------------------------------
static int heading_level(const std::string& line) {
    int level = 0;
    while (level < 6 && level < (int)line.size() && line[level] == '#') ++level;
    if (level > 0 && level < (int)line.size() && line[level] == ' ') return level;
    return 0;
}

static std::string heading_text(const std::string& line) {
    auto pos = line.find(' ');
    return (pos != std::string::npos) ? line.substr(pos + 1) : line;
}

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

    // Strip base64 image data
    text = std::regex_replace(text, std::regex(R"(!\[[^\]]*\]\(data:[^)]+\))"), "");
    text = std::regex_replace(text, std::regex(R"(data:image/[^;]+;base64,[A-Za-z0-9+/=]+)"), "");

    // Collapse multi-space artifacts per line
    std::istringstream lines_stream(text);
    std::string line;
    std::string cleaned;
    while (std::getline(lines_stream, line)) {
        // Collapse 2+ spaces to single space
        line = std::regex_replace(line, std::regex(R"(  +)"), " ");
        cleaned += line + "\n";
    }
    text = std::regex_replace(cleaned, std::regex(R"(\n{3,})"), "\n\n");

    // Split on paragraph boundaries
    std::vector<std::string> paragraphs;
    {
        std::istringstream ss(text);
        std::string para;
        std::string buffer;
        while (std::getline(ss, line)) {
            if (line.empty() || (line.size() == 1 && line[0] == '\r')) {
                if (!buffer.empty()) {
                    // Trim trailing whitespace
                    while (!buffer.empty() && (buffer.back() == '\n' || buffer.back() == '\r' || buffer.back() == ' '))
                        buffer.pop_back();
                    paragraphs.push_back(buffer);
                    buffer.clear();
                }
            } else {
                if (!buffer.empty()) buffer += "\n";
                buffer += line;
            }
        }
        if (!buffer.empty()) {
            while (!buffer.empty() && (buffer.back() == '\n' || buffer.back() == '\r' || buffer.back() == ' '))
                buffer.pop_back();
            paragraphs.push_back(buffer);
        }
    }

    // Heading-aware chunking
    struct HeadingEntry { int level; std::string text; };
    std::vector<HeadingEntry> heading_stack;
    struct Annotated { std::string body; std::string heading_block; std::string section; };
    std::vector<Annotated> annotated;

    auto current_section = [&]() -> std::string {
        std::string s;
        for (auto& h : heading_stack) {
            if (!s.empty()) s += " \xC2\xBB ";  // UTF-8 »
            s += h.text;
        }
        return s;
    };

    auto current_heading_block = [&]() -> std::string {
        std::string s;
        for (auto& h : heading_stack) {
            if (!s.empty()) s += "\n\n";
            s += std::string(h.level, '#') + " " + h.text;
        }
        return s;
    };

    for (auto& para : paragraphs) {
        int level = heading_level(para);
        if (level > 0) {
            while (!heading_stack.empty() && heading_stack.back().level >= level)
                heading_stack.pop_back();
            heading_stack.push_back({level, heading_text(para)});
        } else {
            annotated.push_back({para, current_heading_block(), current_section()});
        }
    }

    // Merge short paragraphs into chunks
    struct Chunk { std::string text; std::string section; };
    std::vector<Chunk> chunks;
    std::string current;
    std::string current_sec;

    for (auto& a : annotated) {
        if (current.empty()) {
            current = a.heading_block.empty() ? a.body : (a.heading_block + "\n\n" + a.body);
            current_sec = a.section;
        } else if ((int)current.size() >= min_chunk_size) {
            chunks.push_back({current, current_sec});
            current = a.heading_block.empty() ? a.body : (a.heading_block + "\n\n" + a.body);
            current_sec = a.section;
        } else {
            if (a.section != current_sec && !a.heading_block.empty()) {
                current += "\n\n" + a.heading_block + "\n\n" + a.body;
            } else {
                current += "\n\n" + a.body;
            }
            current_sec = a.section;
        }
    }
    if (!current.empty()) {
        chunks.push_back({current, current_sec});
    }

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
        ("version,v", CLI_VERSION)
        ("config-file", Diskerror::po::value<std::string>()->default_value(""), CLI_CONFIG_FILE)
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
        std::cout << opts.to_string() << "\n";
        return 0;
    }

    if (opts.count("version")) {
        std::cout << "raggerc " << "0.1.0" << "\n";
        return 0;
    }

    // Load config file
    try {
        ragger::init_config(opts["config-file"].as<std::string>());
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
                std::cerr << "Usage: raggerc import <file> [file...] [--collection name]\n";
                return 1;
            }
            ragger::RaggerMemory memory(db_path, model_dir);
            for (auto& filepath : args) {
                do_import(memory, filepath, collection, min_chunk_size);
            }

        } else if (command == "export") {
            auto args = opts.getParams("args");
            if (args.size() < 2) {
                std::cerr << "Usage: raggerc export <docs|memories|all> <dest-dir> [--collection name]\n";
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
