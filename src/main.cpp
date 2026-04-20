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
#include <grp.h>
#include <pwd.h>
#include <regex>
#include <set>
#include <sstream>
#include <sys/stat.h>
#include <termios.h>
#include <unistd.h>
#include <signal.h>
#include <sys/file.h>
#include <thread>
#include <iomanip>
#include <curl/curl.h>
#include <print>

#include "ProgramOptions.h"
#include "ragger/auth.h"
#include "ragger/chat.h"
#include "ragger/client.h"
#include "ragger/config.h"
#include "ragger/import.h"
#include "ragger/inference.h"
#include "ragger/lang.h"
#include "ragger/logs.h"
#include "ragger/mcp.h"
#include "ragger/memory.h"
#include "ragger/sqlite_backend.h"
#include "ragger/server.h"
#include "ragger/embedder.h"
#include "ragger/storage_types.h"
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
    std::println("Importing {} chunks from {}...", chunks.size(), filename);

    for (size_t i = 0; i < chunks.size(); ++i) {
        nlohmann::json meta = {
            {"source", filepath},
            {"chunk", (int)(i + 1)},
            {"total_chunks", (int)chunks.size()}
        };
        if (!collection.empty()) meta["collection"] = collection;
        if (!chunks[i].section.empty()) meta["section"] = chunks[i].section;

        auto id = memory.store(chunks[i].text, meta);
        std::println("  Chunk {}/{}: {}", (i + 1), chunks.size(), id);
    }
    std::println("✓ Imported {} chunks", chunks.size());
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
        std::println("No documents found in collection '{}'", collection);
        return;
    }

    // Group by source filename
    std::map<std::string, std::vector<ragger::SearchResult*>> files;
    for (auto& r : all) {
        std::string source = r.metadata.value("source", "unknown.md");
        files[source].push_back(&r);
    }

    std::println("Exporting {} documents from '{}'...", files.size(), collection);

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
        std::println("  {} ({} chunks)", fs::path(source).filename().string(), chunks.size());
    }
    std::println("✓ Exported {} documents to {}", files.size(), dest_dir);
}

static void do_export_memories(ragger::RaggerMemory& memory,
                               const std::string& dest_dir,
                               const std::string& group_by) {
    fs::create_directories(dest_dir);

    auto all = memory.load_all("memory");
    if (all.empty()) {
        std::println("No memories to export");
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

    std::println("Exporting {} memories ({} groups by {})...", all.size(), groups.size(), group_by);

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

        std::println("  {}.md ({} entries)", key, entries.size());
    }
    std::println("✓ Exported {} memories to {}", all.size(), dest_dir);
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
static void do_chat(const std::string& db_path, const std::string& model_dir,
                    const std::string& dump_payloads_dir) {
    const auto& cfg = ragger::config();

    // Validate payload dump dir before doing anything else
    if (!dump_payloads_dir.empty()) {
        std::error_code ec;
        bool existed = std::filesystem::exists(dump_payloads_dir);
        std::filesystem::create_directories(dump_payloads_dir, ec);
        if (ec) {
            std::cerr << "Error: cannot create payload dump directory '"
                      << dump_payloads_dir << "': " << ec.message() << "\n";
            return;
        }
        if (!existed) {
            std::cout << "Created payload dump directory: " << dump_payloads_dir << "\n";
        }
    }

    // Build inference client from config
    ragger::InferenceClient inference = ragger::InferenceClient::from_config(cfg);

    if (!dump_payloads_dir.empty()) {
        inference.set_payload_dump_dir(dump_payloads_dir);
    }

    if (inference._endpoints.empty()) {
        std::println("Error: no inference endpoints configured.");
        std::println("Add to settings.ini:");
        std::println("");
        std::println("  [inference]");
        std::println("  api_url = http://localhost:1234/v1");
        std::println("  api_key = lmstudio-local");
        std::println("");
        std::println("Or for multiple endpoints:");
        std::println("");
        std::println("  [inference.local]");
        std::println("  api_url = http://localhost:1234/v1");
        std::println("  api_key = lmstudio-local");
        std::println("  models = qwen/*, llama/*");
        return;
    }

    // Quick connectivity check — GET /models on the first endpoint
    {
        auto& ep = inference._endpoints[0];
        std::string models_url = ep.api_url;
        // Strip trailing /chat/completions or similar, append /models
        auto pos = models_url.rfind("/v1");
        if (pos != std::string::npos)
            models_url = models_url.substr(0, pos + 3) + "/models";
        else
            models_url += "/models";

        CURL* curl = curl_easy_init();
        if (curl) {
            std::string response;
            curl_easy_setopt(curl, CURLOPT_URL, models_url.c_str());
            curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,
                +[](char* ptr, size_t size, size_t nmemb, void* userdata) -> size_t {
                    static_cast<std::string*>(userdata)->append(ptr, size * nmemb);
                    return size * nmemb;
                });
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
            CURLcode res = curl_easy_perform(curl);
            long http_code = 0;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
            curl_easy_cleanup(curl);

            if (res != CURLE_OK) {
                std::cerr << "Error: cannot reach inference endpoint '" << ep.name
                          << "' at " << ep.api_url << "\n"
                          << "  " << curl_easy_strerror(res) << "\n";
                return;
            }
            if (http_code >= 400) {
                std::cerr << "Error: inference endpoint '" << ep.name
                          << "' returned HTTP " << http_code << "\n";
                return;
            }
        }
    }

    // Load memory for context search
    ragger::RaggerMemory memory(db_path, model_dir);

    // Create and run chat session
    ragger::Chat chat(memory, inference, cfg.inference_model);
    chat.run();
}

// -----------------------------------------------------------------------
// Daemon control: start / stop / restart / status
// Wrappers around launchctl (macOS) or systemctl --user (Linux) that
// operate on the user-level service installed by install.sh.
// The daemon process itself is still `ragger serve` (what launchd/systemd run).
// -----------------------------------------------------------------------
static int daemon_control(const std::string& action) {
    struct passwd* pw = getpwuid(getuid());
    std::string home = pw ? pw->pw_dir : (std::getenv("HOME") ? std::getenv("HOME") : "");
    if (home.empty()) {
        std::cerr << "Error: cannot resolve $HOME\n";
        return 1;
    }

#if defined(__APPLE__)
    const std::string label = "com.diskerror.ragger";
    const std::string plist = home + "/Library/LaunchAgents/" + label + ".plist";
    const std::string target = "gui/" + std::to_string(getuid()) + "/" + label;
    const std::string domain = "gui/" + std::to_string(getuid());

    std::string cmd;
    if (action == "start") {
        if (!fs::exists(plist)) {
            std::cerr << "Error: " << plist << " not found. Run ./install.sh first.\n";
            return 1;
        }
        cmd = "launchctl bootstrap " + domain + " " + plist;
    } else if (action == "stop") {
        cmd = "launchctl bootout " + target;
    } else if (action == "restart") {
        std::system(("launchctl bootout " + target + " 2>/dev/null").c_str());
        cmd = "launchctl bootstrap " + domain + " " + plist;
    } else if (action == "status") {
        cmd = "launchctl print " + target;
    } else {
        std::cerr << "Error: unknown daemon action '" << action << "'\n";
        return 1;
    }
#elif defined(__linux__)
    std::string cmd;
    if (action == "start")   cmd = "systemctl --user start ragger.service";
    else if (action == "stop")    cmd = "systemctl --user stop ragger.service";
    else if (action == "restart") cmd = "systemctl --user restart ragger.service";
    else if (action == "status")  cmd = "systemctl --user status ragger.service";
    else {
        std::cerr << "Error: unknown daemon action '" << action << "'\n";
        return 1;
    }
#else
    std::cerr << "Error: daemon control not supported on this platform.\n"
              << "       Run 'ragger serve' manually.\n";
    return 1;
#endif

    int rc = std::system(cmd.c_str());
    return WIFEXITED(rc) ? WEXITSTATUS(rc) : 1;
}

/**
 * Database migration helper - moves memories between user and common databases.
 * Uses StorageBackend::export_memories / import_memories + delete_batch.
 */
static int migrate_memories(const std::string& src_path, const std::string& dst_path,
                            const std::string& direction, const std::string& username,
                            const ragger::MemoryFilter& filter, bool dry_run) {
    ragger::SqliteBackend src(src_path);
    ragger::SqliteBackend dst(dst_path);

    // Export matching records from source
    auto records = src.export_memories(filter);
    if (records.empty()) {
        std::println("No matching records in source DB");
        return 0;
    }

    if (dry_run) {
        std::println("Would move {} records:", records.size());
        int shown = 0;
        for (const auto& r : records) {
            if (shown++ >= 10) {
                std::println("  ... and {} more", (records.size() - 10));
                break;
            }
            std::string preview = r.text.substr(0, 70);
            std::println("  id={} {}", r.id, preview);
        }
        return 0;
    }

    // Resolve user_id for provenance when moving to common
    int user_id = -1;
    if (direction == "to-common") {
        std::println("Note: multi-user mode has been removed. Records will be moved without user_id.");
    }

    // Import into destination
    int imported = dst.import_memories(records, user_id);

    // Delete from source
    std::vector<int> ids;
    ids.reserve(records.size());
    for (const auto& r : records) ids.push_back(static_cast<int>(r.id));
    src.delete_batch(ids);

    std::println("Moved {} records", imported);
    return 0;
}

// -----------------------------------------------------------------------
// Password input (with echo suppression)
// -----------------------------------------------------------------------

static std::string read_password(const std::string& prompt) {
    std::cout << prompt;
    std::cout.flush();
    
    struct termios old_term, new_term;
    tcgetattr(STDIN_FILENO, &old_term);
    new_term = old_term;
    new_term.c_lflag &= ~ECHO;
    tcsetattr(STDIN_FILENO, TCSANOW, &new_term);
    
    std::string password;
    std::getline(std::cin, password);
    
    tcsetattr(STDIN_FILENO, TCSANOW, &old_term);
    std::cout << "\n";
    
    // Trim trailing whitespace/CR (paranoia)
    while (!password.empty() && (password.back() == '\r' || password.back() == '\n' || password.back() == ' '))
        password.pop_back();
    
    return password;
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
    chmod(tok_path.c_str(), 0660);

    // Set ownership if running as root: user owns, ragger group for daemon access
    if (getuid() == 0) {
        struct passwd* pw = getpwnam(username.c_str());
        struct group* rg = getgrnam("ragger");
        if (pw) {
            gid_t gid = rg ? rg->gr_gid : pw->pw_gid;
            chown(ragger_dir.c_str(), pw->pw_uid, gid);
            chmod(ragger_dir.c_str(), 0770);
            chown(tok_path.c_str(), pw->pw_uid, gid);
            // Also fix memories.db if it exists
            std::string db_path = ragger_dir + "/memories.db";
            if (fs::exists(db_path)) {
                chown(db_path.c_str(), pw->pw_uid, gid);
                chmod(db_path.c_str(), 0660);
            }
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
        ("lm-proxy-url", Diskerror::po::value<std::string>(), CLI_LM_PROXY_URL)
        ("collection", Diskerror::po::value<std::string>()->default_value(""), "Collection name")
        ("min-chunk-size", Diskerror::po::value<int>(), "Min chunk size for import")
        ("group-by", Diskerror::po::value<std::string>()->default_value("date"), "Grouping for export (date|category|collection)")
        // admin flags removed — sudo is the admin gate
        ("yes,y", "Skip confirmation prompts (for scripting)")
        ("dump-payloads", Diskerror::po::value<std::string>(), "Write raw request JSON to this directory (one file per prompt)")
    ;
    opts.add_hidden_options()
        ("command", Diskerror::po::value<std::string>()->default_value("help"), CLI_COMMAND)
        ("args", Diskerror::po::value<std::vector<std::string>>(), CLI_ARGS)
        ("ids", Diskerror::po::value<std::string>(), "Comma-separated IDs (move)")
        ("source", Diskerror::po::value<std::string>(), "Source pattern (move)")
        ("category", Diskerror::po::value<std::string>(), "Category filter (move)")
        ("user", Diskerror::po::value<std::string>(), "Target user (move)")
        ("dry-run", "Show what would happen without doing it")
        // --keep-data removed: always keep user data, sudoer can rm manually
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
        std::cout << "  start              Start the background daemon (user LaunchAgent / systemd --user)\n";
        std::cout << "  stop               Stop the background daemon\n";
        std::cout << "  restart            Restart the background daemon\n";
        std::cout << "  status             Show daemon status\n";
        std::cout << "  serve              Run the server in the foreground (what the daemon invokes)\n";
        std::cout << "  search <query>     Search memories by meaning\n";
        std::cout << "  store <text>       Store a new memory\n";
        std::cout << "  count              Show number of stored memories\n";
        std::cout << "  import <file...>   Import files (paragraph-aware chunking)\n";
        std::cout << "  export <mode> <dir>  Export docs|memories|all to directory\n";
        std::cout << "  chat               Interactive chat with memory context\n";
        std::cout << "  mcp                Start MCP server (JSON-RPC over stdin/stdout)\n";
        std::cout << "  add-self           Provision yourself (create token)\n";
        std::cout << "  add-user <name>    Provision a user (requires sudo)\n";
        std::cout << "  add-all            Provision all users (requires sudo, confirms each)\n";
        std::cout << "                     -y/--yes to skip prompts\n";
        std::cout << "  remove-user <name> Remove a user (requires sudo)\n";
        std::cout << "  passwd [<name>]    Change password (own or another user's with sudo)\n";
        std::cout << "  housekeeping       Trigger housekeeping on running daemon\n";
        std::cout << "  reload             Reload config on running daemon (SIGHUP)\n";
        std::cout << "  move <direction>   Move memories between user and common DBs\n";
        std::cout << "                     direction: to-common | to-user\n";
        std::cout << "                     filters: --ids, --source, --collection, --category\n";
        std::cout << "                     options: --user <name>, --dry-run\n";
        // cleanup verb removed — use SQLite CLI or agent-mediated deletion
        std::cout << "  useradd <username>  Add or update a user account (prompts for password)\n";
        std::cout << "  userdel <username>  Remove a user account\n";
        std::cout << "  rebuild-bm25       Rebuild the BM25 keyword index\n";
        std::cout << "  rebuild-embeddings Rebuild embeddings for all memories\n";
        std::cout << "  show-embedding-model  Show current embedding model info\n";
        // (llama and model verbs removed — use external providers)
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
        bool server_cmd = (command == "serve");
        ragger::init_config(opts["config"].as<std::string>(), /*quiet=*/!server_cmd);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    const auto& cfg = ragger::config();

    // CLI overrides
    if (opts.count("lm-proxy-url"))
        ragger::mutable_config().lm_proxy_url = opts["lm-proxy-url"].as<std::string>();

    std::string host       = opts.count("host")      ? opts["host"].as<std::string>()      : cfg.bind_address;
    int         port       = opts.count("port")      ? opts["port"].as<int>()               : cfg.port;
    std::string db_path    = opts.count("db")         ? opts["db"].as<std::string>()         : "";
    std::string model_dir  = opts.count("model-dir")  ? opts["model-dir"].as<std::string>()  : "";
    std::string collection = opts["collection"].as<std::string>();
    int min_chunk_size     = opts.count("min-chunk-size")
                             ? opts["min-chunk-size"].as<int>()
                             : cfg.minimum_chunk_size;
    std::string group_by   = opts["group-by"].as<std::string>();

    try {
        if (command == "start" || command == "stop" ||
            command == "restart" || command == "status") {
            return daemon_control(command);

        } else if (command == "serve") {
            ragger::setup_logging(false, true);
            const auto& cfg = ragger::config();
            std::unique_ptr<ragger::RaggerMemory> mem_ptr;
            // Single-user mode only
            mem_ptr = std::make_unique<ragger::RaggerMemory>(db_path, model_dir);
            auto& memory = *mem_ptr;
            char buf[128];
            std::snprintf(buf, sizeof(buf), MSG_LOADED_MEMORIES, memory.count());
            ragger::log_info(buf);

            ragger::Server server(memory, host, port);
            server.run();

        } else if (command == "chat") {
            ragger::setup_logging(false, false);
            std::string dump_dir = opts.count("dump-payloads")
                ? opts["dump-payloads"].as<std::string>() : "";
            do_chat(db_path, model_dir, dump_dir);

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
            ragger::RaggerClient client(cfg.bind_address, cfg.port, token);
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
            ragger::RaggerClient client(cfg.bind_address, cfg.port, token);
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
            ragger::RaggerClient client(cfg.bind_address, cfg.port, token);
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
            auto mem_ptr = std::make_unique<ragger::RaggerMemory>(db_path, model_dir);
            ragger::run_mcp(*mem_ptr);

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
            std::println("This will re-embed all {} memories. The server should be stopped first.", total_count);
            
            bool proceed = false;
            if (opts.count("yes")) {
                proceed = true;
            } else {
                std::print("Continue? [y/N] ");
                std::string answer;
                std::getline(std::cin, answer);
                proceed = (answer == "y" || answer == "Y");
            }
            
            if (!proceed) {
                std::println("Aborted.");
                return 0;
            }
            
            // Backup the database file
            std::string actual_db_path = db_path.empty() ? cfg.resolved_db_path() : db_path;
            std::string backup_path = actual_db_path + ".bak";
            try {
                fs::copy_file(actual_db_path, backup_path, 
                             fs::copy_options::overwrite_existing);
                std::println("Database backed up to: {}", backup_path);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to create backup: " << e.what() << "\n";
            }
            
            // Rebuild embeddings
            ragger::RaggerMemory memory(db_path, model_dir);
            int count = memory.rebuild_embeddings();
            std::cout << "✓ Embeddings rebuilt: " << count << " documents\n";

        } else if (command == "show-embedding-model") {
            ragger::setup_logging(false, false);
            std::println("Model: {}", cfg.embedding_model);
            std::println("Dimensions: {}", cfg.embedding_dimensions);
            std::string model_path = model_dir.empty() ? cfg.model_dir : model_dir;
            if (!model_path.empty() && fs::is_directory(model_path))
                std::println("Path: {}", model_path);
            else
                std::println("Path: (default)");

        } else if (command == "useradd") {
            ragger::setup_logging(false, false);
            auto args = opts.getParams("args");
            if (args.empty()) {
                std::cerr << "Usage: ragger useradd <username>\n";
                return 1;
            }
            std::string username = args[0];
            
            // Read password with echo disabled
            std::string password = read_password("Password: ");
            if (password.empty()) {
                std::cerr << "Error: password cannot be empty\n";
                return 1;
            }
            
            try {
                ragger::SqliteBackend storage(cfg.resolved_db_path());
                ragger::useradd(storage, username, password);
                std::println("✓ User {} added/updated", username);
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << "\n";
                return 1;
            }

        } else if (command == "userdel") {
            ragger::setup_logging(false, false);
            auto args = opts.getParams("args");
            if (args.empty()) {
                std::cerr << "Usage: ragger userdel <username>\n";
                return 1;
            }
            std::string username = args[0];

            try {
                ragger::SqliteBackend storage(cfg.resolved_db_path());
                ragger::userdel(storage, username);
                std::println("✓ User {} removed", username);
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << "\n";
                return 1;
            }

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
                std::println("✓ Created ~/.ragger/token for {}", username);
            else
                std::println("Token already exists for {}", username);
            std::println("\nYour token: {}", token);
            std::println("Use this in your client config (OpenClaw, Claude Desktop, etc).");
            std::println("Token file: ~/.ragger/token");
            // Register directly in DB
            // Note: Multi-user mode removed. These user management commands are deprecated.
            try {
                std::string reg_db = cfg.resolved_db_path();
                ragger::SqliteBackend backend(reg_db);
                std::string token_hash = ragger::hash_token(token);
                auto existing = backend.get_user_by_username(username);
                if (existing) {
                    if (existing->token_hash != token_hash)
                        backend.update_user_token(username, token_hash);
                    std::println("✓ User exists in database (id: {})", existing->id);
                } else {
                    int user_id = backend.create_user(username, token_hash);
                    std::println("✓ Registered in database (user_id: {})", user_id);
                }
            } catch (const std::exception& e) {
                std::println("Warning: DB registration deferred ({})", e.what());
            }

        // DEPRECATED: Multi-user mode removed. User management commands are no longer needed.
        // Keeping code for reference but commented out.
#if 0
        } else if (command == "add-user") {
            ragger::setup_logging(false, false);
            auto args = opts.getParams("args");
            if (args.empty()) {
                std::cerr << "Usage: ragger add-user <username>\n";
                return 1;
            }
            std::string username = args[0];
            try {
                auto [token, created] = provision_user(username);
                if (created)
                    std::println("✓ Created token for {}", username);
                else
                    std::println("Token already exists for {}", username);
                // Add user to ragger group (requires root)
                if (getuid() == 0) {
#ifdef __APPLE__
                    std::string cmd = "dscl . -append /Groups/ragger GroupMembership " + username;
#else
                    std::string cmd = "usermod -aG ragger " + username;
#endif
                    if (std::system(cmd.c_str()) == 0)
                        std::println("✓ Added {} to ragger group", username);
                    else
                        std::cerr << "Warning: could not add " << username << " to ragger group\n";
                }
                // Register in DB
                std::string reg_db = cfg.resolved_db_path();
                ragger::SqliteBackend umgr(reg_db);
                std::string token_hash = ragger::hash_token(token);
                auto existing = umgr.get_user_by_username(username);
                if (existing) {
                    if (existing->token_hash != token_hash)
                        umgr.update_user_token(username, token_hash);
                    std::println("✓ User exists in database (id: {})", existing->id);
                } else {
                    int user_id = umgr.create_user(username, token_hash);
                    std::println("✓ Registered in database (user_id: {})", user_id);
                }
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << "\n";
                return 1;
            }
            std::println("\nToken file: ~{}/.ragger/token", username);
            std::println("The user will need to update their client config if the token is rotated.");
#endif

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
            // Open DB directly for registration
            std::string reg_db = cfg.resolved_db_path();
            ragger::SqliteBackend umgr(reg_db);

            int count = 0;
            bool auto_yes = opts.count("yes") > 0;
            setpwent();
            while (struct passwd* pw = getpwent()) {
                // Skip users with non-login shells
                if (pw->pw_shell && nologin_shells.count(pw->pw_shell)) continue;
                // Skip users without home directories
                if (!fs::is_directory(pw->pw_dir)) continue;
                // Skip root, nobody, etc.
                if (skip_users.count(pw->pw_name)) continue;
                // Skip service accounts (names starting with _)
                if (pw->pw_name[0] == '_') continue;
                // Skip system home directories
                if (skip_homes.count(pw->pw_dir)) continue;

                std::string uname(pw->pw_name);
                if (!auto_yes) {
                    std::print("  Add {}? [Y/n] ", uname);
                    std::string answer;
                    std::getline(std::cin, answer);
                    // Trim
                    while (!answer.empty() && std::isspace(answer.back())) answer.pop_back();
                    while (!answer.empty() && std::isspace(answer.front())) answer.erase(answer.begin());
                    if (!answer.empty() && std::tolower(answer[0]) != 'y') {
                        std::println("  {}: skipped", uname);
                        continue;
                    }
                }
                try {
                    auto [token, created] = provision_user(uname, pw->pw_dir);
                    std::string status = created ? "created" : "exists";
                    // Add to ragger group
                    {
#ifdef __APPLE__
                        std::string cmd = "dscl . -append /Groups/ragger GroupMembership " + uname;
#else
                        std::string cmd = "usermod -aG ragger " + uname;
#endif
                        if (std::system(cmd.c_str()) == 0)
                            status += ", group added";
                        else
                            status += ", group skipped";
                    }
                    // Register directly in DB
                    try {
                        std::string token_hash = ragger::hash_token(token);
                        auto existing = umgr.get_user_by_username(uname);
                        if (existing) {
                            if (existing->token_hash != token_hash)
                                umgr.update_user_token(uname, token_hash);
                            status += ", registered";
                        } else {
                            umgr.create_user(uname, token_hash);
                            status += ", registered";
                        }
                    } catch (const std::exception& e) {
                        status += ", db error: " + std::string(e.what());
                    }
                    std::println("  {}: {}", uname, status);
                    ++count;
                } catch (const std::exception& e) {
                    std::println("  {}: error ({})", uname, e.what());
                }
            }
            endpwent();
            std::println("✓ Processed {} users", count);

        } else if (command == "remove-user") {
            ragger::setup_logging(false, false);
            auto args = opts.getParams("args");
            if (args.empty()) {
                std::cerr << "Usage: ragger remove-user <username>\n";
                return 1;
            }
            if (getuid() != 0) {
                std::cerr << "Error: remove-user requires sudo\n";
                return 1;
            }
            std::string username = args[0];

            // 1. Remove from ragger OS group
            {
#ifdef __APPLE__
                std::string cmd = "dscl . -delete /Groups/ragger GroupMembership " + username;
#else
                std::string cmd = "gpasswd -d " + username + " ragger";
#endif
                int rc = std::system(cmd.c_str());
                if (rc == 0)
                    std::println("✓ Removed {} from ragger group", username);
                else
                    std::println("  {} was not in ragger group (skipped)", username);
            }

            // 2. Remove from common database
            try {
                std::string reg_db = cfg.resolved_db_path();
                ragger::SqliteBackend umgr(reg_db);
                auto existing = umgr.get_user_by_username(username);
                if (existing) {
                    umgr.delete_user(username);
                    std::println("✓ Removed {} from database", username);
                } else {
                    std::println("  {} not found in database (skipped)", username);
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: database removal: " << e.what() << "\n";
            }

            // 3. Remove token (keep ~/.ragger/ data — sudoer can remove manually)
            struct passwd* pw = getpwnam(username.c_str());
            if (pw) {
                std::string token_path = std::string(pw->pw_dir) + "/.ragger/token";
                if (fs::exists(token_path)) {
                    fs::remove(token_path);
                    std::println("✓ Removed token ({})", token_path);
                } else {
                    std::println("  No token file found (skipped)");
                }
            } else {
                std::println("  User {} not found in system passwd (skipped token)", username);
            }

            std::println("\n✓ User {} removed", username);

        } else if (command == "passwd") {
            ragger::setup_logging(false, false);
            auto args = opts.getParams("args");
            
            // Determine target user
            std::string target_user;
            if (!args.empty()) {
                // Changing another user's password — requires sudo
                target_user = args[0];
                if (getuid() != 0) {
                    std::cerr << "Error: changing another user's password requires sudo\n";
                    return 1;
                }
            } else {
                // Changing own password
                struct passwd* self_pw = getpwuid(getuid());
                if (!self_pw) {
                    std::cerr << "Error: cannot determine username\n";
                    return 1;
                }
                target_user = self_pw->pw_name;
            }
            
            // Open common database directly (no embedder needed for user management)
            // User management always uses the common DB (/var/ragger/memories.db)
            ragger::SqliteBackend umgr(cfg.resolved_db_path());

            // Verify user exists in DB
            auto user_info = umgr.get_user_by_username(target_user);
            if (!user_info) {
                std::cerr << "Error: user '" << target_user << "' not found in database\n";
                std::cerr << "Provision the user first: sudo ragger add-user " << target_user << "\n";
                return 1;
            }

            // If changing own password and not root, verify current password
            if (args.empty() && user_info->id > 0) {
                auto existing = umgr.get_user_password(target_user);
                if (existing) {
                    std::string current = read_password("Current password: ");
                    if (!ragger::verify_password(current, *existing)) {
                        std::cerr << "Error: incorrect password\n";
                        return 1;
                    }
                }
            }
            
            // Read new password
            std::string new_pass = read_password("New password: ");
            if (new_pass.empty()) {
                std::println("Password cleared (web UI access disabled).");
                umgr.set_user_password(target_user, "");
            } else {
                std::string confirm = read_password("Confirm password: ");
                if (new_pass != confirm) {
                    std::cerr << "Error: passwords do not match\n";
                    return 1;
                }
                std::string hash = ragger::hash_password(new_pass);
                umgr.set_user_password(target_user, hash);
                std::println("✓ Password updated for {}", target_user);
            }

        } else if (command == "housekeeping") {
            ragger::setup_logging(false, false);
            // Send SIGUSR1 to the running daemon via server PID file
            namespace fs = std::filesystem;
            pid_t daemon_pid = 0;
            try {
                for (const auto& entry : fs::directory_iterator("/tmp/ragger")) {
                    auto name = entry.path().filename().string();
                    if (name.rfind("server-", 0) == 0 &&
                        name.size() > 4 && name.substr(name.size() - 4) == ".pid") {
                        std::ifstream pf(entry.path());
                        if (pf) pf >> daemon_pid;
                        break;
                    }
                }
            } catch (...) {}
            if (daemon_pid <= 0) {
                std::cerr << "Error: no running ragger daemon found\n";
                return 1;
            }
            if (kill(daemon_pid, 0) != 0) {
                std::cerr << "Error: daemon (pid " << daemon_pid << ") is not running\n";
                return 1;
            }
            if (kill(daemon_pid, SIGUSR1) != 0) {
                if (errno == EPERM) {
                    std::cerr << "Error: permission denied. Use sudo to signal the daemon.\n";
                } else {
                    std::cerr << "Error: failed to signal process: " << strerror(errno) << "\n";
                }
                return 1;
            }
            std::println("✓ Housekeeping triggered (pid {})", daemon_pid);

        } else if (command == "reload") {
            // Send SIGHUP to running daemon to reload config
            namespace fs = std::filesystem;
            pid_t daemon_pid = 0;
            try {
                for (const auto& entry : fs::directory_iterator("/tmp/ragger")) {
                    auto name = entry.path().filename().string();
                    if (name.rfind("server-", 0) == 0 &&
                        name.size() > 4 && name.substr(name.size() - 4) == ".pid") {
                        std::ifstream pf(entry.path());
                        if (pf >> daemon_pid && daemon_pid > 0 && kill(daemon_pid, 0) == 0) {
                            break;
                        }
                        daemon_pid = 0;
                    }
                }
            } catch (...) {}

            if (daemon_pid <= 0) {
                std::cerr << "Error: no running ragger daemon found\n";
                return 1;
            }
            if (kill(daemon_pid, SIGHUP) != 0) {
                std::cerr << "Error: failed to signal process: " << strerror(errno) << "\n";
                return 1;
            }
            std::println("✓ Config reload triggered (pid {})", daemon_pid);

        } else if (command == "move") {
            auto args = opts.getParams("args");
            if (args.empty() || (args[0] != "to-common" && args[0] != "to-user")) {
                std::cerr << "Usage: ragger move <to-common|to-user> [options]\n"
                          << "  --ids ID1,ID2,...    Filter by IDs\n"
                          << "  --source PATTERN     Filter by source (SQL LIKE)\n"
                          << "  --collection NAME    Filter by collection\n"
                          << "  --category NAME      Filter by category\n"
                          << "  --user USERNAME      Target user (default: current)\n"
                          << "  --dry-run            Show what would be moved\n";
                return 1;
            }
            std::string direction = args[0];
            std::string filter_ids = opts.count("ids") ? opts["ids"].as<std::string>() : "";
            std::string filter_source = opts.count("source") ? opts["source"].as<std::string>() : "";
            std::string filter_collection = opts.count("collection") ? opts["collection"].as<std::string>() : "";
            std::string filter_category = opts.count("category") ? opts["category"].as<std::string>() : "";
            std::string target_user = opts.count("user") ? opts["user"].as<std::string>() : "";
            bool dry_run = opts.count("dry-run") > 0;

            if (filter_ids.empty() && filter_source.empty() &&
                filter_collection.empty() && filter_category.empty()) {
                std::cerr << "Error: specify at least one filter (--ids, --source, --collection, --category)\n";
                return 1;
            }

            // Resolve user DB path
            std::string user_db;
            std::string username;
            if (!target_user.empty()) {
                username = target_user;
                struct passwd* pw = getpwnam(target_user.c_str());
                if (!pw) {
                    std::cerr << "Error: cannot resolve home for user '" << target_user << "'\n";
                    return 1;
                }
                user_db = std::string(pw->pw_dir) + "/.ragger/memories.db";
            } else {
                struct passwd* pw = getpwuid(getuid());
                username = pw ? pw->pw_name : "default";
                user_db = ragger::expand_path("~/.ragger/memories.db");
            }

            std::string common_db = cfg.resolved_db_path();

            if (!fs::exists(user_db)) {
                std::cerr << "Error: user DB not found at " << user_db << "\n";
                return 1;
            }
            if (!fs::exists(common_db)) {
                std::cerr << "Error: common DB not found at " << common_db << "\n";
                return 1;
            }

            std::string src_path, dst_path, label_from, label_to;
            if (direction == "to-common") {
                src_path = user_db; dst_path = common_db;
                label_from = "user"; label_to = "common";
            } else {
                src_path = common_db; dst_path = user_db;
                label_from = "common"; label_to = "user";
            }

            // Build filter from CLI args
            ragger::MemoryFilter mfilter;
            if (!filter_ids.empty()) {
                std::istringstream iss(filter_ids);
                std::string tok;
                while (std::getline(iss, tok, ',')) {
                    tok.erase(0, tok.find_first_not_of(' '));
                    tok.erase(tok.find_last_not_of(' ') + 1);
                    if (!tok.empty()) mfilter.ids.push_back(std::stoi(tok));
                }
            }
            if (!filter_source.empty())     mfilter.source_pattern = filter_source;
            if (!filter_collection.empty()) mfilter.collection = filter_collection;
            if (!filter_category.empty())   mfilter.category = filter_category;

            return migrate_memories(src_path, dst_path, direction, username,
                                    mfilter, dry_run);

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
