/**
 * ragger — C++ port of Ragger Memory
 *
 * Verb-style CLI: ragger <verb> [options] [args]
 * No verb or 'help' prints usage.
 */
#include <cstdio>
#include <filesystem>
#include <format>
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
#include "auth.h"
#include "client.h"
#include "config.h"
#include "config_access.h"
#include "export.h"
#include "import.h"
#include "inference.h"
#include "lang.h"
#include "Logger.h"
#include "mcp.h"
#include "memory.h"
#include "vector_codec.h"
#include "recipe_cli.h"
#include "embed_executor.h"
#include "sqlite_backend.h"
#include "user_store.h"
#include "server.h"
#include "embedder.h"
#include "daemon_control.h"
#include "util/fs.h"
#include "util/time.h"
#include "storage_types.h"
#include "nlohmann_json.hpp"

using namespace ragger::lang;
namespace fs = std::filesystem;

// -----------------------------------------------------------------------
// Import: uses ragger::chunk_markdown from import.h
// -----------------------------------------------------------------------
static void do_import(ragger::RaggerMemory &memory,
                      const std::string &filepath,
                      int min_chunk_size,
                      const std::string &title,
                      int year,
                      const std::string &tags) {
    if (!fs::exists(filepath)) {
        throw std::runtime_error(std::format(ragger::lang::ERR_FILE_NOT_FOUND, filepath));
    }

    std::string text = ragger::read_file_to_string(filepath);

    auto chunks = ragger::chunk_markdown(text, min_chunk_size);

    auto filename = fs::path(filepath).filename().string();
    std::println(ragger::lang::MSG_IMPORTING_CHUNKS, chunks.size(), filename);

    // Single timestamp shared across every chunk of this import (issue #48).
    std::string import_ts = ragger::db_timestamp();

    const int total = static_cast<int>(chunks.size());

    // Title/year/tags group and prioritise documents (issue #23 search).
    // `--title` overrides the default (filename stem); `--year`/`--tags`
    // come from the caller. All chunks of one file share these.
    const std::string doc_title = title.empty()
                                      ? fs::path(filepath).stem().string()
                                      : title;

    // Store every chunk first WITHOUT embedding (fast), then embed the bodies
    // out-of-process with bounded concurrency + per-call timeout (issue #41).
    // This keeps a large import from saturating the box: at most
    // embed_max_workers `ragger embed` subprocesses run at once.
    std::vector<int>         ids;
    std::vector<std::string> texts;
    ids.reserve(total);
    texts.reserve(total);
    for (int i = 0; i < total; ++i) {
        ragger::DocumentChunk doc;
        doc.text        = ragger::clean_document_text(chunks[i].text);
        doc.title       = doc_title;
        doc.tags        = tags;
        doc.year        = year;
        doc.path        = ragger::collapse_home(filepath);
        doc.chunk_index = i + 1;
        doc.imported_at = import_ts;

        int id = memory.store_document(doc, /*defer_embedding=*/true);
        ids.push_back(id);
        texts.push_back(doc.text);
        std::println(ragger::lang::MSG_IMPORT_CHUNK, (i + 1), total, std::to_string(id));
    }

    ragger::EmbedExecutor embed_exec;  // timeout / retries / workers from config
    auto vecs = embed_exec.batch(texts);
    int skipped = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (vecs[i]) memory.update_document_embedding(ids[i], *vecs[i]);
        else ++skipped;   // left NULL — `ragger rebuild-embeddings` can retry
    }
    std::println(ragger::lang::MSG_IMPORT_DONE, chunks.size());
    if (skipped > 0)
        Diskerror::Logger::warn(std::format(ragger::lang::WARN_IMPORT_EMBED_SKIPPED,
                                            skipped, total));
}


// -----------------------------------------------------------------------
// Password input (with echo suppression)
// -----------------------------------------------------------------------

static std::string read_password(const std::string &prompt) {
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
    const std::string &username,
    const std::string &home_override = "") {
    std::string home_dir = home_override;
    if (home_dir.empty()) {
        struct passwd *pw = getpwnam(username.c_str());
        if (!pw) throw std::runtime_error(std::format(ragger::lang::ERR_USER_NOT_FOUND, username));
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
        struct passwd *pw = getpwnam(username.c_str());
        struct group *rg = getgrnam("ragger");
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
int main(int argc, char **argv) {
    // Record our own path so embedding can be spawned as `ragger embed`.
    if (argc > 0) ragger::set_executable_path(argv[0]);
    // Record full argv so the daemon can re-exec itself in place (restart).
    ragger::set_full_argv(argc, argv);

    Diskerror::ProgramOptions opts(CLI_DESCRIPTION);
    opts.add_options()
            ("help,h", CLI_HELP)
            ("version,V", CLI_VERSION)
            ("host", Diskerror::po::value<std::string>(), CLI_HOST)
            ("port,p", Diskerror::po::value<int>(), CLI_PORT)
            ("min-chunk-size", Diskerror::po::value<int>(), CLI_MIN_CHUNK_SIZE)
            ("num,n", Diskerror::po::value<int>(),
                "search: number of results to return (default: 3)")
            ("title", Diskerror::po::value<std::string>()->default_value(""), CLI_TITLE)
            ("year", Diskerror::po::value<int>()->default_value(0), CLI_YEAR)
            ("tags", Diskerror::po::value<std::string>()->default_value(""), CLI_TAGS)
            ("status", Diskerror::po::value<std::string>()->default_value(""),
                "decision: status to store/filter by (current|roadmap|superseded|deprecated)")
            // `import conversations` filters & inputs
            ("format", Diskerror::po::value<std::string>()->default_value(""),
                "Conversation source format: code | web | telegram")
            ("self", Diskerror::po::value<std::string>()->default_value(""),
                "Telegram format only: your display name, to split user/assistant turns")
            ("all", "import-conversations: treat a directory, or a file with sibling export "
                "files (memories.json, projects/), as one multi-file import unit")
            ("fdate", "import-conversations: use each input file's mtime as the timestamp "
                "for rows with no derivable date (e.g. memories.json)")
            ("date", Diskerror::po::value<std::string>()->default_value(""),
                "import-conversations: explicit timestamp override "
                "([CC]YYMMDD or YYYY-MM-DD) for rows with no derivable date")
            ("since", Diskerror::po::value<std::string>()->default_value(""),
                "Only turns on/after this date (YYYY-MM-DD)")
            ("until", Diskerror::po::value<std::string>()->default_value(""),
                "Only turns before this date (YYYY-MM-DD)")
            ("session", Diskerror::po::value<std::string>()->default_value(""),
                "Restrict to one session id")
            // `import summaries` input
            ("jsonl", Diskerror::po::value<std::string>()->default_value(""),
                "JSONL file ({text, tags?, timestamp?} per line)")
            // admin flags removed — sudo is the admin gate
            ("yes,y", CLI_YES)
            ("all,a", "config get: show all settings")
            ("json,j", "config get: format output as JSON")
            ("browser,b", "dashboard: open the URL in the default browser")
            ("embeddings,e", CLI_EMBEDDINGS)
            ("missing", "rebuild-phon: only fill rows with a NULL phon column")
            ("output,o", Diskerror::po::value<std::string>(), "Output file");
    opts.add_hidden_options()
            ("command", Diskerror::po::value<std::string>()->default_value("help"), CLI_COMMAND)
            ("args", Diskerror::po::value<std::vector<std::string> >(), CLI_ARGS)
            // Undocumented, testing-only: relocate the entire ~/.ragger base
            // directory (DB, settings.ini, logs, models, recipes, formats,
            // socket, token — everything). No env var equivalent — CLI-only.
            // Must be applied before any config/path resolution happens.
            ("ragger-base", Diskerror::po::value<std::string>()->default_value(""), "Override the ~/.ragger base directory (undocumented, testing only)")
            // --keep-data removed: always keep user data, sudoer can rm manually
            ;
    opts.add_positional("command", 1);
    opts.add_positional("args", -1);

    try {
        opts.run(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << std::format(ragger::lang::ERR_INFERENCE, e.what()) << "\n";
        return 1;
    }

    auto command = opts["command"].as<std::string>();

    if (opts.count("help") || command == "help") {
        std::println(ragger::lang::HELP_VERSION_HEADER, RAGGER_VERSION);
        std::cout << ragger::lang::HELP_SCREEN;
        std::cout << opts.to_string() << "\n";
        return 0;
    }

    if (opts.count("version") || command == "version") {
        std::cout << std::format(ragger::lang::VERSION_FORMAT, RAGGER_VERSION, RAGGER_COMMIT, RAGGER_BUILD_DATE) << "\n";
        return 0;
    }

    // Undocumented, testing-only --ragger-base: relocate the entire ~/.ragger
    // base directory before ANY config/path resolution happens (this must run
    // before init_config(), since the DB's own location depends on it).
    if (auto rh = opts["ragger-base"].as<std::string>(); !rh.empty()) {
        ragger::set_ragger_base_override(rh);
    }

    // Load config: compiled-in defaults overlaid with the settings table.
    try {
        ragger::init_config();
    }
    catch (const std::exception &e) {
        std::cerr << std::format(ragger::lang::ERR_INFERENCE, e.what()) << "\n";
        return 1;
    }

    const auto &cfg = ragger::config();

    //  Initialize static logging system.
    Diskerror::Logger log(cfg.log_file, cfg.log_level,
                           cfg.log_max_size_mb, cfg.log_max_age_days);

    // CLI overrides
    std::string host = opts.count("host") ? opts["host"].as<std::string>() : cfg.bind_address;
    const bool port_overridden = opts.count("port") > 0;
    int port = port_overridden ? opts["port"].as<int>() : cfg.port;
    std::string db_path = cfg.resolved_db_path();
    int min_chunk_size = opts.count("min-chunk-size")
                             ? opts["min-chunk-size"].as<int>()
                             : cfg.minimum_chunk_size;

    try {
        if (command == "start" || command == "stop" ||
            command == "restart" || command == "status") {
            return ragger::daemon_control(command);

        }
        else if (command == "serve") {
            const auto &cfg = ragger::config();

            // Create the memory store first so the DB + schema exist. This is
            // also what the one-time settings.ini migration and the port
            // rectify below need in place.
            std::unique_ptr<ragger::RaggerMemory> mem_ptr;
            mem_ptr = std::make_unique<ragger::RaggerMemory>(db_path);
            auto &memory = *mem_ptr;
            Diskerror::Logger::info(std::format(MSG_LOADED_MEMORIES, memory.count()));

            // One-time legacy config migration: seed the DB `settings` table
            // from settings.ini (if present and not already migrated), retire
            // the file, then re-overlay so the imported values take effect this
            // run. Idempotent via the DB `ini_migrated` marker.
            int imported = ragger::migrate_ini_to_db(db_path);
            if (imported > 0) {
                Diskerror::Logger::info(std::format(
                    "Migrated {} setting(s) from settings.ini into the DB; "
                    "renamed settings.ini -> settings.ini.migrated", imported));
                ragger::overlay_settings_from_db(ragger::mutable_config(),
                                                 cfg.resolved_db_path());
            }

            // Startup port rectify: unless --port was given on the CLI, adopt
            // the dashboard's desired_port into the committed port before
            // binding. This makes ANY restart path (service manager, crash
            // respawn, bare `ragger serve`) pick up a UI port change — not
            // just the dashboard's Restart button. A --port override is
            // transient: it binds the given port but leaves desired_port
            // untouched, so the next plain restart falls back to desired.
            if (!port_overridden && cfg.desired_port != 0 &&
                cfg.desired_port != cfg.port) {
                int adopted = cfg.desired_port;
                Diskerror::Logger::info(std::format(
                    "Startup port rectify: adopting desired_port {} (was {})",
                    adopted, cfg.port));
                auto r = ragger::set_config_persisted("port", std::to_string(adopted),
                                                      /*allow_locked=*/true);
                if (r.has_value()) {
                    port = adopted;  // bind the adopted port this run
                } else {
                    Diskerror::Logger::error(std::format(
                        "Failed to persist rectified port (err={}); binding existing port",
                        static_cast<int>(r.error())));
                }
            }

            // Keep the live config's `port` honest: it must reflect the port
            // actually being bound this run (the CLI override when present, or
            // the rectified/committed value otherwise), NOT persist it. The
            // dashboard's read-only "TCP Port" field reads this, so without
            // this a --port override would show the stale DB value. desired_port
            // is a separate field and is left untouched.
            ragger::mutable_config().port = port;

            ragger::Server server(memory, host, port);
            bool restart = server.run();

            if (restart) {
                // In-place re-exec: rebind listeners with the current config
                // (e.g. a new port). Destroy the server/memory first so the
                // socket + DB handles are released before the new image binds.
                mem_ptr.reset();
                const auto& saved_argv = ragger::full_argv();
                if (!saved_argv.empty()) {
                    // Strip any --host/--port CLI overrides from the replayed
                    // argv so the restarted daemon takes host/port from config
                    // (the DB, which now holds the new port the dashboard set).
                    // Without this, a CLI -p would pin the OLD port on re-exec.
                    std::vector<std::string> filtered;
                    for (size_t i = 0; i < saved_argv.size(); ++i) {
                        const std::string& a = saved_argv[i];
                        if (a == "-p" || a == "--port" || a == "--host") {
                            ++i;  // also skip the following value token
                            continue;
                        }
                        if (a.rfind("--port=", 0) == 0 || a.rfind("--host=", 0) == 0 ||
                            (a.rfind("-p", 0) == 0 && a.size() > 2))  // -p18435
                            continue;
                        filtered.push_back(a);
                    }
                    std::vector<char*> cargv;
                    cargv.reserve(filtered.size() + 1);
                    for (const auto& a : filtered)
                        cargv.push_back(const_cast<char*>(a.c_str()));
                    cargv.push_back(nullptr);
                    Diskerror::Logger::info("Re-executing daemon in place for restart");
                    ::execv(ragger::executable_path().c_str(), cargv.data());
                    // execv only returns on failure.
                    Diskerror::Logger::critical(std::format(
                        "re-exec failed: {} — daemon is stopped", std::strerror(errno)));
                    return 1;
                }
            }

        }
        else if (command == "search") {
            auto args = opts.getParams("args");
            if (args.empty()) {
                Diskerror::Logger::error(CLI_USAGE_SEARCH);
                return 1;
            }
            std::string query;
            for (size_t i = 0; i < args.size(); ++i) {
                if (i > 0) query += " ";
                query += args[i];
            }

            std::vector<std::string> colls;

            // CLI default is 3 results (overridable with -n/--num). This is the
            // human-facing default and is intentionally independent of the
            // daemon/MCP `default_search_limit` (which serves the agent API).
            int limit = opts.count("num") ? opts["num"].as<int>() : 3;
            if (limit < 1) limit = 1;

            // Try daemon first (thin client — no model loading)
            auto token = ragger::load_token();
            ragger::RaggerClient client(cfg.bind_address, cfg.port, token);
            ragger::SearchResponse response;

            if (client.is_available()) {
                response = client.search(query, limit,
                                         cfg.default_min_score, colls);
            }
            else {
                // Fall back to direct DB access
                ragger::RaggerMemory memory(db_path);
                response = memory.search(query, limit,
                                         cfg.default_min_score, colls);
            }

            nlohmann::json output = nlohmann::json::array();
            for (const auto &r: response.results) {
                output.push_back({
                    {"id", r.id}, {"score", r.score},
                    {"text", r.text}, {"metadata", r.metadata}
                });
            }
            std::cout << output.dump(2) << "\n";

        }
        else if (command == "store") {
            auto args = opts.getParams("args");
            if (args.empty()) {
                Diskerror::Logger::error(CLI_USAGE_STORE);
                return 1;
            }
            std::string text;
            for (size_t i = 0; i < args.size(); ++i) {
                if (i > 0) text += " ";
                text += args[i];
            }

            nlohmann::json meta = {};

            // Try daemon first (thin client — no model loading)
            auto token = ragger::load_token();
            ragger::RaggerClient client(cfg.bind_address, cfg.port, token);
            std::string id;

            if (client.is_available()) {
                id = client.store(text, meta);
            }
            else {
                // Fall back to direct DB access
                ragger::RaggerMemory memory(db_path);
                id = memory.store(text, meta);
            }
            std::cout << std::format(ragger::lang::MSG_STORED_WITH_ID, id) << "\n";

        }
        else if (command == "decision") {
            auto args = opts.getParams("args");
            if (args.empty()) {
                Diskerror::Logger::error(ragger::lang::CLI_USAGE_DECISION);
                return 1;
            }
            std::string sub = args[0];
            ragger::RaggerMemory memory(db_path);

            if (sub == "add") {
                if (args.size() < 2) {
                    Diskerror::Logger::error(ragger::lang::CLI_USAGE_DECISION);
                    return 1;
                }
                std::string text;
                for (size_t i = 1; i < args.size(); ++i) {
                    if (i > 1) text += " ";
                    text += args[i];
                }
                std::string status = opts["status"].as<std::string>();
                if (status.empty()) status = "current";
                std::string tags = opts["tags"].as<std::string>();
                int id = memory.store_decision(text, status, tags);
                std::println("Stored decision {} (status: {}).", id, status);
                return 0;
            }
            if (sub == "list") {
                std::string status = opts["status"].as<std::string>();
                if (status.empty()) status = "current";
                int limit = opts.count("num") ? opts["num"].as<int>() : 20;
                if (limit < 1) limit = 1;
                auto items = memory.decisions_by_status(status, limit);
                if (items.empty()) {
                    std::println("No decisions with status \"{}\".", status);
                    return 0;
                }
                for (auto& text : items) {
                    std::println("- {}", text);
                }
                return 0;
            }
            if (sub == "set-status") {
                if (args.size() < 3) {
                    Diskerror::Logger::error(ragger::lang::CLI_USAGE_DECISION);
                    return 1;
                }
                int decision_id = 0;
                try { decision_id = std::stoi(args[1]); } catch (...) {
                    Diskerror::Logger::error("decision set-status: <decision_id> must be a number");
                    return 1;
                }
                std::string status = args[2];
                if (!memory.set_decision_status(decision_id, status)) {
                    Diskerror::Logger::error(
                        "decision set-status: no decision with id " + args[1]);
                    return 1;
                }
                std::println("Decision {} set to status \"{}\".", decision_id, status);
                return 0;
            }
            Diskerror::Logger::error(ragger::lang::CLI_USAGE_DECISION);
            return 1;
        }
        else if (command == "count") {

            // Try daemon first (thin client — no model loading)
            auto token = ragger::load_token();
            ragger::RaggerClient client(cfg.bind_address, cfg.port, token);
            int count;

            if (client.is_available()) {
                count = client.count();
            }
            else {
                // Fall back to direct DB access
                ragger::RaggerMemory memory(db_path);
                count = memory.count();
            }
            std::cout << count << "\n";

        }
        else if (command == "config") {
            // DB-backed config get/set. Reads/writes the settings table via the
            // live (already DB-overlaid) config. No daemon required.
            //
            //   ragger config                      -> help
            //   ragger config get -a [-j]          -> all values
            //   ragger config get <name> [-j]      -> one value
            //   ragger config set <name> <value>   -> validate + persist
            auto args = opts.getParams("args");
            const bool as_json = opts.count("json") > 0;

            auto print_help = []() {
                std::cout <<
                    "Usage:\n"
                    "  ragger config get -a [-j]         Show all settings (‑j = JSON)\n"
                    "  ragger config get <name> [-j]     Show one setting\n"
                    "  ragger config set <name> <value>  Change a setting (blank value = default)\n"
                    "\n"
                    "Settings:\n";
                std::string cur_section;
                for (const auto& m : ragger::lang::config_schema()) {
                    if (m.section != cur_section) {
                        cur_section = std::string(m.section);
                        std::cout << "\n  [" << cur_section << "]\n";
                    }
                    std::cout << std::format("    {:<24} {}\n",
                                             std::string(m.key), std::string(m.help));
                }
            };

            if (args.empty()) {                       // bare `config` -> help
                print_help();
                return 0;
            }

            const std::string sub = args[0];

            if (sub == "get") {
                const auto& cfg = ragger::config();
                // `-a` OR no name -> all. A name -> that one.
                std::string name;
                for (size_t i = 1; i < args.size(); ++i) {
                    if (args[i] == "-a" || args[i] == "--all") continue;
                    if (args[i] == "-j" || args[i] == "--json") continue;
                    name = args[i];
                    break;
                }
                const bool want_all =
                    name.empty() || opts.count("all") > 0 ||
                    std::find(args.begin(), args.end(), "-a") != args.end() ||
                    std::find(args.begin(), args.end(), "--all") != args.end();
                const bool json_out = as_json ||
                    std::find(args.begin(), args.end(), "-j") != args.end();

                if (!want_all && !name.empty()) {
                    auto v = ragger::get_config_value(cfg, name);
                    if (!v) {
                        std::cerr << "Unknown config key: " << name << "\n";
                        return 1;
                    }
                    if (json_out)
                        std::cout << nlohmann::json{{name, *v}}.dump(2) << "\n";
                    else
                        std::cout << *v << "\n";
                    return 0;
                }

                // All keys.
                if (json_out) {
                    nlohmann::json out = nlohmann::json::object();
                    for (auto key : ragger::all_config_keys()) {
                        auto v = ragger::get_config_value(cfg, key);
                        if (v) out[std::string(key)] = *v;
                    }
                    std::cout << out.dump(2) << "\n";
                } else {
                    for (auto key : ragger::all_config_keys()) {
                        auto v = ragger::get_config_value(cfg, key);
                        if (v) std::cout << std::format("{:<24} = {}\n",
                                                        std::string(key), *v);
                    }
                }
                return 0;
            }
            else if (sub == "set") {
                if (args.size() < 2) {
                    std::cerr << "Usage: ragger config set <name> <value>\n";
                    return 1;
                }
                const std::string name = args[1];
                // Value may be empty (reset to default) or contain spaces.
                std::string value;
                for (size_t i = 2; i < args.size(); ++i) {
                    if (i > 2) value += " ";
                    value += args[i];
                }
                auto r = ragger::set_config_persisted(name, value);
                if (!r) {
                    switch (r.error()) {
                        case ragger::ConfigSetError::UnknownKey:
                            std::cerr << "Unknown config key: " << name << "\n"; break;
                        case ragger::ConfigSetError::Locked:
                            std::cerr << "Config key is locked (not user-writable): "
                                      << name << "\n"; break;
                        case ragger::ConfigSetError::InvalidValue:
                            std::cerr << "Invalid value for " << name << ": \""
                                      << value << "\"\n"; break;
                    }
                    return 1;
                }
                auto v = ragger::get_config_value(ragger::config(), name);
                std::cout << name << " = " << (v ? *v : value) << "\n";
                if (r->restart_required)
                    std::cout << "(change saved — restart the daemon to apply)\n";
                if (r->rebuild_required)
                    std::cout << "(change saved — run 'ragger rebuild-embeddings' to apply)\n";
                return 0;
            }
            else {
                print_help();
                return (sub == "help" || sub == "-h" || sub == "--help") ? 0 : 1;
            }
        }
        else if (command == "dashboard") {
            // Print the dashboard URL (same port as normal access) with the
            // access token as a query string. -b/--browser also opens it in
            // the default browser. Access is gated by the token; only a user
            // with RW access to ~/.ragger can read the token file, so holding
            // it proves authorization.
            const auto& cfg = ragger::config();

            // Scheme: HTTPS iff both TLS cert and key are configured.
            const bool tls = !cfg.tls_cert.empty() && !cfg.tls_key.empty();
            const std::string scheme = tls ? "https" : "http";

            // Host: prefer an explicit bind address; fall back to localhost.
            std::string url_host = cfg.bind_address.empty() ? "127.0.0.1"
                                                            : cfg.bind_address;
            if (url_host == "0.0.0.0" || url_host == "::") url_host = "127.0.0.1";

            const std::string token = ragger::ensure_token();

            std::string url = std::format("{}://{}:{}/dashboard",
                                          scheme, url_host, cfg.port);
            if (!token.empty()) url += "?token=" + token;

            std::cout << url << "\n";

            const bool open_browser = opts.count("browser") > 0;
            if (open_browser) {
#if defined(__APPLE__)
                std::string cmd = "open '" + url + "'";
#else
                std::string cmd = "xdg-open '" + url + "' >/dev/null 2>&1 &";
#endif
                if (std::system(cmd.c_str()) != 0)
                    std::cerr << "Could not launch a browser; open the URL above manually.\n";
            }
            return 0;
        }
        else if (command == "import-docs") {
            auto args = opts.getParams("args");
            if (args.empty()) {
                Diskerror::Logger::error(ragger::lang::CLI_USAGE_IMPORT);
                return 1;
            }
            // Markdown/text (and now JSON, auto-extracted to prose) → L5
            // documents. This is the old bare `ragger import <file>` path,
            // given its own verb name so it's no longer an implicit
            // fallthrough default.
            std::string imp_title = opts["title"].as<std::string>();
            int         imp_year  = opts["year"].as<int>();
            std::string imp_tags  = opts["tags"].as<std::string>();
            ragger::RaggerMemory memory(db_path);
            for (auto &filepath: args) {
                do_import(memory, filepath, min_chunk_size,
                          imp_title, imp_year, imp_tags);
            }
        }
        else if (command == "import-conversations") {
            auto args = opts.getParams("args");
            if (args.empty()) {
                Diskerror::Logger::error(ragger::lang::CLI_USAGE_IMPORT_CONVERSATIONS);
                return 1;
            }

            // ISO-8601 (or already-db) timestamp → "YYYY-MM-DD HH:MM:SS".
            auto to_db_ts = [](std::string s) -> std::string {
                if (s.empty()) return s;
                for (auto& c : s) if (c == 'T') c = ' ';
                auto cut = s.find_first_of(".Z+");
                if (cut != std::string::npos) s.erase(cut);
                if (s.size() == 10) s += " 00:00:00";
                return s;
            };

            // `summaries` sub-verb unchanged, just reachable from the new
            // top-level name (hand-authored L4 project summaries).
            if (args[0] == "summaries") {
                std::vector<ragger::SummaryImport> items;
                const std::string jsonl = opts["jsonl"].as<std::string>();
                if (!jsonl.empty()) {
                    items = ragger::load_summary_jsonl(ragger::expand_path(jsonl));
                } else if (args.size() >= 2) {
                    std::vector<std::string> files(args.begin() + 1, args.end());
                    for (auto& p : files) p = ragger::expand_path(p);
                    items = ragger::load_summary_files(files);
                } else {
                    Diskerror::Logger::error(
                        "import-conversations summaries: provide files or --jsonl=FILE");
                    return 1;
                }
                if (items.empty()) {
                    std::println("Nothing to import.");
                    return 0;
                }
                ragger::RaggerMemory memory(db_path);
                int n = 0, n_skip = 0;
                for (const auto& s : items) {
                    std::string ts = to_db_ts(s.timestamp);
                    if (!ts.empty() && memory.summary_exists_exact(s.text, ts)) {
                        ++n_skip;
                        continue;
                    }
                    memory.store_summary(s.text, "project",
                                         /*model_name=*/"",
                                         /*session_guid=*/"",
                                         /*source_timestamp=*/ts,
                                         /*tags=*/s.tags);
                    ++n;
                }
                std::println("Imported {} L4 summaries ({} duplicates skipped).", n, n_skip);
                return 0;
            }

            std::string raw_src = ragger::expand_path(args[0]);
            bool all_flag = opts.count("all") > 0;
            bool is_dir = fs::is_directory(raw_src);

            // Flat Markdown memory files (OpenClaw/nanobot/zeroclaw-style
            // daily logs and MEMORY.md snapshots) — plain `#` headings,
            // no JSON wrapper, dated by filename prefix or file mtime.
            // Handled as its own path, entirely separate from the
            // Claude/Telegram JSON formats below: a single .md file needs
            // no --all (nothing to pull in alongside it), while a
            // directory of them does, same as any other multi-file
            // import.
            bool looks_like_flat_md_dir = is_dir && !fs::exists(raw_src + "/conversations.json");
            if (looks_like_flat_md_dir) {
                bool has_md = false;
                for (auto& e : fs::directory_iterator(raw_src)) {
                    if (e.is_regular_file() && e.path().extension() == ".md") { has_md = true; break; }
                }
                if (has_md && !all_flag) {
                    Diskerror::Logger::error(
                        "import-conversations: " + raw_src + " is a directory of Markdown "
                        "memory files — pass --all to import them all as one unit");
                    return 1;
                }
                if (has_md) {
                    ragger::RaggerMemory memory(db_path);
                    int n_dec = 0, n_dec_skip = 0, n_sum = 0, n_sum_skip = 0, n_files = 0;
                    for (auto& e : fs::directory_iterator(raw_src)) {
                        if (!e.is_regular_file() || e.path().extension() != ".md") continue;
                        auto chunks = ragger::parse_flat_markdown_memory(e.path().string());
                        if (chunks.empty()) continue;
                        ++n_files;
                        for (auto& c : chunks) {
                            std::string ts = to_db_ts(c.date);
                            if (c.is_decision_like) {
                                if (memory.decision_exists_exact(c.text, ts)) { ++n_dec_skip; continue; }
                                memory.store_decision(c.text, "current", "flat-memory-import", ts);
                                ++n_dec;
                            } else {
                                if (memory.summary_exists_exact(c.text, ts)) { ++n_sum_skip; continue; }
                                memory.store_summary(c.text, "session",
                                                     /*model_name=*/"", /*session_guid=*/"",
                                                     /*source_timestamp=*/ts,
                                                     /*tags=*/"flat-memory-import");
                                ++n_sum;
                            }
                        }
                    }
                    std::println(
                        "Imported {} Markdown memory files: {} session summaries ({} duplicates "
                        "skipped), {} decisions ({} duplicates skipped).",
                        n_files, n_sum, n_sum_skip, n_dec, n_dec_skip);
                    return 0;
                }
                // Directory has neither conversations.json nor any .md
                // files — fall through to the conversations.json-required
                // error path below, which gives the right message for an
                // unrecognized directory.
            }
            if (!is_dir && fs::path(raw_src).extension() == ".md") {
                auto chunks = ragger::parse_flat_markdown_memory(raw_src);
                if (chunks.empty()) {
                    std::println("Nothing to import (empty file or recognized as agent scaffolding).");
                    return 0;
                }
                ragger::RaggerMemory memory(db_path);
                int n_dec = 0, n_dec_skip = 0, n_sum = 0, n_sum_skip = 0;
                for (auto& c : chunks) {
                    std::string ts = to_db_ts(c.date);
                    if (c.is_decision_like) {
                        if (memory.decision_exists_exact(c.text, ts)) { ++n_dec_skip; continue; }
                        memory.store_decision(c.text, "current", "flat-memory-import", ts);
                        ++n_dec;
                    } else {
                        if (memory.summary_exists_exact(c.text, ts)) { ++n_sum_skip; continue; }
                        memory.store_summary(c.text, "session",
                                             /*model_name=*/"", /*session_guid=*/"",
                                             /*source_timestamp=*/ts,
                                             /*tags=*/"flat-memory-import");
                        ++n_sum;
                    }
                }
                std::println(
                    "Imported {} session summaries ({} duplicates skipped), "
                    "{} decisions ({} duplicates skipped).",
                    n_sum, n_sum_skip, n_dec, n_dec_skip);
                return 0;
            }

            // Special case: the input file itself IS memories.json (not
            // conversations.json with a memories.json sibling). This is a
            // standalone import of just that one file — no siblings to
            // pull in, no "point at conversations.json instead" detour.
            // Handle it here and return, before the sibling-detection
            // logic below (which is about conversations.json's siblings,
            // not about being handed memories.json directly).
            if (!is_dir && fs::path(raw_src).filename() == "memories.json") {
                bool fdate = opts.count("fdate") > 0;
                std::string date_override = opts["date"].as<std::string>();
                std::string mem_ts;
                if (!date_override.empty()) {
                    mem_ts = to_db_ts(date_override);
                } else if (fdate) {
                    auto ftime = fs::last_write_time(raw_src);
                    auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                        ftime - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
                    std::time_t tt = std::chrono::system_clock::to_time_t(sctp);
                    char buf[32];
                    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&tt));
                    mem_ts = buf;
                }
                if (mem_ts.empty()) {
                    Diskerror::Logger::error(
                        "import-conversations: " + raw_src + " has no derivable date — "
                        "pass --fdate (use the file's mtime) or --date=DATE, or import it via "
                        "import-docs instead");
                    return 1;
                }
                std::string raw = ragger::read_file_to_string(raw_src);
                std::string text = ragger::extract_text_from_json(raw);
                if (text.empty()) {
                    std::println("Nothing to import.");
                    return 0;
                }
                ragger::RaggerMemory memory(db_path);
                if (memory.decision_exists_exact(text, mem_ts)) {
                    std::println("memories.json already imported (dated {}) — skipped.", mem_ts);
                    return 0;
                }
                // TODO: LLM-driven split into discrete claims. For now,
                // store the narrative as one decision row.
                memory.store_decision(text, "current", "claude-memory", mem_ts);
                std::println("Imported memories.json as 1 decision (dated {}).", mem_ts);
                return 0;
            }

            // Directory input, or a file with export siblings, both need
            // --all: it's the single "yes, treat this as one multi-file
            // import unit" switch. No implicit default either way.
            if (is_dir && !all_flag) {
                Diskerror::Logger::error(
                    "import-conversations: " + raw_src + " is a directory (looks like a "
                    "multi-file export) — pass --all to import it as one unit");
                return 1;
            }

            // Resolve the actual conversations.json to parse, plus any
            // sibling files (memories.json, projects/*.json) --all pulls in.
            std::string conv_path;
            std::string memories_path;
            std::vector<std::string> project_paths;
            if (is_dir) {
                conv_path = raw_src + "/conversations.json";
                if (!fs::exists(conv_path)) {
                    Diskerror::Logger::error(
                        "import-conversations: " + raw_src + " has no conversations.json");
                    return 1;
                }
                std::string cand_mem = raw_src + "/memories.json";
                if (fs::exists(cand_mem)) memories_path = cand_mem;
                std::string proj_dir = raw_src + "/projects";
                if (fs::is_directory(proj_dir)) {
                    for (auto& e : fs::directory_iterator(proj_dir)) {
                        if (e.is_regular_file() && e.path().extension() == ".json")
                            project_paths.push_back(e.path().string());
                    }
                }
            } else {
                conv_path = raw_src;
                if (all_flag) {
                    std::string dir = fs::path(raw_src).parent_path().string();
                    if (dir.empty()) dir = ".";
                    std::string cand_mem = dir + "/memories.json";
                    if (fs::exists(cand_mem)) memories_path = cand_mem;
                    std::string proj_dir = dir + "/projects";
                    if (fs::is_directory(proj_dir)) {
                        for (auto& e : fs::directory_iterator(proj_dir)) {
                            if (e.is_regular_file() && e.path().extension() == ".json")
                                project_paths.push_back(e.path().string());
                        }
                    }
                } else {
                    // Single file, no --all: if it has export siblings
                    // sitting right next to it, say so rather than
                    // silently ignoring them.
                    std::string dir = fs::path(raw_src).parent_path().string();
                    if (dir.empty()) dir = ".";
                    bool has_siblings = fs::exists(dir + "/memories.json") ||
                                       fs::is_directory(dir + "/projects");
                    if (has_siblings) {
                        Diskerror::Logger::error(
                            "import-conversations: " + raw_src + " has sibling export files "
                            "(memories.json / projects/) — pass --all to import them together, "
                            "or point at just this file to import only conversations");
                        return 1;
                    }
                }
            }

            // Auto-detect the conversation file's shape; --format overrides
            // when the sniff is wrong or the file is ambiguous.
            std::string fmt = opts["format"].as<std::string>();
            if (fmt.empty()) {
                switch (ragger::detect_conversation_format(conv_path)) {
                    case ragger::ConversationFormat::ClaudeWeb:      fmt = "web"; break;
                    case ragger::ConversationFormat::ClaudeCode:     fmt = "code"; break;
                    case ragger::ConversationFormat::Telegram:       fmt = "telegram"; break;
                    case ragger::ConversationFormat::ClaudeMemories:
                        Diskerror::Logger::error(
                            "import-conversations: " + conv_path + " looks like a Claude "
                            "memories.json (no per-message timestamps) — point --all at the "
                            "export directory so dates come from conversations.json, or import "
                            "it directly with --fdate/--date=DATE, or use import-docs instead");
                        return 1;
                    case ragger::ConversationFormat::Unknown:
                        Diskerror::Logger::error(
                            "import-conversations: could not detect the format of " + conv_path +
                            " — pass --format=code|web|telegram to override");
                        return 1;
                }
            }
            if (fmt != "code" && fmt != "web" && fmt != "telegram") {
                Diskerror::Logger::error(
                    "import-conversations: --format must be code|web|telegram");
                return 1;
            }

            ragger::TurnFilter f;
            f.session = opts["session"].as<std::string>();
            f.since   = opts["since"].as<std::string>();
            f.until   = opts["until"].as<std::string>();

            std::vector<ragger::ImportTurn> turns;
            if (fmt == "code") {
                turns = ragger::parse_claude_code(conv_path);
            } else if (fmt == "web") {
                turns = ragger::parse_claude_web(conv_path);
            } else { // telegram
                std::string self_name = opts["self"].as<std::string>();
                if (self_name.empty()) {
                    Diskerror::Logger::error(
                        "import-conversations --format=telegram: --self=\"Your Display Name\" is required");
                    return 1;
                }
                turns = ragger::parse_telegram(conv_path, self_name);
            }
            turns = ragger::filter_turns(turns, f);
            if (turns.empty() && memories_path.empty() && project_paths.empty()) {
                std::println("No turns matched.");
                return 0;
            }

            ragger::RaggerMemory memory(db_path);
            int n = 0, n_skip = 0, n_merged = 0;
            std::vector<std::pair<int, std::string>> inserted;  // telegram backfill scope
            bool is_telegram = (fmt == "telegram");
            for (const auto& t : turns) {
                std::string ts = to_db_ts(t.timestamp);
                if (!ts.empty() && !t.user_text.empty() &&
                    memory.turn_exists_fuzzy(t.user_text, ts, 90)) {
                    ++n_skip;
                    continue;
                }
                if (!is_telegram && !t.user_text.empty()) {
                    auto existing = memory.find_turn_by_text(t.user_text);
                    if (existing) {
                        if (ragger::is_synthetic_session_guid(existing->session_guid) &&
                            !t.session_id.empty()) {
                            memory.update_turn_meta(existing->turn_id, /*timestamp=*/"",
                                                    /*session_guid=*/t.session_id);
                            ++n_merged;
                        } else {
                            ++n_skip;
                        }
                        continue;
                    }
                }
                int new_id = memory.store_turn(t.user_text, t.assistant_text,
                                  /*model_name=*/t.source + "-import",
                                  /*defer_embedding=*/false,
                                  /*session_guid=*/t.session_id,
                                  /*source_timestamp=*/ts);
                if (is_telegram) inserted.emplace_back(new_id, t.user_text);
                if (++n % 25 == 0) {
                    std::println(std::cerr, "  imported {} turns...", n);
                }
            }
            int n_backfilled = 0;
            if (is_telegram && !inserted.empty()) {
                std::println(std::cerr, "Backfilling session ids for {} imported turns...",
                             inserted.size());
                for (const auto& [id, user_text] : inserted) {
                    if (user_text.empty()) continue;
                    auto match = memory.find_turn_by_text(user_text);
                    if (match && match->turn_id != id &&
                        !ragger::is_synthetic_session_guid(match->session_guid)) {
                        memory.update_turn_meta(id, /*timestamp=*/"",
                                                /*session_guid=*/match->session_guid);
                        ++n_backfilled;
                    }
                }
            }
            std::println("Imported {} conversation turns ({} duplicates skipped, {} session ids merged, {} backfilled).",
                         n, n_skip, n_merged, n_backfilled);

            // Per-conversation `summary` field (claude-web only) → one L3
            // session-summary row each, dated by that conversation's own
            // created_at (already real, no --fdate/--date needed).
            if (fmt == "web") {
                std::ifstream in(conv_path);
                nlohmann::json doc;
                try { in >> doc; } catch (...) { doc = nullptr; }
                nlohmann::json convs;
                if (doc.is_array()) convs = doc;
                else if (doc.is_object() && doc.contains("conversations")) convs = doc["conversations"];
                int n_sessions = 0, n_sessions_skip = 0;
                for (const auto& conv : convs) {
                    if (!conv.is_object()) continue;
                    std::string summary = conv.value("summary", "");
                    std::string sid = conv.value("uuid", "");
                    std::string ts = to_db_ts(conv.value("created_at", ""));
                    if (summary.empty() || ts.empty()) continue;
                    if (memory.summary_exists_exact(summary, ts)) { ++n_sessions_skip; continue; }
                    memory.store_summary(summary, "session",
                                         /*model_name=*/"claude",
                                         /*session_guid=*/sid,
                                         /*source_timestamp=*/ts,
                                         /*tags=*/"session");
                    ++n_sessions;
                }
                if (n_sessions || n_sessions_skip) {
                    std::println("Imported {} session summaries ({} duplicates skipped).",
                                 n_sessions, n_sessions_skip);
                }
            }

            // memories.json → L5/decisions. No derivable per-claim date —
            // requires --fdate (file mtime) or --date=DATE.
            if (!memories_path.empty()) {
                bool fdate = opts.count("fdate") > 0;
                std::string date_override = opts["date"].as<std::string>();
                std::string mem_ts;
                if (!date_override.empty()) {
                    mem_ts = to_db_ts(date_override);
                } else if (fdate) {
                    auto ftime = fs::last_write_time(memories_path);
                    auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                        ftime - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
                    std::time_t tt = std::chrono::system_clock::to_time_t(sctp);
                    char buf[32];
                    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&tt));
                    mem_ts = buf;
                }
                if (mem_ts.empty()) {
                    Diskerror::Logger::error(
                        "import-conversations: " + memories_path + " has no derivable date — "
                        "pass --fdate (use the file's mtime) or --date=DATE, or import it via "
                        "import-docs instead");
                } else {
                    std::string raw = ragger::read_file_to_string(memories_path);
                    std::string text = ragger::extract_text_from_json(raw);
                    if (!text.empty()) {
                        if (memory.decision_exists_exact(text, mem_ts)) {
                            std::println("memories.json already imported (dated {}) — skipped.", mem_ts);
                        } else {
                            // TODO: LLM-driven split into discrete claims.
                            // For now, store the narrative as one decision
                            // row — still searchable, still dated, just
                            // not yet broken into separate atomic claims.
                            memory.store_decision(text, "current", "claude-memory", mem_ts);
                            std::println("Imported memories.json as 1 decision (dated {}).", mem_ts);
                        }
                    }
                }
            }

            // Project files (L4): name + description → one summary row,
            // dated by the project's own created_at (real, no flag needed).
            if (!project_paths.empty()) {
                int n_proj = 0, n_proj_skip = 0;
                for (auto& pp : project_paths) {
                    std::ifstream pin(pp);
                    nlohmann::json pdoc;
                    try { pin >> pdoc; } catch (...) { continue; }
                    if (!pdoc.is_object()) continue;
                    std::string name = pdoc.value("name", "");
                    std::string desc = pdoc.value("description", "");
                    std::string ts = to_db_ts(pdoc.value("created_at", ""));
                    if (name.empty() || ts.empty()) continue;
                    std::string text = desc.empty() ? name : (name + "\n\n" + desc);
                    if (memory.summary_exists_exact(text, ts)) { ++n_proj_skip; continue; }
                    memory.store_summary(text, "project",
                                         /*model_name=*/"claude",
                                         /*session_guid=*/"",
                                         /*source_timestamp=*/ts,
                                         /*tags=*/"claude-project");
                    ++n_proj;
                }
                if (n_proj || n_proj_skip) {
                    std::println("Imported {} project summaries ({} duplicates skipped).",
                                 n_proj, n_proj_skip);
                }
            }

            return 0;
        }
        else if (command == "import") {
            // Deprecated alias for import-docs (the old implicit-fallthrough
            // behavior). Kept working so existing scripts don't break.
            auto args = opts.getParams("args");
            if (args.empty()) {
                Diskerror::Logger::error(ragger::lang::CLI_USAGE_IMPORT);
                return 1;
            }
            Diskerror::Logger::warn(
                "\"ragger import\" is deprecated — use \"ragger import-docs\" "
                "(docs) or \"ragger import-conversations\" (conversations/memories/projects)");
            std::string imp_title = opts["title"].as<std::string>();
            int         imp_year  = opts["year"].as<int>();
            std::string imp_tags  = opts["tags"].as<std::string>();
            ragger::RaggerMemory memory(db_path);
            for (auto &filepath: args) {
                do_import(memory, filepath, min_chunk_size,
                          imp_title, imp_year, imp_tags);
            }
        }
        else if (command == "export") {
            auto args = opts.getParams("args");
            if (args.empty()) {
                Diskerror::Logger::error(ragger::lang::CLI_USAGE_EXPORT);
                return 1;
            }
            std::string target = args[0];

            // Resolve DB path (same logic as other commands that don't need embedder)
            std::string resolved_db = db_path.empty()
                ? ragger::config().resolved_db_path() : ragger::expand_path(db_path);

            ragger::ExportOptions export_opts;
            export_opts.include_embeddings = opts.count("embeddings") > 0;

            if (target != "all") {
                auto tables = ragger::export_list_tables(resolved_db);
                bool found = false;
                for (auto& t : tables) { if (t == target) { found = true; break; } }
                if (!found) {
                    std::string avail;
                    for (auto& t : tables) {
                        if (!avail.empty()) avail += ", ";
                        avail += t;
                    }
                    Diskerror::Logger::error(std::format(
                        ragger::lang::MSG_EXPORT_TABLE_NOT_FOUND, target, avail));
                    return 1;
                }
                export_opts.table = target;
            }

            if (opts.count("output")) {
                std::string outpath = opts["output"].as<std::string>();
                std::ofstream outfile(outpath);
                if (!outfile) {
                    Diskerror::Logger::error(std::format("Cannot open: {}", outpath));
                    return 1;
                }
                int rows = ragger::export_sql(outfile, resolved_db, export_opts);
                std::println(ragger::lang::MSG_EXPORT_WROTE_FILE, outpath);
                std::println(ragger::lang::MSG_EXPORT_DONE, rows);
            } else {
                ragger::export_sql(std::cout, resolved_db, export_opts);
            }

        }
        else if (command == "mcp") {
            auto mem_ptr = std::make_unique<ragger::RaggerMemory>(db_path);
            ragger::run_mcp(*mem_ptr);

        }
        else if (command == "recipe") {
            // `ragger recipe [name]` — pick / inspect; writes the choice
            // to DB `settings.recipe`. Sentinel "default" tracks the INI.
            auto recipe_args = opts.getParams("args");
            return ragger::run_recipe_cli(recipe_args, db_path);
        }
        else if (command == "rebuild-embeddings") {

            // Get count first (before loading full memory). Skip the guard so
            // a pending model/dtype/dims change doesn't block the count/confirm.
            // Count across all four embedded tables — what the rebuild touches,
            // not just summaries (count()).
            ragger::RaggerMemory memory_temp(db_path,
                                             /*skip_embedding_guard=*/true);
            int total_count = memory_temp.backend()->count_embeddable_rows();
            memory_temp.close();

            // Warning + confirmation prompt
            std::println(ragger::lang::MSG_REBUILD_CONFIRM, total_count);

            bool proceed = false;
            if (opts.count("yes")) {
                proceed = true;
            }
            else {
                std::print("{}", ragger::lang::PROMPT_CONTINUE);
                std::string answer;
                std::getline(std::cin, answer);
                proceed = (answer == "y" || answer == "Y");
            }

            if (!proceed) {
                std::println("{}", ragger::lang::MSG_ABORTED);
                return 0;
            }

            // Backup the database file
            std::string actual_db_path = db_path.empty() ? cfg.resolved_db_path() : db_path;
            std::string backup_path = actual_db_path + ".bak";
            try {
                fs::copy_file(actual_db_path, backup_path,
                              fs::copy_options::overwrite_existing);
                std::println(ragger::lang::MSG_DB_BACKED_UP, backup_path);
            }
            catch (const std::exception &e) {
                Diskerror::Logger::critical(std::format(ragger::lang::WARN_BACKUP_FAILED, e.what()));
            }

            // Rebuild embeddings. Skip the drift guard — re-encoding at the
            // new config is exactly the point — then rewrite the settings
            // identity so the new model/dtype/dimensions "take". Doing this
            // *after* the re-encode means an aborted/failed rebuild leaves
            // settings≠config, so the guard still catches it next startup.
            // Log start/finish to the activity log (not just stdout) so the
            // re-embed is visible in the audit trail alongside the daemon's
            // degraded-mode / recovery messages.
            Diskerror::Logger::info(std::format(
                "rebuild-embeddings started: {} row(s), model '{}', {} {}-dim",
                total_count, cfg.embedding_model,
                ragger::vector_codec::canonical(cfg.embedding_vector_type),
                cfg.embedding_dimensions));
            ragger::RaggerMemory memory(db_path,
                                        /*skip_embedding_guard=*/true);
            int count = memory.rebuild_embeddings();
            ragger::UserStore settings_store(db_path);
            settings_store.set_setting(
                "embedding_model", cfg.embedding_model);
            settings_store.set_setting(
                "vector_type", ragger::vector_codec::canonical(cfg.embedding_vector_type));
            settings_store.set_setting(
                "dimensions", std::to_string(cfg.embedding_dimensions));
            Diskerror::Logger::info(std::format(
                "rebuild-embeddings finished: {} row(s) re-encoded; settings "
                "identity updated", count));
            std::cout << std::format(ragger::lang::MSG_EMBEDDINGS_REBUILT, count) << "\n";

        }
        else if (command == "rebuild-phon") {
            // Recompute the phon (Double Metaphone "sounds-like") column from
            // text across all four context tables. Pure string work — no
            // embedder, no model-identity change. `--missing` only fills
            // phon-NULL rows (cheap post-migration backfill); default recomputes
            // all (e.g. after a phonize() change). Backs up the DB first — cheap
            // rollback, no drift guard needed.
            bool only_missing = opts.count("missing") > 0;

            std::string actual_db_path = db_path.empty() ? cfg.resolved_db_path() : db_path;
            std::string backup_path = actual_db_path + ".bak";
            try {
                fs::copy_file(actual_db_path, backup_path,
                              fs::copy_options::overwrite_existing);
                std::println(ragger::lang::MSG_DB_BACKED_UP, backup_path);
            }
            catch (const std::exception &e) {
                Diskerror::Logger::critical(std::format(ragger::lang::WARN_BACKUP_FAILED, e.what()));
            }

            ragger::RaggerMemory memory(db_path,
                                        /*skip_embedding_guard=*/true);
            int count = memory.rebuild_phon(only_missing, /*progress=*/true);
            std::println("Rebuilt phonetic codes for {} row(s).", count);
        }
        else if (command == "show-embedding-model") {
            std::println(ragger::lang::MSG_EMBEDDING_MODEL_NAME, cfg.embedding_model);
            std::println(ragger::lang::MSG_EMBEDDING_DIMENSIONS, cfg.embedding_dimensions);
            std::string model_path = cfg.resolved_model_dir();
            if (!model_path.empty() && fs::is_directory(model_path))
                std::println(ragger::lang::MSG_EMBEDDING_PATH, model_path);
            else
                std::println(ragger::lang::MSG_EMBEDDING_PATH_DEFAULT);
        }
        else if (command == "embed") {
            std::string text((std::istreambuf_iterator<char>(std::cin)),
                             std::istreambuf_iterator<char>());
            if (text.empty()) {
                std::cerr << "Error: empty input" << "\n";
                return 1;
            }
            try {
                std::unique_ptr<ragger::Embedder> embedder;
                if (cfg.embedding_engine == "external") {
                    embedder = std::make_unique<ragger::Embedder>(
                        cfg.embedding_external_host,
                        cfg.embedding_external_port,
                        cfg.embedding_external_model,
                        cfg.embedding_external_api_key,
                        cfg.embedding_dimensions);
                } else {
                    embedder = std::make_unique<ragger::Embedder>(cfg.resolved_model_dir());
                }
                std::vector<float> vec = embedder->encode(text);
                nlohmann::json arr = vec;
                std::cout << arr.dump() << "\n";
            } catch (const std::exception &e) {
                std::cerr << std::format(ragger::lang::ERR_INFERENCE, e.what()) << "\n";
                return 1;
            }
        }
        else if (command == "useradd") {
            // Create a new user and issue a bearer token. Token is printed
            // exactly once — caller must save it. Errors if user exists
            // (use `usermod <name>` to rotate an existing user's token).
            auto args = opts.getParams("args");
            if (args.empty()) {
                Diskerror::Logger::critical(ragger::lang::CLI_USAGE_USERADD);
                return 1;
            }
            std::string username = args[0];

            try {
                ragger::UserStore storage(cfg.resolved_db_path());
                if (storage.get_user_by_username(username)) {
                    Diskerror::Logger::error(std::format(ragger::lang::ERR_USERADD_EXISTS, username) + "\n"
                                          + std::format(ragger::lang::ERR_USERADD_EXISTS_HINT, username));
                    return 1;
                }
                std::string token = ragger::generate_token();
                std::string token_hash = ragger::hash_token(token);
                storage.create_user(username, token_hash);
                std::println(ragger::lang::MSG_USER_ADDED, username);
                std::println("");
                std::println(ragger::lang::MSG_TOKEN_VALUE, token);
                std::println("");
                std::println("{}", ragger::lang::MSG_TOKEN_SAVE_WARNING);
            }
            catch (const std::exception &e) {
                Diskerror::Logger::critical(e.what());
                return 1;
            }

        }
        else if (command == "usermod") {
            // Rotate an existing user's bearer token. Prints the new token once.
            // Errors if the user does not exist.
            auto args = opts.getParams("args");
            if (args.empty()) {
                Diskerror::Logger::error(ragger::lang::CLI_USAGE_USERMOD);
                return 1;
            }
            std::string username = args[0];

            try {
                ragger::UserStore storage(cfg.resolved_db_path());
                if (!storage.get_user_by_username(username)) {
                    Diskerror::Logger::error(std::format(ragger::lang::ERR_USERMOD_MISSING, username) + "\n"
                                          + std::format(ragger::lang::ERR_USERMOD_MISSING_HINT, username));
                    return 1;
                }
                std::string token = ragger::generate_token();
                std::string token_hash = ragger::hash_token(token);
                storage.update_user_token(username, token_hash);
                std::println(ragger::lang::MSG_TOKEN_ROTATED, username);
                std::println("");
                std::println(ragger::lang::MSG_TOKEN_VALUE, token);
                std::println("");
                std::println("{}", ragger::lang::MSG_TOKEN_SAVE_WARNING);
            }
            catch (const std::exception &e) {
                Diskerror::Logger::critical(e.what());
                return 1;
            }

        }
        else if (command == "userdel") {
            auto args = opts.getParams("args");
            if (args.empty()) {
                Diskerror::Logger::critical(ragger::lang::CLI_USAGE_USERDEL);
                return 1;
            }
            std::string username = args[0];

            try {
                ragger::UserStore storage(cfg.resolved_db_path());
                ragger::userdel(storage, username);
                std::println(ragger::lang::MSG_USER_REMOVED, username);
            }
            catch (const std::exception &e) {
                Diskerror::Logger::critical(e.what());
                return 1;
            }

        }
        else if (command == "add-self") {
            // getpwuid is more reliable than getlogin in non-TTY contexts
            struct passwd *self_pw = getpwuid(getuid());
            char *login = self_pw ? self_pw->pw_name : nullptr;
            if (!login) {
                Diskerror::Logger::error(ragger::lang::ERR_UNKNOWN_USER);
                return 1;
            }
            std::string username(login);
            auto [token, created] = provision_user(username);
            if (created)
                std::println(ragger::lang::MSG_TOKEN_CREATED, username);
            else
                std::println(ragger::lang::MSG_TOKEN_EXISTS, username);
            std::println(ragger::lang::MSG_YOUR_TOKEN, token);
            std::println("{}", ragger::lang::MSG_TOKEN_USE_HINT);
            std::println("{}", ragger::lang::MSG_TOKEN_FILE_HINT);
            // Register directly in DB
            // Note: Multi-user mode removed. These user management commands are deprecated.
            try {
                std::string reg_db = cfg.resolved_db_path();
                ragger::UserStore backend(reg_db);
                std::string token_hash = ragger::hash_token(token);
                auto existing = backend.get_user_by_username(username);
                if (existing) {
                    if (existing->token_hash != token_hash)
                        backend.update_user_token(username, token_hash);
                    std::println(ragger::lang::MSG_USER_IN_DB, existing->id);
                }
                else {
                    int user_id = backend.create_user(username, token_hash);
                    std::println(ragger::lang::MSG_USER_REGISTERED, user_id);
                }
            }
            catch (const std::exception &e) {
                std::println(ragger::lang::WARN_DB_DEFERRED, e.what());
            }

        }
        else if (command == "passwd") {
            // Set (or clear) a user's web-UI login password. Only needed for
            // remote users who log in through the browser — local (127.0.0.1
            // / unix socket) sessions auto-authenticate as the daemon owner.
            // Empty password clears web-UI access for that user.
            auto args = opts.getParams("args");
            if (args.empty()) {
                Diskerror::Logger::error(ragger::lang::CLI_USAGE_PASSWD);
                return 1;
            }
            std::string target_user = args[0];

            try {
                ragger::UserStore umgr(cfg.resolved_db_path());
                auto user_info = umgr.get_user_by_username(target_user);
                if (!user_info) {
                    Diskerror::Logger::error(std::format(ragger::lang::ERR_USERMOD_MISSING, target_user) + "\n"
                                          + std::format(ragger::lang::ERR_PASSWD_MISSING_HINT, target_user));
                    return 1;
                }

                std::string new_pass = read_password(ragger::lang::PROMPT_NEW_PASSWORD);
                if (new_pass.empty()) {
                    umgr.set_user_password(target_user, "");
                    std::println(ragger::lang::MSG_PASSWORD_CLEARED, target_user);
                }
                else {
                    std::string confirm = read_password(ragger::lang::PROMPT_CONFIRM_PASSWORD);
                    if (new_pass != confirm) {
                        Diskerror::Logger::critical(ragger::lang::ERR_PASSWORDS_DIFFER);
                        return 1;
                    }
                    std::string hash = ragger::hash_password(new_pass);
                    umgr.set_user_password(target_user, hash);
                    std::println(ragger::lang::MSG_PASSWORD_SET, target_user);
                }
            }
            catch (const std::exception &e) {
                Diskerror::Logger::critical(e.what());
                return 1;
            }

        }
        else if (command == "housekeeping") {
            // Send SIGUSR1 to the running daemon via server PID file
            namespace fs = std::filesystem;
            pid_t daemon_pid = 0;
            try {
                for (const auto &entry: fs::directory_iterator("/tmp/ragger")) {
                    auto name = entry.path().filename().string();
                    if (name.rfind("server-", 0) == 0 &&
                        name.size() > 4 && name.substr(name.size() - 4) == ".pid") {
                        std::ifstream pf(entry.path());
                        if (pf) pf >> daemon_pid;
                        break;
                    }
                }
            }
            catch (...) {
            }
            if (daemon_pid <= 0) {
                Diskerror::Logger::error(ragger::lang::ERR_DAEMON_NOT_FOUND);
                return 1;
            }
            if (kill(daemon_pid, 0) != 0) {
                Diskerror::Logger::error(std::format(ragger::lang::ERR_DAEMON_PID_NOT_RUNNING, daemon_pid));
                return 1;
            }
            if (kill(daemon_pid, SIGUSR1) != 0) {
                if (errno == EPERM) {
                    Diskerror::Logger::error(ragger::lang::ERR_PERMISSION_DENIED_SIGNAL);
                }
                else {
                    Diskerror::Logger::error(std::format(ragger::lang::ERR_SIGNAL_FAILED, strerror(errno)));
                }
                return 1;
            }
            Diskerror::Logger::info(std::format(ragger::lang::MSG_HOUSEKEEPING_TRIGGERED, daemon_pid));

        }
        else if (command == "reload") {
            // Send SIGHUP to running daemon to reload config
            namespace fs = std::filesystem;
            pid_t daemon_pid = 0;
            try {
                for (const auto &entry: fs::directory_iterator("/tmp/ragger")) {
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
            }
            catch (...) {
            }

            if (daemon_pid <= 0) {
                Diskerror::Logger::error(ragger::lang::ERR_DAEMON_NOT_FOUND);
                return 1;
            }
            if (kill(daemon_pid, SIGHUP) != 0) {
                std::cerr << std::format(ragger::lang::ERR_SIGNAL_FAILED, strerror(errno)) << "\n";
                return 1;
            }
            Diskerror::Logger::error(std::format(ragger::lang::MSG_CONFIG_RELOAD_OK, daemon_pid));

        }
        else {
            std::cerr << std::format(ragger::lang::CLI_UNKNOWN_COMMAND, command) << "\n";
            return 1;
        }
    }
    catch (const std::exception &e) {
        Diskerror::Logger::critical(e.what());
        return 1;
    }

    return 0;
}
