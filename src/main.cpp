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

#include "diskerror/program_options.h"
#include "ragger/auth.h"
#include "ragger/client.h"
#include "ragger/config.h"
#include "ragger/export.h"
#include "ragger/import.h"
#include "ragger/inference.h"
#include "ragger/lang.h"
#include "diskerror/logger.h"
#include "ragger/mcp.h"
#include "ragger/memory.h"
#include "ragger/onboard.h"
#include "ragger/recipe_cli.h"
#include "ragger/embed_executor.h"
#include "ragger/sqlite_backend.h"
#include "ragger/user_store.h"
#include "ragger/server.h"
#include "ragger/embedder.h"
#include "ragger/daemon_control.h"
#include "ragger/util/fs.h"
#include "ragger/util/time.h"
#include "ragger/storage_types.h"
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
        doc.text        = chunks[i].text;
        doc.title       = doc_title;
        doc.tags        = tags;
        doc.year        = year;
        doc.path        = filepath;
        doc.chunk_index = i + 1;
        doc.imported_at = import_ts;

        int id = memory.store_document(doc, /*defer_embedding=*/true);
        ids.push_back(id);
        texts.push_back(chunks[i].text);
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
        Diskerror::logger::warn(std::format(ragger::lang::WARN_IMPORT_EMBED_SKIPPED,
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

    Diskerror::ProgramOptions opts(CLI_DESCRIPTION);
    opts.add_options()
            ("help,h", CLI_HELP)
            ("version,V", CLI_VERSION)
            ("config", Diskerror::po::value<std::string>()->default_value(""), CLI_CONFIG_FILE)
            ("host", Diskerror::po::value<std::string>(), CLI_HOST)
            ("port,p", Diskerror::po::value<int>(), CLI_PORT)
            ("model-dir", Diskerror::po::value<std::string>(), CLI_MODEL_DIR)
            ("min-chunk-size", Diskerror::po::value<int>(), CLI_MIN_CHUNK_SIZE)
            ("num,n", Diskerror::po::value<int>(),
                "search: number of results to return (default: 3)")
            ("title", Diskerror::po::value<std::string>()->default_value(""), CLI_TITLE)
            ("year", Diskerror::po::value<int>()->default_value(0), CLI_YEAR)
            ("tags", Diskerror::po::value<std::string>()->default_value(""), CLI_TAGS)
            // `import conversations` filters & inputs
            ("format", Diskerror::po::value<std::string>()->default_value(""),
                "Conversation source format: code | web")
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
            ("embeddings,e", CLI_EMBEDDINGS)
            ("missing", "rebuild-phon: only fill rows with a NULL phon column")
            ("output,o", Diskerror::po::value<std::string>(), "Output file");
    opts.add_hidden_options()
            ("command", Diskerror::po::value<std::string>()->default_value("help"), CLI_COMMAND)
            ("args", Diskerror::po::value<std::vector<std::string> >(), CLI_ARGS)
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

    // Load config file
    try {
        bool server_cmd = (command == "serve");
        ragger::init_config(opts["config"].as<std::string>());
    }
    catch (const std::exception &e) {
        std::cerr << std::format(ragger::lang::ERR_INFERENCE, e.what()) << "\n";
        return 1;
    }

    const auto &cfg = ragger::config();

    //  Initialize static logging system.
    Diskerror::logger log(cfg.log_file, cfg.log_level);

    // CLI overrides
    std::string host = opts.count("host") ? opts["host"].as<std::string>() : cfg.bind_address;
    int port = opts.count("port") ? opts["port"].as<int>() : cfg.port;
    std::string db_path = cfg.resolved_db_path();
    std::string model_dir = opts.count("model-dir") ? opts["model-dir"].as<std::string>() : "";
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
            std::unique_ptr<ragger::RaggerMemory> mem_ptr;
            // Single-user mode only
            mem_ptr = std::make_unique<ragger::RaggerMemory>(db_path, model_dir);
            auto &memory = *mem_ptr;
            Diskerror::logger::info(std::format(MSG_LOADED_MEMORIES, memory.count()));

            ragger::Server server(memory, host, port);
            server.run();

        }
        else if (command == "search") {
            auto args = opts.getParams("args");
            if (args.empty()) {
                Diskerror::logger::error(CLI_USAGE_SEARCH);
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
                ragger::RaggerMemory memory(db_path, model_dir);
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
                Diskerror::logger::error(CLI_USAGE_STORE);
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
                ragger::RaggerMemory memory(db_path, model_dir);
                id = memory.store(text, meta);
            }
            std::cout << std::format(ragger::lang::MSG_STORED_WITH_ID, id) << "\n";

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
                ragger::RaggerMemory memory(db_path, model_dir);
                count = memory.count();
            }
            std::cout << count << "\n";

        }
        else if (command == "import") {
            auto args = opts.getParams("args");
            if (args.empty()) {
                Diskerror::logger::error(ragger::lang::CLI_USAGE_IMPORT);
                return 1;
            }

            // ISO-8601 (or already-db) timestamp → "YYYY-MM-DD HH:MM:SS".
            // Tolerant: keeps the first 10 chars of a date and the time
            // portion if present; drops fractional seconds and zone.
            auto to_db_ts = [](std::string s) -> std::string {
                if (s.empty()) return s;
                // 'T' separator → space
                for (auto& c : s) if (c == 'T') c = ' ';
                // Cut at fractional / zone marker
                auto cut = s.find_first_of(".Z+");
                if (cut != std::string::npos) s.erase(cut);
                // Date-only input: pad with midnight
                if (s.size() == 10) s += " 00:00:00";
                return s;
            };

            // Subcommand: "conversations" or "summaries" — anything else is
            // the legacy document import.
            const std::string& sub = args[0];

            if (sub == "conversations") {
                const std::string fmt = opts["format"].as<std::string>();
                if (fmt != "code" && fmt != "web") {
                    Diskerror::logger::error(
                        "import conversations: --format=code|web is required");
                    return 1;
                }
                if (args.size() < 2) {
                    Diskerror::logger::error(
                        "import conversations: path to JSONL file or directory required");
                    return 1;
                }
                std::string src = ragger::expand_path(args[1]);

                ragger::TurnFilter f;
                f.session = opts["session"].as<std::string>();
                f.since   = opts["since"].as<std::string>();
                f.until   = opts["until"].as<std::string>();

                auto turns = (fmt == "code") ? ragger::parse_claude_code(src)
                                              : ragger::parse_claude_web(src);
                turns = ragger::filter_turns(turns, f);
                if (turns.empty()) {
                    std::println("No turns matched.");
                    return 0;
                }

                ragger::RaggerMemory memory(db_path, model_dir);
                int n = 0;
                for (const auto& t : turns) {
                    std::string text = "User: " + t.user_text;
                    if (!t.assistant_text.empty())
                        text += "\n\nAssistant: " + t.assistant_text;
                    // Land at level="turn" so imports cluster with L2 turn
                    // summaries rather than polluting the session-summary
                    // bucket. tags carry provenance (source + session id).
                    std::string tags = t.source;
                    if (!t.session_id.empty()) tags += "," + t.session_id;
                    nlohmann::json meta = {
                        {"level", "turn"},
                        {"tags",  tags},
                    };
                    std::string ts = to_db_ts(t.timestamp);
                    if (!ts.empty()) meta["timestamp"] = ts;
                    memory.store(text, meta);
                    if (++n % 25 == 0) {
                        std::println(std::cerr, "  imported {} turns...", n);
                    }
                }
                std::println("Imported {} conversation turns.", n);
                return 0;
            }

            if (sub == "summaries") {
                std::vector<ragger::SummaryImport> items;
                const std::string jsonl = opts["jsonl"].as<std::string>();
                if (!jsonl.empty()) {
                    items = ragger::load_summary_jsonl(ragger::expand_path(jsonl));
                } else if (args.size() >= 2) {
                    std::vector<std::string> files(args.begin() + 1, args.end());
                    for (auto& p : files) p = ragger::expand_path(p);
                    items = ragger::load_summary_files(files);
                } else {
                    Diskerror::logger::error(
                        "import summaries: provide files or --jsonl=FILE");
                    return 1;
                }
                if (items.empty()) {
                    std::println("Nothing to import.");
                    return 0;
                }
                ragger::RaggerMemory memory(db_path, model_dir);
                int n = 0;
                for (const auto& s : items) {
                    memory.store_summary(s.text, "project", "complete",
                                         /*model_name=*/"",
                                         /*session_guid=*/"",
                                         /*source_timestamp=*/to_db_ts(s.timestamp),
                                         /*tags=*/s.tags);
                    ++n;
                }
                std::println("Imported {} L4 summaries.", n);
                return 0;
            }

            // Legacy: ragger import <file> [file...] — markdown/text chunks
            std::string imp_title = opts["title"].as<std::string>();
            int         imp_year  = opts["year"].as<int>();
            std::string imp_tags  = opts["tags"].as<std::string>();
            ragger::RaggerMemory memory(db_path, model_dir);
            for (auto &filepath: args) {
                do_import(memory, filepath, min_chunk_size,
                          imp_title, imp_year, imp_tags);
            }

        }
        else if (command == "export") {
            auto args = opts.getParams("args");
            if (args.empty()) {
                Diskerror::logger::error(ragger::lang::CLI_USAGE_EXPORT);
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
                    Diskerror::logger::error(std::format(
                        ragger::lang::MSG_EXPORT_TABLE_NOT_FOUND, target, avail));
                    return 1;
                }
                export_opts.table = target;
            }

            if (opts.count("output")) {
                std::string outpath = opts["output"].as<std::string>();
                std::ofstream outfile(outpath);
                if (!outfile) {
                    Diskerror::logger::error(std::format("Cannot open: {}", outpath));
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
            auto mem_ptr = std::make_unique<ragger::RaggerMemory>(db_path, model_dir);
            ragger::run_mcp(*mem_ptr);

        }
        else if (command == "recipe") {
            // `ragger recipe [name]` — pick / inspect; writes the choice
            // to DB `settings.recipe`. Sentinel "default" tracks the INI.
            auto recipe_args = opts.getParams("args");
            return ragger::run_recipe_cli(recipe_args, db_path);
        }
        else if (command == "onboard") {
            // Guided first-run setup — capture/build, recipe, inference,
            // memory model, daemon start. Idempotent.
            auto ob_args = opts.getParams("args");
            return ragger::run_onboard(ob_args, db_path);
        }
        else if (command == "rebuild-embeddings") {

            // Get count first (before loading full memory). Skip the guard so
            // a pending model/dtype/dims change doesn't block the count/confirm.
            // Count across all four embedded tables — what the rebuild touches,
            // not just summaries (count()).
            ragger::RaggerMemory memory_temp(db_path, model_dir,
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
                Diskerror::logger::critical(std::format(ragger::lang::WARN_BACKUP_FAILED, e.what()));
            }

            // Rebuild embeddings. Skip the drift guard — re-encoding at the
            // new config is exactly the point — then rewrite the settings
            // identity so the new model/dtype/dimensions "take". Doing this
            // *after* the re-encode means an aborted/failed rebuild leaves
            // settings≠config, so the guard still catches it next startup.
            ragger::RaggerMemory memory(db_path, model_dir,
                                        /*skip_embedding_guard=*/true);
            int count = memory.rebuild_embeddings();
            ragger::UserStore settings_store(db_path);
            settings_store.set_setting(
                "embedding_model", cfg.resolve_model(cfg.embedding_model));
            settings_store.set_setting(
                "vector_type", cfg.embedding_vector_type == "f32" ? "f32" : "f16");
            settings_store.set_setting(
                "dimensions", std::to_string(cfg.embedding_dimensions));
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
                Diskerror::logger::critical(std::format(ragger::lang::WARN_BACKUP_FAILED, e.what()));
            }

            ragger::RaggerMemory memory(db_path, model_dir,
                                        /*skip_embedding_guard=*/true);
            int count = memory.rebuild_phon(only_missing, /*progress=*/true);
            std::println("Rebuilt phonetic codes for {} row(s).", count);
        }
        else if (command == "show-embedding-model") {
            std::println(ragger::lang::MSG_EMBEDDING_MODEL_NAME, cfg.embedding_model);
            std::println(ragger::lang::MSG_EMBEDDING_DIMENSIONS, cfg.embedding_dimensions);
            std::string model_path = model_dir.empty() ? cfg.model_dir : model_dir;
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
                ragger::Embedder embedder(cfg.resolved_model_dir());
                std::vector<float> vec = embedder.encode(text);
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
                Diskerror::logger::critical(ragger::lang::CLI_USAGE_USERADD);
                return 1;
            }
            std::string username = args[0];

            try {
                ragger::UserStore storage(cfg.resolved_db_path());
                if (storage.get_user_by_username(username)) {
                    Diskerror::logger::error(std::format(ragger::lang::ERR_USERADD_EXISTS, username) + "\n"
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
                Diskerror::logger::critical(e.what());
                return 1;
            }

        }
        else if (command == "usermod") {
            // Rotate an existing user's bearer token. Prints the new token once.
            // Errors if the user does not exist.
            auto args = opts.getParams("args");
            if (args.empty()) {
                Diskerror::logger::error(ragger::lang::CLI_USAGE_USERMOD);
                return 1;
            }
            std::string username = args[0];

            try {
                ragger::UserStore storage(cfg.resolved_db_path());
                if (!storage.get_user_by_username(username)) {
                    Diskerror::logger::error(std::format(ragger::lang::ERR_USERMOD_MISSING, username) + "\n"
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
                Diskerror::logger::critical(e.what());
                return 1;
            }

        }
        else if (command == "userdel") {
            auto args = opts.getParams("args");
            if (args.empty()) {
                Diskerror::logger::critical(ragger::lang::CLI_USAGE_USERDEL);
                return 1;
            }
            std::string username = args[0];

            try {
                ragger::UserStore storage(cfg.resolved_db_path());
                ragger::userdel(storage, username);
                std::println(ragger::lang::MSG_USER_REMOVED, username);
            }
            catch (const std::exception &e) {
                Diskerror::logger::critical(e.what());
                return 1;
            }

        }
        else if (command == "add-self") {
            // getpwuid is more reliable than getlogin in non-TTY contexts
            struct passwd *self_pw = getpwuid(getuid());
            char *login = self_pw ? self_pw->pw_name : nullptr;
            if (!login) {
                Diskerror::logger::error(ragger::lang::ERR_UNKNOWN_USER);
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
                Diskerror::logger::error(ragger::lang::CLI_USAGE_PASSWD);
                return 1;
            }
            std::string target_user = args[0];

            try {
                ragger::UserStore umgr(cfg.resolved_db_path());
                auto user_info = umgr.get_user_by_username(target_user);
                if (!user_info) {
                    Diskerror::logger::error(std::format(ragger::lang::ERR_USERMOD_MISSING, target_user) + "\n"
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
                        Diskerror::logger::critical(ragger::lang::ERR_PASSWORDS_DIFFER);
                        return 1;
                    }
                    std::string hash = ragger::hash_password(new_pass);
                    umgr.set_user_password(target_user, hash);
                    std::println(ragger::lang::MSG_PASSWORD_SET, target_user);
                }
            }
            catch (const std::exception &e) {
                Diskerror::logger::critical(e.what());
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
                Diskerror::logger::error(ragger::lang::ERR_DAEMON_NOT_FOUND);
                return 1;
            }
            if (kill(daemon_pid, 0) != 0) {
                Diskerror::logger::error(std::format(ragger::lang::ERR_DAEMON_PID_NOT_RUNNING, daemon_pid));
                return 1;
            }
            if (kill(daemon_pid, SIGUSR1) != 0) {
                if (errno == EPERM) {
                    Diskerror::logger::error(ragger::lang::ERR_PERMISSION_DENIED_SIGNAL);
                }
                else {
                    Diskerror::logger::error(std::format(ragger::lang::ERR_SIGNAL_FAILED, strerror(errno)));
                }
                return 1;
            }
            Diskerror::logger::info(std::format(ragger::lang::MSG_HOUSEKEEPING_TRIGGERED, daemon_pid));

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
                Diskerror::logger::error(ragger::lang::ERR_DAEMON_NOT_FOUND);
                return 1;
            }
            if (kill(daemon_pid, SIGHUP) != 0) {
                std::cerr << std::format(ragger::lang::ERR_SIGNAL_FAILED, strerror(errno)) << "\n";
                return 1;
            }
            Diskerror::logger::error(std::format(ragger::lang::MSG_CONFIG_RELOAD_OK, daemon_pid));

        }
        else {
            std::cerr << std::format(ragger::lang::CLI_UNKNOWN_COMMAND, command) << "\n";
            return 1;
        }
    }
    catch (const std::exception &e) {
        Diskerror::logger::critical(e.what());
        return 1;
    }

    return 0;
}
