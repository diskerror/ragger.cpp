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
    std::println(ragger::lang::MSG_IMPORTING_CHUNKS, chunks.size(), filename);

    for (size_t i = 0; i < chunks.size(); ++i) {
        nlohmann::json meta = {
            {"source", filepath},
            {"chunk", (int)(i + 1)},
            {"total_chunks", (int)chunks.size()}
        };
        if (!collection.empty()) meta["collection"] = collection;
        if (!chunks[i].section.empty()) meta["section"] = chunks[i].section;

        auto id = memory.store(chunks[i].text, meta);
        std::println(ragger::lang::MSG_IMPORT_CHUNK, (i + 1), chunks.size(), id);
    }
    std::println(ragger::lang::MSG_IMPORT_DONE, chunks.size());
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
            std::cerr << std::format(ragger::lang::ERR_PAYLOAD_DUMP_DIR, dump_payloads_dir, ec.message()) << "\n";
            return;
        }
        if (!existed) {
            std::cout << std::format(ragger::lang::MSG_PAYLOAD_DUMP_CREATED, dump_payloads_dir) << "\n";
        }
    }

    // Build inference client from config
    ragger::InferenceClient inference = ragger::InferenceClient::from_config(cfg);

    if (!dump_payloads_dir.empty()) {
        inference.set_payload_dump_dir(dump_payloads_dir);
    }

    if (inference._endpoints.empty()) {
        std::println("{}", ragger::lang::ERR_NO_ENDPOINTS);
        std::println(ragger::lang::MSG_INFERENCE_HINT);
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
                std::cerr << std::format(ragger::lang::ERR_ENDPOINT_UNREACHABLE, ep.name, ep.api_url)
                          << "\n  " << curl_easy_strerror(res) << "\n";
                return;
            }
            if (http_code >= 400) {
                std::cerr << std::format(ragger::lang::ERR_ENDPOINT_HTTP, ep.name, http_code) << "\n";
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
//
// Design: idempotent + friendly.
//   start   → "already running" if it is, else launch it
//   stop    → "not running" if it isn't, else stop it
//   restart → cycle it whether up or down
//   status  → one-line summary (running pid / stopped / not loaded)
// -----------------------------------------------------------------------
namespace {

/// Run a shell command, return captured stdout. Silent on failure (returns "").
std::string capture_output(const std::string& cmd) {
    FILE* pipe = popen((cmd + " 2>/dev/null").c_str(), "r");
    if (!pipe) return "";
    std::string out;
    char buf[4096];
    while (std::fgets(buf, sizeof(buf), pipe)) out += buf;
    pclose(pipe);
    return out;
}

/// Run a shell command, discard output, return true iff exit code 0.
bool run_quiet(const std::string& cmd) {
    int rc = std::system((cmd + " >/dev/null 2>&1").c_str());
    return WIFEXITED(rc) && WEXITSTATUS(rc) == 0;
}

/// Kill every `ragger` process owned by the current user except this one.
/// Used by `stop` and `restart` to sweep MCP servers and stray CLI
/// invocations — MCP is typically launched by a client (OpenClaw, Claude
/// Desktop) and has no service manager of its own, so we have to reap it here.
/// Sends SIGTERM, waits briefly, then SIGKILL to any survivors.
/// Returns the number of processes signaled.
int kill_other_ragger_instances() {
    pid_t self = getpid();
    auto out = capture_output(
        "pgrep -u " + std::to_string(getuid()) + " -x ragger");
    std::vector<pid_t> pids;
    std::istringstream iss(out);
    std::string line;
    while (std::getline(iss, line)) {
        try {
            pid_t pid = std::stoi(line);
            if (pid != self && pid > 1) pids.push_back(pid);
        } catch (...) { /* skip malformed */ }
    }
    if (pids.empty()) return 0;

    for (pid_t p : pids) ::kill(p, SIGTERM);
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    for (pid_t p : pids) {
        if (::kill(p, 0) == 0) ::kill(p, SIGKILL);  // still alive → force
    }
    return static_cast<int>(pids.size());
}

} // anonymous namespace

static int daemon_control(const std::string& action) {
    struct passwd* pw = getpwuid(getuid());
    std::string home = pw ? pw->pw_dir : (std::getenv("HOME") ? std::getenv("HOME") : "");
    if (home.empty()) {
        std::cerr << ragger::lang::ERR_HOME_NOT_FOUND << "\n";
        return 1;
    }

#if defined(__APPLE__)
    const std::string label  = "com.diskerror.ragger";
    const std::string plist  = home + "/Library/LaunchAgents/" + label + ".plist";
    const std::string target = "gui/" + std::to_string(getuid()) + "/" + label;
    const std::string domain = "gui/" + std::to_string(getuid());

    // Loaded = bootstrapped into the gui domain. Running = has a live pid.
    auto is_loaded = [&]() { return run_quiet("launchctl print " + target); };
    auto get_pid   = [&]() -> std::string {
        if (!is_loaded()) return "";
        auto out = capture_output("launchctl print " + target);
        auto pos = out.find("\n\tpid = ");
        if (pos == std::string::npos) pos = out.find("\n    pid = ");
        if (pos == std::string::npos) return "";
        auto s = out.find('=', pos) + 1;
        while (s < out.size() && out[s] == ' ') ++s;
        auto e = out.find('\n', s);
        return out.substr(s, e - s);
    };
    auto is_running = [&]() { return !get_pid().empty(); };

    auto ensure_plist = [&]() -> bool {
        if (fs::exists(plist)) return true;
        std::cerr << std::format(ragger::lang::ERR_PLIST_NOT_FOUND, plist) << "\n";
        return false;
    };

    if (action == "start") {
        if (!ensure_plist()) return 1;
        if (is_running()) {
            std::cout << std::format(ragger::lang::MSG_ALREADY_RUNNING, get_pid()) << "\n";
            return 0;
        }
        if (is_loaded()) {
            if (!run_quiet("launchctl kickstart " + target)) {
                std::cerr << ragger::lang::ERR_LAUNCHCTL_KICKSTART << "\n"; return 1;
            }
        } else {
            if (!run_quiet("launchctl bootstrap " + domain + " " + plist)) {
                std::cerr << ragger::lang::ERR_LAUNCHCTL_BOOTSTRAP << "\n"; return 1;
            }
        }
        auto pid = get_pid();
        if (!pid.empty())
            std::cout << std::format(ragger::lang::MSG_STARTED_PID, pid) << "\n";
        else
            std::cout << ragger::lang::MSG_STARTED << "\n";
        return 0;
    }

    if (action == "stop") {
        bool daemon_was_running = is_loaded() && is_running();
        if (daemon_was_running) {
            if (!run_quiet("launchctl bootout " + target)) {
                std::cerr << ragger::lang::ERR_LAUNCHCTL_BOOTOUT << "\n"; return 1;
            }
        }
        // Sweep any other ragger processes this user owns (MCP servers
        // launched by clients, stray CLI runs, etc.).
        int extras = kill_other_ragger_instances();

        if (daemon_was_running && extras > 0) {
            if (extras == 1)
                std::cout << std::format(ragger::lang::MSG_STOPPED_EXTRA_1, extras) << "\n";
            else
                std::cout << std::format(ragger::lang::MSG_STOPPED_EXTRA_N, extras) << "\n";
        } else if (daemon_was_running) {
            std::cout << ragger::lang::MSG_STOPPED << "\n";
        } else if (extras > 0) {
            if (extras == 1)
                std::cout << std::format(ragger::lang::MSG_EXTRAS_ONLY_1, extras) << "\n";
            else
                std::cout << std::format(ragger::lang::MSG_EXTRAS_ONLY_N, extras) << "\n";
        } else {
            std::cout << ragger::lang::MSG_NOT_RUNNING << "\n";
        }
        return 0;
    }

    if (action == "restart") {
        if (!ensure_plist()) return 1;
        bool was_running = is_running();
        if (is_loaded()) {
            run_quiet("launchctl bootout " + target);
            // bootout returns before launchd completes teardown — wait for
            // the domain to actually release the label (up to 3s).
            for (int i = 0; i < 30 && is_loaded(); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        // Sweep MCP + stray instances before relaunching the daemon.
        // (Clients will need to reconnect their MCP subprocess themselves.)
        kill_other_ragger_instances();

        if (!run_quiet("launchctl bootstrap " + domain + " " + plist)) {
            std::cerr << ragger::lang::ERR_LAUNCHCTL_BOOTSTRAP << "\n"; return 1;
        }
        auto pid = get_pid();
        if (!pid.empty()) {
            if (was_running)
                std::cout << std::format(ragger::lang::MSG_RESTARTED_PID, pid) << "\n";
            else
                std::cout << std::format(ragger::lang::MSG_STARTED_PID, pid) << "\n";
        } else {
            std::cout << (was_running ? ragger::lang::MSG_RESTARTED
                                      : ragger::lang::MSG_STARTED) << "\n";
        }
        return 0;
    }

    if (action == "status") {
        if (!is_loaded()) {
            std::cout << ragger::lang::MSG_NOT_LOADED << "\n";
            return 0;
        }
        auto pid = get_pid();
        if (pid.empty()) {
            std::cout << ragger::lang::MSG_LOADED_NOT_RUNNING << "\n";
            return 0;
        }
        std::cout << std::format(ragger::lang::MSG_RUNNING_PID, pid) << "\n";
        return 0;
    }

    std::cerr << std::format(ragger::lang::ERR_UNKNOWN_ACTION, action) << "\n";
    return 1;

#elif defined(__linux__)
    const std::string unit = "ragger.service";

    auto is_active = [&]() {
        return run_quiet("systemctl --user is-active --quiet " + unit);
    };
    auto main_pid = [&]() -> std::string {
        auto out = capture_output(
            "systemctl --user show -p MainPID --value " + unit);
        // trim trailing newline/whitespace
        while (!out.empty() && (out.back() == '\n' || out.back() == ' '))
            out.pop_back();
        if (out == "0" || out.empty()) return "";
        return out;
    };

    if (action == "start") {
        if (is_active()) {
            auto pid = main_pid();
            if (!pid.empty())
                std::cout << std::format(ragger::lang::MSG_ALREADY_RUNNING, pid) << "\n";
            else
                std::cout << ragger::lang::MSG_IS_RUNNING << "\n";
            return 0;
        }
        if (!run_quiet("systemctl --user start " + unit)) {
            std::cerr << ragger::lang::ERR_SYSTEMCTL_START << "\n"; return 1;
        }
        auto pid = main_pid();
        if (!pid.empty())
            std::cout << std::format(ragger::lang::MSG_STARTED_PID, pid) << "\n";
        else
            std::cout << ragger::lang::MSG_STARTED << "\n";
        return 0;
    }

    if (action == "stop") {
        bool daemon_was_running = is_active();
        if (daemon_was_running) {
            if (!run_quiet("systemctl --user stop " + unit)) {
                std::cerr << ragger::lang::ERR_SYSTEMCTL_STOP << "\n"; return 1;
            }
        }
        int extras = kill_other_ragger_instances();

        if (daemon_was_running && extras > 0) {
            if (extras == 1)
                std::cout << std::format(ragger::lang::MSG_STOPPED_EXTRA_1, extras) << "\n";
            else
                std::cout << std::format(ragger::lang::MSG_STOPPED_EXTRA_N, extras) << "\n";
        } else if (daemon_was_running) {
            std::cout << ragger::lang::MSG_STOPPED << "\n";
        } else if (extras > 0) {
            if (extras == 1)
                std::cout << std::format(ragger::lang::MSG_EXTRAS_ONLY_1, extras) << "\n";
            else
                std::cout << std::format(ragger::lang::MSG_EXTRAS_ONLY_N, extras) << "\n";
        } else {
            std::cout << ragger::lang::MSG_NOT_RUNNING << "\n";
        }
        return 0;
    }

    if (action == "restart") {
        bool was_active = is_active();
        // Stop daemon and sweep MCP + strays before relaunching.
        if (was_active) run_quiet("systemctl --user stop " + unit);
        kill_other_ragger_instances();
        if (!run_quiet("systemctl --user start " + unit)) {
            std::cerr << ragger::lang::ERR_SYSTEMCTL_START << "\n"; return 1;
        }
        auto pid = main_pid();
        if (!pid.empty()) {
            if (was_active)
                std::cout << std::format(ragger::lang::MSG_RESTARTED_PID, pid) << "\n";
            else
                std::cout << std::format(ragger::lang::MSG_STARTED_PID, pid) << "\n";
        } else {
            std::cout << (was_active ? ragger::lang::MSG_RESTARTED
                                     : ragger::lang::MSG_STARTED) << "\n";
        }
        return 0;
    }

    if (action == "status") {
        if (is_active()) {
            auto pid = main_pid();
            if (!pid.empty())
                std::cout << std::format(ragger::lang::MSG_RUNNING_PID, pid) << "\n";
            else
                std::cout << ragger::lang::MSG_IS_RUNNING << "\n";
        } else {
            std::cout << ragger::lang::MSG_NOT_RUNNING << "\n";
        }
        return 0;
    }

    std::cerr << std::format(ragger::lang::ERR_UNKNOWN_ACTION, action) << "\n";
    return 1;
#else
    std::cerr << ragger::lang::ERR_DAEMON_UNSUPPORTED << "\n";
    return 1;
#endif
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
        // admin flags removed — sudo is the admin gate
        ("yes,y", "Skip confirmation prompts (for scripting)")
        ("dump-payloads", Diskerror::po::value<std::string>(), "Write raw request JSON to this directory (one file per prompt)")
    ;
    opts.add_hidden_options()
        ("command", Diskerror::po::value<std::string>()->default_value("help"), CLI_COMMAND)
        ("args", Diskerror::po::value<std::vector<std::string>>(), CLI_ARGS)
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
        std::cout << "  chat               Interactive chat with memory context\n";
        std::cout << "  mcp                Start MCP server (JSON-RPC over stdin/stdout)\n";
        std::cout << "  useradd <name>     Create a user and issue a bearer token (printed once)\n";
        std::cout << "  usermod <name>     Rotate an existing user's bearer token (printed once)\n";
        std::cout << "  userdel <name>     Remove a user and revoke their token\n";
        std::cout << "  passwd <name>      Set (or clear) a user's web-UI login password\n";
        std::cout << "  add-self           Bootstrap ~/.ragger/token for the current user\n";
        std::cout << "  housekeeping       Trigger housekeeping on running daemon\n";
        std::cout << "  reload             Reload config on running daemon (SIGHUP)\n";
        std::cout << "                     options: --user <name>, --dry-run\n";
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
                std::cerr << ragger::lang::CLI_USAGE_IMPORT << "\n";
                return 1;
            }
            ragger::RaggerMemory memory(db_path, model_dir);
            for (auto& filepath : args) {
                do_import(memory, filepath, collection, min_chunk_size);
            }

        } else if (command == "mcp") {
            ragger::setup_logging(false, false);
            auto mem_ptr = std::make_unique<ragger::RaggerMemory>(db_path, model_dir);
            ragger::run_mcp(*mem_ptr);

        } else if (command == "rebuild-bm25") {
            ragger::setup_logging(false, false);
            ragger::RaggerMemory memory(db_path, model_dir);
            int count = memory.rebuild_bm25();
            std::cout << std::format(ragger::lang::MSG_BM25_REBUILT, count) << "\n";

        } else if (command == "rebuild-embeddings") {
            ragger::setup_logging(false, false);
            
            // Get count first (before loading full memory)
            ragger::RaggerMemory memory_temp(db_path, model_dir);
            int total_count = memory_temp.count();
            memory_temp.close();
            
            // Warning + confirmation prompt
            std::println(ragger::lang::MSG_REBUILD_CONFIRM, total_count);

            bool proceed = false;
            if (opts.count("yes")) {
                proceed = true;
            } else {
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
            } catch (const std::exception& e) {
                std::cerr << std::format(ragger::lang::WARN_BACKUP_FAILED, e.what()) << "\n";
            }

            // Rebuild embeddings
            ragger::RaggerMemory memory(db_path, model_dir);
            int count = memory.rebuild_embeddings();
            std::cout << std::format(ragger::lang::MSG_EMBEDDINGS_REBUILT, count) << "\n";

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
            // Create a new user and issue a bearer token. Token is printed
            // exactly once — caller must save it. Errors if user exists
            // (use `usermod <name>` to rotate an existing user's token).
            ragger::setup_logging(false, false);
            auto args = opts.getParams("args");
            if (args.empty()) {
                std::cerr << ragger::lang::CLI_USAGE_USERADD << "\n";
                return 1;
            }
            std::string username = args[0];

            try {
                ragger::SqliteBackend storage(cfg.resolved_db_path());
                if (storage.get_user_by_username(username)) {
                    std::cerr << std::format(ragger::lang::ERR_USERADD_EXISTS, username) << "\n"
                              << std::format(ragger::lang::ERR_USERADD_EXISTS_HINT, username) << "\n";
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
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << "\n";
                return 1;
            }

        } else if (command == "usermod") {
            // Rotate an existing user's bearer token. Prints the new token once.
            // Errors if the user does not exist.
            ragger::setup_logging(false, false);
            auto args = opts.getParams("args");
            if (args.empty()) {
                std::cerr << ragger::lang::CLI_USAGE_USERMOD << "\n";
                return 1;
            }
            std::string username = args[0];

            try {
                ragger::SqliteBackend storage(cfg.resolved_db_path());
                if (!storage.get_user_by_username(username)) {
                    std::cerr << std::format(ragger::lang::ERR_USERMOD_MISSING, username) << "\n"
                              << std::format(ragger::lang::ERR_USERMOD_MISSING_HINT, username) << "\n";
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
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << "\n";
                return 1;
            }

        } else if (command == "userdel") {
            ragger::setup_logging(false, false);
            auto args = opts.getParams("args");
            if (args.empty()) {
                std::cerr << ragger::lang::CLI_USAGE_USERDEL << "\n";
                return 1;
            }
            std::string username = args[0];

            try {
                ragger::SqliteBackend storage(cfg.resolved_db_path());
                ragger::userdel(storage, username);
                std::println(ragger::lang::MSG_USER_REMOVED, username);
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
                std::cerr << ragger::lang::ERR_UNKNOWN_USER << "\n";
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
                ragger::SqliteBackend backend(reg_db);
                std::string token_hash = ragger::hash_token(token);
                auto existing = backend.get_user_by_username(username);
                if (existing) {
                    if (existing->token_hash != token_hash)
                        backend.update_user_token(username, token_hash);
                    std::println(ragger::lang::MSG_USER_IN_DB, existing->id);
                } else {
                    int user_id = backend.create_user(username, token_hash);
                    std::println(ragger::lang::MSG_USER_REGISTERED, user_id);
                }
            } catch (const std::exception& e) {
                std::println(ragger::lang::WARN_DB_DEFERRED, e.what());
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
            // Set (or clear) a user's web-UI login password. Only needed for
            // remote users who log in through the browser — local (127.0.0.1
            // / unix socket) sessions auto-authenticate as the daemon owner.
            // Empty password clears web-UI access for that user.
            ragger::setup_logging(false, false);
            auto args = opts.getParams("args");
            if (args.empty()) {
                std::cerr << ragger::lang::CLI_USAGE_PASSWD << "\n";
                return 1;
            }
            std::string target_user = args[0];

            try {
                ragger::SqliteBackend umgr(cfg.resolved_db_path());
                auto user_info = umgr.get_user_by_username(target_user);
                if (!user_info) {
                    std::cerr << std::format(ragger::lang::ERR_USERMOD_MISSING, target_user) << "\n"
                              << std::format(ragger::lang::ERR_PASSWD_MISSING_HINT, target_user) << "\n";
                    return 1;
                }

                std::string new_pass = read_password(ragger::lang::PROMPT_NEW_PASSWORD);
                if (new_pass.empty()) {
                    umgr.set_user_password(target_user, "");
                    std::println(ragger::lang::MSG_PASSWORD_CLEARED, target_user);
                } else {
                    std::string confirm = read_password(ragger::lang::PROMPT_CONFIRM_PASSWORD);
                    if (new_pass != confirm) {
                        std::cerr << ragger::lang::ERR_PASSWORDS_DIFFER << "\n";
                        return 1;
                    }
                    std::string hash = ragger::hash_password(new_pass);
                    umgr.set_user_password(target_user, hash);
                    std::println(ragger::lang::MSG_PASSWORD_SET, target_user);
                }
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << "\n";
                return 1;
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
