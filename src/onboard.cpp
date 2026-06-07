/**
 * `ragger onboard` — guided first-run setup. See include/ragger/onboard.h.
 */

#include "ragger/onboard.h"

#include "ragger/config.h"
#include "ragger/daemon_control.h"
#include "ragger/recipe_cli.h"
#include "ragger/util/fs.h"

#include <curl/curl.h>
#include <nlohmann_json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <print>
#include <regex>
#include <sstream>
#include <string>
#include <unistd.h>

namespace ragger {
namespace {

namespace fs = std::filesystem;
using json   = nlohmann::json;

// --------------------------------------------------------------------------
// Small console helpers — line-based, no raw-mode termios. The recipe step
// reuses recipe_cli.cpp, which already handles arrow-key picking.
// --------------------------------------------------------------------------

void hr() { std::println(""); std::println("─────────────────────────────────────────────"); std::println(""); }

void section(const std::string& title) {
    hr();
    std::println("⌘ {}", title);
    std::println("");
}

/// Prompt for a single line. Empty input → default. Returns the user's
/// answer (trimmed) or the default if they just hit Enter.
std::string prompt(const std::string& label, const std::string& def = "") {
    if (def.empty()) std::print("{}: ", label);
    else             std::print("{} [{}]: ", label, def);
    std::cout.flush();
    std::string s;
    if (!std::getline(std::cin, s)) return def;
    // Trim
    auto issp = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!s.empty() && issp((unsigned char)s.back()))  s.pop_back();
    while (!s.empty() && issp((unsigned char)s.front())) s.erase(s.begin());
    return s.empty() ? def : s;
}

bool prompt_yn(const std::string& label, bool def) {
    while (true) {
        std::string ans = prompt(label + (def ? " (Y/n)" : " (y/N)"),
                                 def ? "Y" : "N");
        if (ans.empty()) return def;
        char c = static_cast<char>(std::tolower((unsigned char)ans[0]));
        if (c == 'y') return true;
        if (c == 'n') return false;
        std::println("  Please answer y or n.");
    }
}

// --------------------------------------------------------------------------
// settings.ini editing — line-based so comments and section ordering survive.
// --------------------------------------------------------------------------

std::string settings_path() {
    return expand_path("~/.ragger/settings.ini");
}

std::string read_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) return "";
    std::stringstream ss; ss << in.rdbuf();
    return ss.str();
}

bool write_file(const std::string& path, const std::string& body) {
    std::ofstream out(path);
    if (!out) return false;
    out << body;
    return out.good();
}

/// Replace `key = value` (anywhere in the file, first match) with the new
/// value. Preserves leading whitespace and any inline `# comment` tail.
/// Returns true if a replacement happened; false if the key wasn't found.
bool set_ini_value(std::string& body, const std::string& key,
                   const std::string& value) {
    std::regex re("(^|\\n)([ \\t]*" + key + "[ \\t]*=)[ \\t]*([^\\n#]*)(#[^\\n]*)?",
                  std::regex::ECMAScript);
    std::smatch m;
    if (!std::regex_search(body, m, re)) return false;
    std::string replacement =
        m[1].str() + m[2].str() + " " + value +
        (m[4].matched ? std::string("  ") + m[4].str() : std::string(""));
    body = m.prefix().str() + replacement + m.suffix().str();
    return true;
}

/// Read the current text value of a key (first match), trimming whitespace
/// and any inline comment. Empty string if not present.
std::string get_ini_value(const std::string& body, const std::string& key) {
    std::regex re("(^|\\n)[ \\t]*" + key + "[ \\t]*=[ \\t]*([^\\n#]*)",
                  std::regex::ECMAScript);
    std::smatch m;
    if (!std::regex_search(body, m, re)) return "";
    std::string v = m[2].str();
    auto issp = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!v.empty() && issp((unsigned char)v.back()))  v.pop_back();
    while (!v.empty() && issp((unsigned char)v.front())) v.erase(v.begin());
    return v;
}

bool ini_bool(const std::string& v, bool fallback) {
    if (v.empty()) return fallback;
    std::string s; s.reserve(v.size());
    for (char c : v) s += static_cast<char>(std::tolower((unsigned char)c));
    if (s == "true" || s == "1" || s == "yes" || s == "on")  return true;
    if (s == "false"|| s == "0" || s == "no"  || s == "off") return false;
    return fallback;
}

// --------------------------------------------------------------------------
// Inference probe — GET /v1/models from a candidate endpoint.
// --------------------------------------------------------------------------

size_t curl_collect(char* p, size_t s, size_t n, void* ud) {
    static_cast<std::string*>(ud)->append(p, s * n);
    return s * n;
}

/// Returns the list of model IDs the endpoint reports, or empty on failure.
std::vector<std::string> probe_models(const std::string& base_url) {
    std::vector<std::string> ids;
    CURL* curl = curl_easy_init();
    if (!curl) return ids;
    std::string url = base_url + "/models";
    std::string buf;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 4L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_collect);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
    CURLcode rc = curl_easy_perform(curl);
    long http = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http);
    curl_easy_cleanup(curl);
    if (rc != CURLE_OK || http != 200) return ids;
    try {
        json j = json::parse(buf);
        for (const auto& m : j.value("data", json::array())) {
            if (m.is_object() && m.contains("id")) {
                ids.push_back(m["id"].get<std::string>());
            }
        }
    } catch (...) {}
    return ids;
}

/// Numbered list with prompt for a choice. Returns the selected string, or
/// empty if the user typed nothing (caller treats as skip).
std::string pick_from_list(const std::vector<std::string>& options,
                           const std::string& label) {
    for (size_t i = 0; i < options.size(); ++i) {
        std::println("  [{}] {}", i + 1, options[i]);
    }
    std::println("  [0] skip / leave empty");
    while (true) {
        std::string ans = prompt(label, "0");
        try {
            int n = std::stoi(ans);
            if (n == 0) return "";
            if (n >= 1 && n <= (int)options.size()) return options[n - 1];
        } catch (...) {}
        std::println("  Pick a number between 0 and {}.", options.size());
    }
}

// --------------------------------------------------------------------------
// Embedding model presence
// --------------------------------------------------------------------------

bool embedding_model_present() {
    std::string mdir = expand_path("~/.ragger/models/all-MiniLM-L6-v2");
    return fs::exists(mdir + "/model.onnx");
}

} // namespace

// =========================================================================
// run_onboard — drives all sections sequentially.
// =========================================================================

int run_onboard(const std::vector<std::string>& /*args*/,
                const std::string& db_path) {
    const std::string ini_path = settings_path();
    std::string body = read_file(ini_path);
    if (body.empty()) {
        std::println(stderr,
            "No settings file at {} — run ./scripts/install.sh first.",
            ini_path);
        return 2;
    }

    std::println("");
    std::println("Ragger onboarding");
    std::println("=================");
    std::println("");
    std::println("This walks through the choices that don't have one safe default.");
    std::println("Each section shows the current value; press Enter to keep it.");
    std::println("Re-run `ragger onboard` any time to change a single section.");

    // -------- 1. Storage --------
    section("Storage");
    std::println("Ragger keeps everything under ~/.ragger/:");
    std::println("  db:        {}", config().resolved_db_path());
    std::println("  models:    {}", expand_path("~/.ragger/models"));
    std::println("  recipes:   {}",
                 config().recipes_dir.empty()
                     ? expand_path("~/.ragger/recipes") : config().recipes_dir);
    std::println("");
    std::println("To put any of these on a different disk, symlink the path:");
    std::println("  ln -s /mnt/big/ragger ~/.ragger");
    std::println("(Onboarding leaves ~/.ragger in place — it just tells you the option.)");

    // -------- 2. Capture & context --------
    section("Turn capture & context");
    bool cur_cap = ini_bool(get_ini_value(body, "capture_turns"), true);
    bool cur_bld = ini_bool(get_ini_value(body, "build_context"), false);
    std::println("capture_turns — write side");
    std::println("  When on, every conversation turn the agent pushes via");
    std::println("  POST /turn or the capture_turn MCP tool is saved to memory.");
    std::println("  Default: on (recommended — this is what builds your memory).");
    std::println("");
    bool new_cap = prompt_yn("Enable turn capture?", cur_cap);
    std::println("");
    std::println("build_context — read side");
    std::println("  When on, GET /session/<id> and the build_context MCP tool");
    std::println("  assemble a recipe-shaped summary payload for the agent to");
    std::println("  inject as context. Only meaningful when capture_turns is on.");
    std::println("  Default: off — most agents request context explicitly rather");
    std::println("  than having it injected automatically every turn.");
    std::println("");
    bool new_bld = prompt_yn("Enable automatic context assembly?",
                             cur_bld && new_cap);
    if (new_bld && !new_cap) {
        std::println("  (build_context needs capture_turns on — turning it off.)");
        new_bld = false;
    }
    if (new_cap != cur_cap)
        set_ini_value(body, "capture_turns", new_cap ? "true" : "false");
    if (new_bld != cur_bld)
        set_ini_value(body, "build_context", new_bld ? "true" : "false");

    // -------- 3. Default recipe --------
    section("Default build_context recipe");
    std::println("Recipes shape how raw turns + summaries are layered into a");
    std::println("payload. Pick one now or skip (run `ragger recipe` later).");
    std::println("");
    if (prompt_yn("Open the recipe picker?", new_bld)) {
        run_recipe_cli({}, db_path);   // reuses the existing TUI picker
    }

    // -------- 4. Inference endpoint --------
    section("Inference endpoint (for summarization)");
    std::string cur_url   = get_ini_value(body, "api_url");
    std::string cur_model = get_ini_value(body, "model");
    if (cur_url.empty()) cur_url = "http://localhost:1234/v1";
    std::println("Ragger summarizes turns with an OpenAI-compatible endpoint.");
    std::println("LM Studio's default is http://localhost:1234/v1.");
    std::println("");
    std::string new_url = prompt("Endpoint URL (blank to skip)", cur_url);
    std::string new_model = cur_model;
    if (!new_url.empty()) {
        std::println("");
        std::println("Probing {}...", new_url);
        auto ids = probe_models(new_url);
        if (ids.empty()) {
            std::println("  No response (or empty model list). You can still");
            std::println("  fill the model name by hand — it'll work once the");
            std::println("  endpoint is up. Until then summaries get the draft");
            std::println("  fallback (tagged `draft`, rewritten on retry).");
            std::println("");
            new_model = prompt("Model name to use (blank to leave empty)", cur_model);
        } else {
            std::println("  Found {} models.", ids.size());
            std::println("");
            std::string picked = pick_from_list(ids,
                "Pick the summarization model");
            if (!picked.empty()) new_model = picked;
        }
    }
    set_ini_value(body, "api_url", new_url);
    set_ini_value(body, "model",   new_model);

    // -------- 5. Write settings --------
    section("Saving settings");
    if (!write_file(ini_path, body)) {
        std::println(stderr, "Failed to write {}", ini_path);
        return 1;
    }
    std::println("Wrote {}", ini_path);

    // -------- 6. Embedding model --------
    section("Embedding model");
    if (embedding_model_present()) {
        std::println("✓ all-MiniLM-L6-v2 already present at ~/.ragger/models/");
    } else {
        std::println("The all-MiniLM-L6-v2 ONNX model (~90 MB) isn't installed.");
        std::println("install.sh handles this in one pass:");
        std::println("");
        std::println("  cd /path/to/ragger && ./scripts/install.sh");
        std::println("");
        std::println("(Re-running install.sh is safe — it skips anything already done.)");
    }

    // -------- 7. Daemon --------
    section("Daemon");
    std::println("The daemon owns the summarizer worker, the HTTP/MCP listeners,");
    std::println("and the housekeeping loop. It runs under your user account.");
    std::println("");
    if (prompt_yn("Start (or restart) the daemon now?", true)) {
        // daemon_control("restart") is the safe option — works whether the
        // daemon was already running or not.
        int rc = daemon_control("restart");
        if (rc != 0) {
            std::println("  daemon_control returned {}; you can run `ragger start`",
                         rc);
            std::println("  yourself to see the full message.");
        }
    } else {
        std::println("  Start it later with: ragger start");
    }

    hr();
    std::println("Done.");
    std::println("");
    std::println("Next steps:");
    std::println("  • Wire Ragger into Claude Code:  ./scripts/install-claude-code.sh");
    std::println("  • Wire into Claude Desktop:      ./scripts/install-claude-desktop.sh");
    std::println("  • Tail what the daemon is doing: tail -f ~/.ragger/activity.log");
    std::println("");
    return 0;
}

} // namespace ragger
