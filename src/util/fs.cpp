/**
 * Filesystem helpers implementation. See include/ragger/util/fs.h.
 */
#include "util/fs.h"

#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include <pwd.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace ragger {

std::string read_file_to_string(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return {};
    return std::string((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
}

std::string home_dir() {
    const char* home = std::getenv("HOME");
    if (home && *home) return home;
    // Fallback: passwd entry for the current uid (e.g. when HOME is unset
    // under a daemon).
    struct passwd* pw = getpwuid(getuid());
    if (pw && pw->pw_dir) return pw->pw_dir;
    return {};
}

std::string collapse_home(const std::string& path) {
    std::string home = home_dir();
    if (home.empty()) return path;
    // Ensure home doesn't end with '/' for a clean prefix check.
    if (home.back() == '/') home.pop_back();
    if (path.size() > home.size() &&
        path.compare(0, home.size(), home) == 0 &&
        path[home.size()] == '/') {
        return "~" + path.substr(home.size());
    }
    if (path == home) return "~";
    return path;
}

// Testing-only base-dir override, set once at startup from the hidden
// --ragger-base CLI flag. No env var equivalent — CLI-only, by design.
static std::string _ragger_base_override;

void set_ragger_base_override(const std::string& path) {
    _ragger_base_override = path;
}

std::string ragger_base_dir() {
    if (!_ragger_base_override.empty()) return _ragger_base_override;
    std::string home = home_dir();
    if (home.empty()) return {};
    return home + "/.ragger";
}

namespace {
std::string shell_quote_path(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}
}  // namespace

std::string archive_db_files(const std::string& db_path) {
    fs::path src(db_path);
    std::string stem = src.stem().string();               // e.g. "memories"
    std::string ts;
    {
        std::time_t now = std::time(nullptr);
        std::tm tmv{};
#if defined(_WIN32)
        localtime_s(&tmv, &now);
#else
        localtime_r(&now, &tmv);
#endif
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tmv);
        ts = buf;
    }
    std::string prefix = stem + "_BACKUP_" + ts;

    fs::path parent = src.parent_path();
    std::string db_filename = src.filename().string();          // "memories.db"
    std::string wal_filename = db_filename + "-wal";
    std::string shm_filename = db_filename + "-shm";
    bool has_wal = fs::exists(parent / wal_filename);
    bool has_shm = fs::exists(parent / shm_filename);

    // -- tar.gz --
    {
        fs::path tar_path = parent / (prefix + ".tar.gz");
        std::string cmd = "tar -czf " + shell_quote_path(tar_path.string()) +
            " -C " + shell_quote_path(parent.string()) +
            " " + shell_quote_path(db_filename);
        if (has_wal) cmd += " " + shell_quote_path(wal_filename);
        if (has_shm) cmd += " " + shell_quote_path(shm_filename);
        if (std::system(cmd.c_str()) == 0 && fs::exists(tar_path)) {
            return tar_path.string();
        }
    }

    // -- zip fallback --
    {
        fs::path zip_path = parent / (prefix + ".zip");
        std::string cmd = "cd " + shell_quote_path(parent.string()) +
            " && zip -q " + shell_quote_path(zip_path.string()) +
            " -j " + shell_quote_path(db_filename);
        if (has_wal) cmd += " " + shell_quote_path(wal_filename);
        if (has_shm) cmd += " " + shell_quote_path(shm_filename);
        if (std::system(cmd.c_str()) == 0 && fs::exists(zip_path)) {
            return zip_path.string();
        }
    }

    // -- plain copy fallback --
    fs::path copy_path = parent / (prefix + ".db");
    if (fs::exists(copy_path)) {
        throw std::runtime_error(
            "backup target already exists: " + copy_path.string());
    }
    fs::copy_file(src, copy_path);
    if (has_wal) fs::copy_file(parent / wal_filename, parent / (prefix + ".db-wal"));
    if (has_shm) fs::copy_file(parent / shm_filename, parent / (prefix + ".db-shm"));
    return copy_path.string();
}

} // namespace ragger
