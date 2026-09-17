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

std::string archive_db_files(const std::string& db_path,
                              const std::vector<std::string>& extra_db_paths) {
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

    // Collect every file to bundle as (containing_dir, filename) pairs: the
    // primary DB + its -wal/-shm siblings, then the same for each extra DB
    // path (e.g. stats.db). A path that doesn't exist is silently skipped --
    // an optional sibling DB (or one with no WAL/SHM at rest) isn't an error.
    struct FileRef { fs::path dir; std::string name; };
    std::vector<FileRef> files;
    auto add_db_and_siblings = [&](const std::string& path) {
        fs::path p(path);
        if (!fs::exists(p)) return;
        fs::path dir = p.parent_path();
        std::string name = p.filename().string();
        files.push_back({dir, name});
        for (const char* suffix : {"-wal", "-shm"}) {
            fs::path sib = dir / (name + suffix);
            if (fs::exists(sib)) files.push_back({dir, name + suffix});
        }
    };
    add_db_and_siblings(db_path);
    for (const auto& extra : extra_db_paths) add_db_and_siblings(extra);

    // -- tar.gz -- (-C per entry lets bsdtar/GNU tar pull files from
    // different source directories into one flat archive)
    {
        fs::path tar_path = parent / (prefix + ".tar.gz");
        std::string cmd = "tar -czf " + shell_quote_path(tar_path.string());
        for (const auto& f : files) {
            cmd += " -C " + shell_quote_path(f.dir.string()) +
                   " " + shell_quote_path(f.name);
        }
        if (std::system(cmd.c_str()) == 0 && fs::exists(tar_path)) {
            return tar_path.string();
        }
    }

    // -- zip fallback -- (zip has no per-file -C; cd into each dir and
    // append, since the source dirs may differ across files)
    {
        fs::path zip_path = parent / (prefix + ".zip");
        bool ok = true;
        for (const auto& f : files) {
            std::string cmd = "cd " + shell_quote_path(f.dir.string()) +
                " && zip -q " + shell_quote_path(zip_path.string()) +
                " -j " + shell_quote_path(f.name);
            if (std::system(cmd.c_str()) != 0) { ok = false; break; }
        }
        if (ok && fs::exists(zip_path)) {
            return zip_path.string();
        }
    }

    // -- plain copy fallback (primary DB + its siblings only -- extras are
    // best-effort and skipped here since there's no archive to bundle them
    // into; tar/zip failing at all is already an unusual environment) --
    std::string db_filename = src.filename().string();
    std::string wal_filename = db_filename + "-wal";
    std::string shm_filename = db_filename + "-shm";
    bool has_wal = fs::exists(parent / wal_filename);
    bool has_shm = fs::exists(parent / shm_filename);
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
