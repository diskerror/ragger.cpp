/**
 * Filesystem helpers implementation. See include/ragger/util/fs.h.
 */
#include "util/fs.h"

#include <cstdlib>
#include <fstream>
#include <iterator>

#include <pwd.h>
#include <unistd.h>

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

} // namespace ragger
