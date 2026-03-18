#include "ragger/config.h"
#include <cstdlib>

namespace ragger {

std::string expand_path(const std::string& path) {
    if (path.empty() || path[0] != '~') return path;
    const char* home = std::getenv("HOME");
    if (!home) return path;
    return std::string(home) + path.substr(1);
}

std::string sqlite_db_path() { return expand_path(SQLITE_PATH); }
std::string log_dir_path()   { return expand_path(LOG_DIR); }

} // namespace ragger
