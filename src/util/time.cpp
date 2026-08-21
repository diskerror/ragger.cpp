/**
 * Timestamp helpers implementation. See include/ragger/util/time.h.
 */
#include "util/time.h"

#include <chrono>
#include <iomanip>
#include <sstream>

namespace ragger {

std::string db_timestamp(std::time_t when) {
    std::tm local{};
    localtime_r(&when, &local);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%F %T", &local);
    return std::string(buf);
}

std::string db_timestamp() {
    return db_timestamp(
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
}

std::optional<std::time_t> parse_db_timestamp(const std::string& s) {
    std::tm tm{};
    std::istringstream ss(s);
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    if (ss.fail()) return std::nullopt;
    tm.tm_isdst = -1; // let mktime determine DST for the local zone
    return std::mktime(&tm);
}

int64_t db_epoch(std::time_t when) {
    return static_cast<int64_t>(when);
}

int64_t db_epoch() {
    return db_epoch(
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
}

std::string file_stamp() {
    auto now = std::chrono::system_clock::now();
    auto tt  = std::chrono::system_clock::to_time_t(now);
    auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                   now.time_since_epoch()).count() % 1000;
    std::tm local{};
    localtime_r(&tt, &local);
    std::ostringstream out;
    out << std::put_time(&local, "%Y%m%d_%H%M%S")
        << '_' << std::setfill('0') << std::setw(3) << ms;
    return out.str();
}

} // namespace ragger
