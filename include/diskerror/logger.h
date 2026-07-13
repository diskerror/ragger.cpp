//
// Created by Reid Woodbury Jr on 4/28/26.
//

#ifndef DISKERROR_LOGGER_H
#define DISKERROR_LOGGER_H

#include <functional>
#include <mutex>
#include <string>

namespace Diskerror {

// Built-in log rotation, no external tool or root privilege required —
// works identically on macOS, Linux, and (whenever it's ported) Windows,
// since it's plain std::filesystem rather than an OS-specific facility
// (newsyslog/logrotate/Scheduled Task all behave differently per platform).
//
// Each log call opens the file, appends one line, and closes it again
// (rather than holding a persistent handle for the process lifetime).
// This is slightly more I/O per call, but it means the file is never held
// open across a rotation, the file is trivially re-created if something
// external ever does remove/rename it out from under the daemon, and a
// fresh log_max_size_mb/log_max_age_days check can piggyback on every
// append with no separate timer/thread needed.
class logger {
    static unsigned int instanceCount;
    static std::mutex logMutex;
    static std::string logFileName;
    static long maxSizeBytes;   // 0 = size-based rotation disabled
    static int  maxAgeDays;     // 0 = age-based backup cleanup disabled

    static void append_line(const std::string& tag, const std::string& message);
    static void rotate_locked();          // caller must hold logMutex
    static void sweep_backups_locked();   // caller must hold logMutex; called after rotate_locked()

public:
    // maxSizeMb: rotate activity.log -> activity.log.<timestamp> once it
    //   reaches this size (MB). 0 disables size-based rotation.
    // maxAgeDays: delete rotated backups older than this many days. 0
    //   disables age-based cleanup (backups accumulate forever).
    // Creates the log directory and the log file itself if either is
    // missing (fails fast with a thrown runtime_error if the path is
    // genuinely unwritable, e.g. permission denied).
    logger(const std::string& logFileName, const std::string& level = "info",
           long maxSizeMb = 0, int maxAgeDays = 0);
    ~logger();

    static std::string get_timestamp();

    static std::function<void(std::string const&)> trace;
    static std::function<void(std::string const&)> debug;
    static std::function<void(std::string const&)> info;
    static std::function<void(std::string const&)> warn;
    static std::function<void(std::string const&)> error;
    static std::function<void(std::string const&)> critical;
};

} // namespace Diskerror

#endif //DISKERROR_LOGGER_H
