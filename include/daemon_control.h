/**
 * Daemon / service control — manage the user-level `ragger` service via
 * launchctl (macOS) or systemctl --user (Linux). Split out of main.cpp so the
 * CLI entry point isn't carrying ~300 lines of platform service plumbing.
 */
#pragma once

#include <string>

namespace ragger {

/// Manage the user-level ragger service. `action` is one of start / stop /
/// restart / status. Returns a process exit code (0 = success). On an
/// unsupported platform it logs an error and returns 1.
int daemon_control(const std::string& action);

} // namespace ragger
