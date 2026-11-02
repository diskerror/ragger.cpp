/**
 * Daemon / service control — manage the user-level `ragger` service via
 * launchctl (macOS) or systemctl --user (Linux). Split out of main.cpp so the
 * CLI entry point isn't carrying ~300 lines of platform service plumbing.
 */
#pragma once

#include <sys/types.h>

#include <string>
#include <vector>

namespace ragger {

/// Manage the user-level ragger service. `action` is one of start / stop /
/// restart / status. Returns a process exit code (0 = success). On an
/// unsupported platform it logs an error and returns 1.
int daemon_control(const std::string& action);

/// S21 (C m18): true iff some argument TOKEN equals "serve" or "mcp" — the
/// only processes the stop/restart sweep may signal (daemon and MCP servers;
/// an unrelated `ragger re-embed` in another terminal must survive). Options
/// before the verb (--ragger-base X, …) do not matter; a token must equal the
/// verb exactly.
bool is_sweepable_cmdline(const std::vector<std::string>& argv);

/// S22 (C m8): PID of a live ragger daemon (first readable
/// /tmp/ragger/server-*.pid whose process answers kill(pid, 0)), or 0.
/// The single PID-file scanner; every former hand-rolled loop calls this.
pid_t find_live_daemon_pid();

/// S23 (C m9): argv for the in-place daemon re-exec — strips the host/port
/// CLI overrides (and their values) so the restarted daemon takes them from
/// config: `-p V`, `--port V`, `--host V`, `-H V` (flag + value), and the
/// attached forms `--port=V`, `--host=V`, `-pV`, `-HV`. A trailing flag with
/// no value drops only the flag. Everything else passes through untouched.
std::vector<std::string> filter_restart_argv(const std::vector<std::string>& argv);

} // namespace ragger
