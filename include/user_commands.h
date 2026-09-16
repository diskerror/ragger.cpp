/**
 * User-management CLI commands — useradd/usermod/userdel/add-self/passwd,
 * plus the token-provisioning + password-prompt helpers they share.
 *
 * Extracted from main.cpp so the CLI dispatcher stays a thin switchboard.
 * Each cmd_* function owns one verb end-to-end (arg parsing via the passed
 * ProgramOptions, UserStore access, messages, error handling) and returns
 * the process exit code main() should return for that verb.
 */
#pragma once

#include <string>
#include <utility>

namespace Diskerror { class ProgramOptions; }

namespace ragger {

/// Prompt on stdout, read a line from stdin with terminal echo suppressed
/// (used for password entry). Trims trailing \\r/\\n/space.
std::string read_password(const std::string &prompt);

/// Provision a user: create ~/.ragger/ and a token file for `username`.
/// Returns {token, created}; if a token file already exists, returns the
/// existing token with created=false. `home_override` is for testing —
/// production callers leave it empty and it's resolved via getpwnam.
std::pair<std::string, bool> provision_user(const std::string &username,
                                            const std::string &home_override = "");

/// `ragger useradd <name>` — create a new user + bearer token.
int cmd_useradd(Diskerror::ProgramOptions &opts, const std::string &db_path);

/// `ragger usermod <name>` — rotate an existing user's bearer token.
int cmd_usermod(Diskerror::ProgramOptions &opts, const std::string &db_path);

/// `ragger userdel <name>` — remove a user.
int cmd_userdel(Diskerror::ProgramOptions &opts, const std::string &db_path);

/// `ragger add-self` — provision (or reuse) a token for the OS-login user
/// running the command, and register it in the DB.
int cmd_add_self(const std::string &db_path);

/// `ragger passwd <name>` — set or clear a user's web-UI login password.
int cmd_passwd(Diskerror::ProgramOptions &opts, const std::string &db_path);

} // namespace ragger
