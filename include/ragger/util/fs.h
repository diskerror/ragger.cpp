/**
 * Filesystem helpers — small utilities that were previously open-coded in
 * several modules (reading a file into a string, resolving the user's home
 * directory).
 *
 * home_dir() is the single source of truth for "where is HOME": it honors the
 * HOME environment variable (so a relocated home such as an external-drive
 * mount is respected) and falls back to the passwd database when HOME is unset
 * (e.g. under a daemon). `expand_path()` (config.h) is built on top of it.
 */
#pragma once

#include <string>

namespace ragger {

/// Read an entire file into a string. Returns empty string if the file does
/// not exist or cannot be opened.
std::string read_file_to_string(const std::string& path);

/// The user's home directory: $HOME if set, else the passwd entry for the
/// current uid. Returns empty string if neither is available.
std::string home_dir();

/// Set the Ragger base-directory override. Testing-only escape hatch, set
/// once at startup from the hidden `--ragger-base <path>` CLI flag before
/// any config/path resolution happens. No environment-variable equivalent
/// by design — CLI-only. Empty (default, unset) means "use $HOME/.ragger".
void set_ragger_base_override(const std::string& path);

/// The Ragger base directory: the `--ragger-base` override if set, else
/// `$HOME/.ragger`. This is the ONE place every other Ragger path (DB,
/// settings.ini, logs, models, recipes, formats, socket, token) is rooted
/// from — so pointing this at a throwaway directory relocates the entire
/// on-disk footprint for testing, with nothing else to configure.
std::string ragger_base_dir();

} // namespace ragger
