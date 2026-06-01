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

} // namespace ragger
