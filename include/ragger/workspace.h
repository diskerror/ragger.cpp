/**
 * Workspace / persona loading — assembles the system-prompt context from the
 * user's workspace files. Shared by the CLI chat REPL (Chat) and the HTTP
 * server's session manager (ChatSessionManager) so both produce an identical
 * persona block.
 *
 * Load order: the configured system_prompt_file first (default ~/.ragger/
 * SYSTEM.md), then SOUL.md / USER.md / MEMORY.md from persona_dir. Each file's
 * trailing whitespace is trimmed, empties are skipped, and sections are joined
 * with "\n\n---\n\n".
 */
#pragma once

#include <optional>
#include <string>

namespace ragger {

/// Load and concatenate the workspace/persona files into a single string.
///
/// When `char_budget` is set, files are loaded in priority order until the
/// budget is exhausted; the file that would overflow is truncated on a
/// paragraph boundary (whole paragraphs only) with a "[... <file> truncated
/// ...]" marker, and no further files are loaded. When `char_budget` is
/// nullopt the files are loaded in full.
std::string load_workspace(std::optional<int> char_budget);

} // namespace ragger
