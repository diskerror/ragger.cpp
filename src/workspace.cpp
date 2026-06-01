/**
 * Workspace / persona loading implementation. See include/ragger/workspace.h.
 *
 * Previously this logic was duplicated in Chat::load_workspace_files
 * (src/chat.cpp) and ChatSessionManager::load_workspace_files
 * (src/chat_sessions.cpp); the two copies had already drifted apart.
 */
#include "ragger/workspace.h"

#include "ragger/config.h"
#include "ragger/lang.h"
#include "ragger/util/fs.h"
#include "diskerror/logger.h"

#include <filesystem>
#include <format>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

namespace ragger {

namespace {

/// Trim trailing whitespace; returns empty if the content is all whitespace.
std::string rtrim(std::string s) {
    size_t end = s.find_last_not_of(" \t\r\n");
    if (end == std::string::npos) return {};
    s.resize(end + 1);
    return s;
}

/// The ordered list of workspace file paths: system_prompt_file first, then
/// the persona files, skipping any that resolve to the system_prompt_file.
std::vector<std::string> workspace_paths(const std::string& sys_path,
                                         const std::string& persona_dir) {
    std::string sys_norm = fs::path(sys_path).lexically_normal().string();
    std::vector<std::string> paths;
    paths.push_back(sys_path);
    for (const char* fname : {"SOUL.md", "USER.md", "MEMORY.md"}) {
        std::string p = persona_dir + "/" + fname;
        if (fs::path(p).lexically_normal().string() != sys_norm) paths.push_back(p);
    }
    return paths;
}

/// Keep as many leading whole paragraphs (separated by blank lines) of
/// `content` as fit within `limit` characters.
std::string truncate_paragraphs(const std::string& content, int limit) {
    std::vector<std::string> paragraphs;
    std::istringstream stream(content);
    std::string buffer;
    for (std::string line; std::getline(stream, line);) {
        if (line.empty() && !buffer.empty()) {
            paragraphs.push_back(buffer);
            buffer.clear();
        } else {
            if (!buffer.empty()) buffer += "\n";
            buffer += line;
        }
    }
    if (!buffer.empty()) paragraphs.push_back(buffer);

    std::string out;
    for (const auto& p : paragraphs) {
        if (static_cast<int>(out.size() + p.size() + 2) > limit) break;
        if (!out.empty()) out += "\n\n";
        out += p;
    }
    return out;
}

} // namespace

std::string load_workspace(std::optional<int> char_budget) {
    const auto& cfg = config();
    std::string persona_dir = expand_path(cfg.persona_dir);
    std::string sys_path = expand_path(cfg.system_prompt_file);

    std::vector<std::string> sections;
    int total_chars = 0;

    for (const auto& path : workspace_paths(sys_path, persona_dir)) {
        if (!fs::exists(path)) continue;
        std::string content = rtrim(read_file_to_string(path));
        if (content.empty()) continue;

        if (char_budget) {
            // Account for the "\n\n---\n\n" (8 chars) joining each section.
            int separator_overhead = static_cast<int>(sections.size()) * 8;
            int remaining = *char_budget - total_chars - separator_overhead;
            if (remaining <= 0) break;

            if (static_cast<int>(content.size()) > remaining) {
                std::string truncated = truncate_paragraphs(content, remaining);
                if (!truncated.empty()) {
                    std::string label = fs::path(path).filename().string();
                    truncated += "\n\n[... " + label + " truncated ...]";
                    sections.push_back(truncated);
                    total_chars += static_cast<int>(truncated.size());
                }
                break; // budget reached — stop loading further files
            }
        }

        sections.push_back(content);
        total_chars += static_cast<int>(content.size());
    }

    if (sections.empty()) {
        Diskerror::logger::info(std::format(lang::MSG_NO_PERSONA_FILES, persona_dir));
    }

    std::string result;
    for (size_t i = 0; i < sections.size(); ++i) {
        if (i > 0) result += "\n\n---\n\n";
        result += sections[i];
    }
    return result;
}

} // namespace ragger
