/**
 * Chat implementation — conversation REPL with persistence
 */
#include "ragger/chat.h"
#include "ragger/config.h"
#include "ragger/lang.h"
#include "ragger/memory.h"
#include "ragger/inference.h"
#include <unistd.h>  // fork, _exit
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <regex>
#include "nlohmann_json.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace ragger::lang;

namespace ragger {

// Persona sizing threshold — below this context size, apply percentage-based sizing
static const int PERSONA_SIZING_THRESHOLD = 32768;

Chat::Chat(RaggerMemory& memory, InferenceClient& inference, const std::string& model)
    : memory_(memory)
    , inference_(inference)
    , model_(model)
{
    const auto& cfg = config();
    
    // Load config
    store_turns_ = cfg.chat_store_turns;
    summarize_on_pause_ = cfg.chat_summarize_on_pause;
    pause_minutes_ = cfg.chat_pause_minutes;
    summarize_on_quit_ = cfg.chat_summarize_on_quit;
    max_memory_results_ = cfg.chat_max_memory_results;
    
    // Initialize activity timestamp
    update_activity();
}

std::string Chat::load_workspace_files(int max_context) {
    /**
     * Load workspace MD files with priority-based truncation.
     * 
     * Priority order: SOUL.md > USER.md > AGENTS.md > TOOLS.md > MEMORY.md
     * 
     * For small contexts (< 32768 tokens):
     *   - Calculate persona budget: max_context * chars_per_token * (persona_pct / 100)
     *   - Apply user ceiling (chat_max_persona_chars) if set and smaller
     *   - Load files in priority order
     *   - Paragraph-aware truncation for last file that doesn't fit
     * 
     * For large/unknown contexts:
     *   - Load everything, apply user ceiling only
     */
    const auto& cfg = config();
    
    std::string ragger_dir = expand_path("~/.ragger");

    // Calculate persona budget
    int max_persona_chars = 0;
    if (max_context > 0 && max_context < PERSONA_SIZING_THRESHOLD) {
        // Small context — apply percentage-based sizing
        int persona_budget = static_cast<int>(
            max_context * cfg.chat_chars_per_token * (cfg.chat_persona_pct / 100.0f)
        );
        persona_budget = std::max(persona_budget, 500);  // minimum budget
        
        // Apply user ceiling if set and smaller
        if (cfg.chat_max_persona_chars > 0) {
            max_persona_chars = std::min(cfg.chat_max_persona_chars, persona_budget);
        } else {
            max_persona_chars = persona_budget;
        }
    } else {
        // Large or unknown context — user ceiling only (0 = unlimited)
        max_persona_chars = cfg.chat_max_persona_chars;
    }
    
    // system_prompt_file (default: SYSTEM.md) is loaded first.
    // Persona files follow; the list excludes any file that resolves to the same path.
    std::string sys_path = expand_path(cfg.system_prompt_file);
    std::string sys_norm = fs::path(sys_path).lexically_normal().string();

    // Build ordered list: system_prompt_file by full path, then persona files by name
    std::vector<std::string> full_paths;
    full_paths.push_back(sys_path);
    for (const char* fname : {"SOUL.md", "USER.md", "MEMORY.md"}) {
        std::string p = ragger_dir + "/" + fname;
        if (fs::path(p).lexically_normal().string() != sys_norm) full_paths.push_back(p);
    }

    std::vector<std::string> sections;
    int total_chars = 0;

    for (const auto& path : full_paths) {
        std::string label = fs::path(path).filename().string();
        if (!fs::exists(path)) {
            continue;  // file not found — silently skip
        }

        // Read file
        std::ifstream file(path);
        if (!file) continue;
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());

        // Trim trailing whitespace
        size_t end = content.find_last_not_of(" \t\r\n");
        if (end != std::string::npos) {
            content = content.substr(0, end + 1);
        } else {
            content.clear();
        }

        if (content.empty()) continue;

        // Check if we need to truncate
        if (max_persona_chars > 0) {
            // Account for separator size (sections.size() gives count of separators needed)
            int separator_overhead = static_cast<int>(sections.size()) * 8;  // "\n\n---\n\n" = 8 chars
            int remaining = max_persona_chars - total_chars - separator_overhead;

            if (remaining <= 0) {
                break;  // No space left
            }

            if (static_cast<int>(content.size()) > remaining) {
                // Paragraph-aware truncation
                std::vector<std::string> paragraphs;
                std::istringstream stream(content);
                std::string buffer;

                // Split by double newline
                for (std::string line; std::getline(stream, line); ) {
                    if (line.empty() && !buffer.empty()) {
                        paragraphs.push_back(buffer);
                        buffer.clear();
                    } else {
                        if (!buffer.empty()) buffer += "\n";
                        buffer += line;
                    }
                }
                if (!buffer.empty()) {
                    paragraphs.push_back(buffer);
                }

                // Keep as many complete paragraphs as fit
                std::string truncated;
                for (const auto& p : paragraphs) {
                    if (static_cast<int>(truncated.size() + p.size() + 2) > remaining) {
                        break;
                    }
                    if (!truncated.empty()) truncated += "\n\n";
                    truncated += p;
                }

                // If we got at least some content, use it
                if (!truncated.empty()) {
                    truncated += "\n\n[... " + label + " truncated ...]";
                    sections.push_back(truncated);
                    total_chars += static_cast<int>(truncated.size());
                }
                // If even first paragraph doesn't fit, skip this file entirely
                break;  // Stop loading more files
            }
        }
        
        sections.push_back(content);
        total_chars += static_cast<int>(content.size());
    }
    
    // Join sections with separator
    std::string result;
    for (size_t i = 0; i < sections.size(); ++i) {
        if (i > 0) result += "\n\n---\n\n";
        result += sections[i];
    }
    
    return result;
}

int Chat::store_partial_turn(const std::string& user_text) {
    if (store_turns_ == "false") return -1;

    try {
        json meta = {
            {"collection", "memory"},
            {"category", "conversation"},
            {"source", "ragger-chat"},
            {"role", "exchange"},
            {"partial", true}  // assistant reply not yet appended
        };
        // Defer embedding: finalize_turn will replace this row's text in
        // a few seconds anyway. Embedding now would be discarded work.
        auto id_str = memory_.store(user_text, meta,
                                    /*common=*/false, /*defer_embedding=*/true);
        return std::stoi(id_str);
    } catch (const std::exception& e) {
        std::cerr << std::format(ragger::lang::WARN_STORE_TURN, e.what()) << "\n";
        return -1;
    }
}

int Chat::finalize_turn(int partial_id,
                        const std::string& user_text,
                        const std::string& assistant_text) {
    if (store_turns_ == "false") return -1;

    // ASCII Unit Separator (U+001F) between user and assistant halves —
    // structural marker, near-zero semantic weight in tokenizers, can't
    // collide with content the way "---" or "Assistant:" can.
    std::string turn_text = user_text + "\n\x1F\n" + assistant_text;
    json meta = {
        {"collection", "memory"},
        {"category", "conversation"},
        {"source", "ragger-chat"},
        {"role", "exchange"}
    };

    int row_id = -1;
    try {
        // Update the partial row in place — preserves id and the original
        // (prompt-arrival) timestamp, which is the correct value for
        // recency-weighted retrieval. Embedding is deferred; the fork
        // below picks up this row plus any leftover NULLs.
        if (partial_id >= 0 && memory_.update_text(partial_id, turn_text, meta,
                                                   /*defer_embedding=*/true)) {
            row_id = partial_id;
        } else {
            // Partial row missing/protected — fall back to a fresh row.
            auto id_str = memory_.store(turn_text, meta,
                                        /*common=*/false, /*defer_embedding=*/true);
            row_id = std::stoi(id_str);
        }
    } catch (const std::exception& e) {
        std::cerr << std::format(ragger::lang::WARN_STORE_TURN, e.what()) << "\n";
        return -1;
    }

    // Spawn a backfill child. SQLite isn't fork-safe across an open
    // connection, so the child opens its own RaggerMemory — whose
    // constructor already runs backfill_embeddings() and exits. Same
    // fork pattern as bg_summarize.
    pid_t pid = fork();
    if (pid == 0) {
        try {
            const auto& cfg = config();
            RaggerMemory child_memory(cfg.resolved_db_path(), cfg.resolved_model_dir());
            child_memory.close();
        } catch (...) {
            // Silent — next finalize or startup will retry.
        }
        _exit(0);
    }
    return row_id;
}

std::vector<std::pair<std::string, std::string>> Chat::turns_as_pairs() const {
    std::vector<std::pair<std::string, std::string>> out;
    for (size_t i = 0; i + 1 < unsummarized_turns_.size(); ) {
        const auto& a = unsummarized_turns_[i];
        const auto& b = unsummarized_turns_[i + 1];
        if (a.role == "user" && b.role == "assistant") {
            std::string utxt, atxt;
            for (const auto& c : a.content) utxt += c.text;
            for (const auto& c : b.content) atxt += c.text;
            out.push_back({utxt, atxt});
            i += 2;
        } else {
            ++i;  // skip stray entry; keep walking
        }
    }
    return out;
}

void Chat::check_orphaned_turns() {
    if (store_turns_ == "false" || store_turns_ == "session") {
        return;  // Only applies to per-turn mode
    }
    
    try {
        // Find all conversation turns
        json filter = {
            {"category", "conversation"},
            {"source", "ragger-chat"}
        };
        auto turns = memory_.search_by_metadata(filter);
        
        if (turns.empty()) {
            return;
        }
        
        // Filter out session-mode entries
        std::vector<SearchResult> orphaned_turns;
        for (const auto& turn : turns) {
            if (turn.metadata.value("mode", "") != "session") {
                // Check if timestamp is old (> 5 minutes means from previous session)
                std::tm tm = {};
                std::istringstream ss(turn.timestamp);
                ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
                
                if (ss.fail()) continue;  // couldn't parse
                
                auto tp = std::chrono::system_clock::from_time_t(std::mktime(&tm));
                auto now = std::chrono::system_clock::now();
                auto age_minutes = std::chrono::duration_cast<std::chrono::minutes>(now - tp).count();
                
                // Consider orphaned if > 5 minutes old (from previous session)
                if (age_minutes > 5) {
                    orphaned_turns.push_back(turn);
                }
            }
        }
        
        if (orphaned_turns.empty()) {
            return;
        }
        
        std::cout << std::format(ragger::lang::MSG_ORPHAN_FOUND, orphaned_turns.size()) << "\n";
        
        // Reconstruct user/assistant pairs by splitting on the U+001F
        // unit separator. Rows without the separator are partial turns
        // (assistant reply never landed) — feed them as user-only.
        std::vector<std::pair<std::string, std::string>> turn_pairs;
        for (const auto& turn : orphaned_turns) {
            const std::string& text = turn.text;
            size_t sep = text.find('\x1F');
            if (sep == std::string::npos) {
                turn_pairs.push_back({text, ""});
                continue;
            }
            // Strip the surrounding \n that brackets the separator.
            size_t u_end = (sep > 0 && text[sep - 1] == '\n') ? sep - 1 : sep;
            size_t a_start = (sep + 1 < text.size() && text[sep + 1] == '\n')
                             ? sep + 2 : sep + 1;
            turn_pairs.push_back({text.substr(0, u_end), text.substr(a_start)});
        }
        
        if (!turn_pairs.empty()) {
            // Summarize orphaned turns
            std::string summary = summarize_conversation(turn_pairs);
            
            if (!summary.empty()) {
                // Store summary
                json meta = {
                    {"collection", "memory"},
                    {"category", "session-summary"},
                    {"source", "ragger-chat"},
                    {"turns", turn_pairs.size()},
                    {"recovered", true}
                };
                memory_.store(summary, meta);
                
                // Delete orphaned raw turns (delete_batch will respect keep tags)
                std::vector<int> orphaned_ids;
                for (const auto& turn : orphaned_turns) {
                    orphaned_ids.push_back(turn.id);
                }
                int deleted = memory_.delete_batch(orphaned_ids);
                
                std::cout << std::format(ragger::lang::MSG_ORPHAN_RECOVERED,
                                         turn_pairs.size(), deleted) << "\n";
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << std::format(ragger::lang::WARN_ORPHAN_CHECK, e.what()) << "\n";
    }
}


void Chat::bg_summarize(
        const std::vector<std::pair<std::string, std::string>>& turns_to_summarize) {
    if (turns_to_summarize.empty()) return;

    auto turns_copy = turns_to_summarize;

    pid_t pid = fork();
    if (pid != 0) {
        // Parent: return immediately
        return;
    }

    // Child process: fresh connections (SQLite not fork-safe)
    try {
        const auto& cfg = config();
        RaggerMemory child_memory(cfg.resolved_db_path(), cfg.resolved_model_dir());
        InferenceClient child_inference = InferenceClient::from_config(cfg);

        std::string conversation_text;
        for (const auto& turn : turns_copy) {
            conversation_text += "**User:** " + turn.first + "\n\n";
            conversation_text += "**Assistant:** " + turn.second + "\n\n";
        }

        std::vector<Message> summary_messages = {
            {"system", "Summarize this conversation into a concise memory entry. "
                       "Extract: key facts, decisions, questions asked, topics discussed. "
                       "Write in third person past tense. Be brief — this will be stored "
                       "as a memory chunk for future retrieval."},
            {"user", conversation_text}
        };

        std::string summary = child_inference.chat(summary_messages, model_);

        if (!summary.empty()) {
            json meta = {
                {"collection", "memory"},
                {"category", "session-summary"},
                {"source", "ragger-chat"},
                {"turns", turns_copy.size()}
            };
            child_memory.store(summary, meta);
        }

        child_memory.close();
    } catch (...) {
        // Silent failure in background child
    }
    _exit(0);
}

void Chat::check_pause_summary() {
    if (!summarize_on_pause_ || unsummarized_turns_.empty()) {
        return;
    }

    int idle = idle_seconds();
    if (idle < pause_minutes_ * 60) {
        return;
    }

    bg_summarize(turns_as_pairs());
    unsummarized_turns_.clear();
}

void Chat::quit_summary() {
    if (!summarize_on_quit_ || unsummarized_turns_.empty()) {
        return;
    }

    std::cout << ragger::lang::MSG_SUMMARIZING << "\n";
    bg_summarize(turns_as_pairs());
}

std::string Chat::summarize_conversation(const std::vector<std::pair<std::string, std::string>>& turns) {
    if (turns.empty()) {
        return "";
    }
    
    // Build conversation text
    std::string conversation_text;
    for (const auto& turn : turns) {
        conversation_text += "**User:** " + turn.first + "\n\n";
        conversation_text += "**Assistant:** " + turn.second + "\n\n";
    }
    
    // Build summary request
    std::vector<Message> summary_messages = {
        {"system", "Summarize this conversation into a concise memory entry. "
                   "Extract: key facts, decisions, questions asked, topics discussed. "
                   "Write in third person past tense. Be brief — this will be stored "
                   "as a memory chunk for future retrieval."},
        {"user", conversation_text}
    };
    
    try {
        return inference_.chat(summary_messages, model_);
    } catch (const std::exception& e) {
        std::cerr << std::format(ragger::lang::WARN_SUMMARY, e.what()) << "\n";
        return "";
    }
}

void Chat::run() {
    const auto& cfg = config();
    
    // Get endpoint to determine context size
    auto& endpoint = inference_._endpoints[0];  // Assume first endpoint for now
    int max_context = 0;
    
    // Find matching endpoint for model
    for (auto& ep : inference_._endpoints) {
        if (ep.matches(model_)) {
            endpoint = ep;
            max_context = endpoint.name == "local" ? 0 : 0;  // TODO: get from config
            break;
        }
    }
    
    // Load from config
    for (const auto& ep_cfg : cfg.inference_endpoints) {
        if (ep_cfg.name == endpoint.name) {
            max_context = ep_cfg.max_context;
            break;
        }
    }
    
    // Load workspace files with context sizing
    std::string workspace = load_workspace_files(max_context);

    if (workspace.empty()) {
        std::cerr << ragger::lang::ERR_NO_SYSTEM_PROMPT
                  << ragger::lang::MSG_NO_SYSTEM_PROMPT_HINT << "\n";
    }

    // Initialize conversation
    if (!workspace.empty()) {
        messages_.push_back({"system", workspace});
    }
    
    // Print startup banner
    std::cout << std::format(ragger::lang::MSG_CHAT_BANNER, model_) << "\n";
    std::cout << std::format(ragger::lang::MSG_CHAT_TURN_STORAGE, store_turns_) << "\n";

    if (max_context > 0 && max_context < PERSONA_SIZING_THRESHOLD) {
        int persona_chars = cfg.chat_max_persona_chars;
        if (persona_chars == 0) {
            persona_chars = static_cast<int>(
                max_context * cfg.chat_chars_per_token * (cfg.chat_persona_pct / 100.0f)
            );
        }
        std::cout << std::format(ragger::lang::MSG_CHAT_CONTEXT_SIZED,
                                 max_context, endpoint.name,
                                 cfg.chat_persona_pct, persona_chars) << "\n";
    } else {
        std::string ctx_info = max_context > 0
            ? std::format(ragger::lang::MSG_CHAT_CTX_TOKENS, max_context)
            : ragger::lang::MSG_CHAT_CTX_UNKNOWN;
        std::string persona_info = cfg.chat_max_persona_chars > 0
            ? std::format(ragger::lang::MSG_CHAT_PERSONA_CHARS, cfg.chat_max_persona_chars)
            : ragger::lang::MSG_CHAT_PERSONA_NONE;
        std::cout << std::format(ragger::lang::MSG_CHAT_CONTEXT_OPEN,
                                 ctx_info, endpoint.name, persona_info) << "\n";
    }

    std::cout << ragger::lang::MSG_CHAT_QUIT_HINT << "\n\n";
    
    // Check for orphaned turns from previous session (crash recovery)
    check_orphaned_turns();
    
    
    // Main REPL loop
    std::string line;
    while (true) {
        // Check for pause summary before waiting for input
        check_pause_summary();
        
        std::cout << ragger::lang::CHAT_PROMPT_USER << std::flush;
        if (!std::getline(std::cin, line)) {
            std::cout << "\n" << ragger::lang::MSG_CHAT_GOODBYE << "\n";
            break;
        }
        
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        
        if (line.empty()) continue;
        
        if (line == "/quit" || line == "/exit") {
            std::cout << ragger::lang::MSG_CHAT_GOODBYE << "\n";
            break;
        }

        // ---- /models — query active endpoint for available models ----
        if (line == "/models") {
            const auto& cfg = config();

            // Determine which endpoint to query
            Endpoint* active_ep = nullptr;
            if (!inference_.forced_endpoint().empty()) {
                for (auto& ep : inference_._endpoints) {
                    if (ep.name == inference_.forced_endpoint()) { active_ep = &ep; break; }
                }
            }
            if (!active_ep && !inference_._endpoints.empty()) {
                // Use the endpoint that would handle the current model
                for (auto& ep : inference_._endpoints) {
                    if (model_.empty() || ep.matches(model_)) { active_ep = &ep; break; }
                }
                if (!active_ep) active_ep = &inference_._endpoints.back();
            }

            if (active_ep) {
                std::cout << std::format(ragger::lang::MSG_MODELS_QUERYING,
                                         active_ep->name, active_ep->api_url) << "\n";
                auto models = active_ep->list_models();
                if (models.empty()) {
                    std::cout << ragger::lang::MSG_MODELS_NONE << "\n";
                } else {
                    for (const auto& m : models) {
                        std::cout << "  " << m << "\n";
                    }
                    std::cout << std::format(ragger::lang::MSG_MODELS_COUNT, models.size()) << "\n";
                }
            } else {
                std::cout << ragger::lang::MSG_MODELS_NO_ENDPOINTS << "\n";
            }

            if (!cfg.model_aliases.empty()) {
                std::cout << "\n" << ragger::lang::MSG_MODELS_ALIASES << "\n";
                for (const auto& [alias, full] : cfg.model_aliases) {
                    std::cout << std::format(ragger::lang::MSG_MODELS_ALIAS_LINE, alias, full) << "\n";
                }
            }

            std::cout << "\n" << std::format(ragger::lang::MSG_MODELS_CURRENT,
                                              model_.empty() ? ragger::lang::MSG_MODEL_DEFAULT : model_)
                      << "\n\n";
            continue;
        }

        // ---- /model [name] — show or switch model ----
        if (line == "/model") {
            std::cout << std::format(ragger::lang::MSG_MODEL_CURRENT,
                                     model_.empty() ? ragger::lang::MSG_MODEL_DEFAULT : model_) << "\n";
            std::cout << ragger::lang::MSG_MODEL_USAGE_SWITCH << "\n\n";
            continue;
        }

        if (line.substr(0, 7) == "/model ") {
            std::string new_model = line.substr(7);
            new_model.erase(0, new_model.find_first_not_of(" \t"));
            new_model.erase(new_model.find_last_not_of(" \t") + 1);
            if (new_model.empty()) {
                std::cout << ragger::lang::MSG_MODEL_USAGE << "\n\n";
                continue;
            }
            // Resolve alias
            std::string resolved = config().resolve_model(new_model);
            model_ = resolved;
            std::cout << std::format(ragger::lang::MSG_MODEL_SWITCHED, resolved);
            if (resolved != new_model) {
                std::cout << std::format(ragger::lang::MSG_MODEL_ALIAS_SUFFIX, new_model);
            }
            std::cout << "\n\n";
            continue;
        }

        // ---- /endpoints — list endpoints with live status ----
        if (line == "/endpoints" || line == "/services") {
            std::cout << ragger::lang::MSG_ENDPOINTS_HEADER << "\n";
            for (auto& ep : inference_._endpoints) {
                bool up = ep.is_reachable();
                std::string marker = up ? "✓" : "✗";
                std::string forced = (ep.name == inference_.forced_endpoint())
                    ? ragger::lang::MSG_ENDPOINT_ACTIVE : "";
                std::cout << std::format(ragger::lang::MSG_ENDPOINT_LINE,
                                         marker, ep.name, ep.api_url, ep.format_name, forced) << "\n";
            }
            std::string current = inference_.forced_endpoint().empty() ? "auto" : inference_.forced_endpoint();
            std::cout << "\n" << std::format(ragger::lang::MSG_ENDPOINT_ROUTING, current) << "\n";
            std::cout << ragger::lang::MSG_ENDPOINT_USAGE << "\n\n";
            continue;
        }

        // ---- /endpoint [name|auto] — force or auto-route endpoint ----
        if (line == "/endpoint" || line == "/service") {
            std::string current = inference_.forced_endpoint().empty() ? "auto" : inference_.forced_endpoint();
            std::cout << std::format(ragger::lang::MSG_ENDPOINT_CURRENT, current) << "\n";
            std::cout << ragger::lang::MSG_ENDPOINT_USAGE << "\n\n";
            continue;
        }

        if (line.substr(0, 10) == "/endpoint " || line.substr(0, 9) == "/service ") {
            size_t skip = (line[1] == 'e') ? 10 : 9;
            std::string name = line.substr(skip);
            name.erase(0, name.find_first_not_of(" \t"));
            name.erase(name.find_last_not_of(" \t") + 1);
            if (name.empty()) {
                std::cout << ragger::lang::MSG_ENDPOINT_USAGE_ARG << "\n\n";
                continue;
            }
            if (name == "auto") {
                inference_.set_forced_endpoint("");
                std::cout << ragger::lang::MSG_ENDPOINT_AUTO << "\n\n";
                continue;
            }
            try {
                inference_.set_forced_endpoint(name);
                // Check reachability
                for (auto& ep : inference_._endpoints) {
                    if (ep.name == name) {
                        if (!ep.is_reachable()) {
                            std::cout << std::format(ragger::lang::WARN_ENDPOINT_DOWN, name) << "\n";
                        }
                        break;
                    }
                }
                std::cout << std::format(ragger::lang::MSG_ENDPOINT_FORCED, name) << "\n\n";
            } catch (const std::exception& e) {
                std::cout << std::format(ragger::lang::WARN_CHAT_ERROR, e.what()) << "\n";
                std::string available;
                for (size_t i = 0; i < inference_._endpoints.size(); ++i) {
                    if (i > 0) available += ", ";
                    available += inference_._endpoints[i].name;
                }
                std::cout << std::format(ragger::lang::MSG_ENDPOINT_AVAILABLE, available) << "\n\n";
            }
            continue;
        }

        // ---- /help ----
        if (line == "/help") {
            std::cout << ragger::lang::MSG_HELP_HEADER    << "\n"
                      << ragger::lang::MSG_HELP_MODELS    << "\n"
                      << ragger::lang::MSG_HELP_MODEL     << "\n"
                      << ragger::lang::MSG_HELP_ENDPOINTS << "\n"
                      << ragger::lang::MSG_HELP_ENDPOINT  << "\n"
                      << ragger::lang::MSG_HELP_HELP      << "\n"
                      << ragger::lang::MSG_HELP_QUIT      << "\n\n";
            continue;
        }

        // Unrecognized input (including unknown /commands) passes through to the LLM
        
        update_activity();
        
        // Search memory for context
        std::vector<std::string> context_chunks;
        try {
            auto result = memory_.search(line, max_memory_results_, 0.3f);
            for (const auto& r : result.results) {
                context_chunks.push_back(r.text);
            }
        } catch (const std::exception& e) {
            std::cerr << std::format(ragger::lang::WARN_MEMORY_SEARCH, e.what()) << "\n";
        }
        
        // Build message with context
        std::vector<Message> current_messages = messages_;
        
        if (!context_chunks.empty()) {
            std::string context_text;
            for (size_t i = 0; i < context_chunks.size(); ++i) {
                if (i > 0) context_text += "\n\n---\n\n";
                context_text += context_chunks[i];
            }
            std::string memory_block = "\n\n## Relevant memories:\n\n" + context_text;
            
            // Append to existing system message or create one
            if (!current_messages.empty() && current_messages[0].role == "system") {
                current_messages[0].content += memory_block;
            } else {
                current_messages.insert(current_messages.begin(),
                                       {"system", memory_block});
            }
        }
        
        current_messages.push_back({"user", line});

        // Persist the user prompt before the LLM call. If the call crashes
        // mid-stream the row stays — durable evidence of what was asked.
        int partial_id = store_partial_turn(line);

        // Send to inference API (streaming)
        std::cout << ragger::lang::CHAT_PROMPT_ASSISTANT << std::flush;
        std::string response_text;

        try {
            inference_.chat_stream(current_messages, [&](const std::string& token) {
                std::cout << token << std::flush;
                response_text += token;
            }, model_);
            std::cout << "\n";
        } catch (const std::exception& e) {
            std::cout << "\n" << std::format(ragger::lang::ERR_INFERENCE, e.what()) << "\n";
            // Leave the partial row in place — that's the whole point.
            continue;
        }

        // Update conversation history
        messages_.push_back({"user", line});
        messages_.push_back({"assistant", response_text});

        // Replace the partial row with the full exchange and track both
        // halves as Turn entries (memory_id on the assistant turn points
        // at the persisted exchange row, since exchanges are stored as a
        // single combined row today).
        int exchange_id = finalize_turn(partial_id, line, response_text);
        unsummarized_turns_.push_back({"user", {{ "text", line }}, -1});
        unsummarized_turns_.push_back({"assistant", {{ "text", response_text }}, exchange_id});
        update_activity();
        
        std::cout << "\n";  // blank line between exchanges
    }
    
    // Final summary
    quit_summary();
}

} // namespace ragger
