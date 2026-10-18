/**
 * sqlite/status.h — THE status vocabulary (v0.17 rework, PHASE3 Step 4, B §2.6).
 *
 * Each list keeps EXACTLY the membership the code had before this refactor.
 * The sites disagree on purpose or by accident; unifying membership is
 * Phase 7's decision, not this refactor's. Named lists replace the eight
 * hand-copied status sets; `transition_ok` is the agent_update transition
 * table moved verbatim from agent_write.cpp.
 */
#pragma once

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ragger::sqlite::status {

// cache.cpp dec/ins loaders + search.cpp structured passes: rows whose
// vectors participate in scoring.
inline constexpr std::string_view kCacheLoaded[] = {"current", "provisional", "roadmap"};

// "active" filters: agent_search active mode (agent_read.cpp),
// supersede-hint candidates (write.cpp).
inline constexpr std::string_view kSearchActive[] = {"current", "provisional"};

// Statuses accepted by the create path (store_structured_row) and the MCP
// store schema.
inline constexpr std::string_view kCreatable[] = {"current", "provisional", "pending"};

// Statuses the 0.16 -> 0.17 migration leg recognizes as already-valid
// (anything else leaked in as a category word and moves to tags).
inline constexpr std::string_view kMigrationKnown[] = {
    "current", "provisional", "pending", "superseded", "retired", "roadmap", "deprecated"};

// Fifth list (not one of the four the phase doc anticipated): the target
// statuses agent_update accepts. Excludes 'superseded' (reachable only via
// `supersedes` on a new row) and 'deprecated' (legacy, not settable).
inline constexpr std::string_view kAgentSettable[] = {
    "current", "provisional", "pending", "roadmap", "retired"};

// -> "'current','provisional'" — for SQL IN (...) clauses.
inline std::string in_list_sql(std::span<const std::string_view> values) {
    std::string out;
    for (const auto& v : values) {
        if (!out.empty()) out += ",";
        out += "'";
        out += v;
        out += "'";
    }
    return out;
}

// Membership test for the lists above.
template <std::size_t N>
bool is_in(const std::string_view (&values)[N], std::string_view v) {
    for (const auto& s : values)
        if (s == v) return true;
    return false;
}

// agent_update transition table, MOVED VERBATIM from agent_write.cpp
// (kStatuses map). 'superseded' is reachable only through `supersedes` on a
// new decision, never here; 'retired' is terminal.
bool transition_ok(std::string_view from, std::string_view to);

// ---- knowledge kinds (PHASE7) --------------------------------------------
// One row per knowledge(kind). The rules are exactly the old per-table ones:
// decision requires a trigger; insight requires >=1 evidence; note needs
// neither. `noun` builds the existing error messages verbatim.
struct KindRule { std::string_view name; bool trigger_required; bool evidence_required; std::string_view noun; };
inline constexpr KindRule kKinds[] = {
    {"decision", true,  false, "decision"},
    {"insight",  false, true,  "insight"},
    {"note",     false, false, "note"},
};
inline constexpr const KindRule* find_kind(std::string_view kind) {
    for (const auto& k : kKinds)
        if (kind == k.name) return &k;
    return nullptr;
}

} // namespace ragger::sqlite::status
