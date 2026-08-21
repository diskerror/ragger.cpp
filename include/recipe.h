/**
 * Recipe — deterministic, layered context assembly.
 *
 * A recipe walks back from the latest prompt in a session, consuming
 * turns/summaries layer by layer to balance recency-of-detail against
 * token cost. The default `natural_fading` recipe gives the agent the
 * last few raw turns verbatim, then earlier turns as L2 summaries, then
 * the session summary, then the project summary, then pertinent
 * decisions — mimicking how a person remembers a conversation.
 *
 * Recipes are JSON files in `~/.ragger/recipes/<name>.json`. The loader
 * scans the directory at server startup; built-in defaults are returned
 * when the directory is missing or empty. Layer kinds:
 *
 *   raw_turn         — N raw turns, newest-first (consumes the walk)
 *   turn_summary     — N L2 summaries, newest-first (consumes the walk)
 *   session_summary  — current session's L3 (latest), up to `limit`
 *   project_summary  — recent L4 project summaries, up to `limit`
 *   decisions        — current L6 decisions, up to `limit`
 *   general_search   — up to `limit` relevance-ranked results from across
 *                      ALL summaries+documents+decisions (session-agnostic),
 *                      excluding anything earlier layers already emitted.
 *                      Query = current session's latest user turn + up to 2
 *                      most recent turn summaries.
 *
 * `limit` is per-layer. `max_tokens` is a
 * hard ceiling; assembly truncates from the oldest end once exceeded.
 */
#pragma once

#include <string>
#include <vector>

#include "nlohmann_json.hpp"

namespace ragger {

enum class LayerKind {
    RawTurn,
    TurnSummary,
    SessionSummary,
    ProjectSummary,
    Decisions,
    GeneralSearch,
};

struct RecipeLayer {
    LayerKind kind;
    int       limit = 0;  // 0 = unlimited for that layer
};

struct Recipe {
    std::string              name;
    std::string              description;
    std::vector<RecipeLayer> layers;
    int                      max_tokens = 8000;
    // chars-per-token approximation used for max_tokens enforcement.
    // 4.0 is a reasonable default for English-language LLM tokenizers.
    float                    chars_per_token = 4.0f;
};

/// Parse a recipe from JSON. Throws on malformed input; unknown layer
/// kinds are skipped with a logged warning. `source` is purely for
/// error messages.
Recipe parse_recipe(const nlohmann::json& j, const std::string& source = "");

/// Load every recipe from `dir` (`*.json`, non-recursive). Missing dir
/// returns empty. Each file may contain a single recipe object or an
/// array of recipes. Names are taken from the JSON `name` field; the
/// filename is used as a fallback when `name` is absent.
std::vector<Recipe> load_recipes_from_dir(const std::string& dir);

/// Built-in recipes returned when the user has no custom files. Always
/// contains at least `natural_fading` (the default) and `raw_only`.
std::vector<Recipe> builtin_recipes();

/// Convenience: find a recipe by name. Returns nullptr if absent.
const Recipe* find_recipe(const std::vector<Recipe>& recipes,
                          const std::string& name);

} // namespace ragger
