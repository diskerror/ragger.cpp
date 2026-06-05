/**
 * Recipe loader. See include/ragger/recipe.h.
 */

#include "ragger/recipe.h"

#include "diskerror/logger.h"

#include <filesystem>
#include <fstream>
#include <unordered_map>

namespace ragger {
namespace fs = std::filesystem;

namespace {

const std::unordered_map<std::string, LayerKind>& layer_kind_map() {
    static const std::unordered_map<std::string, LayerKind> m = {
        {"raw_turn",        LayerKind::RawTurn},
        {"turn_summary",    LayerKind::TurnSummary},
        {"session_summary", LayerKind::SessionSummary},
        {"project_summary", LayerKind::ProjectSummary},
        {"decisions",       LayerKind::Decisions},
    };
    return m;
}

void parse_layers_into(Recipe& recipe, const nlohmann::json& layers,
                       const std::string& source) {
    const auto& kinds = layer_kind_map();
    for (const auto& layer : layers) {
        const std::string kind_str = layer.value("kind", "");
        auto it = kinds.find(kind_str);
        if (it == kinds.end()) {
            Diskerror::logger::warn(
                "recipe '" + source + "': unknown layer kind '" + kind_str +
                "', skipping");
            continue;
        }
        int count = layer.value("count", 0);
        // `decisions` historically used `limit`; accept both for ergonomics.
        if (count == 0) count = layer.value("limit", 0);
        recipe.layers.push_back({it->second, count});
    }
}

} // namespace

Recipe parse_recipe(const nlohmann::json& j, const std::string& source) {
    Recipe r;
    r.name            = j.value("name", source);
    r.description     = j.value("description", "");
    r.max_tokens      = j.value("max_tokens", 8000);
    r.chars_per_token = j.value("chars_per_token", 4.0f);
    if (j.contains("layers") && j["layers"].is_array()) {
        parse_layers_into(r, j["layers"], r.name.empty() ? source : r.name);
    }
    return r;
}

std::vector<Recipe> load_recipes_from_dir(const std::string& dir) {
    std::vector<Recipe> out;
    std::error_code ec;
    if (dir.empty() || !fs::exists(dir, ec) || !fs::is_directory(dir, ec)) {
        return out;
    }

    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) continue;
        const auto path = entry.path();
        if (path.extension() != ".json") continue;
        std::ifstream f(path);
        if (!f) continue;
        try {
            nlohmann::json doc = nlohmann::json::parse(f);
            const auto stem = path.stem().string();
            if (doc.is_array()) {
                for (const auto& item : doc) {
                    out.push_back(parse_recipe(item, stem));
                }
            } else {
                out.push_back(parse_recipe(doc, stem));
            }
        } catch (const std::exception& e) {
            Diskerror::logger::warn(
                std::string("recipe load failed for ") + path.string() +
                ": " + e.what());
        }
    }
    return out;
}

std::vector<Recipe> builtin_recipes() {
    std::vector<Recipe> out;

    // natural_fading — the default. Mimics how memory decays:
    // freshly-spoken turns stay verbatim, older ones fade to summaries,
    // then to a session-level digest, then to project context, then to
    // pertinent decisions.
    Recipe nf;
    nf.name        = "natural_fading";
    nf.description = "Recent raw → turn summaries → session → project → decisions";
    nf.layers      = {
        {LayerKind::RawTurn,        2},
        {LayerKind::TurnSummary,    5},
        {LayerKind::SessionSummary, 1},
        {LayerKind::ProjectSummary, 1},
        {LayerKind::Decisions,      3},
    };
    nf.max_tokens = 8000;
    out.push_back(std::move(nf));

    // raw_only — preserves today's pre-recipe behavior under an explicit
    // name. Useful as a debugging baseline and for short sessions.
    Recipe ro;
    ro.name        = "raw_only";
    ro.description = "Raw turns of the active session, oldest first";
    ro.layers      = { {LayerKind::RawTurn, 0} };
    ro.max_tokens  = 16000;
    out.push_back(std::move(ro));

    return out;
}

const Recipe* find_recipe(const std::vector<Recipe>& recipes,
                          const std::string& name) {
    if (name.empty()) return nullptr;
    for (const auto& r : recipes) {
        if (r.name == name) return &r;
    }
    return nullptr;
}

} // namespace ragger
