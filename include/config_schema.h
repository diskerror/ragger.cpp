/**
 * Config schema — the single source of truth for every user-editable Ragger
 * setting. One row per key drives BOTH faces:
 *
 *   - CLI  (`ragger config get/set`, help text, validation)
 *   - Dashboard (labels, control type, popup options, input filtering, help)
 *
 * The actual entries (with their pretty names and help text) live in
 * include/ragger/lang/en.h so they are translatable in one place. This header
 * only defines the shape.
 *
 * Values are stored as TEXT rows in the DB `settings` table. A missing or
 * empty row means "use `default_value`" — which is exactly the dashboard's
 * "blank box reverts to default" behaviour.
 */
#pragma once

#include <string_view>
#include <span>
#include <cstddef>

namespace ragger::lang {

// How a value is typed, validated, and rendered in the dashboard.
enum class CfgType {
    Boolean,   // true/false  → checkbox
    Integer,   // digits only → text box, [0-9-] filter
    Float,     // decimal     → text box, [0-9.-] filter
    Enum,      // fixed set   → popup menu (see `options`)
    String,    // free text
    Path,      // filesystem path (free text, path-ish)
    Text,      // multi-line free text (e.g. summarizer prompt)
};

// Editability class. Server-locked keys are shown read-only in the dashboard
// and refused by `ragger config set` unless the special path allows it;
// RebuildRequired keys are editable but flag that a re-encode / restart is
// needed before the change fully takes effect.
enum class CfgEdit {
    Live,            // hot-applies immediately
    RestartRequired, // needs daemon restart to take effect
    RebuildRequired, // needs `ragger rebuild-embeddings` (model/dims/vec type)
    Locked,          // read-only (bootstrap/system config owns it)
};

struct ConfigMeta {
    std::string_view key;          // DB row key, e.g. "bm25_weight"
    std::string_view section;      // dashboard tab, e.g. "server"
    std::string_view pretty;       // UI label, e.g. "BM25 Weight"
    CfgType          type;
    CfgEdit          edit;
    std::string_view default_value; // canonical default as TEXT
    std::string_view options;       // Enum only: comma-separated choices
    std::string_view help;          // explanation (CLI help + dashboard tooltip)
};

// Defined in include/ragger/lang/en.h (kConfigSchema[]).
std::span<const ConfigMeta> config_schema();

// Look up one entry by key; returns nullptr if unknown.
const ConfigMeta* config_meta(std::string_view key);

} // namespace ragger::lang
