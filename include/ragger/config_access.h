/**
 * Generic string-keyed access to Config fields, driven by the schema in
 * config_schema.h / en.h. This is the bridge that lets the CLI (`ragger
 * config get/set`) and the dashboard (GET/PUT /config) read and write any
 * setting by its string key without a bespoke handler per field.
 *
 * Value round-tripping is TEXT, matching the DB `settings` table:
 *   - bool  -> "true" / "false"
 *   - int   -> decimal
 *   - float -> shortest round-trippable decimal
 *   - enum/string/path/text -> as-is
 *
 * DB is the source of truth. Flow:
 *   init_config() loads INI seed -> overlay_settings_from_db() applies rows
 *   -> live Config. `config set` validates, writes the DB row, and updates
 *   the live Config in place.
 */
#pragma once

#include "ragger/config.h"
#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <expected>

namespace ragger {

// Result of a validate/set attempt.
enum class ConfigSetError {
    UnknownKey,
    Locked,        // key is CfgEdit::Locked — not user-writable
    InvalidValue,  // failed type/enum validation
};

struct ConfigSetResult {
    bool restart_required = false;   // change stored but needs daemon restart
    bool rebuild_required = false;   // needs `ragger rebuild-embeddings`
};

/// Read one setting's current value from a Config as TEXT. Returns nullopt
/// for an unknown key. An empty stored string is returned as the schema
/// default (so "blank == default" holds at the read boundary too).
std::optional<std::string> get_config_value(const Config& cfg, std::string_view key);

/// Validate `value` against the key's schema type/enum. Does NOT write.
/// Empty `value` is always valid — it means "reset to default".
std::expected<void, ConfigSetError> validate_config_value(std::string_view key,
                                                          std::string_view value);

/// Apply a validated value onto a live Config in memory. Empty value resets
/// the field to the schema default. Returns which restart/rebuild flags the
/// change implies. Caller is responsible for persisting to the DB.
std::expected<ConfigSetResult, ConfigSetError> apply_config_value(
    Config& cfg, std::string_view key, std::string_view value,
    bool allow_locked = false);

/// Overlay every row present in the DB `settings` table onto `cfg`, for keys
/// that appear in the schema. Unknown/reserved keys (db_version, boundary
/// watermarks) are ignored. Called once after INI load in init_config().
/// `db_path` is the resolved main DB. Silent no-op if the DB is absent.
void overlay_settings_from_db(Config& cfg, const std::string& db_path);

/// All schema keys in declaration order (for `config get -a` and the CLI
/// help listing).
std::vector<std::string_view> all_config_keys();

/// Validate, persist to the DB `settings` table, AND update the live global
/// config in one call. This is the single write path used by both the CLI
/// (`ragger config set`) and the dashboard (PUT /config/<key>). An empty
/// value deletes the row (reverting the key to its schema default) and
/// resets the live field. Requires init_config() to have run.
std::expected<ConfigSetResult, ConfigSetError> set_config_persisted(
    std::string_view key, std::string_view value, bool allow_locked = false);

/// One-time migration: seed the DB `settings` table from a legacy
/// settings.ini, then retire the file. Idempotent via a DB marker
/// (`ini_migrated`) so it never runs twice, even though the bootstrap may
/// recreate a default settings.ini. For every schema key whose INI-parsed
/// value differs from the schema default, a row is written to the DB (unless
/// a row already exists — existing DB values always win). On success the INI
/// is renamed to `<path>.migrated`. `db_path` must point at an existing,
/// schema-initialized DB. Returns the number of keys imported (0 if the
/// migration was already done or there was no INI). Best-effort: never throws.
int migrate_ini_to_db(const std::string& db_path);

} // namespace ragger
