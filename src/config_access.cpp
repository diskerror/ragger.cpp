/**
 * Generic string-keyed Config access — see config_access.h.
 *
 * Maps each schema key to its Config struct field via member pointers,
 * grouped by field type. Keeping the mapping here (one table per C++ type)
 * is the single place that knows "key X lives in field Y"; everything else
 * (CLI, dashboard, DB overlay) goes through the generic get/validate/apply.
 */
#include "config_access.h"
#include "lang.h"          // ragger::lang::config_schema / config_meta
#include "user_store.h"
#include "util/fs.h"

#include <charconv>
#include <cstdlib>
#include <cerrno>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <unordered_map>
#include <filesystem>

namespace ragger {

using lang::CfgType;
using lang::CfgEdit;
using lang::ConfigMeta;

namespace {

// ---- key -> Config member pointer, one map per storage type -------------
// Key names sometimes differ from field names (socket_enable ->
// socket_enabled, bind -> bind_address, default_limit ->
// default_search_limit, cert/key -> tls_cert/tls_key, normalize_home ->
// normalize_home_path); the map is the authority.

const std::unordered_map<std::string_view, bool Config::*>& bool_fields() {
    static const std::unordered_map<std::string_view, bool Config::*> m = {
        {"socket_enable",  &Config::socket_enabled},
        {"tcp_enable",     &Config::tcp_enabled},
        {"capture_turns",  &Config::capture_turns},
        {"build_context",  &Config::build_context},
        {"bm25_enabled",   &Config::bm25_enabled},
        {"normalize_home", &Config::normalize_home_path},
    };
    return m;
}

const std::unordered_map<std::string_view, int Config::*>& int_fields() {
    static const std::unordered_map<std::string_view, int Config::*> m = {
        {"port",                  &Config::port},
        {"desired_port",          &Config::desired_port},
        {"embedding_dimensions",  &Config::embedding_dimensions},
        {"desired_embedding_dimensions", &Config::desired_embedding_dimensions},
        {"embed_timeout_ms",      &Config::embed_timeout_ms},
        {"embed_retries",         &Config::embed_retries},
        {"embed_max_workers",     &Config::embed_max_workers},
        {"default_limit",         &Config::default_search_limit},
        {"max_search_limit",      &Config::max_search_limit},
        {"summarizer_max_tokens", &Config::summarizer_max_tokens},
        {"summarizer_target_pct", &Config::summarizer_target_pct},
        {"summarizer_max_pct",    &Config::summarizer_max_pct},
        {"log_max_age_days",      &Config::log_max_age_days},
        {"housekeeping_interval", &Config::housekeeping_interval},
        {"episode_idle_minutes",  &Config::episode_idle_minutes},
        {"catch_up_batch_size",   &Config::catch_up_batch_size},
        {"project_gap_days",      &Config::project_gap_days},
        {"max_turn_failures",     &Config::max_turn_failures},
        {"minimum_chunk_size",    &Config::minimum_chunk_size},
    };
    return m;
}

const std::unordered_map<std::string_view, float Config::*>& float_fields() {
    static const std::unordered_map<std::string_view, float Config::*> m = {
        {"default_min_score",       &Config::default_min_score},
        {"bm25_weight",             &Config::bm25_weight},
        {"vector_weight",           &Config::vector_weight},
        {"phon_weight",             &Config::phon_weight},
        {"cleanup_max_age_hours",   &Config::cleanup_max_age_hours},
        {"episode_threshold_base",  &Config::episode_threshold_base},
        {"episode_threshold_cap",   &Config::episode_threshold_cap},
        {"episode_threshold_step",  &Config::episode_threshold_step},
        {"episode_step_minutes",    &Config::episode_step_minutes},
    };
    return m;
}

const std::unordered_map<std::string_view, std::string Config::*>& string_fields() {
    static const std::unordered_map<std::string_view, std::string Config::*> m = {
        {"bind",                  &Config::bind_address},
        {"server_name",           &Config::server_name},
        {"cert",                  &Config::tls_cert},
        {"key",                   &Config::tls_key},
        {"embedding_model",       &Config::embedding_model},
        {"embedding_vector_type", &Config::embedding_vector_type},
        {"desired_embedding_model",       &Config::desired_embedding_model},
        {"desired_embedding_vector_type", &Config::desired_embedding_vector_type},
        {"summarizer_model",      &Config::summarizer_model},
        {"summarizer_api_url",    &Config::summarizer_api_url},
        {"summarizer_api_key",    &Config::summarizer_api_key},
        {"summarizer_prompt",     &Config::summarizer_prompt},
        {"log_level",             &Config::log_level},
        {"default_recipe",        &Config::default_recipe},
    };
    return m;
}

// log_max_size_mb is the lone `long` field — handled specially.

std::string fmt_float(float v) {
    // Shortest round-trippable representation, no trailing zeros.
    char buf[32];
    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), v);
    if (ec != std::errc{}) return "0";
    return std::string(buf, ptr);
}

bool parse_bool_strict(std::string_view v, bool& out) {
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    if (s == "true" || s == "yes" || s == "1")  { out = true;  return true; }
    if (s == "false"|| s == "no"  || s == "0")  { out = false; return true; }
    return false;
}

bool parse_int_strict(std::string_view v, long& out) {
    auto s = v;
    while (!s.empty() && std::isspace((unsigned char)s.front())) s.remove_prefix(1);
    while (!s.empty() && std::isspace((unsigned char)s.back()))  s.remove_suffix(1);
    if (s.empty()) return false;
    long val{};
    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), val);
    if (ec != std::errc{} || ptr != s.data() + s.size()) return false;
    out = val;
    return true;
}

bool parse_float_strict(std::string_view v, double& out) {
    auto s = v;
    while (!s.empty() && std::isspace((unsigned char)s.front())) s.remove_prefix(1);
    while (!s.empty() && std::isspace((unsigned char)s.back()))  s.remove_suffix(1);
    if (s.empty()) return false;
    // std::from_chars(double) is unavailable on this macOS SDK; use strtod on
    // a NUL-terminated copy and require the whole token to be consumed.
    std::string str(s);
    char* end = nullptr;
    errno = 0;
    double val = std::strtod(str.c_str(), &end);
    if (end != str.c_str() + str.size() || errno == ERANGE) return false;
    out = val;
    return true;
}

} // namespace

std::optional<std::string> get_config_value(const Config& cfg, std::string_view key) {
    const ConfigMeta* meta = lang::config_meta(key);
    if (!meta) return std::nullopt;

    if (auto it = bool_fields().find(key); it != bool_fields().end())
        return (cfg.*(it->second)) ? "true" : "false";
    if (auto it = int_fields().find(key); it != int_fields().end())
        return std::to_string(cfg.*(it->second));
    if (auto it = float_fields().find(key); it != float_fields().end())
        return fmt_float(cfg.*(it->second));
    if (auto it = string_fields().find(key); it != string_fields().end())
        return cfg.*(it->second);
    if (key == "log_max_size_mb")
        return std::to_string(cfg.log_max_size_mb);

    return std::nullopt;
}

std::expected<void, ConfigSetError> validate_config_value(std::string_view key,
                                                          std::string_view value) {
    const ConfigMeta* meta = lang::config_meta(key);
    if (!meta) return std::unexpected(ConfigSetError::UnknownKey);
    // Empty = reset to default: always valid.
    if (value.empty()) return {};

    switch (meta->type) {
        case CfgType::Boolean: {
            bool b;
            if (!parse_bool_strict(value, b))
                return std::unexpected(ConfigSetError::InvalidValue);
            return {};
        }
        case CfgType::Integer: {
            long n;
            if (!parse_int_strict(value, n))
                return std::unexpected(ConfigSetError::InvalidValue);
            return {};
        }
        case CfgType::Float: {
            double d;
            if (!parse_float_strict(value, d))
                return std::unexpected(ConfigSetError::InvalidValue);
            return {};
        }
        case CfgType::Enum: {
            // value must be one of the comma-separated options.
            std::string_view opts = meta->options;
            size_t pos = 0;
            while (pos <= opts.size()) {
                size_t comma = opts.find(',', pos);
                std::string_view tok = (comma == std::string_view::npos)
                    ? opts.substr(pos)
                    : opts.substr(pos, comma - pos);
                if (tok == value) return {};
                if (comma == std::string_view::npos) break;
                pos = comma + 1;
            }
            return std::unexpected(ConfigSetError::InvalidValue);
        }
        case CfgType::String:
        case CfgType::Path:
        case CfgType::Text:
            return {};
    }
    return {};
}

std::expected<ConfigSetResult, ConfigSetError> apply_config_value(
    Config& cfg, std::string_view key, std::string_view value,
    bool allow_locked) {

    const ConfigMeta* meta = lang::config_meta(key);
    if (!meta) return std::unexpected(ConfigSetError::UnknownKey);
    if (meta->edit == CfgEdit::Locked && !allow_locked)
        return std::unexpected(ConfigSetError::Locked);

    if (auto v = validate_config_value(key, value); !v.has_value())
        return std::unexpected(v.error());

    // Empty means "reset to schema default" — re-run apply with the default.
    std::string_view eff = value.empty() ? meta->default_value : value;

    if (auto it = bool_fields().find(key); it != bool_fields().end()) {
        bool b; parse_bool_strict(eff, b);
        cfg.*(it->second) = b;
    } else if (auto it = int_fields().find(key); it != int_fields().end()) {
        long n{}; parse_int_strict(eff, n);
        cfg.*(it->second) = static_cast<int>(n);
    } else if (auto it = float_fields().find(key); it != float_fields().end()) {
        double d{}; parse_float_strict(eff, d);
        cfg.*(it->second) = static_cast<float>(d);
    } else if (auto it = string_fields().find(key); it != string_fields().end()) {
        cfg.*(it->second) = std::string(eff);
    } else if (key == "log_max_size_mb") {
        long n{}; parse_int_strict(eff, n);
        cfg.log_max_size_mb = n;
    } else {
        return std::unexpected(ConfigSetError::UnknownKey);
    }

    ConfigSetResult r;
    r.restart_required = (meta->edit == CfgEdit::RestartRequired);
    r.rebuild_required = (meta->edit == CfgEdit::RebuildRequired);
    return r;
}

void overlay_settings_from_db(Config& cfg, const std::string& db_path) {
    namespace fs = std::filesystem;
    if (db_path.empty() || !fs::exists(db_path)) return;

    UserStore store(db_path);
    for (const auto& meta : lang::config_schema()) {
        auto v = store.get_setting(std::string(meta.key));
        if (!v.has_value()) continue;        // no row -> keep INI/default
        // Best-effort: ignore invalid rows rather than fail startup.
        // allow_locked: DB rows are the authoritative persisted state, so
        // they must apply even for Locked keys (e.g. `port`). Locked only
        // gates USER writes (CLI/dashboard), not the DB->live overlay.
        (void)apply_config_value(cfg, meta.key, *v, /*allow_locked=*/true);
    }

    // Seed the desired_* embedding identity from current when unset, so the
    // dashboard shows the live values as the starting target rather than
    // blanks. "current" here is the config's embedding_* (which the drift
    // guard keeps in sync with the stored vectors).
    if (cfg.desired_embedding_model.empty())
        cfg.desired_embedding_model = cfg.embedding_model;
    if (cfg.desired_embedding_vector_type.empty())
        cfg.desired_embedding_vector_type = cfg.embedding_vector_type;
    if (cfg.desired_embedding_dimensions == 0)
        cfg.desired_embedding_dimensions = cfg.embedding_dimensions;

    // Seed desired_port from the committed port when unset, so the dashboard
    // shows the live port as the starting target. The startup rectify (see
    // main serve) then adopts any UI change into `port` on the next restart.
    if (cfg.desired_port == 0)
        cfg.desired_port = cfg.port;
}

std::vector<std::string_view> all_config_keys() {
    std::vector<std::string_view> keys;
    for (const auto& meta : lang::config_schema())
        keys.push_back(meta.key);
    return keys;
}

std::expected<ConfigSetResult, ConfigSetError> set_config_persisted(
    std::string_view key, std::string_view value, bool allow_locked) {

    const ConfigMeta* meta = lang::config_meta(key);
    if (!meta) return std::unexpected(ConfigSetError::UnknownKey);
    if (meta->edit == CfgEdit::Locked && !allow_locked)
        return std::unexpected(ConfigSetError::Locked);

    // Update the live config first (also validates). On failure, nothing is
    // persisted.
    Config& cfg = mutable_config();
    auto applied = apply_config_value(cfg, key, value, allow_locked);
    if (!applied.has_value()) return std::unexpected(applied.error());

    // Persist to the DB settings table. Empty value stores an empty row,
    // which overlay_settings_from_db() re-interprets as "use default" — so
    // the round-trip stays consistent. Best-effort: if the DB isn't there
    // yet, the live change still stands.
    const std::string db_path = cfg.resolved_db_path();
    if (std::filesystem::exists(db_path)) {
        UserStore store(db_path);
        store.set_setting(std::string(key), std::string(value));
    }

    return applied;
}

int migrate_ini_to_db(const std::string& db_path) {
    namespace fs = std::filesystem;

    // Need an existing, schema-initialized DB to write into.
    if (db_path.empty() || !fs::exists(db_path)) return 0;

    UserStore store(db_path);

    // Idempotency marker — lives outside the config schema, so
    // overlay_settings_from_db() ignores it. Once set, the migration is done
    // for good, even though bootstrap_user_config() may recreate a default
    // settings.ini on a later launch.
    if (store.get_setting("ini_migrated").has_value()) return 0;

    // Resolve the INI path from the live config's base dir.
    const std::string ini_path = mutable_config().resolved_settings_path();
    if (!fs::exists(ini_path)) {
        // No legacy file to migrate — mark done so we don't re-check forever.
        store.set_setting("ini_migrated", "true");
        return 0;
    }

    // Parse the INI into a throwaway Config. On parse failure, leave the file
    // in place and do NOT mark migrated — a human should look.
    auto parsed = load_config(ini_path);
    if (!parsed.has_value()) return 0;
    const Config& ini_cfg = *parsed;

    // A fresh default Config gives us the "schema default" baseline to compare
    // against, so we only import values the user actually customized.
    Config defaults;

    int imported = 0;
    for (const auto& meta : lang::config_schema()) {
        // Never migrate the desired_* mirrors or the locked drift-guard
        // current-embedding identity; those are managed by their own routines.
        // (They simply won't differ from defaults in a legacy INI anyway.)
        auto ini_val = get_config_value(ini_cfg, meta.key);
        auto def_val = get_config_value(defaults, meta.key);
        if (!ini_val.has_value()) continue;
        if (def_val.has_value() && *ini_val == *def_val) continue;  // unchanged

        // Existing DB rows always win — never clobber a value already set.
        if (store.get_setting(std::string(meta.key)).has_value()) continue;

        store.set_setting(std::string(meta.key), *ini_val);
        ++imported;
    }

    // Retire the INI so it never seeds again. If the rename fails (permissions),
    // still mark migrated: the DB now owns the values and re-importing would be
    // wrong (it would resurrect stale INI values over later DB edits).
    std::error_code ec;
    fs::rename(ini_path, ini_path + ".migrated", ec);

    store.set_setting("ini_migrated", "true");
    return imported;
}

} // namespace ragger
