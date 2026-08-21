/**
 * Recipe CLI — `ragger recipe [name]`
 *
 * No args: interactive picker over the available recipes (↑/↓ + Enter,
 * `q` quits without choosing). The picker has a synthetic top entry
 * `default` that means "track [server] default_recipe in settings.ini" —
 * pick it to undo a previous override.
 *
 * Named arg: write that recipe (or the literal `default` sentinel) as
 * the active choice and print its details. Non-interactive equivalent
 * of the picker.
 *
 * Persistence: written to the DB `settings` table under the key `recipe`.
 * Resolution precedence at lookup time (in build_context):
 *   1. DB `settings.recipe` if present AND not the sentinel "default"
 *   2. settings.ini `[server] default_recipe`
 *   3. First available built-in
 *
 * The DB write takes effect immediately — no daemon restart needed.
 */
#pragma once

#include <string>
#include <vector>

namespace ragger {

/// Entry point for the `recipe` verb. `args` is everything after the
/// verb token; one optional element = recipe name. `db_path` is forwarded
/// to the DB-only SqliteBackend so the choice can be persisted into the
/// `settings` table. Returns a process exit code (0 success, non-zero
/// on error / no recipes found / unknown name).
int run_recipe_cli(const std::vector<std::string>& args,
                   const std::string& db_path);

} // namespace ragger
