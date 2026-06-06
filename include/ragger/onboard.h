/**
 * `ragger onboard` — guided first-run setup.
 *
 * Walks the user through the choices that don't have a single safe default
 * for everyone: capture / build_context, default recipe, inference endpoint
 * + model, embedding model download, daemon start.
 *
 * Idempotent — re-runs surface the current state and let you change one
 * section without disturbing the rest. Plugin (Claude Code / Claude
 * Desktop) wiring stays in its own scripts.
 */
#pragma once

#include <string>
#include <vector>

namespace ragger {

/// Run the interactive onboarding flow. Returns 0 on completion, non-zero
/// on a fatal error (file write failure, missing settings.ini, etc.).
int run_onboard(const std::vector<std::string>& args,
                const std::string& db_path);

} // namespace ragger
