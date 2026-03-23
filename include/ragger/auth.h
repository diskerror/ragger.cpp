/**
 * Authentication utilities for Ragger Memory
 *
 * Token management, hashing, and persistence.
 */
#pragma once

#include <string>
#include <optional>

namespace ragger {

/// SHA-256 hash of a token string (hex digest)
std::string hash_token(const std::string& token);

/// Path to user's token file (~/.ragger/token)
std::string token_path();

/// Load token from file, or empty string if not found
std::string load_token();

/// Generate a cryptographically secure token
std::string generate_token();

/// Load existing token or generate new one. Creates ~/.ragger/ if needed.
std::string ensure_token();

} // namespace ragger
