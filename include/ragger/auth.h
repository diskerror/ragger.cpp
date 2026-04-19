/**
 * Authentication utilities for Ragger Memory
 *
 * Password hashing (PBKDF2) and time-seeded token issuance/verification.
 * All persistent user state is accessed through StorageBackend.
 */
#pragma once

#include <string>
#include <optional>
#include <utility>

namespace ragger {

class StorageBackend;

struct AuthResult {
    bool        ok = false;
    std::string username;
};

// --- Low-level primitives (no DB) ---

/// SHA-256 hash of a string (hex digest).
std::string hash_token(const std::string& token);

/// Generate a cryptographically secure random token (hex-encoded).
std::string generate_random_token(int bytes = 32);

/// Generate a cryptographically secure token (base64url-encoded, 32 bytes).
std::string generate_token();

/// Hash a password using PBKDF2-SHA256. Returns "pbkdf2:iterations:salt:hash".
std::string hash_password(const std::string& password);

/// Verify a password against a stored PBKDF2 hash string.
bool verify_password(const std::string& password, const std::string& stored_hash);

/// Path to user's token file (~/.ragger/token).
std::string token_path();

/// Load token from file, or empty string if not found.
std::string load_token();

/// Load existing token or generate new one. Creates ~/.ragger/ if needed.
std::string ensure_token();

// --- High-level DB-backed API ---

/// Insert a new user or update an existing user's password.
void useradd(StorageBackend& db, const std::string& user, const std::string& pw);

/// Delete a user. No-op if user doesn't exist.
void userdel(StorageBackend& db, const std::string& user);

/// Verify a password for a user by looking up their stored hash.
/// Returns false if user doesn't exist or password doesn't match.
bool verify_password(StorageBackend& db, const std::string& user, const std::string& pw);

/// Issue a time-seeded token for a user. The token is sha256 of
/// (user + ":" + epoch_minute + ":" + server_secret). The hash is
/// stored via update_user_token and the rotated_at timestamp is set.
/// Returns the raw token (the client-facing credential).
std::string issue_token(StorageBackend& db, const std::string& user);

/// Verify a token. Returns ok=true with username if the token's stored
/// hash is found and the rotated_at timestamp is within the last 24 hours.
AuthResult verify_token(StorageBackend& db, const std::string& token);

} // namespace ragger
