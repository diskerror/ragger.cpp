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
//
// Note: user creation / token rotation happens directly in the CLI handlers
// (src/main.cpp) via StorageBackend::{create_user, update_user_token,
// set_user_password}. The one-shot `useradd(db, user, pw)` helper was
// removed — it conflated account creation with password setting, which
// are now distinct verbs (useradd/usermod vs. passwd).

/// Delete a user. No-op if user doesn't exist.
void userdel(StorageBackend& db, const std::string& user);

/// Verify a password for a user by looking up their stored hash.
/// Returns false if user doesn't exist or password doesn't match.
bool verify_password(StorageBackend& db, const std::string& user, const std::string& pw);

/// Issue a fresh random token for a user. Stores the SHA-256 hash in the
/// DB and records the issue timestamp. Returns the raw token — caller
/// must deliver it to the user once; it will not be recoverable afterward.
/// Valid until explicitly revoked (no automatic expiry).
std::string issue_token(StorageBackend& db, const std::string& user);

/// Verify a bearer token. Hashes the presented value and looks it up in
/// the DB. Returns ok=true with username if found; no time window applied.
AuthResult verify_token(StorageBackend& db, const std::string& token);

} // namespace ragger
