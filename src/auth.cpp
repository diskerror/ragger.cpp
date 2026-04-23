/**
 * Authentication utilities implementation
 */
#include "ragger/auth.h"
#include "ragger/config.h"
#include "ragger/storage_backend.h"
#include "ragger/storage_types.h"

#include <openssl/sha.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <random>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <pwd.h>
#include <sys/stat.h>
#include <unistd.h>
#include <format>

namespace ragger {

namespace fs = std::filesystem;

// Base64url encoding (RFC 4648 section 5)
static std::string base64url_encode(const unsigned char* data, size_t len) {
    static const char* base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    
    std::string result;
    result.reserve((len * 4 / 3) + 3);
    
    for (size_t i = 0; i < len; ) {
        uint32_t octet_a = i < len ? data[i++] : 0;
        uint32_t octet_b = i < len ? data[i++] : 0;
        uint32_t octet_c = i < len ? data[i++] : 0;
        uint32_t triple = (octet_a << 16) + (octet_b << 8) + octet_c;
        
        result.push_back(base64_chars[(triple >> 18) & 0x3F]);
        result.push_back(base64_chars[(triple >> 12) & 0x3F]);
        result.push_back(base64_chars[(triple >> 6) & 0x3F]);
        result.push_back(base64_chars[triple & 0x3F]);
    }
    
    // Remove padding (base64url doesn't use padding)
    size_t padding = (3 - (len % 3)) % 3;
    if (padding > 0) {
        result.resize(result.size() - padding);
    }
    
    return result;
}

std::string hash_token(const std::string& token) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(token.c_str()), 
           token.size(), hash);
    
    // Convert to hex string
    std::ostringstream oss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<int>(hash[i]);
    }
    return oss.str();
}

std::string token_path() {
    return expand_path("~/.ragger/token");
}

std::string load_token() {
    std::string path = token_path();
    std::ifstream f(path);
    if (!f.is_open()) return "";
    
    std::string token;
    std::getline(f, token);
    
    // Trim whitespace
    size_t start = token.find_first_not_of(" \t\r\n");
    size_t end = token.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    return token.substr(start, end - start + 1);
}

std::string generate_token() {
    unsigned char random_bytes[32];
    
    // Read from /dev/urandom
    FILE* f = fopen("/dev/urandom", "rb");
    if (!f) {
        throw std::runtime_error("Failed to open /dev/urandom");
    }
    size_t read_count = fread(random_bytes, 1, 32, f);
    fclose(f);
    
    if (read_count != 32) {
        throw std::runtime_error("Failed to read enough random bytes");
    }
    
    return base64url_encode(random_bytes, 32);
}

std::string ensure_token() {
    std::string path = token_path();
    
    // Check if token file exists
    if (fs::exists(path)) {
        std::string token = load_token();
        if (!token.empty()) return token;
    }
    
    // Create parent directory
    fs::create_directories(fs::path(path).parent_path());
    
    // Generate new token
    std::string token = generate_token();
    
    // Write to file
    std::ofstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to write token file: " + path);
    }
    f << token << std::endl;
    f.close();
    
    // Set permissions to 0640 (owner read/write + group read)
    // Group read allows the daemon (_ragger) to read user tokens
    chmod(path.c_str(), 0640);
    
    return token;
}

// --- Static ISO timestamp (informational use only) ---

static std::string iso_timestamp_now() {
    auto tt = std::chrono::system_clock::to_time_t(
                  std::chrono::system_clock::now());
    std::tm gm{};
    gmtime_r(&tt, &gm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &gm);
    return buf;
}

// NOTE: The old time-seeded token rotation system (server.secret +
// token_for_minute + 24h verify window) has been removed. Tokens are now
// plain random values: generate_token() → hash_token() → stored in DB.
// The framework (users.token_hash, users.token_rotated_at, issue_token,
// verify_token) is kept for the planned labeled-token system (see GitHub
// issue: "Labeled API tokens").

// PBKDF2 password hashing
static const int PBKDF2_ITERATIONS = 600000;
static const int PBKDF2_SALT_LEN = 16;
static const int PBKDF2_KEY_LEN = 32;

static std::string bytes_to_hex(const unsigned char* data, size_t len) {
    std::ostringstream oss;
    for (size_t i = 0; i < len; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<int>(data[i]);
    }
    return oss.str();
}

static std::vector<unsigned char> hex_to_bytes(const std::string& hex) {
    std::vector<unsigned char> bytes;
    for (size_t i = 0; i + 1 < hex.size(); i += 2) {
        unsigned int byte;
        std::sscanf(hex.c_str() + i, "%02x", &byte);
        bytes.push_back(static_cast<unsigned char>(byte));
    }
    return bytes;
}

std::string generate_random_token(int bytes) {
    std::vector<unsigned char> buf(bytes);
    if (RAND_bytes(buf.data(), bytes) != 1) {
        throw std::runtime_error("Failed to generate random bytes");
    }
    // URL-safe base64-like: hex encoding (simple, always works)
    return bytes_to_hex(buf.data(), bytes);
}

std::string hash_password(const std::string& password) {
    unsigned char salt[PBKDF2_SALT_LEN];
    if (RAND_bytes(salt, PBKDF2_SALT_LEN) != 1) {
        throw std::runtime_error("Failed to generate random salt");
    }
    
    unsigned char key[PBKDF2_KEY_LEN];
    if (PKCS5_PBKDF2_HMAC(
            password.c_str(), static_cast<int>(password.size()),
            salt, PBKDF2_SALT_LEN,
            PBKDF2_ITERATIONS,
            EVP_sha256(),
            PBKDF2_KEY_LEN, key) != 1) {
        throw std::runtime_error("PBKDF2 failed");
    }
    
    // Format: pbkdf2:iterations:salt_hex:key_hex
    return "pbkdf2:" + std::to_string(PBKDF2_ITERATIONS) + ":"
         + bytes_to_hex(salt, PBKDF2_SALT_LEN) + ":"
         + bytes_to_hex(key, PBKDF2_KEY_LEN);
}

bool verify_password(const std::string& password, const std::string& stored_hash) {
    // Parse "pbkdf2:iterations:salt_hex:key_hex"
    if (stored_hash.substr(0, 7) != "pbkdf2:") return false;
    
    size_t pos1 = 7;
    size_t pos2 = stored_hash.find(':', pos1);
    if (pos2 == std::string::npos) return false;
    size_t pos3 = stored_hash.find(':', pos2 + 1);
    if (pos3 == std::string::npos) return false;
    
    int iterations = std::stoi(stored_hash.substr(pos1, pos2 - pos1));
    auto salt = hex_to_bytes(stored_hash.substr(pos2 + 1, pos3 - pos2 - 1));
    auto expected_key = hex_to_bytes(stored_hash.substr(pos3 + 1));
    
    unsigned char key[PBKDF2_KEY_LEN];
    if (PKCS5_PBKDF2_HMAC(
            password.c_str(), static_cast<int>(password.size()),
            salt.data(), static_cast<int>(salt.size()),
            iterations,
            EVP_sha256(),
            PBKDF2_KEY_LEN, key) != 1) {
        return false;
    }
    
    // Constant-time comparison
    return CRYPTO_memcmp(key, expected_key.data(),
                         std::min(static_cast<size_t>(PBKDF2_KEY_LEN), expected_key.size())) == 0;
}

// --- DB-backed API ---

void userdel(StorageBackend& db, const std::string& user) {
    db.delete_user(user);
}

bool verify_password(StorageBackend& db, const std::string& user, const std::string& pw) {
    auto stored = db.get_user_password(user);
    if (!stored || stored->empty()) return false;
    return verify_password(pw, *stored);
}

std::string issue_token(StorageBackend& db, const std::string& user) {
    // Generate a fresh random token, store its hash, record issue time.
    // Returns the raw token — caller must deliver it to the user once;
    // only the hash is stored in the DB.
    std::string token = generate_token();
    std::string token_hash = hash_token(token);
    db.update_user_token(user, token_hash);
    db.update_user_token_rotated_at(user, iso_timestamp_now());
    return token;
}

AuthResult verify_token(StorageBackend& db, const std::string& token) {
    // Hash the presented token and look it up. Valid until explicitly
    // revoked — no time window. (The old 24h rotation window is gone.)
    std::string token_hash = hash_token(token);
    auto user_info = db.get_user_by_token_hash(token_hash);
    if (!user_info) return {};
    return {true, user_info->username};
}

} // namespace ragger
