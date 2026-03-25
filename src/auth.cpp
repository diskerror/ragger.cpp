/**
 * Authentication utilities implementation
 */
#include "ragger/auth.h"
#include "ragger/config.h"

#include <CommonCrypto/CommonDigest.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <random>
#include <cstdio>
#include <pwd.h>
#include <sys/stat.h>
#include <unistd.h>

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
    unsigned char hash[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(token.c_str(), static_cast<CC_LONG>(token.size()), hash);
    
    // Convert to hex string
    std::ostringstream oss;
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) {
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

std::pair<std::string, std::string> rotate_token_for_user(const std::string& username) {
    // Generate new token
    std::string new_token = generate_token();
    std::string new_hash = hash_token(new_token);
    
    // Determine token file path for the user
    // For multi-user setup, tokens live in each user's home directory
    // Get user's home directory
    std::string user_home;
    struct passwd* pw = getpwnam(username.c_str());
    if (pw) {
        user_home = pw->pw_dir;
    } else {
        // Fallback: assume current user if username lookup fails
        const char* home = std::getenv("HOME");
        if (home) user_home = home;
    }
    
    if (user_home.empty()) {
        throw std::runtime_error("Cannot determine home directory for user: " + username);
    }
    
    std::string token_file = user_home + "/.ragger/token";
    
    // Ensure .ragger directory exists
    fs::create_directories(fs::path(token_file).parent_path());
    
    // Write new token
    std::ofstream f(token_file);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to write token file: " + token_file);
    }
    f << new_token << std::endl;
    f.close();
    
    // Set permissions to 0640
    chmod(token_file.c_str(), 0640);
    
    return {new_token, new_hash};
}

} // namespace ragger
