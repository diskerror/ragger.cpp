/**
 * Auth module tests
 *
 * Tests token generation, hashing, and persistence.
 */
#include "ragger/auth.h"
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <print>
#include <unistd.h>

namespace fs = std::filesystem;

void test_hash_token() {
    std::print("  test_hash_token..."); std::cout.flush();

    // Known SHA-256 of "test" = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    auto hash = ragger::hash_token("test");
    assert(hash == "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08");

    // Different inputs → different hashes
    assert(ragger::hash_token("a") != ragger::hash_token("b"));

    // Same input → same hash (deterministic)
    assert(ragger::hash_token("hello") == ragger::hash_token("hello"));

    // Empty string should also hash
    auto empty_hash = ragger::hash_token("");
    assert(!empty_hash.empty());
    assert(empty_hash == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");

    std::println(" OK");
}

void test_generate_token() {
    std::print("  test_generate_token..."); std::cout.flush();

    auto t1 = ragger::generate_token();
    auto t2 = ragger::generate_token();

    assert(!t1.empty());
    assert(!t2.empty());
    assert(t1 != t2);  // Unique each time
    assert(t1.length() >= 20);  // Reasonable length

    std::println(" OK");
}

void test_token_path() {
    std::print("  test_token_path..."); std::cout.flush();

    auto path = ragger::token_path();
    assert(!path.empty());
    assert(path.find("/.ragger/token") != std::string::npos);

    std::println(" OK");
}

void test_load_token() {
    std::print("  test_load_token..."); std::cout.flush();

    // This test depends on whether ~/.ragger/token exists
    // We'll just verify it doesn't crash and returns a string
    auto token = ragger::load_token();
    // Token might be empty if file doesn't exist, which is fine

    std::println(" OK");
}

void test_ensure_token_with_temp_dir() {
    std::print("  test_ensure_token_with_temp_dir..."); std::cout.flush();

    // Save original HOME
    const char* original_home = std::getenv("HOME");
    assert(original_home != nullptr);

    // Create temp directory for testing
    std::string temp_home = "/tmp/ragger_test_home_" + std::to_string(getpid());
    fs::create_directories(temp_home);

    // Set temp HOME
    setenv("HOME", temp_home.c_str(), 1);

    // ensure_token should create the directory and token
    auto token1 = ragger::ensure_token();
    assert(!token1.empty());

    // Verify the token file was created
    std::string token_file = temp_home + "/.ragger/token";
    assert(fs::exists(token_file));

    // ensure_token again should return the same token
    auto token2 = ragger::ensure_token();
    assert(token1 == token2);

    // Clean up
    fs::remove_all(temp_home);

    // Restore original HOME
    setenv("HOME", original_home, 1);

    std::println(" OK");
}

void test_rotate_token() {
    std::print("  test_rotate_token..."); std::cout.flush();

    // Save original HOME
    const char* original_home = std::getenv("HOME");
    assert(original_home != nullptr);

    // Create temp directory for testing
    std::string temp_home = "/tmp/ragger_test_rotate_" + std::to_string(getpid());
    fs::create_directories(temp_home + "/.ragger");

    // Set temp HOME
    setenv("HOME", temp_home.c_str(), 1);

    // Create initial token
    std::string initial_token = ragger::generate_token();
    std::string token_file = temp_home + "/.ragger/token";
    std::ofstream f(token_file);
    f << initial_token << std::endl;
    f.close();

    // For testing, pass a fake username and ensure HOME is set correctly
    // The rotate_token_for_user will use $HOME when the username lookup fails
    std::string test_username = "nonexistent_test_user";

    // Rotate the token
    auto [new_token, new_hash] = ragger::rotate_token_for_user(test_username);

    // Verify new token is different
    assert(new_token != initial_token);
    assert(!new_token.empty());
    assert(!new_hash.empty());

    // Verify token file was updated
    std::ifstream f_read(token_file);
    std::string file_token;
    std::getline(f_read, file_token);
    f_read.close();
    
    // Trim whitespace
    size_t start = file_token.find_first_not_of(" \t\r\n");
    size_t end = file_token.find_last_not_of(" \t\r\n");
    if (start != std::string::npos) {
        file_token = file_token.substr(start, end - start + 1);
    }

    assert(file_token == new_token);

    // Verify hash is correct
    assert(new_hash == ragger::hash_token(new_token));

    // Clean up
    fs::remove_all(temp_home);

    // Restore original HOME
    setenv("HOME", original_home, 1);

    std::println(" OK");
}

int main() {
    std::println("Running auth tests:");

    test_hash_token();
    test_generate_token();
    test_token_path();
    test_load_token();
    test_ensure_token_with_temp_dir();
    test_rotate_token();

    std::println("test_auth: all passed");
    return 0;
}
