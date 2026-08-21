/**
 * Auth module tests
 *
 * Tests token generation, hashing, and persistence.
 */
#include "auth.h"
#include "config.h"
#include "user_store.h"
#include "util/fs.h"
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <print>
#include <unistd.h>

namespace fs = std::filesystem;

// Test helper: create-or-update a user with a password. The one-shot
// ragger::useradd() API was retired when account creation and password
// setting became distinct verbs; tests still need the combined action.
static void seed_user_with_password(
        ragger::UserStore& db,
        const std::string& user,
        const std::string& pw) {
    std::string pwhash = ragger::hash_password(pw);
    if (!db.get_user_by_username(user)) {
        db.create_user(user, "");
    }
    db.set_user_password(user, pwhash);
}

void test_hash_token() {
    std::println("  test_hash_token...");

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
    std::println("  test_generate_token...");

    auto t1 = ragger::generate_token();
    auto t2 = ragger::generate_token();

    assert(!t1.empty());
    assert(!t2.empty());
    assert(t1 != t2);  // Unique each time
    assert(t1.length() >= 20);  // Reasonable length

    std::println(" OK");
}

void test_token_path() {
    std::println("  test_token_path...");

    // Isolate against a throwaway --ragger-base so this never touches the
    // real ~/.ragger/token.
    std::string ragger_base = "/tmp/ragger_test_home_" + std::to_string(getpid()) + "/.ragger";
    ragger::set_ragger_base_override(ragger_base);

    auto path = ragger::token_path();
    assert(!path.empty());
    assert(path.find("/.ragger/token") != std::string::npos);
    assert(path.rfind(ragger_base, 0) == 0);  // rooted under the override

    ragger::set_ragger_base_override("");
    std::println(" OK");
}

void test_load_token() {
    std::println("  test_load_token...");

    // Isolated, empty ragger-base — no token file exists yet, so this
    // exercises the "not found" path without touching the real ~/.ragger.
    std::string ragger_base = "/tmp/ragger_test_home_" + std::to_string(getpid()) + "_load/.ragger";
    ragger::set_ragger_base_override(ragger_base);

    auto token = ragger::load_token();
    assert(token.empty());  // no token file in the fresh isolated dir

    ragger::set_ragger_base_override("");
    fs::remove_all(ragger_base);
    std::println(" OK");
}

void test_ensure_token_with_temp_dir() {
    std::println("  test_ensure_token_with_temp_dir...");

    // Use the --ragger-base override mechanism (not HOME-swapping) to
    // isolate this test — this is exactly what --ragger-base is for.
    std::string ragger_base = "/tmp/ragger_test_home_" + std::to_string(getpid()) + "/.ragger";
    ragger::set_ragger_base_override(ragger_base);

    // ensure_token should create the directory and token
    auto token1 = ragger::ensure_token();
    assert(!token1.empty());

    // Verify the token file was created under the override, not real ~/.ragger
    std::string token_file = ragger_base + "/token";
    assert(fs::exists(token_file));

    // ensure_token again should return the same token
    auto token2 = ragger::ensure_token();
    assert(token1 == token2);

    // Clean up
    ragger::set_ragger_base_override("");
    fs::remove_all(ragger_base);

    std::println(" OK");
}

void test_useradd_and_verify_password() {
    std::print("  test_useradd_and_verify_password...");
    
    std::string db_path = "/tmp/ragger_test_auth_useradd.db";
    fs::remove(db_path);
    {
        ragger::UserStore db(db_path);
        seed_user_with_password(db, "alice", "correct-horse");
        assert(ragger::verify_password(db, "alice", "correct-horse"));
        assert(!ragger::verify_password(db, "alice", "wrong-password"));
        assert(!ragger::verify_password(db, "bob", "anything")); // no such user
    }
    fs::remove(db_path);
    std::println(" OK");
}

void test_useradd_update_existing() {
    std::print("  test_useradd_update_existing...");
    
    std::string db_path = "/tmp/ragger_test_auth_update.db";
    fs::remove(db_path);
    {
        ragger::UserStore db(db_path);
        seed_user_with_password(db, "alice", "pw1");
        seed_user_with_password(db, "alice", "pw2"); // same user, new password — should update
        assert(ragger::verify_password(db, "alice", "pw2"));
        assert(!ragger::verify_password(db, "alice", "pw1"));
    }
    fs::remove(db_path);
    std::println(" OK");
}

void test_userdel() {
    std::print("  test_userdel...");
    
    std::string db_path = "/tmp/ragger_test_auth_del.db";
    fs::remove(db_path);
    {
        ragger::UserStore db(db_path);
        seed_user_with_password(db, "alice", "pw");
        ragger::userdel(db, "alice");
        assert(!ragger::verify_password(db, "alice", "pw"));
        // userdel should not throw on nonexistent user
        ragger::userdel(db, "nonexistent");
    }
    fs::remove(db_path);
    std::println(" OK");
}

void test_token_roundtrip() {
    std::print("  test_token_roundtrip...");
    
    std::string db_path = "/tmp/ragger_test_auth_tokens.db";
    fs::remove(db_path);
    {
        ragger::UserStore db(db_path);
        seed_user_with_password(db, "alice", "pw");
        auto token = ragger::issue_token(db, "alice");
        assert(!token.empty());
        
        // FIXME: verify_token is failing even for valid tokens.
        // This appears to be a bug in the token verification logic,
        // possibly related to token rotation timestamp handling or
        // time-seeded token hashing. The token is issued correctly
        // and stored in the database, but verify_token returns false.
        auto result = ragger::verify_token(db, token);
        assert(result.ok);  // BUG: This assertion fails - investigate verify_token logic
        assert(result.username == "alice");
        
        auto bad = ragger::verify_token(db, "not-a-real-token");
        assert(!bad.ok);
    }
    fs::remove(db_path);
    std::println(" OK");
}

int main() {
    std::println("Running auth tests:");

    // token_path()/load_token()/ensure_token() now resolve through
    // Config::resolved_token_path(), which requires config() to be
    // initialized. Isolate against a throwaway --ragger-base for the whole
    // test binary so nothing here ever touches the real ~/.ragger.
    std::string isolated_base = "/tmp/ragger_test_auth_base_" + std::to_string(getpid()) + "/.ragger";
    ragger::set_ragger_base_override(isolated_base);
    ragger::init_config("");

    test_hash_token();
    test_generate_token();
    test_token_path();
    test_load_token();
    test_ensure_token_with_temp_dir();

    test_useradd_and_verify_password();
    test_useradd_update_existing();
    test_userdel();
    test_token_roundtrip();

    ragger::set_ragger_base_override("");
    fs::remove_all(isolated_base);

    std::println("test_auth: all passed");
    return 0;
}
