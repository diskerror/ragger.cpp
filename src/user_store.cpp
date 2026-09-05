/**
 * UserStore — user and settings management delegated through StorageBackend.
 */
#include "user_store.h"
#include "sqlite_backend.h"
#include "config.h"

#include <filesystem>
#include <stdexcept>
#include <string>

namespace ragger {

namespace fs = std::filesystem;

struct UserStore::Impl {
    std::unique_ptr<SqliteBackend> backend;

    explicit Impl(const std::string& path) {
        std::string resolved = expand_path(path);
        fs::create_directories(fs::path(resolved).parent_path());
        backend = std::make_unique<SqliteBackend>(resolved);
    }
};

UserStore::UserStore(const std::string& db_path)
    : pImpl(std::make_unique<Impl>(db_path)) {}

UserStore::~UserStore() = default;

std::optional<UserInfo> UserStore::get_user_by_username(const std::string& username) {
    return pImpl->backend->get_user_by_username(username);
}

std::optional<std::string> UserStore::get_user_password(const std::string& username) {
    return pImpl->backend->get_user_password(username);
}

void UserStore::update_user_token(const std::string& username, const std::string& new_hash) {
    pImpl->backend->update_user_token(username, new_hash);
}

int UserStore::create_user(const std::string& username, const std::string& token_hash) {
    return pImpl->backend->create_user(username, token_hash);
}

bool UserStore::delete_user(const std::string& username) {
    return pImpl->backend->delete_user(username);
}

void UserStore::set_user_password(const std::string& username,
                                  const std::string& password_hash) {
    pImpl->backend->set_user_password(username, password_hash);
}

std::optional<UserInfo> UserStore::get_user_by_token_hash(const std::string& token_hash) {
    return pImpl->backend->get_user_by_token_hash(token_hash);
}

std::optional<std::string> UserStore::get_setting(const std::string& key) {
    return pImpl->backend->get_setting(key);
}

void UserStore::set_setting(const std::string& key, const std::string& value) {
    pImpl->backend->set_setting(key, value);
}

} // namespace ragger
