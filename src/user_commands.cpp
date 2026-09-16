/**
 * User-management CLI commands. See user_commands.h.
 */
#include "user_commands.h"

#include "ProgramOptions.h"
#include "auth.h"
#include "config.h"
#include "lang.h"
#include "Logger.h"
#include "user_store.h"

#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <grp.h>
#include <iostream>
#include <print>
#include <pwd.h>
#include <sys/stat.h>
#include <termios.h>
#include <unistd.h>

namespace ragger {

namespace fs = std::filesystem;

std::string read_password(const std::string &prompt) {
    std::cout << prompt;
    std::cout.flush();

    struct termios old_term, new_term;
    tcgetattr(STDIN_FILENO, &old_term);
    new_term = old_term;
    new_term.c_lflag &= ~ECHO;
    tcsetattr(STDIN_FILENO, TCSANOW, &new_term);

    std::string password;
    std::getline(std::cin, password);

    tcsetattr(STDIN_FILENO, TCSANOW, &old_term);
    std::cout << "\n";

    // Trim trailing whitespace/CR (paranoia)
    while (!password.empty() && (password.back() == '\r' || password.back() == '\n' || password.back() == ' '))
        password.pop_back();

    return password;
}

std::pair<std::string, bool> provision_user(const std::string &username,
                                            const std::string &home_override) {
    std::string home_dir = home_override;
    if (home_dir.empty()) {
        struct passwd *pw = getpwnam(username.c_str());
        if (!pw) throw std::runtime_error(std::format(ragger::lang::ERR_USER_NOT_FOUND, username));
        home_dir = pw->pw_dir;
    }

    std::string ragger_dir = home_dir + "/.ragger";
    std::string tok_path = ragger_dir + "/token";

    // Check existing
    if (fs::exists(tok_path)) {
        std::ifstream f(tok_path);
        std::string token;
        std::getline(f, token);
        // trim
        size_t s = token.find_first_not_of(" \t\r\n");
        size_t e = token.find_last_not_of(" \t\r\n");
        if (s != std::string::npos) {
            token = token.substr(s, e - s + 1);
            if (!token.empty()) return {token, false};
        }
    }

    // Create directory and token
    fs::create_directories(ragger_dir);
    std::string token = ragger::generate_token();
    {
        std::ofstream f(tok_path);
        f << token << "\n";
    }
    chmod(tok_path.c_str(), 0660);

    // Set ownership if running as root: user owns, ragger group for daemon access
    if (getuid() == 0) {
        struct passwd *pw = getpwnam(username.c_str());
        struct group *rg = getgrnam("ragger");
        if (pw) {
            gid_t gid = rg ? rg->gr_gid : pw->pw_gid;
            chown(ragger_dir.c_str(), pw->pw_uid, gid);
            chmod(ragger_dir.c_str(), 0770);
            chown(tok_path.c_str(), pw->pw_uid, gid);
            // Also fix memories.db if it exists
            std::string db_path = ragger_dir + "/memories.db";
            if (fs::exists(db_path)) {
                chown(db_path.c_str(), pw->pw_uid, gid);
                chmod(db_path.c_str(), 0660);
            }
        }
    }

    return {token, true};
}

int cmd_useradd(Diskerror::ProgramOptions &opts, const std::string &db_path) {
    // Create a new user and issue a bearer token. Token is printed
    // exactly once — caller must save it. Errors if user exists
    // (use `usermod <name>` to rotate an existing user's token).
    auto args = opts.getParams("args");
    if (args.empty()) {
        Diskerror::Logger::critical(ragger::lang::CLI_USAGE_USERADD);
        return 1;
    }
    std::string username = args[0];

    try {
        ragger::UserStore storage(db_path);
        if (storage.get_user_by_username(username)) {
            Diskerror::Logger::error(std::format(ragger::lang::ERR_USERADD_EXISTS, username) + "\n"
                                  + std::format(ragger::lang::ERR_USERADD_EXISTS_HINT, username));
            return 1;
        }
        std::string token = ragger::generate_token();
        std::string token_hash = ragger::hash_token(token);
        storage.create_user(username, token_hash);
        std::println(ragger::lang::MSG_USER_ADDED, username);
        std::println("");
        std::println(ragger::lang::MSG_TOKEN_VALUE, token);
        std::println("");
        std::println("{}", ragger::lang::MSG_TOKEN_SAVE_WARNING);
    }
    catch (const std::exception &e) {
        Diskerror::Logger::critical(e.what());
        return 1;
    }
    return 0;
}

int cmd_usermod(Diskerror::ProgramOptions &opts, const std::string &db_path) {
    // Rotate an existing user's bearer token. Prints the new token once.
    // Errors if the user does not exist.
    auto args = opts.getParams("args");
    if (args.empty()) {
        Diskerror::Logger::error(ragger::lang::CLI_USAGE_USERMOD);
        return 1;
    }
    std::string username = args[0];

    try {
        ragger::UserStore storage(db_path);
        if (!storage.get_user_by_username(username)) {
            Diskerror::Logger::error(std::format(ragger::lang::ERR_USERMOD_MISSING, username) + "\n"
                                  + std::format(ragger::lang::ERR_USERMOD_MISSING_HINT, username));
            return 1;
        }
        std::string token = ragger::generate_token();
        std::string token_hash = ragger::hash_token(token);
        storage.update_user_token(username, token_hash);
        std::println(ragger::lang::MSG_TOKEN_ROTATED, username);
        std::println("");
        std::println(ragger::lang::MSG_TOKEN_VALUE, token);
        std::println("");
        std::println("{}", ragger::lang::MSG_TOKEN_SAVE_WARNING);
    }
    catch (const std::exception &e) {
        Diskerror::Logger::critical(e.what());
        return 1;
    }
    return 0;
}

int cmd_userdel(Diskerror::ProgramOptions &opts, const std::string &db_path) {
    auto args = opts.getParams("args");
    if (args.empty()) {
        Diskerror::Logger::critical(ragger::lang::CLI_USAGE_USERDEL);
        return 1;
    }
    std::string username = args[0];

    try {
        ragger::UserStore storage(db_path);
        ragger::userdel(storage, username);
        std::println(ragger::lang::MSG_USER_REMOVED, username);
    }
    catch (const std::exception &e) {
        Diskerror::Logger::critical(e.what());
        return 1;
    }
    return 0;
}

int cmd_add_self(const std::string &db_path) {
    // getpwuid is more reliable than getlogin in non-TTY contexts
    struct passwd *self_pw = getpwuid(getuid());
    char *login = self_pw ? self_pw->pw_name : nullptr;
    if (!login) {
        Diskerror::Logger::error(ragger::lang::ERR_UNKNOWN_USER);
        return 1;
    }
    std::string username(login);
    auto [token, created] = provision_user(username);
    if (created)
        std::println(ragger::lang::MSG_TOKEN_CREATED, username);
    else
        std::println(ragger::lang::MSG_TOKEN_EXISTS, username);
    std::println(ragger::lang::MSG_YOUR_TOKEN, token);
    std::println("{}", ragger::lang::MSG_TOKEN_USE_HINT);
    std::println("{}", ragger::lang::MSG_TOKEN_FILE_HINT);
    // Register directly in DB
    // Note: Multi-user mode removed. These user management commands are deprecated.
    try {
        ragger::UserStore backend(db_path);
        std::string token_hash = ragger::hash_token(token);
        auto existing = backend.get_user_by_username(username);
        if (existing) {
            if (existing->token_hash != token_hash)
                backend.update_user_token(username, token_hash);
            std::println(ragger::lang::MSG_USER_IN_DB, existing->id);
        }
        else {
            int user_id = backend.create_user(username, token_hash);
            std::println(ragger::lang::MSG_USER_REGISTERED, user_id);
        }
    }
    catch (const std::exception &e) {
        std::println(ragger::lang::WARN_DB_DEFERRED, e.what());
    }
    return 0;
}

int cmd_passwd(Diskerror::ProgramOptions &opts, const std::string &db_path) {
    // Set (or clear) a user's web-UI login password. Only needed for
    // remote users who log in through the browser — local (127.0.0.1
    // / unix socket) sessions auto-authenticate as the daemon owner.
    // Empty password clears web-UI access for that user.
    auto args = opts.getParams("args");
    if (args.empty()) {
        Diskerror::Logger::error(ragger::lang::CLI_USAGE_PASSWD);
        return 1;
    }
    std::string target_user = args[0];

    try {
        ragger::UserStore umgr(db_path);
        auto user_info = umgr.get_user_by_username(target_user);
        if (!user_info) {
            Diskerror::Logger::error(std::format(ragger::lang::ERR_USERMOD_MISSING, target_user) + "\n"
                                  + std::format(ragger::lang::ERR_PASSWD_MISSING_HINT, target_user));
            return 1;
        }

        std::string new_pass = read_password(ragger::lang::PROMPT_NEW_PASSWORD);
        if (new_pass.empty()) {
            umgr.set_user_password(target_user, "");
            std::println(ragger::lang::MSG_PASSWORD_CLEARED, target_user);
        }
        else {
            std::string confirm = read_password(ragger::lang::PROMPT_CONFIRM_PASSWORD);
            if (new_pass != confirm) {
                Diskerror::Logger::critical(ragger::lang::ERR_PASSWORDS_DIFFER);
                return 1;
            }
            std::string hash = ragger::hash_password(new_pass);
            umgr.set_user_password(target_user, hash);
            std::println(ragger::lang::MSG_PASSWORD_SET, target_user);
        }
    }
    catch (const std::exception &e) {
        Diskerror::Logger::critical(e.what());
        return 1;
    }
    return 0;
}

} // namespace ragger
