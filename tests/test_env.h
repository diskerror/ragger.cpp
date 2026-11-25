/**
 * test_env.h — isolated ragger base for one test binary.
 *
 * Why this exists: several test binaries called ragger::init_config() with no
 * --ragger-base override, so Server's constructor opened the LIVE
 * ~/.ragger/memories.db (and ran ensure_token()) from the test suite. This
 * fixture points the whole process at a throwaway base before any config or
 * DB access. RULE (RULES.md #17): no test may open ~/.ragger — construct a
 * TestEnv as the FIRST thing in main().
 *
 * Header-only and namespace-less, like check.h.
 */
#pragma once

#include "check.h"
#include "config.h"
#include "util/fs.h"

#include <cstdlib>
#include <filesystem>
#include <string>
#include <unistd.h>

struct TestEnv {
    explicit TestEnv(const std::string& tag) {
        // Capture the REAL model dir before any override exists:
        // ragger_base_dir() is still the developer's ~/.ragger here, and
        // Config{}.embedding_model is the compiled-in default. If a
        // developer's live settings name a different model, tests will SKIP
        // for lack of a model — the "no new SKIPPED" check surfaces that;
        // do not work around it.
        const std::string real_models_root = ragger::ragger_base_dir() + "/models";
        model_dir_ = real_models_root + "/" + ragger::Config{}.embedding_model;

        const char* tmpdir = std::getenv("TMPDIR");
        std::string tmp = (tmpdir && *tmpdir) ? tmpdir : "/tmp";
        if (!tmp.empty() && tmp.back() == '/') tmp.pop_back();  // macOS $TMPDIR has a trailing slash
        base_ = tmp + "/ragger_test_" + tag + "_" + std::to_string(getpid());
        std::filesystem::create_directories(base_);

        // Symlink the real models root into the scratch base so code that
        // resolves the model dir through config() finds the same model.
        // Tests SKIP (not silently pass) when the model is missing.
        if (std::filesystem::exists(real_models_root)) {
            std::error_code ec;
            std::filesystem::create_directory_symlink(real_models_root, base_ + "/models", ec);
            CHECK(!ec);
        }

        // From here on the overlay reads only the (empty) scratch base.
        ragger::set_ragger_base_override(base_);
        ragger::init_config();

        // Safety net: the config must resolve under the scratch base.
        CHECK(ragger::ragger_base_dir().rfind(base_, 0) == 0);
    }

    ~TestEnv() {
        ragger::set_ragger_base_override("");
        std::error_code ec;
        std::filesystem::remove_all(base_, ec);  // removes the models symlink, never its target
    }

    const std::string& base() const { return base_; }
    const std::string& model_dir() const { return model_dir_; }

    TestEnv(const TestEnv&) = delete;
    TestEnv& operator=(const TestEnv&) = delete;

  private:
    std::string base_;
    std::string model_dir_;
};
