#include "ragger/config.h"
#include <cassert>
#include <cstdlib>
#include <iostream>

int main() {
    // expand_path with ~
    std::string expanded = ragger::expand_path("~/.ragger/memories.db");
    const char* home = std::getenv("HOME");
    assert(home != nullptr);
    assert(expanded.find('~') == std::string::npos);
    assert(expanded.find(home) == 0);

    // expand_path without ~
    assert(ragger::expand_path("/absolute/path") == "/absolute/path");
    assert(ragger::expand_path("relative/path") == "relative/path");

    // constants
    assert(ragger::EMBEDDING_DIMENSIONS == 384);
    assert(ragger::DEFAULT_PORT == 8432);

    std::cout << "test_config: all passed\n";
    return 0;
}
