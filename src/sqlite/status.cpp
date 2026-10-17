/**
 * sqlite/status.cpp — agent_update status transition table (PHASE3 Step 4),
 * moved verbatim from agent_write.cpp's decision_transition_ok.
 */
#include "sqlite/status.h"

#include <map>
#include <set>
#include <string>

namespace ragger::sqlite::status {

bool transition_ok(std::string_view from, std::string_view to) {
    static const std::map<std::string, std::set<std::string>> kAllowed = {
        {"pending",     {"current", "provisional", "retired"}},
        {"provisional", {"current", "retired"}},
        {"current",     {"provisional", "retired"}},
        {"roadmap",     {"current", "provisional", "retired"}},
        {"superseded",  {}},
        {"retired",     {}},
    };
    auto it = kAllowed.find(std::string(from));
    if (it == kAllowed.end()) return false;
    return it->second.count(std::string(to)) > 0;
}

} // namespace ragger::sqlite::status
