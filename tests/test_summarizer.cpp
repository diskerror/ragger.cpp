#include "ragger/summarizer.h"

#include <cassert>
#include <print>
#include <string>

using ragger::is_system_injected_turn;

// System-injected user turns: should be detected (summarizer skips them).
void test_detects_system_markers() {
    std::print("  test_detects_system_markers...");

    assert(is_system_injected_turn(
        "[System note: Your previous turn was interrupted before you could "
        "process the last tool result(s)."));
    assert(is_system_injected_turn(
        "[Note: model was just switched from claude-sonnet-4-6 to "
        "claude-opus-4-8 via Anthropic. Adjust your self-identification.]"));
    assert(is_system_injected_turn(
        "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted."));
    assert(is_system_injected_turn(
        "[IMPORTANT: Background process proc_b623911a9a5d completed (exit code 1)."));

    // Leading whitespace must not defeat detection.
    assert(is_system_injected_turn("   \n[System note: foo"));

    std::println(" ok");
}

// Genuine human messages: must NOT be treated as system-injected.
void test_ignores_human_turns() {
    std::print("  test_ignores_human_turns...");

    assert(!is_system_injected_turn("Next item: storing system messages."));
    assert(!is_system_injected_turn("stop. leaving it blank is appropriate."));
    assert(!is_system_injected_turn(""));
    // A bracketed but non-system opener is left alone.
    assert(!is_system_injected_turn("[note] check this out"));
    // OUT-OF-BAND wraps a real human message — intentionally NOT a marker.
    assert(!is_system_injected_turn(
        "[OUT-OF-BAND USER MESSAGE] actually, do it the other way"));

    std::println(" ok");
}

int main() {
    std::println("test_summarizer:");
    test_detects_system_markers();
    test_ignores_human_turns();
    std::println("All summarizer tests passed.");
    return 0;
}
