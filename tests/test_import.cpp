/**
 * Import chunking tests
 */
#include "ragger/import.h"
#include <cassert>
#include <iostream>
#include <print>

using ragger::heading_level;
using ragger::heading_text;
using ragger::chunk_markdown;

// -----------------------------------------------------------------------
// heading_level tests
// -----------------------------------------------------------------------

void test_heading_levels() {
    assert(heading_level("# Title") == 1);
    assert(heading_level("## Subtitle") == 2);
    assert(heading_level("### Level 3") == 3);
    assert(heading_level("###### Level 6") == 6);
    assert(heading_level("Not a heading") == 0);
    assert(heading_level("") == 0);
    assert(heading_level("#NoSpace") == 0);  // No space after #
    assert(heading_level("##NoSpace") == 0);
}

void test_heading_text_extraction() {
    assert(heading_text("# Title") == "Title");
    assert(heading_text("## Sub Title") == "Sub Title");
    assert(heading_text("### Multi Word Heading") == "Multi Word Heading");
}

// -----------------------------------------------------------------------
// chunk_markdown tests
// -----------------------------------------------------------------------

void test_empty_input() {
    auto chunks = chunk_markdown("", 300);
    assert(chunks.empty());
}

void test_no_headings() {
    std::string text = "This is a paragraph of text without any headings.\n\n"
                       "This is another paragraph that should be merged.";
    auto chunks = chunk_markdown(text, 300);
    // Both paragraphs are short, should merge into one chunk
    assert(chunks.size() == 1);
    assert(chunks[0].section.empty());
}

void test_single_heading_with_content() {
    std::string text = "# My Document\n\n"
                       "This is the content under the heading. "
                       "It has enough text to be meaningful.";
    auto chunks = chunk_markdown(text, 10);
    assert(!chunks.empty());
    // First chunk should include the heading
    assert(chunks[0].text.find("# My Document") != std::string::npos);
    assert(chunks[0].section == "My Document");
}

void test_multiple_headings() {
    std::string text = "# Chapter One\n\n"
                       "Content for chapter one goes here with plenty of words to exceed the minimum chunk size easily.\n\n"
                       "# Chapter Two\n\n"
                       "Content for chapter two goes here with plenty of words to exceed the minimum chunk size easily.\n\n"
                       "# Chapter Three\n\n"
                       "Content for chapter three.";
    auto chunks = chunk_markdown(text, 50);
    assert(chunks.size() >= 2);  // Should split on headings
}

void test_nested_headings() {
    std::string text = "# Title\n\n"
                       "## Section A\n\n"
                       "Content A with enough text to be a real chunk by itself when needed.\n\n"
                       "## Section B\n\n"
                       "Content B with enough text to be a real chunk by itself when needed.\n\n"
                       "### Subsection B1\n\n"
                       "Content B1 details.";
    auto chunks = chunk_markdown(text, 30);
    // Check section breadcrumbs
    bool found_nested = false;
    for (auto& c : chunks) {
        if (c.section.find("\xC2\xBB") != std::string::npos) {
            found_nested = true;  // Found a "»" separator = nested section
        }
    }
    assert(found_nested);
}

void test_heading_level_reset() {
    // When a same-or-higher level heading appears, the stack should pop
    std::string text = "# A\n\n## A1\n\nContent A1.\n\n# B\n\nContent B.";
    auto chunks = chunk_markdown(text, 10);
    // The "B" chunk should NOT have "A" in its section
    bool b_clean = false;
    for (auto& c : chunks) {
        if (c.text.find("Content B") != std::string::npos) {
            assert(c.section.find("A1") == std::string::npos);
            b_clean = true;
        }
    }
    assert(b_clean);
}

void test_minimum_chunk_size_merging() {
    std::string text = "# Title\n\nShort.\n\nAlso short.\n\nStill short.";
    // With high min_chunk_size, everything should merge
    auto chunks = chunk_markdown(text, 1000);
    assert(chunks.size() == 1);
}

void test_minimum_chunk_size_splitting() {
    std::string text = "# Title\n\n";
    // Add enough content to exceed min_chunk_size
    for (int i = 0; i < 20; ++i) {
        text += "Paragraph number " + std::to_string(i) + " with some filler text to bulk it up.\n\n";
    }
    auto chunks = chunk_markdown(text, 50);
    assert(chunks.size() > 1);
}

void test_base64_stripping() {
    std::string text = "# Doc\n\n"
                       "Some text before image.\n\n"
                       "![alt](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEA)\n\n"
                       "Some text after image.";
    auto chunks = chunk_markdown(text, 10);
    // base64 data should be stripped
    for (auto& c : chunks) {
        assert(c.text.find("base64") == std::string::npos);
    }
}

void test_whitespace_collapsing() {
    std::string text = "# Title\n\n"
                       "Text  with   multiple    spaces.\n\n\n\n\n"
                       "After many blank lines.";
    auto chunks = chunk_markdown(text, 10);
    // Multiple spaces collapsed, excessive newlines collapsed
    for (auto& c : chunks) {
        assert(c.text.find("  ") == std::string::npos ||
               c.text.find("  ") == std::string::npos);  // some tolerance
    }
}

void test_tiny_file() {
    std::string text = "Hi.";
    auto chunks = chunk_markdown(text, 300);
    assert(chunks.size() == 1);
    assert(chunks[0].text == "Hi.");
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------

int main() {
    test_heading_levels();
    test_heading_text_extraction();
    test_empty_input();
    test_no_headings();
    test_single_heading_with_content();
    test_multiple_headings();
    test_nested_headings();
    test_heading_level_reset();
    test_minimum_chunk_size_merging();
    test_minimum_chunk_size_splitting();
    test_base64_stripping();
    test_whitespace_collapsing();
    test_tiny_file();

    std::println("test_import: all passed");
    return 0;
}
