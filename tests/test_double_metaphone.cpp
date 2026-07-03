// Unit tests for the Double Metaphone phonetic encoder (dolphining sounds-like).
// Vectors validated against the reference `metaphone` implementation, capped to
// the canonical 4-char keys. main()-based per this project's test convention.

#include "ragger/double_metaphone.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

using namespace ragger;

static int failures = 0;

static void expect(const std::string& word,
                   const std::string& want_primary,
                   const std::string& want_alt = "") {
    auto codes = double_metaphone(word);
    std::string p = codes.empty() ? "" : codes[0];
    std::string a = codes.size() > 1 ? codes[1] : "";
    if (p != want_primary || a != want_alt) {
        std::printf("FAIL %-12s got [%s,%s] want [%s,%s]\n",
                    word.c_str(), p.c_str(), a.c_str(),
                    want_primary.c_str(), want_alt.c_str());
        ++failures;
    }
}

int main() {
    // Primary-only words.
    expect("dolphin",   "TLFN");
    expect("Thompson",  "TMPS");
    expect("knight",    "NT");
    expect("bob",       "PP");
    expect("hello",     "HL");
    expect("panic",     "PNK");
    expect("caesar",    "SSR");
    expect("focaccia",  "FKX");
    expect("accident",  "AKST");
    expect("ghost",     "KST");
    expect("psalm",     "SLM");
    expect("pneumonia", "NMN");

    // Words with a distinct alternate pronunciation.
    expect("night",     "NT");
    expect("smith",     "SM0",  "XMT");
    expect("schmidt",   "XMT",  "SMT");
    expect("Catherine", "K0RN", "KTRN");
    expect("richard",   "RXRT", "RKRT");
    expect("czar",      "SR",   "XR");
    expect("xavier",    "SF",   "SFR");
    expect("Wednesday", "ATNS", "FTNS");

    // Non-alpha / empty input yields no codes.
    assert(double_metaphone("").empty());
    assert(double_metaphone("123!@#").empty());

    // phonize: no stopword filtering — every word contributes, both codes emitted.
    assert(phonize("Don't panic") == "TNT PNK");
    assert(phonize("the quick brown fox") == "0 T KK PRN FKS");
    assert(phonize("   ") == "");

    if (failures == 0) {
        std::printf("test_double_metaphone: ALL PASS\n");
        return 0;
    }
    std::printf("test_double_metaphone: %d FAILURE(S)\n", failures);
    return 1;
}
