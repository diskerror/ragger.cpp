/**
 * check.h — assertion macro that SURVIVES -DNDEBUG.
 *
 * Why this exists: the default CMAKE_BUILD_TYPE here is Release, which adds
 * -DNDEBUG, and -DNDEBUG compiles `assert(expr)` away to nothing -- INCLUDING
 * expr itself. Any test that wrapped a side-effecting call in assert(), e.g.
 *
 *     assert(sqlite3_open(":memory:", &db) == SQLITE_OK);
 *
 * silently never ran that call in a Release build. test_text_index did exactly
 * this: the open never happened, `db` stayed nullptr, and the first real
 * sqlite3_exec() returned SQLITE_MISUSE(21) with a NULL error message --
 * surfacing as the cryptic "SqliteTextIndex schema: unknown error". Every other
 * assertion in those files was also a no-op, so the tests asserted NOTHING.
 *
 * CHECK() always evaluates its expression exactly once, in every build type,
 * and aborts with file/line context on failure.
 *
 * RULE FOR THIS TEST SUITE: never put a function call inside assert(). Use
 * CHECK() for anything with side effects; plain assert() is acceptable only for
 * pure comparisons of already-computed values (and even then CHECK is better,
 * since it still fires in Release).
 */
#pragma once

#include <cstdio>
#include <cstdlib>

#define CHECK(expr)                                                        \
    do {                                                                   \
        if (!(expr)) {                                                     \
            std::fprintf(stderr, "CHECK FAILED: %s\n  at %s:%d\n",         \
                         #expr, __FILE__, __LINE__);                       \
            std::fflush(stderr);                                           \
            std::abort();                                                  \
        }                                                                  \
    } while (0)
