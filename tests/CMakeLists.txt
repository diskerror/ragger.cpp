# Tests for ragger
# Using simple main()-based tests for now; can switch to Catch2/GoogleTest later.

add_executable(test_config test_config.cpp)
target_link_libraries(test_config PRIVATE ragger_core)
add_test(NAME test_config COMMAND test_config)

add_executable(test_bm25 test_bm25.cpp)
target_link_libraries(test_bm25 PRIVATE ragger_core)
add_test(NAME test_bm25 COMMAND test_bm25)

add_executable(test_sqlite_backend test_sqlite_backend.cpp)
target_link_libraries(test_sqlite_backend PRIVATE ragger_core)
add_test(NAME test_sqlite_backend COMMAND test_sqlite_backend)

add_executable(test_import test_import.cpp)
target_link_libraries(test_import PRIVATE ragger_core)
add_test(NAME test_import COMMAND test_import)
