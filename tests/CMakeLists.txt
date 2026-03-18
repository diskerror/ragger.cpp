# Tests for raggerc
# Using simple main()-based tests for now; can switch to Catch2/GoogleTest later.

add_executable(test_config test_config.cpp)
target_link_libraries(test_config PRIVATE ragger_core)
add_test(NAME test_config COMMAND test_config)
