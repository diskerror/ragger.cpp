# Tests for ragger
# Using simple main()-based tests for now; can switch to Catch2/GoogleTest later.

add_executable(test_config test_config.cpp)
target_link_libraries(test_config PRIVATE ragger_core)
add_test(NAME test_config COMMAND test_config)

add_executable(test_sqlite_backend test_sqlite_backend.cpp)
target_link_libraries(test_sqlite_backend PRIVATE ragger_core)
add_test(NAME test_sqlite_backend COMMAND test_sqlite_backend)

add_executable(test_import test_import.cpp)
target_link_libraries(test_import PRIVATE ragger_core)
add_test(NAME test_import COMMAND test_import)

add_executable(test_export test_export.cpp)
target_link_libraries(test_export PRIVATE ragger_core)
add_test(NAME test_export COMMAND test_export)

add_executable(test_auth test_auth.cpp)
target_link_libraries(test_auth PRIVATE ragger_core)
add_test(NAME test_auth COMMAND test_auth)

add_executable(test_server test_server.cpp)
target_link_libraries(test_server PRIVATE ragger_core)
add_test(NAME test_server COMMAND test_server)

add_executable(test_client test_client.cpp)
target_link_libraries(test_client PRIVATE ragger_core)
add_test(NAME test_client COMMAND test_client)

add_executable(test_memory test_memory.cpp)
target_link_libraries(test_memory PRIVATE ragger_core)
add_test(NAME test_memory COMMAND test_memory)


add_executable(test_path_normalization test_path_normalization.cpp)
target_link_libraries(test_path_normalization PRIVATE ragger_core)
add_test(NAME test_path_normalization COMMAND test_path_normalization)

add_executable(test_embedder test_embedder.cpp)
target_link_libraries(test_embedder PRIVATE ragger_core)
add_test(NAME test_embedder COMMAND test_embedder)

add_executable(test_logger test_logger.cpp)
target_link_libraries(test_logger PRIVATE ragger_core)
add_test(NAME test_logger COMMAND test_logger)

add_executable(test_embed_executor test_embed_executor.cpp)
target_link_libraries(test_embed_executor PRIVATE ragger_core)
# Point the test at the freshly-built ragger binary it spawns.
target_compile_definitions(test_embed_executor PRIVATE RAGGER_BIN="$<TARGET_FILE:ragger>")
add_test(NAME test_embed_executor COMMAND test_embed_executor)

add_executable(test_summarizer test_summarizer.cpp)
target_link_libraries(test_summarizer PRIVATE ragger_core)
add_test(NAME test_summarizer COMMAND test_summarizer)

add_executable(test_double_metaphone test_double_metaphone.cpp)
target_link_libraries(test_double_metaphone PRIVATE ragger_core)
add_test(NAME test_double_metaphone COMMAND test_double_metaphone)

# Stats logger tests — only built/registered when RAGGER_STATS is ON.
# (ragger_core carries the RAGGER_STATS compile definition transitively, so
# the test body's #ifdef sees it; we still gate the target so a default build
# doesn't compile or run it at all.)
if(RAGGER_STATS)
    add_executable(test_stats test_stats.cpp)
    target_link_libraries(test_stats PRIVATE ragger_core)
    add_test(NAME test_stats COMMAND test_stats)
endif()
