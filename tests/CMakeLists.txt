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

add_executable(test_agent_parse test_agent_parse.cpp)
target_link_libraries(test_agent_parse PRIVATE ragger_core)
add_test(NAME test_agent_parse COMMAND test_agent_parse)

add_executable(test_rebuild_check test_rebuild_check.cpp)
target_link_libraries(test_rebuild_check PRIVATE ragger_core)
add_test(NAME test_rebuild_check COMMAND test_rebuild_check)


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

add_executable(test_text_index test_text_index.cpp)
target_link_libraries(test_text_index PRIVATE ragger_core)
add_test(NAME test_text_index COMMAND test_text_index)
add_test(NAME test_double_metaphone COMMAND test_double_metaphone)

add_executable(test_vector_codec test_vector_codec.cpp)
target_link_libraries(test_vector_codec PRIVATE ragger_core)
add_test(NAME test_vector_codec COMMAND test_vector_codec)

add_executable(test_fts_tokenize test_fts_tokenize.cpp)
target_link_libraries(test_fts_tokenize PRIVATE ragger_core)
add_test(NAME test_fts_tokenize COMMAND test_fts_tokenize)

# S53 (C §7 items 1/3): live-HTTP route tests against an in-process Server on
# an ephemeral port (harness/http_fixture.h). The fixture polls /health for up
# to 5 s; the binary itself runs in a few seconds, so no TIMEOUT property.
# ${CMAKE_CURRENT_SOURCE_DIR} is on the include path so harness/http_fixture.h
# finds check.h / test_env.h beside it.
add_executable(test_http test_http.cpp)
target_link_libraries(test_http PRIVATE ragger_core)
target_include_directories(test_http PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME test_http COMMAND test_http)

# LR S91: the daemon jobs runner (single-flight, cancel, SSE `job` frames,
# shutdown join) against a live daemon whose inference endpoint is a local
# stub (harness/http_fixture.h after_env hook). The slow-latch tests make it
# a slower binary; the shutdown test can wait out one model call.
add_executable(test_jobs test_jobs.cpp)
target_link_libraries(test_jobs PRIVATE ragger_core)
target_include_directories(test_jobs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME test_jobs COMMAND test_jobs)
set_tests_properties(test_jobs PROPERTIES TIMEOUT 120)

# Stats logger tests — only built/registered when RAGGER_STATS is ON.
# (ragger_core carries the RAGGER_STATS compile definition transitively, so
# the test body's #ifdef sees it; we still gate the target so a default build
# doesn't compile or run it at all.)
if(RAGGER_STATS)
    add_executable(test_stats test_stats.cpp)
    target_link_libraries(test_stats PRIVATE ragger_core)
    add_test(NAME test_stats COMMAND test_stats)
endif()

# S52 (C m23): the Hermes-plugin robustness script. It is self-contained — it
# stubs `agent.memory_provider` and loads `hermes-plugin/__init__.py` by path —
# so no Hermes install is needed on sys.path. It is a PLAIN SCRIPT, not a
# unittest/pytest module (no TestCase, no __main__ guard): it runs top to
# bottom, raises AssertionError via its own check() helper and prints "OK", so
# the interpreter itself is the runner and its exit status is the verdict.
# When no interpreter is found the test is simply not registered.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
    add_test(NAME test_plugin_robustness
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/test_plugin_robustness.py)
    set_tests_properties(test_plugin_robustness PROPERTIES TIMEOUT 120)
endif()
