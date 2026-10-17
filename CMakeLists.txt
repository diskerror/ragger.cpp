cmake_minimum_required(VERSION 3.24)

# Linux: prefer clang (c_lib's WAVE.h uses a clang-accepted layout GCC rejects).
# Only when the user hasn't chosen a compiler (CC/CXX env or -DCMAKE_CXX_COMPILER)
# and only when this is the top-level project.
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux" AND NOT CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX}
   AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    foreach(_v "" -21 -20 -19 -18 -17 -16)
        find_program(_clangxx_${_v} NAMES clang++${_v} NO_CACHE)
        find_program(_clang_${_v}   NAMES clang${_v}   NO_CACHE)
        if(_clangxx_${_v} AND _clang_${_v})
            set(CMAKE_CXX_COMPILER "${_clangxx_${_v}}" CACHE FILEPATH "C++ compiler")
            set(CMAKE_C_COMPILER   "${_clang_${_v}}"   CACHE FILEPATH "C compiler")
            message(STATUS "Using clang${_v} (preferred on Linux)")
            break()
        endif()
    endforeach()
endif()
if(APPLE)
    # Has compiled successfully with earlier versions of MacOS. Caveat emptor.
    set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING "Minimum macOS version")
endif()
project(ragger VERSION 0.16.2.4 LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build metadata (version, commit, build date) is generated at BUILD time —
# see "build_info" below — so a new commit/tag never needs a reconfigure.

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(RAGGER_BUILD_TESTS "Build test suite" ON)
option(RAGGER_STATS "Log retrieval stats to a discardable stats.db (instrumentation; default OFF)" OFF)

# Fetched dependencies (tokenizers-cpp, c_lib fallback, ONNX checksums): pinned
# in cmake/pins.cmake; -DDEPS_LATEST=ON tracks upstream instead. See Deps.cmake.
include(cmake/Deps.cmake)

# ---------------------------------------------------------------------------
# Boost (for ProgramOptions via c_lib)
# ---------------------------------------------------------------------------
if(APPLE)
    set(BOOST_ROOT "/opt/local/libexec/boost/1.88" CACHE PATH "Boost root")
    find_package(Boost 1.88 REQUIRED COMPONENTS program_options)
else()
    find_package(Boost 1.74 REQUIRED COMPONENTS program_options)
endif()

# ---------------------------------------------------------------------------
# SQLite3 — c_lib's vendored build (diskerror_sqlite3, foreign keys default
# ON). -DRAGGER_SYSTEM_SQLITE=ON links the system library instead; FK
# enforcement then relies only on the per-connection PRAGMA.
# ---------------------------------------------------------------------------
option(RAGGER_SYSTEM_SQLITE "Link the system SQLite instead of c_lib's diskerror_sqlite3" OFF)
if(RAGGER_SYSTEM_SQLITE)
    find_package(SQLite3 REQUIRED)
    set(RAGGER_SQLITE_TARGET SQLite::SQLite3)
    message(STATUS "SQLite: system library (foreign keys NOT compiled-in default)")
else()
    set(RAGGER_SQLITE_TARGET diskerror_sqlite3)
endif()

# ---------------------------------------------------------------------------
# ONNX Runtime — vendored if present, else fetched at configure time
# ---------------------------------------------------------------------------
# Repo-local override: drop the matching upstream archive in vendor/onnxruntime/
# and the lib + headers will be picked up as-is (handy for offline builds).
# Otherwise we fetch the right tarball for the host platform from the
# microsoft/onnxruntime GitHub release.
set(ORT_VERSION "1.24.4" CACHE STRING "ONNX Runtime version to use")

if(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(ORT_PKG  "onnxruntime-osx-arm64-${ORT_VERSION}")
    else()
        set(ORT_PKG  "onnxruntime-osx-x86_64-${ORT_VERSION}")
    endif()
    set(ORT_LIB_NAME "libonnxruntime.dylib")
elseif(UNIX)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(ORT_PKG  "onnxruntime-linux-aarch64-${ORT_VERSION}")
    else()
        set(ORT_PKG  "onnxruntime-linux-x64-${ORT_VERSION}")
    endif()
    set(ORT_LIB_NAME "libonnxruntime.so")
else()
    message(FATAL_ERROR "Unsupported platform for ONNX Runtime auto-fetch")
endif()

set(ORT_VENDOR_DIR "${CMAKE_SOURCE_DIR}/vendor/onnxruntime")
if(EXISTS "${ORT_VENDOR_DIR}/lib/${ORT_LIB_NAME}"
   AND EXISTS "${ORT_VENDOR_DIR}/include/onnxruntime_cxx_api.h")
    set(ORT_DIR "${ORT_VENDOR_DIR}")
    message(STATUS "ONNX Runtime: using vendored ${ORT_DIR}")
else()
    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_PKG}.tgz")
    set(ORT_CACHE "${CMAKE_BINARY_DIR}/_deps/onnxruntime/${ORT_PKG}")
    if(NOT EXISTS "${ORT_CACHE}/lib/${ORT_LIB_NAME}")
        message(STATUS "ONNX Runtime: fetching ${ORT_URL}")
        set(_ort_hash "")
        if(DEFINED ORT_SHA256_${ORT_PKG})
            set(_ort_hash EXPECTED_HASH SHA256=${ORT_SHA256_${ORT_PKG}})
        else()
            message(WARNING "ONNX Runtime: no pinned SHA-256 for ${ORT_PKG} (cmake/pins.cmake); download is unverified")
        endif()
        file(DOWNLOAD "${ORT_URL}" "${CMAKE_BINARY_DIR}/_deps/${ORT_PKG}.tgz"
             SHOW_PROGRESS STATUS _ort_dl ${_ort_hash})
        list(GET _ort_dl 0 _ort_rc)
        if(NOT _ort_rc EQUAL 0)
            message(FATAL_ERROR "ONNX Runtime download failed: ${_ort_dl}")
        endif()
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/_deps/onnxruntime")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_BINARY_DIR}/_deps/${ORT_PKG}.tgz"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/_deps/onnxruntime"
            RESULT_VARIABLE _ort_tar)
        if(NOT _ort_tar EQUAL 0)
            message(FATAL_ERROR "ONNX Runtime extract failed: ${_ort_tar}")
        endif()
    endif()
    set(ORT_DIR "${ORT_CACHE}")
endif()

set(ORT_LIB "${ORT_DIR}/lib/${ORT_LIB_NAME}")
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_DIR}/include"
)

# ---------------------------------------------------------------------------
# tokenizers-cpp (HuggingFace tokenizer via Rust FFI)
# ---------------------------------------------------------------------------
if(APPLE)
    # Use nightly Rust for unstable features needed by tokenizers
    find_program(RUSTUP rustup)
    if(RUSTUP)
        execute_process(
            COMMAND ${RUSTUP} default nightly-aarch64-apple-darwin
            OUTPUT_QUIET
        )
    endif()
endif()
deps_add(tokenizers_cpp main)

# ---------------------------------------------------------------------------
# Eigen3 (MacPorts)
# ---------------------------------------------------------------------------
if(APPLE)
    find_package(Eigen3 3.4 REQUIRED NO_MODULE
        HINTS /opt/local/share/eigen3/cmake
    )
else()
    find_package(Eigen3 3.4 REQUIRED NO_MODULE)
endif()

# ---------------------------------------------------------------------------
# Header-only vendored libs (cpp-httplib, nlohmann/json)
# ---------------------------------------------------------------------------
add_library(vendor_headers INTERFACE)
target_include_directories(vendor_headers INTERFACE
    "${CMAKE_SOURCE_DIR}/vendor"
)
# cpp-httplib with OpenSSL support
target_compile_definitions(vendor_headers INTERFACE CPPHTTPLIB_OPENSSL_SUPPORT)

# ---------------------------------------------------------------------------
# c_lib — prebuilt libraries from sibling ../c_lib/build (optimized,
# built and tested by c_lib itself; see ../c_lib/cmake/UseCLib.cmake). A
# Ragger clean never recompiles c_lib. Without a local ../c_lib (fresh
# clone / CI), falls back to building it from GitHub as a subproject.
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/../c_lib/cmake/UseCLib.cmake")
    include("${CMAKE_SOURCE_DIR}/../c_lib/cmake/UseCLib.cmake")
    use_c_lib(0.2)
else()
    message(STATUS "c_lib: fetching from GitHub (no local ../c_lib)")
    deps_add(c_lib master)
endif()

# c_lib targets provide their own include directories via
# target_include_directories(... PUBLIC), so consumers just link the targets.
# Convenience alias for ragger_core:
add_library(c_lib_utils INTERFACE)
target_link_libraries(c_lib_utils INTERFACE
    diskerror_program_options
    diskerror_logger
)

# ---------------------------------------------------------------------------
# libcurl (for inference client)
# ---------------------------------------------------------------------------
find_package(CURL REQUIRED)

# ---------------------------------------------------------------------------
# OpenSSL (for SHA-256 hashing in auth)
# ---------------------------------------------------------------------------
find_package(OpenSSL REQUIRED)

# ---------------------------------------------------------------------------
# Embed default-settings.txt into a generated header — the single source of
# truth for config.cpp's compiled-in DEFAULT_CONFIG string. This file is
# NOT installed or parsed as an INI file at runtime; it only ever exists as
# this build-time input.
# ---------------------------------------------------------------------------
set(RAGGER_GENERATED_DIR "${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${RAGGER_GENERATED_DIR}")
add_custom_command(
    OUTPUT "${RAGGER_GENERATED_DIR}/default_config.inc"
    COMMAND ${CMAKE_COMMAND}
            -DSRC=${CMAKE_SOURCE_DIR}/default-settings.txt
            -DDST=${RAGGER_GENERATED_DIR}/default_config.inc
            -P ${CMAKE_SOURCE_DIR}/cmake/embed_ini.cmake
    DEPENDS "${CMAKE_SOURCE_DIR}/default-settings.txt" "${CMAKE_SOURCE_DIR}/cmake/embed_ini.cmake"
    COMMENT "Embedding default-settings.txt -> default_config.inc"
)
add_custom_target(generate_default_config DEPENDS "${RAGGER_GENERATED_DIR}/default_config.inc")

# Embed the dashboard HTML/JS into the binary (served from memory by the
# daemon at /dashboard). web/dashboard.html is the single source of truth.
add_custom_command(
    OUTPUT "${RAGGER_GENERATED_DIR}/dashboard_html.inc"
    COMMAND ${CMAKE_COMMAND}
            -DSRC=${CMAKE_SOURCE_DIR}/web/dashboard.html
            -DDST=${RAGGER_GENERATED_DIR}/dashboard_html.inc
            -P ${CMAKE_SOURCE_DIR}/cmake/embed_html.cmake
    DEPENDS "${CMAKE_SOURCE_DIR}/web/dashboard.html" "${CMAKE_SOURCE_DIR}/cmake/embed_html.cmake"
    COMMENT "Embedding web/dashboard.html -> dashboard_html.inc"
)
add_custom_target(generate_dashboard_html DEPENDS "${RAGGER_GENERATED_DIR}/dashboard_html.inc")

# ---------------------------------------------------------------------------
# ragger library (core logic, no main())
# ---------------------------------------------------------------------------
add_library(ragger_core STATIC
    src/client.cpp
    src/config.cpp
    src/config_access.cpp
    src/util/time.cpp
    src/util/fs.cpp
    src/util/http_client.cpp
    src/summarizer.cpp
    src/summarizer_service.cpp
    src/recipe.cpp
    src/recipe_cli.cpp
    src/embedder.cpp
    src/embed_executor.cpp
    src/tokenizer_wrapper.cpp
    src/fts_tokenize.cpp
    src/text_index.cpp
    src/sqlite/text_index.cpp
    src/sqlite/backend.cpp
    src/sqlite/schema.cpp
    src/sqlite/admin.cpp
    src/sqlite/cache.cpp
    src/sqlite/write.cpp
    src/sqlite/search.cpp
    src/sqlite/maintenance.cpp
    src/user_store.cpp
    src/import.cpp
    src/user_commands.cpp
    src/export.cpp
    src/memory.cpp
    src/stats_logger.cpp
    src/server.cpp
    src/inference.cpp
    src/api_formats.cpp
    src/auth.cpp
    src/mcp.cpp
)
target_include_directories(ragger_core PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
)
target_include_directories(ragger_core PRIVATE
    "${RAGGER_GENERATED_DIR}"
)
add_dependencies(ragger_core generate_default_config)
add_dependencies(ragger_core generate_dashboard_html)
# Build info: regenerated on every build, rewritten only when the
# version/commit changes (then only build_info.cpp recompiles).
set(RAGGER_BUILD_INFO_CPP "${CMAKE_BINARY_DIR}/generated/build_info.cpp")
add_custom_target(ragger_build_info
    COMMAND "${CMAKE_COMMAND}" -DSRC_DIR="${CMAKE_SOURCE_DIR}"
            -DOUT="${RAGGER_BUILD_INFO_CPP}" -DVERSION="${PROJECT_VERSION}"
            -P "${CMAKE_SOURCE_DIR}/cmake/build_info.cmake"
    BYPRODUCTS "${RAGGER_BUILD_INFO_CPP}"
    COMMENT "ragger: build info"
)
target_sources(ragger_core PRIVATE "${RAGGER_BUILD_INFO_CPP}")
add_dependencies(ragger_core ragger_build_info)
if(TARGET c_lib_build)
    add_dependencies(ragger_core c_lib_build)
endif()
if(RAGGER_STATS)
    target_compile_definitions(ragger_core PUBLIC RAGGER_STATS)
    message(STATUS "RAGGER_STATS: ON — retrieval stats logged to ~/.ragger/stats.db")
endif()
target_link_libraries(ragger_core PUBLIC
    c_lib_utils
    diskerror_embedding_codec
    diskerror_double_metaphone
    diskerror_stemmer_snowball
    ${RAGGER_SQLITE_TARGET}
    onnxruntime
    tokenizers_cpp
    Eigen3::Eigen
    vendor_headers
    Boost::boost
    CURL::libcurl
    OpenSSL::Crypto
    OpenSSL::SSL
)

# ---------------------------------------------------------------------------
# ragger executable
# ---------------------------------------------------------------------------
add_executable(ragger src/main.cpp src/daemon_control.cpp)
target_link_libraries(ragger PRIVATE
    ragger_core)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(RAGGER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Install prefix" FORCE)
install(TARGETS ragger RUNTIME DESTINATION bin)
