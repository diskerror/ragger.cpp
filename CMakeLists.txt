cmake_minimum_required(VERSION 3.24)
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS version")
endif()
project(ragger VERSION 0.9.11 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build metadata: version, commit, build date
string(TIMESTAMP RAGGER_BUILD_DATE "%Y-%m-%d %H:%M:%S")

# Git commit (short hash)
execute_process(
    COMMAND git rev-parse --short=7 HEAD
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE RAGGER_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE GIT_HASH_RESULT
)
if(NOT GIT_HASH_RESULT EQUAL 0)
    set(RAGGER_COMMIT "unknown")
endif()

# Use tag as version if on a tagged commit, otherwise version+commit
execute_process(
    COMMAND git describe --tags --exact-match
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE GIT_TAG
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE GIT_TAG_RESULT
)
if(GIT_TAG_RESULT EQUAL 0)
    set(RAGGER_BUILD_VERSION "${PROJECT_VERSION}")
else()
    set(RAGGER_BUILD_VERSION "${PROJECT_VERSION}+${RAGGER_COMMIT}")
endif()
message(STATUS "VERSION: ${RAGGER_BUILD_VERSION}")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(RAGGER_BUILD_TESTS "Build test suite" ON)

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
# SQLite3
# ---------------------------------------------------------------------------
find_package(SQLite3 REQUIRED)

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
        file(DOWNLOAD "${ORT_URL}" "${CMAKE_BINARY_DIR}/_deps/${ORT_PKG}.tgz"
             SHOW_PROGRESS STATUS _ort_dl)
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
add_subdirectory(vendor/tokenizers-cpp)

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
# c_lib — ProgramOptions (hard-linked from c_lib)
# ---------------------------------------------------------------------------
add_library(c_lib_po STATIC
    src/program_options.cpp
)
target_include_directories(c_lib_po PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
)
target_link_libraries(c_lib_po PUBLIC Boost::program_options)

# ---------------------------------------------------------------------------
# libcurl (for inference client)
# ---------------------------------------------------------------------------
find_package(CURL REQUIRED)

# ---------------------------------------------------------------------------
# OpenSSL (for SHA-256 hashing in auth)
# ---------------------------------------------------------------------------
find_package(OpenSSL REQUIRED)

# ---------------------------------------------------------------------------
# ragger library (core logic, no main())
# ---------------------------------------------------------------------------
add_library(ragger_core STATIC
    src/client.cpp
    src/config.cpp
    src/util/time.cpp
    src/util/fs.cpp
    src/summarizer.cpp
    src/summarizer_service.cpp
    src/recipe.cpp
    src/recipe_cli.cpp
    src/onboard.cpp
    src/embedder.cpp
    src/embed_executor.cpp
    src/tokenizer_wrapper.cpp
    src/sqlite_backend.cpp
    src/user_store.cpp
    src/import.cpp
    src/export.cpp
    src/memory.cpp
    src/server.cpp
    src/inference.cpp
    src/api_formats.cpp
    src/logger.cpp
    src/auth.cpp
    src/mcp.cpp
)
target_include_directories(ragger_core PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
)
target_compile_definitions(ragger_core PUBLIC
    RAGGER_VERSION="${PROJECT_VERSION}"
    RAGGER_COMMIT="${RAGGER_COMMIT}"
    RAGGER_BUILD_DATE="${RAGGER_BUILD_DATE}"
)
target_link_libraries(ragger_core PUBLIC
    SQLite::SQLite3
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
    ragger_core
    c_lib_po)

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
