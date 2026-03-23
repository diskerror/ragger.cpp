cmake_minimum_required(VERSION 3.24)
project(ragger VERSION 0.7.1 LANGUAGES CXX)

# Build version: tag or date suffix
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
    string(TIMESTAMP BUILD_TIMESTAMP "%Y-%m-%d_%H:%M:%S")
    set(RAGGER_BUILD_VERSION "${PROJECT_VERSION}-${BUILD_TIMESTAMP}")
endif()
message(STATUS "VERSION: ${RAGGER_BUILD_VERSION}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(RAGGER_BUILD_TESTS "Build test suite" ON)

# ---------------------------------------------------------------------------
# Boost (for ProgramOptions via c_lib, and Boost.Asio for Crow)
# ---------------------------------------------------------------------------
# MacPorts boost188 lives here; adjust if using a different version
set(BOOST_ROOT "/opt/local/libexec/boost/1.88" CACHE PATH "Boost root")
find_package(Boost 1.76 REQUIRED COMPONENTS program_options)

# ---------------------------------------------------------------------------
# SQLite3
# ---------------------------------------------------------------------------
find_package(SQLite3 REQUIRED)

# ---------------------------------------------------------------------------
# ONNX Runtime (vendored pre-built)
# ---------------------------------------------------------------------------
set(ORT_DIR "${CMAKE_SOURCE_DIR}/vendor/onnxruntime")
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_DIR}/lib/libonnxruntime.dylib"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_DIR}/include"
)

# ---------------------------------------------------------------------------
# tokenizers-cpp (HuggingFace tokenizer via Rust FFI)
# ---------------------------------------------------------------------------
add_subdirectory(vendor/tokenizers-cpp)

# ---------------------------------------------------------------------------
# Eigen3 (MacPorts)
# ---------------------------------------------------------------------------
find_package(Eigen3 3.4 REQUIRED NO_MODULE
    HINTS /opt/local/share/eigen3/cmake
)

# ---------------------------------------------------------------------------
# Header-only vendored libs (Crow, nlohmann/json)
# ---------------------------------------------------------------------------
add_library(vendor_headers INTERFACE)
target_include_directories(vendor_headers INTERFACE
    "${CMAKE_SOURCE_DIR}/vendor"
)
# Crow: use Boost.Asio instead of standalone Asio
target_compile_definitions(vendor_headers INTERFACE CROW_USE_BOOST)

# ---------------------------------------------------------------------------
# c_lib — ProgramOptions (hard-linked from c_lib)
# ---------------------------------------------------------------------------
add_library(c_lib_po STATIC
    ProgramOptions.cpp
)
target_include_directories(c_lib_po PUBLIC
    "${CMAKE_SOURCE_DIR}"
)
target_link_libraries(c_lib_po PUBLIC Boost::program_options)

# ---------------------------------------------------------------------------
# libcurl (for inference client)
# ---------------------------------------------------------------------------
find_package(CURL REQUIRED)

# ---------------------------------------------------------------------------
# ragger library (core logic, no main())
# ---------------------------------------------------------------------------
add_library(ragger_core STATIC
    src/chat.cpp
    src/config.cpp
    src/embedder.cpp
    src/tokenizer_wrapper.cpp
    src/sqlite_backend.cpp
    src/bm25.cpp
    src/import.cpp
    src/memory.cpp
    src/server.cpp
    src/inference.cpp
    src/api_formats.cpp
    src/logs.cpp
    src/auth.cpp
)
target_include_directories(ragger_core PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
)
target_compile_definitions(ragger_core PUBLIC
    RAGGER_VERSION="${RAGGER_BUILD_VERSION}"
)
target_link_libraries(ragger_core PUBLIC
    SQLite::SQLite3
    onnxruntime
    tokenizers_cpp
    Eigen3::Eigen
    vendor_headers
    Boost::boost
    CURL::libcurl
)

# ---------------------------------------------------------------------------
# ragger executable
# ---------------------------------------------------------------------------
add_executable(ragger src/main.cpp)
target_link_libraries(ragger PRIVATE
    ragger_core
    c_lib_po
)

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
