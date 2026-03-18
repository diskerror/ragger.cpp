cmake_minimum_required(VERSION 3.24)
project(raggerc VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(RAGGER_BUILD_TESTS "Build test suite" ON)

# ---------------------------------------------------------------------------
# Boost (for ProgramOptions via c_lib)
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
# Crow (HTTP framework, fetched at configure time)
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    Crow
    GIT_REPOSITORY https://github.com/CrowCpp/Crow.git
    GIT_TAG        v1.2.1.2
    GIT_SHALLOW    TRUE
)
set(CROW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CROW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CROW_USE_BOOST ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Crow)

# ---------------------------------------------------------------------------
# Header-only vendored libs
# ---------------------------------------------------------------------------
add_library(vendor_headers INTERFACE)
target_include_directories(vendor_headers INTERFACE
    "${CMAKE_SOURCE_DIR}/vendor"          # nlohmann_json.hpp
)

# ---------------------------------------------------------------------------
# c_lib — ProgramOptions (symlinked)
# ---------------------------------------------------------------------------
add_library(c_lib_po STATIC
    ProgramOptions.cpp
)
target_include_directories(c_lib_po PUBLIC
    "${CMAKE_SOURCE_DIR}"
)
target_link_libraries(c_lib_po PUBLIC Boost::program_options)

# ---------------------------------------------------------------------------
# ragger library (core logic, no main())
# ---------------------------------------------------------------------------
add_library(ragger_core STATIC
    src/config.cpp
    src/embedder.cpp
    src/tokenizer_wrapper.cpp
    src/sqlite_backend.cpp
    src/bm25.cpp
    src/memory.cpp
    src/server.cpp
)
target_include_directories(ragger_core PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
)
target_link_libraries(ragger_core PUBLIC
    SQLite::SQLite3
    onnxruntime
    tokenizers_cpp
    Eigen3::Eigen
    Crow::Crow
    vendor_headers
    Boost::boost           # headers only (for misc utilities)
)

# ---------------------------------------------------------------------------
# raggerc executable
# ---------------------------------------------------------------------------
add_executable(raggerc src/main.cpp)
target_link_libraries(raggerc PRIVATE
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
install(TARGETS raggerc RUNTIME DESTINATION bin)
