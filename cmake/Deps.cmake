# Deps.cmake — fetch third-party source dependencies at configure time.
#
# Pinned by default: every dependency is checked out at the exact commit
# recorded in cmake/pins.cmake, so any tag or fresh clone builds the same code.
#
# Development: configure with -DDEPS_LATEST=ON (put it in your local
# CMakeUserPresets.json dev preset) to track each upstream's newest commit
# instead. The commits actually used are written to
# <build>/deps-resolved.cmake; run scripts/pin-deps.sh before tagging a
# release to copy them into pins.cmake and commit that.
#
# Offline / local checkout of one dependency:
#   -DFETCHCONTENT_SOURCE_DIR_<NAME>=/path/to/checkout     (NAME upper-case,
#   e.g. TOKENIZERS_CPP, C_LIB). Overridden dependencies are never recorded.
#
# Refresh to newest upstream on demand (leaves DEPS_LATEST=ON in the cache):
#   cmake --build <build> --target update_deps
#
# This file is duplicated in sibling projects; keep the copies identical.

include_guard(GLOBAL)
include(FetchContent)
if(POLICY CMP0169)
    # We use the one-argument FetchContent_Populate() so a dependency can be
    # added EXCLUDE_FROM_ALL (FetchContent_Declare's own flag needs CMake 3.28).
    cmake_policy(SET CMP0169 OLD)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/pins.cmake")

option(DEPS_LATEST "Track upstream's newest commit instead of cmake/pins.cmake (development)" OFF)

set(DEPS_RESOLVED_FILE "${CMAKE_BINARY_DIR}/deps-resolved.cmake")
if(DEPS_LATEST)
    file(WRITE "${DEPS_RESOLVED_FILE}"
         "# Commits used by the last DEPS_LATEST configure. Feed to scripts/pin-deps.sh.\n")
    message(STATUS "deps: DEPS_LATEST=ON — tracking upstream's newest commits")
endif()

# deps_add(<name> <latest_ref> [EXCLUDE_FROM_ALL])
#   <name>        FetchContent name, e.g. tokenizers_cpp; pins.cmake must define
#                 <NAME>_REPO and <NAME>_SHA (upper-case).
#   <latest_ref>  branch to follow when DEPS_LATEST=ON.
function(deps_add name latest_ref)
    cmake_parse_arguments(A "EXCLUDE_FROM_ALL" "" "" ${ARGN})
    string(TOUPPER "${name}" up)
    string(TOLOWER "${name}" lc)

    if(NOT DEFINED ${up}_REPO OR NOT DEFINED ${up}_SHA)
        message(FATAL_ERROR "deps: cmake/pins.cmake must define ${up}_REPO and ${up}_SHA")
    endif()
    if(DEPS_LATEST)
        set(ref "${latest_ref}")
    else()
        set(ref "${${up}_SHA}")
    endif()

    FetchContent_Declare(${name} GIT_REPOSITORY "${${up}_REPO}" GIT_TAG "${ref}")
    FetchContent_GetProperties(${name})
    if(NOT ${lc}_POPULATED)
        FetchContent_Populate(${name})
    endif()

    if(FETCHCONTENT_SOURCE_DIR_${up})
        message(STATUS "deps: ${name} from local ${FETCHCONTENT_SOURCE_DIR_${up}}")
    elseif(DEPS_LATEST)
        execute_process(COMMAND git -C "${${lc}_SOURCE_DIR}" rev-parse HEAD
                        OUTPUT_VARIABLE got OUTPUT_STRIP_TRAILING_WHITESPACE)
        file(APPEND "${DEPS_RESOLVED_FILE}" "set(${up}_SHA ${got})\n")
        message(STATUS "deps: ${name} = ${got} (latest ${latest_ref})")
    else()
        message(STATUS "deps: ${name} @ ${ref} (pinned)")
    endif()

    if(A_EXCLUDE_FROM_ALL)
        add_subdirectory("${${lc}_SOURCE_DIR}" "${${lc}_BINARY_DIR}" EXCLUDE_FROM_ALL)
    else()
        add_subdirectory("${${lc}_SOURCE_DIR}" "${${lc}_BINARY_DIR}")
    endif()
endfunction()

if(NOT TARGET update_deps)
    add_custom_target(update_deps
        COMMAND "${CMAKE_COMMAND}" -S "${CMAKE_SOURCE_DIR}" -B "${CMAKE_BINARY_DIR}" -DDEPS_LATEST=ON
        COMMENT "deps: reconfiguring against upstream's newest commits"
        USES_TERMINAL)
endif()
