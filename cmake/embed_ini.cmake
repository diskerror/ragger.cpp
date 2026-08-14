# Wraps default-settings.txt in a raw-string C++ literal so config.cpp's
# embedded DEFAULT_CONFIG stays in perfect sync with the build-time template —
# default-settings.txt is the single source of truth; nothing here duplicates it.
# NOTE: despite its historic INI-flavored content, this file is not installed
# or parsed at runtime — it exists solely as this build-time input.
if(NOT DEFINED SRC OR NOT DEFINED DST)
    message(FATAL_ERROR "embed_ini.cmake requires -DSRC=<path> -DDST=<path>")
endif()

file(READ "${SRC}" _ini_content)

set(_out "// Auto-generated from default-settings.txt by cmake/embed_ini.cmake.\n")
string(APPEND _out "// DO NOT EDIT — edit default-settings.txt instead and rebuild.\n")
string(APPEND _out "static constexpr const char* DEFAULT_CONFIG = R\"RAGGERCFG(\n")
string(APPEND _out "${_ini_content}")
string(APPEND _out ")RAGGERCFG\";\n")

file(WRITE "${DST}" "${_out}")
