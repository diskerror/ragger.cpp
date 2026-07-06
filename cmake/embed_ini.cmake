# Wraps example-settings.ini in a raw-string C++ literal so config.cpp's
# embedded DEFAULT_CONFIG stays in perfect sync with the shipped template —
# example-settings.ini is the single source of truth; nothing here duplicates it.
if(NOT DEFINED SRC OR NOT DEFINED DST)
    message(FATAL_ERROR "embed_ini.cmake requires -DSRC=<path> -DDST=<path>")
endif()

file(READ "${SRC}" _ini_content)

set(_out "// Auto-generated from example-settings.ini by cmake/embed_ini.cmake.\n")
string(APPEND _out "// DO NOT EDIT — edit example-settings.ini instead and rebuild.\n")
string(APPEND _out "static constexpr const char* DEFAULT_CONFIG = R\"RAGGERCFG(\n")
string(APPEND _out "${_ini_content}")
string(APPEND _out ")RAGGERCFG\";\n")

file(WRITE "${DST}" "${_out}")
