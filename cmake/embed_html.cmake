# Wraps the dashboard HTML/JS file in a raw-string C++ literal so the daemon
# can serve it from memory (no on-disk asset to lose). dashboard.html is the
# single source of truth; this only mirrors it into a generated header.
if(NOT DEFINED SRC OR NOT DEFINED DST)
    message(FATAL_ERROR "embed_html.cmake requires -DSRC=<path> -DDST=<path>")
endif()

file(READ "${SRC}" _html_content)

set(_out "// Auto-generated from dashboard.html by cmake/embed_html.cmake.\n")
string(APPEND _out "// DO NOT EDIT — edit web/dashboard.html instead and rebuild.\n")
string(APPEND _out "static constexpr const char* DASHBOARD_HTML = R\"RAGGERHTML(\n")
string(APPEND _out "${_html_content}")
string(APPEND _out ")RAGGERHTML\";\n")

file(WRITE "${DST}" "${_out}")
