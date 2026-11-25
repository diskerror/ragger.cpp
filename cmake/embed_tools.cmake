# LR S95: embed every tools/*.json into ONE generated header so the tool
# definitions (name, description, inputSchema) live in exactly one place —
# the JSON files — and both surfaces (MCP tools/list and GET /tools) read
# them from the binary. Same raw-string mechanism as embed_ini.cmake /
# embed_html.cmake; the array is keyed by file stem, and mcp.cpp assembles
# the list order and applies the config gates.
if(NOT DEFINED SRC_DIR OR NOT DEFINED DST)
    message(FATAL_ERROR "embed_tools.cmake requires -DSRC_DIR=<path> -DDST=<path>")
endif()

file(GLOB tool_files LIST_DIRECTORIES FALSE "${SRC_DIR}/*.json")
list(SORT tool_files)

if(tool_files STREQUAL "")
    message(FATAL_ERROR "embed_tools.cmake found no *.json in ${SRC_DIR}")
endif()

set(_out "// Auto-generated from tools/*.json by cmake/embed_tools.cmake.\n")
string(APPEND _out "// DO NOT EDIT — edit the JSON files and rebuild.\n")
string(APPEND _out "static const struct { const char* name; const char* json; } TOOLS_JSON[] = {\n")
foreach(f ${tool_files})
    get_filename_component(stem "${f}" NAME_WE)
    file(READ "${f}" content)
    string(APPEND _out "    {\"${stem}\", R\"RAGGERTOOL(${content})RAGGERTOOL\"},\n")
endforeach()
string(APPEND _out "};\n")

file(WRITE "${DST}" "${_out}")
