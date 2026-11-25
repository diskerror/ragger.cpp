#!/usr/bin/env python3
"""Regenerate the AGENT_TOOLS block of openclaw-plugin/index.ts from tools/*.json.

tools/*.json is the single definition of the agent tools (the daemon and
`ragger mcp` serve the same files). The OpenClaw plugin is installed as two
loose files, so it cannot read them at run time; this script inlines them
between the GENERATED markers. Run after any tools/*.json change:

    ./scripts/gen-openclaw-tools.py          # rewrite index.ts
    ./scripts/gen-openclaw-tools.py --check  # exit 1 if index.ts is stale
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TS = ROOT / "openclaw-plugin" / "index.ts"
BEGIN = "// BEGIN GENERATED AGENT_TOOLS (scripts/gen-openclaw-tools.py; do not edit by hand)"
END = "// END GENERATED AGENT_TOOLS"

# tool name -> (HTTP path, label). Order is registration order.
TOOLS = [
    ("memory_search",   "/memory/search",    "Memory Search"),
    ("memory_get",      "/memory/get",       "Memory Get"),
    ("memory_timeline", "/memory/timeline",  "Memory Timeline"),
    ("memory_note",     "/memory/note",      "Memory Note"),
    ("memory_decision", "/decision",         "Memory Decision"),
    ("memory_update",   "/memory/update",    "Memory Update"),
    ("document_search", "/memory/documents", "Document Search"),
]

def block() -> str:
    out = [BEGIN, "const AGENT_TOOLS: Array<{ name: string; path: string; label: string; description: string; parameters: Record<string, unknown> }> = ["]
    for name, path, label in TOOLS:
        j = json.loads((ROOT / "tools" / f"{name}.json").read_text())
        assert j["name"] == name, name
        out.append("  {")
        out.append(f"    name: {json.dumps(name)},")
        out.append(f"    path: {json.dumps(path)},")
        out.append(f"    label: {json.dumps(label)},")
        out.append(f"    description: {json.dumps(j['description'], ensure_ascii=False)},")
        out.append(f"    parameters: {json.dumps(j['inputSchema'], ensure_ascii=False)},")
        out.append("  },")
    out.append("];")
    out.append(END)
    return "\n".join(out)

def main() -> int:
    s = TS.read_text()
    a, b = s.index(BEGIN), s.index(END) + len(END)
    new = s[:a] + block() + s[b:]
    if "--check" in sys.argv:
        if new != s:
            print("openclaw-plugin/index.ts AGENT_TOOLS is stale; run scripts/gen-openclaw-tools.py")
            return 1
        print("openclaw-plugin AGENT_TOOLS up to date")
        return 0
    TS.write_text(new)
    print("rewrote AGENT_TOOLS block")
    return 0

sys.exit(main())
