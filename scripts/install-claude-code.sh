#!/bin/bash
# install-claude-code.sh — Wire Ragger into the Claude Code CLI (TUI).
#
# Usage:
#   ./scripts/install-claude-code.sh              # install / update
#   ./scripts/install-claude-code.sh --uninstall  # remove the integration
#
# Run AFTER install-bin.sh. Requires: ragger binary at ~/.local/bin/ragger.
#
# What install does:
#   1. Installs docs/agent-memory-instructions.md → ~/.ragger/ (only if
#      absent — `ragger mcp` serves it via the MCP initialize block).
#   2. Registers `ragger mcp` as an MCP server in ~/.claude.json so the
#      search / store / capture_turn / build_context tools appear in
#      every Claude Code session.
#   3. Installs ~/.ragger/hooks/claude-code-capture-turn.sh and wires it
#      as a Stop hook in ~/.claude/settings.json — every assistant
#      response posts its turn to Ragger via POST /turn.
#
# Both config edits are conservative: a dated backup is written before
# any change, existing entries for other servers/hooks are preserved.
#
# Idempotent. Re-running upgrades the binary path and refreshes the
# hook script without touching anything else.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
fail() { echo -e "${RED}[!]${NC} $*" >&2; exit 1; }

if [ "$(id -u)" -eq 0 ]; then
    fail "Do NOT run install-claude-code.sh as root."
fi

cd "$(dirname "$0")/.."
SRC="$(pwd)"

UNINSTALL=0
[ "${1:-}" = "--uninstall" ] && UNINSTALL=1

command -v python3 >/dev/null 2>&1 || fail "python3 is required to edit Claude Code config."

# --- Paths ---
MCP_CFG="$HOME/.claude.json"                    # MCP server registry
HOOK_CFG="$HOME/.claude/settings.json"          # user settings (incl. hooks)
HOOK_DIR="$HOME/.ragger/hooks"
HOOK_SCRIPT="$HOOK_DIR/claude-code-capture-turn.sh"
INSTR_SRC="$SRC/docs/agent-memory-instructions.md"
INSTR_DST="$HOME/.ragger/agent-memory-instructions.md"

# ============================================================
# Uninstall
# ============================================================
if [ "$UNINSTALL" -eq 1 ]; then
    if [ -f "$MCP_CFG" ]; then
        python3 - "$MCP_CFG" <<'PYEOF'
import json, sys, pathlib, shutil, datetime
p = pathlib.Path(sys.argv[1])
try:
    data = json.loads(p.read_text() or "{}")
except Exception as e:
    print(f"[!] Could not parse {p} ({e})", file=sys.stderr); sys.exit(0)
servers = data.get("mcpServers", {})
if "ragger" in servers:
    bak = p.with_name(p.name + f".bak-{datetime.date.today().isoformat()}")
    if not bak.exists(): shutil.copy2(p, bak)
    del servers["ragger"]
    if not servers: data.pop("mcpServers", None)
    p.write_text(json.dumps(data, indent=2) + "\n")
    print("[+]   removed mcpServers.ragger from .claude.json")
else:
    print("[+]   no ragger MCP entry — nothing to remove")
PYEOF
    fi

    if [ -f "$HOOK_CFG" ]; then
        HOOK_SCRIPT="$HOOK_SCRIPT" python3 - "$HOOK_CFG" <<'PYEOF'
import json, os, sys, pathlib, shutil, datetime
p = pathlib.Path(sys.argv[1])
try:
    data = json.loads(p.read_text() or "{}")
except Exception as e:
    print(f"[!] Could not parse {p} ({e})", file=sys.stderr); sys.exit(0)
target = os.environ["HOOK_SCRIPT"]
hooks = data.get("hooks", {})
stop = hooks.get("Stop", [])
removed = False
new_stop = []
for entry in stop:
    inner = [h for h in entry.get("hooks", [])
             if h.get("command") != target]
    if inner:
        entry["hooks"] = inner
        new_stop.append(entry)
    else:
        removed = True
if removed:
    bak = p.with_name(p.name + f".bak-{datetime.date.today().isoformat()}")
    if not bak.exists(): shutil.copy2(p, bak)
    if new_stop:
        hooks["Stop"] = new_stop
    else:
        hooks.pop("Stop", None)
    if not hooks: data.pop("hooks", None)
    p.write_text(json.dumps(data, indent=2) + "\n")
    print("[+]   removed Stop hook from settings.json")
else:
    print("[+]   no ragger Stop hook present — nothing to remove")
PYEOF
    fi

    info "Done. The instructions file at $INSTR_DST and the hook script were"
    info "left in place; remove them by hand if you want a fully clean uninstall."
    exit 0
fi

# ============================================================
# Install
# ============================================================

# 1. Locate the ragger binary (absolute path — hooks/MCP launchers get a
# minimal PATH and won't resolve ~).
RAGGER_BIN="$HOME/.local/bin/ragger"
if [ ! -x "$RAGGER_BIN" ]; then
    RAGGER_BIN="$(command -v ragger || true)"
fi
[ -n "$RAGGER_BIN" ] && [ -x "$RAGGER_BIN" ] || \
    fail "ragger binary not found — run ./scripts/install-bin.sh first."
RAGGER_BIN="$(cd "$(dirname "$RAGGER_BIN")" && pwd)/$(basename "$RAGGER_BIN")"

echo ""
info "Wiring Ragger into Claude Code"
info "  binary: $RAGGER_BIN"

# 2. Agent usage instructions (idempotent — preserves edits)
mkdir -p "$HOME/.ragger"
if [ ! -f "$INSTR_SRC" ]; then
    warn "Instructions source missing at $INSTR_SRC — skipping."
elif [ -f "$INSTR_DST" ]; then
    info "  kept existing $INSTR_DST"
else
    cp "$INSTR_SRC" "$INSTR_DST"
    info "  installed $INSTR_DST"
fi

# 3. Install / refresh the Stop-hook helper. Always rewrite so improvements
# ship with each install (the script reads from your config at run time,
# nothing to migrate).
mkdir -p "$HOOK_DIR"
cat > "$HOOK_SCRIPT" <<'HOOKEOF'
#!/bin/bash
# claude-code-capture-turn.sh — Claude Code "Stop" hook → ragger POST /turn.
#
# Stdin is the hook payload from Claude Code (JSON with `transcript_path`
# and `session_id`). We lift the most recent human user prompt and the
# assistant reply that just landed and hand them to Ragger. Anything
# that fails here is silently swallowed so a broken capture never breaks
# the user's session.
set -u
SOCKET="$HOME/.ragger/ragger.sock"
TOKEN_FILE="$HOME/.ragger/token"
[ -S "$SOCKET" ] || exit 0      # daemon not running → nothing to do
TOKEN=""
[ -r "$TOKEN_FILE" ] && TOKEN="$(cat "$TOKEN_FILE")"

# Snapshot the hook JSON before invoking python — python3's heredoc
# below would otherwise consume stdin and leave nothing for the parser.
HOOK_JSON="$(cat)"

PAYLOAD="$(HOOK_JSON="$HOOK_JSON" python3 <<'PYEOF' 2>/dev/null
import json, os, sys

def text_of(content):
    if isinstance(content, str): return content.strip()
    if isinstance(content, list):
        return "\n\n".join(
            b.get("text","").strip() for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    return ""

def is_tool_result(content):
    """True if this user message is a tool result, not a human prompt."""
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        )
    return False

def is_synthetic(txt):
    """True for injected non-human messages (task notifications, local command output)."""
    return txt.startswith(("<task-notification>", "<local-command-stdout>"))

try:
    inp = json.loads(os.environ.get("HOOK_JSON","") or "{}")
except Exception:
    sys.exit(0)
tp  = inp.get("transcript_path", "")
sid = inp.get("session_id", "")
if not tp or not os.path.exists(tp):
    sys.exit(0)

events = []
with open(tp, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: events.append(json.loads(line))
        except Exception: continue

assistant_text, assistant_model, user_text = "", "", ""
seen_assistant = False
for ev in reversed(events):
    t   = ev.get("type")
    msg = ev.get("message", {}) or {}
    content = msg.get("content")

    if not seen_assistant:
        if t == "assistant":
            txt = text_of(content)
            if txt:
                assistant_text  = txt
                assistant_model = msg.get("model", "")
                seen_assistant  = True
    else:
        if t == "user":
            # Skip tool-result messages — keep walking back to the
            # human prompt (text content, not tool_result blocks).
            if is_tool_result(content):
                continue
            txt = text_of(content)
            # Skip synthetic injected messages (task notifications etc.)
            if txt and is_synthetic(txt):
                continue
            if txt:
                user_text = txt
                break

if not assistant_text:
    sys.exit(0)

print(json.dumps({
    "user":       user_text,
    "assistant":  assistant_text,
    "model":      assistant_model,
    "session_id": sid,
}))
PYEOF
)"

[ -n "$PAYLOAD" ] || exit 0

curl -sS --max-time 5 \
     --unix-socket "$SOCKET" \
     -H "Content-Type: application/json" \
     ${TOKEN:+-H "Authorization: Bearer *** \
     -X POST "http://localhost/turn" \
     -d "$PAYLOAD" >/dev/null 2>&1 || true
HOOKEOF
chmod 0755 "$HOOK_SCRIPT"
info "  installed $HOOK_SCRIPT"

# 4. Register the MCP server in ~/.claude.json
[ -f "$MCP_CFG" ] || printf '{}\n' > "$MCP_CFG"
RAGGER_BIN="$RAGGER_BIN" python3 - "$MCP_CFG" <<'PYEOF'
import json, os, sys, pathlib, shutil, datetime
p = pathlib.Path(sys.argv[1])
try:
    data = json.loads(p.read_text() or "{}")
except Exception as e:
    print(f"[!] Could not parse {p} ({e}) — aborting MCP edit", file=sys.stderr); sys.exit(1)
servers = data.setdefault("mcpServers", {})
desired = {"command": os.environ["RAGGER_BIN"], "args": ["mcp"]}
if servers.get("ragger") == desired:
    print("[+]   mcpServers.ragger already up to date in .claude.json")
    sys.exit(0)
bak = p.with_name(p.name + f".bak-{datetime.date.today().isoformat()}")
if not bak.exists():
    shutil.copy2(p, bak)
    print(f"[+]   backed up → {bak.name}")
if "ragger" in servers:
    servers["ragger"].update(desired)
    print("[+]   refreshed mcpServers.ragger in .claude.json")
else:
    servers["ragger"] = desired
    print("[+]   added mcpServers.ragger to .claude.json")
p.write_text(json.dumps(data, indent=2) + "\n")
PYEOF

# 5. Wire the Stop hook into ~/.claude/settings.json
mkdir -p "$(dirname "$HOOK_CFG")"
[ -f "$HOOK_CFG" ] || printf '{}\n' > "$HOOK_CFG"
HOOK_SCRIPT="$HOOK_SCRIPT" python3 - "$HOOK_CFG" <<'PYEOF'
import json, os, sys, pathlib, shutil, datetime
p = pathlib.Path(sys.argv[1])
try:
    data = json.loads(p.read_text() or "{}")
except Exception as e:
    print(f"[!] Could not parse {p} ({e}) — aborting hook edit", file=sys.stderr); sys.exit(1)

hooks = data.setdefault("hooks", {})
stop  = hooks.setdefault("Stop", [])
target = os.environ["HOOK_SCRIPT"]
desired = {"type": "command", "command": target}

# Look for an existing matcher entry with our hook; refresh in place if found.
present = False
for entry in stop:
    for h in entry.get("hooks", []):
        if h.get("command") == target:
            present = True
            h.update(desired)
            break
    if present: break

if not present:
    bak = p.with_name(p.name + f".bak-{datetime.date.today().isoformat()}")
    if not bak.exists():
        shutil.copy2(p, bak)
        print(f"[+]   backed up → {bak.name}")
    stop.append({"matcher": "", "hooks": [desired]})
    print("[+]   added Stop hook → " + target)
else:
    print("[+]   Stop hook already wired — refreshed in place")
p.write_text(json.dumps(data, indent=2) + "\n")
PYEOF

echo ""
info "Done."
info "Restart your Claude Code session (exit + reopen) so it picks up the"
info "MCP registration and Stop hook. The ragger daemon must be running"
info "for turn capture: 'ragger start'."
