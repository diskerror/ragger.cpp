#!/bin/bash
# install-hermes.sh — Wire Ragger as a memory provider plugin for Hermes Agent.
#
# Usage:
#   ./scripts/install-hermes.sh              # install / update
#   ./scripts/install-hermes.sh --uninstall  # remove plugin and revert config
#
# Run this AFTER install-bin.sh if you have Hermes Agent installed.
# Requires: ragger already installed (~/.local/bin/ragger)
#
# What install does:
#   - Copies hermes-plugin/ files into ~/.hermes/plugins/ragger-memory/
#   - Sets memory.provider: ragger in ~/.hermes/config.yaml
#     (conservative merge — only sets the key if it is currently empty)
#
# What uninstall does:
#   - Removes ~/.hermes/plugins/ragger-memory/
#   - Clears memory.provider back to '' if it is currently 'ragger'
#
# Idempotent — safe to re-run.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
fail() { echo -e "${RED}[!]${NC} $*" >&2; exit 1; }

HERMES_HOME="$HOME/.hermes"
PLUGIN_DST="$HERMES_HOME/plugins/ragger"
HERMES_CFG="$HERMES_HOME/config.yaml"

# ============================================================
# Uninstall
# ============================================================

if [ "${1:-}" = "--uninstall" ]; then
    echo ""
    info "Uninstalling Ragger memory plugin from Hermes"

    # Remove plugin directory
    if [ -d "$PLUGIN_DST" ]; then
        rm -rf "$PLUGIN_DST"
        info "  removed $PLUGIN_DST"
    else
        info "  $PLUGIN_DST not found — already removed"
    fi

    # Revert memory.provider in config.yaml
    if [ -f "$HERMES_CFG" ] && command -v python3 >/dev/null 2>&1; then
        python3 - "$HERMES_CFG" << 'PYEOF'
import sys, pathlib, shutil, datetime, re

cfg_path = pathlib.Path(sys.argv[1])
lines = cfg_path.read_text(encoding="utf-8").splitlines(keepends=True)

in_memory = False
changed = False

for i, line in enumerate(lines):
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]

    if re.match(r'^memory:\s*(?:#.*)?$', line.rstrip()):
        in_memory = True
        continue

    if in_memory:
        if stripped and not stripped.startswith("#") and not line[0].isspace():
            break  # left the memory block

        m = re.match(r'^(\s+provider:\s*)(.*)$', line.rstrip())
        if m:
            current_val = m.group(2).strip().strip("'\"")
            if current_val == "ragger":
                lines[i] = f"{m.group(1)}''\n"
                changed = True
                print("[+]   memory.provider cleared (was 'ragger')")
            elif current_val:
                print(f"[+]   memory.provider is '{current_val}', not 'ragger' — leaving alone")
            else:
                print("[+]   memory.provider already empty — no change needed")
            break

if changed:
    bak = cfg_path.with_suffix(f".yaml.bak-{datetime.date.today().isoformat()}")
    if not bak.exists():
        shutil.copy2(cfg_path, bak)
        print(f"[+]   backed up → {bak.name}")
    cfg_path.write_text("".join(lines), encoding="utf-8")
    print(f"[+]   updated {cfg_path}")
PYEOF
    elif [ -f "$HERMES_CFG" ]; then
        warn "python3 not found — clear memory.provider manually in $HERMES_CFG"
    fi

    echo ""
    info "Uninstall complete. Restart the Hermes gateway (hermes gateway restart) to pick up the change."
    exit 0
fi

# ============================================================
# Install
# ============================================================

if [ "$(id -u)" -eq 0 ]; then
    fail "Do NOT run install-hermes.sh as root."
fi

cd "$(dirname "$0")/.."
SRC="$(pwd)"

PLUGIN_SRC="$SRC/hermes-plugin"

# --- Preflight ---

if [ ! -d "$HERMES_HOME" ]; then
    fail "Hermes not found at $HERMES_HOME — install Hermes Agent first."
fi

if [ ! -d "$PLUGIN_SRC" ]; then
    fail "Plugin source not found at $PLUGIN_SRC — run from the Ragger repo root."
fi

if [ ! -f "$HOME/.local/bin/ragger" ] && ! command -v ragger >/dev/null 2>&1; then
    warn "ragger binary not found on PATH — install Ragger first (./scripts/install-bin.sh)."
    warn "Continuing anyway; plugin will be inactive until ragger is installed."
fi

echo ""
info "Installing Ragger memory plugin into Hermes at $HERMES_HOME"

# --- Copy plugin files ---

mkdir -p "$PLUGIN_DST"

for f in plugin.yaml __init__.py; do
    if [ -f "$PLUGIN_SRC/$f" ]; then
        cp "$PLUGIN_SRC/$f" "$PLUGIN_DST/$f"
        info "  wrote $PLUGIN_DST/$f"
    else
        fail "Missing $PLUGIN_SRC/$f — repo may be incomplete."
    fi
done

# --- Wire memory.provider in config.yaml ---

if [ ! -f "$HERMES_CFG" ]; then
    warn "$HERMES_CFG not found — run Hermes once to create it, then re-run this script."
    echo ""
    warn "When config.yaml exists, add manually:"
    warn "    memory:"
    warn "      provider: ragger"
    exit 0
fi

if ! command -v python3 >/dev/null 2>&1; then
    warn "python3 not found — cannot update $HERMES_CFG automatically."
    warn "Add the following manually:"
    warn "    memory:"
    warn "      provider: ragger"
    exit 0
fi

python3 - "$HERMES_CFG" << 'PYEOF'
import sys, pathlib, shutil, datetime, re

cfg_path = pathlib.Path(sys.argv[1])
lines = cfg_path.read_text(encoding="utf-8").splitlines(keepends=True)

in_memory = False
found = False
changed = False
memory_header_idx = None   # index of the "memory:" line
memory_indent = "  "        # indent used for keys inside memory: (default 2 spaces)
insert_idx = None           # where to insert a new provider: line if missing

for i, line in enumerate(lines):
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]

    # Detect the memory: section header (top-level key)
    if re.match(r'^memory:\s*(?:#.*)?$', line.rstrip()):
        in_memory = True
        memory_header_idx = i
        insert_idx = i + 1
        continue

    if in_memory:
        # A new top-level key ends the memory block
        if stripped and not stripped.startswith("#") and not line[0].isspace():
            in_memory = False
            break

        # Track the indent + the last line still inside the memory block,
        # so we know where/how to insert a provider: key if it's missing.
        if stripped and line[0].isspace():
            memory_indent = indent
            insert_idx = i + 1

        # Look for "  provider: <value>"
        m = re.match(r'^(\s+provider:\s*)(.*)$', line.rstrip())
        if m:
            found = True
            current_val = m.group(2).strip().strip("'\"")
            if current_val in ("", "null"):
                lines[i] = f"{m.group(1)}ragger\n"
                changed = True
                print("[+]   memory.provider = ragger")
            elif current_val == "ragger":
                print("[+]   memory.provider already set to ragger — no change needed")
            else:
                print(f"[!]   memory.provider already set to '{current_val}' — leaving alone")
                print( "      To switch to Ragger, set it manually: memory.provider: ragger")
            break

if not found:
    if memory_header_idx is not None:
        # memory: block exists but has no provider key — insert one.
        lines.insert(insert_idx, f"{memory_indent}provider: ragger\n")
        changed = True
        print("[+]   memory.provider key missing — inserted: ragger")
    else:
        # No memory: block at all — append a fresh one.
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append("memory:\n")
        lines.append("  provider: ragger\n")
        changed = True
        print("[+]   memory: block missing — appended with provider: ragger")

if changed:
    bak = cfg_path.with_suffix(f".yaml.bak-{datetime.date.today().isoformat()}")
    if not bak.exists():
        shutil.copy2(cfg_path, bak)
        print(f"[+]   backed up → {bak.name}")
    cfg_path.write_text("".join(lines), encoding="utf-8")
    print(f"[+]   updated {cfg_path}")
PYEOF

# --- Patch Hermes core to pass the live model name to memory providers ---
#
# Hermes core calls MemoryProvider.on_turn_start() from agent/turn_context.py
# WITHOUT the model name (the provider docstring says kwargs "may include
# model", but nothing sends it), so the plugin can't record which model is
# conversing (turns.model_id stays NULL). We add model= to that call.
#
# This is NOT shipped upstream. `hermes update` autostashes local edits and
# can fail to re-apply them when turn_context.py moves (it did on 2026-09-15,
# which is when model_id capture silently stopped), so re-run this installer
# after every Hermes update. Idempotent; handles both known call shapes.

if command -v python3 >/dev/null 2>&1; then
    python3 - << 'PYEOF'
import pathlib, re, sys

path = pathlib.Path.home() / ".hermes" / "hermes-agent" / "agent" / "turn_context.py"
if not path.exists():
    print(f"[!]   {path} not found — skipping core patch")
    print("      (model name won't be recorded until this is patched by hand)")
    sys.exit(0)

src = path.read_text(encoding="utf-8")
call = re.search(r"on_turn_start\(\s*agent\._user_turn_count,\s*(_query|_turn_msg)\s*(,[^)]*)?\)", src, re.S)
if call is None:
    print(f"[!]   on_turn_start call site not found in {path}")
    print("      Hermes internals may have changed — patch model= by hand")
elif "model=" in call.group(0):
    print("[+]   core model patch already present — no change needed")
else:
    old = call.group(0)
    arg = call.group(1)
    new = old.replace(arg, arg + ', model=getattr(agent, "model", "") or ""', 1)
    path.write_text(src.replace(old, new, 1), encoding="utf-8")
    print(f"[+]   patched {path} (pass model= to on_turn_start)")
PYEOF
else
    warn "python3 not found — cannot apply Hermes core model patch."
    warn "Manually add 'model=getattr(agent, \"model\", \"\") or \"\"' to the on_turn_start call in"
    warn "  ~/.hermes/hermes-agent/agent/turn_context.py"
fi

# --- Patch Hermes core to fire a post_turn_finalized hook ---
#
# The plugin captures INTERRUPTED turns (user /stop) through a
# "post_turn_finalized" plugin hook, but Hermes core has no such hook: the
# only hook that fires on an interrupted turn (on_session_end) carries no
# message text. Tracking upstream: NousResearch/hermes-agent#108808. Until that
# lands we add one call to agent/turn_finalizer.py, right after the external
# memory sync in finalize_turn(). Idempotent. Like the model patch above, a
# `hermes update` can undo it, so re-run this installer after every update.

if command -v python3 >/dev/null 2>&1; then
    python3 - << 'PYEOF'
import pathlib, sys

path = pathlib.Path.home() / ".hermes" / "hermes-agent" / "agent" / "turn_finalizer.py"
if not path.exists():
    print(f"[!]   {path} not found — skipping post_turn_finalized patch")
    print("      (interrupted turns won't be captured until this is patched by hand)")
    sys.exit(0)

src = path.read_text(encoding="utf-8")
marker = "post_turn_finalized"
anchor = "    # Background memory/skill review runs AFTER delivery"
patch = '''    # Plugin hook: post_turn_finalized (local patch from Ragger's install-hermes.sh;
    # not an upstream hook, see hermes-agent#108808). Fires for EVERY turn,
    # including interrupted ones, so a memory plugin can capture them.
    _invoke_hook_safely(
        "post_turn_finalized", logger,
        turn_data={
            "user": _summarize_user_message_for_log(original_user_message, sep="\\n") if original_user_message else "",
            "assistant": final_response if isinstance(final_response, str) else "",
            "model": getattr(agent, "model", "") or "",
            "session_id": agent.session_id or "",
            "interrupted": bool(interrupted),
            "turn_id": turn_id,
        },
    )

'''

if marker in src:
    print("[+]   post_turn_finalized patch already present — no change needed")
elif src.count(anchor) == 1 and "_invoke_hook_safely" in src and "_summarize_user_message_for_log" in src:
    path.write_text(src.replace(anchor, patch + anchor, 1), encoding="utf-8")
    print(f"[+]   patched {path} (fire post_turn_finalized from finalize_turn)")
else:
    print(f"[!]   finalize_turn anchor not found in {path}")
    print("      Hermes internals may have changed — patch post_turn_finalized by hand")
PYEOF
else
    warn "python3 not found — cannot apply the Hermes core post_turn_finalized patch."
fi

# --- Done ---

echo ""
info "Hermes plugin installed."
echo ""
echo "  Start the Ragger daemon:  ragger start"
echo "  Restart Hermes gateway:   hermes gateway restart  (or restart the Telegram bot)"
echo ""
echo "  To verify: send '/memory' or start a conversation — Hermes will show"
echo "  'Ragger Memory' in the system prompt context block."
echo ""
echo "  To remove:  ./scripts/install-hermes.sh --uninstall"
echo ""
if ! ragger status >/dev/null 2>&1; then
    warn "Ragger daemon is not currently running. Start it with: ragger start"
fi
