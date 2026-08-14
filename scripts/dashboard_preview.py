#!/usr/bin/env python3
"""
Dev preview server for the Ragger dashboard.

Serves web/dashboard.html with a MOCK backend so you can iterate on the
HTML/CSS/JS in a browser without building the C++ daemon. It fakes the two
real endpoints the dashboard talks to:

    GET  /config          -> config entries built from default-settings.txt-ish
                             schema (mirrors the C++ config_schema in en.h)
    GET  /config/<key>    -> one entry
    PUT  /config/<key>    -> validate + echo back (in-memory only)
    GET  /events          -> SSE stream: 'stats' every 2s + occasional 'activity'
    GET  /dashboard       -> the HTML file itself

Usage:
    python3 scripts/dashboard_preview.py [--port 8899] [--open]

This is a DEV TOOL ONLY. The real values, validation, and persistence live in
the C++ daemon; this just lets you see and click the UI. The schema below is a
hand-mirror of include/ragger/lang/en.h kConfigSchema — if you add config keys
there, add them here too (or regenerate).
"""
import argparse
import json
import os
import re
import sys
import time
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.normpath(os.path.join(HERE, "..", "web", "dashboard.html"))

# Mirror of the C++ kConfigSchema (include/ragger/lang/en.h). Each entry:
# (key, section, label, type, edit, default, options, help)
SCHEMA = [
    # server
    ("socket_enable", "server", "Socket Enable", "boolean", "restart", "true", "",
     "Unix domain socket at ~/.ragger/ragger.sock."),
    ("bind", "server", "TCP Bind Address", "string", "restart", "", "",
     "TCP bind address. Empty = no TCP listener."),
    ("port", "server", "TCP Port", "integer", "locked", "8432", "",
     "The port the running daemon is bound to. Read-only — set Desired Port and restart."),
    ("desired_port", "server", "Desired Port", "integer", "restart", "", "",
     "The TCP port to bind on the next restart. Any restart adopts it."),
    ("server_name", "server", "Server Name", "string", "restart", "", "",
     "Hostname for cpp-httplib (optional)."),
    ("cert", "server", "TLS Certificate", "path", "restart", "", "",
     "TLS certificate chain (PEM). Both cert and key enable HTTPS."),
    ("key", "server", "TLS Private Key", "path", "restart", "", "",
     "TLS private key (PEM)."),
    ("capture_turns", "server", "Capture Turns", "boolean", "live", "true", "",
     "Accept agent-pushed turns into the turns table."),
    # embedding — current (read-only) + desired (editable) + embed subprocess
    ("embedding_model", "embedding", "Current Model", "string", "locked", "all-MiniLM-L6-v2", "",
     "The embedding model the stored vectors were built with. Read-only."),
    ("embedding_dimensions", "embedding", "Current Dimensions", "integer", "locked", "384", "",
     "Vector dimensionality of the stored vectors. Read-only."),
    ("embedding_vector_type", "embedding", "Current Vector Type", "enum", "locked", "f16", "f16,f32",
     "On-disk precision of the stored vectors. Read-only."),
    ("desired_embedding_model", "embedding", "Desired Model", "enum", "rebuild", "all-MiniLM-L6-v2", "all-MiniLM-L6-v2",
     "The embedding model to switch to. Choices come from ~/.ragger/models/."),
    ("desired_embedding_vector_type", "embedding", "Desired Vector Type", "enum", "rebuild", "f16", "f16,f32",
     "Target on-disk vector precision: f16 or f32."),
    ("desired_embedding_dimensions", "embedding", "Desired Dimensions", "integer", "rebuild", "384", "",
     "Target vector dimensionality. Model-determined."),
    ("embed_timeout_ms", "embedding", "Embed Timeout (ms)", "integer", "live", "10000", "",
     "Per-embed subprocess timeout in milliseconds."),
    ("embed_retries", "embedding", "Embed Retries", "integer", "live", "1", "",
     "Retries on a failed or timed-out embed."),
    ("embed_max_workers", "embedding", "Max Workers", "integer", "live", "8", "",
     "Cap on concurrent embed subprocesses."),
    # search
    ("default_limit", "search", "Default Limit", "integer", "live", "5", "",
     "Number of search results returned by default."),
    ("default_min_score", "search", "Default Min Score", "float", "live", "0.4", "",
     "Minimum similarity score for results (0.0 - 1.0)."),
    ("bm25_enabled", "search", "BM25 Enabled", "boolean", "live", "true", "",
     "Enable BM25 keyword matching."),
    ("bm25_weight", "search", "BM25 Weight", "float", "live", "4", "",
     "Weight for keyword matching (ratio)."),
    ("vector_weight", "search", "Vector Weight", "float", "live", "8", "",
     "Weight for semantic similarity (ratio)."),
    ("phon_weight", "search", "Phonetic Weight", "float", "live", "1", "",
     "Sounds-like (dolphining) weight. 0 disables."),
    ("max_search_limit", "search", "Max Search Limit", "integer", "locked", "0", "",
     "System ceiling on default_limit (0 = no ceiling)."),
    # summarizer
    ("summarizer_model", "summarizer", "Model", "string", "live", "qwen3-4b-instruct-2507", "",
     "Model used for L2/L3 summarization."),
    ("summarizer_api_url", "summarizer", "API URL", "string", "live", "http://localhost:1234/v1", "",
     "Inference endpoint (LM Studio convention)."),
    ("summarizer_api_key", "summarizer", "API Key", "string", "live", "", "",
     "API key for the summarizer endpoint."),
    ("summarizer_max_tokens", "summarizer", "Max Tokens", "integer", "live", "1024", "",
     "Max tokens for a summary. 0 inherits global default."),
    ("summarizer_target_pct", "summarizer", "Target %", "integer", "live", "40", "",
     "Target summary length as % of raw turn; 0 = 30%."),
    ("summarizer_max_pct", "summarizer", "Max %", "integer", "live", "60", "",
     "Hard cap as % of raw turn; 0 = 60%."),
    ("summarizer_prompt", "summarizer", "Prompt", "text", "live", "", "",
     "System prompt. Empty = built-in default. Two {} placeholders required."),
    # logging
    ("log_level", "logging", "Log Level", "enum", "live", "warn",
     "trace,debug,info,warn,error,critical",
     "Verbosity of the single activity log."),
    ("log_max_size_mb", "logging", "Log Max Size (MB)", "integer", "live", "1", "",
     "Rotate activity.log at this size. 0 disables."),
    ("log_max_age_days", "logging", "Log Max Age (days)", "integer", "live", "14", "",
     "Delete rotated backups older than this. 0 disables."),
    # housekeeping
    ("cleanup_max_age_hours", "housekeeping", "Cleanup Max Age (hrs)", "float", "live", "0", "",
     "Raw-turn lifetime (0 = keep forever)."),
    ("housekeeping_interval", "housekeeping", "Housekeeping Interval (sec)", "integer", "live", "60", "",
     "Seconds between passes (0 disables)."),
    ("episode_idle_minutes", "housekeeping", "Episode Idle Minutes", "integer", "live", "15", "",
     "Idle gap (min) that closes an episode."),
    ("episode_threshold_base", "housekeeping", "Episode Threshold Base", "float", "live", "0.2", "",
     "Starting similarity threshold for episode boundaries."),
    ("episode_threshold_cap", "housekeeping", "Episode Threshold Cap", "float", "live", "0.5", "",
     "Maximum threshold after time scaling."),
    ("episode_threshold_step", "housekeeping", "Episode Threshold Step", "float", "live", "0.01", "",
     "Threshold increment per step."),
    ("episode_step_minutes", "housekeeping", "Episode Step Minutes", "float", "live", "1.0", "",
     "Minutes of idle per increment."),
    ("catch_up_batch_size", "housekeeping", "Catch-up Batch Size", "integer", "live", "8", "",
     "Per-tick cap on unsummarized turns / retries."),
    ("project_gap_days", "housekeeping", "Project Gap Days", "integer", "live", "7", "",
     "Min gap (days) that triggers a project-boundary close."),
    # import
    ("minimum_chunk_size", "import", "Minimum Chunk Size", "integer", "live", "300", "",
     "Minimum size of an imported document chunk."),
    ("normalize_home", "import", "Normalize Home", "boolean", "live", "true", "",
     "Replace /Users/<you> or /home/<user> with ~ in stored paths."),
]

# In-memory current values (start = defaults).
VALUES = {row[0]: row[5] for row in SCHEMA}

# Mock embedding state for the preview.
MOCK_MODELS = ["all-MiniLM-L6-v2"]
EMBED = {"reembedding": False}
# Mock restart state: the port the "daemon" is bound to (vs config port).
BOUND = {"port": 8432, "bind": "127.0.0.1"}


def entry_json(row):
    key, section, label, typ, edit, default, options, help_ = row
    e = {
        "key": key, "section": section, "label": label, "type": typ,
        "edit": edit, "default": default, "help": help_,
        "value": VALUES.get(key, default),
    }
    if typ == "enum":
        e["options"] = [o for o in options.split(",") if o]
    return e


def validate(row, value):
    typ = row[3]
    if value == "":
        return True  # blank = default
    if typ == "integer":
        return re.fullmatch(r"-?\d+", value) is not None
    if typ == "float":
        return re.fullmatch(r"-?\d*\.?\d+", value) is not None
    if typ == "boolean":
        return value in ("true", "false")
    if typ == "enum":
        return value in row[6].split(",")
    return True


SCHEMA_BY_KEY = {row[0]: row for row in SCHEMA}

# Fake activity lines the SSE stream will emit periodically.
FAKE_ACTIVITY = [
    "GET /search 200 (12ms)",
    "POST /store 200",
    "housekeeping: closed episode (idle 15m)",
    "summarizer: wrote L2 summary for turn 8841",
    "GET /count 200",
    "backfill_embeddings: 3 rows updated",
]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # quiet

    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/dashboard"):
            try:
                with open(HTML_PATH, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(404, "dashboard.html not found — build it first",
                           "text/plain")
            return
        if path == "/config":
            self._send(200, json.dumps({"config": [entry_json(r) for r in SCHEMA]}))
            return
        m = re.fullmatch(r"/config/([A-Za-z0-9_]+)", path)
        if m:
            row = SCHEMA_BY_KEY.get(m.group(1))
            if not row:
                self._send(404, json.dumps({"error": "unknown key"}))
            else:
                self._send(200, json.dumps(entry_json(row)))
            return
        if path == "/stats":
            self._send(200, json.dumps(self._stats()))
            return
        if path == "/models":
            self._send(200, json.dumps({"models": MOCK_MODELS}))
            return
        if path == "/embedding/status":
            self._send(200, json.dumps(self._embed_status()))
            return
        if path == "/restart/status":
            self._send(200, json.dumps(self._restart_status()))
            return
        if path == "/health":
            self._send(200, json.dumps({"status": "ok"}))
            return
        if path == "/events":
            self._sse()
            return
        self._send(404, "not found", "text/plain")

    def _restart_status(self):
        # desired_port (editable) falls back to the bound port when unset.
        dp = VALUES.get("desired_port", "")
        desired = int(dp) if dp else BOUND["port"]
        cfg_bind = VALUES.get("bind", "127.0.0.1") or "127.0.0.1"
        tls = bool(VALUES.get("cert")) and bool(VALUES.get("key"))
        return {
            "bound_port": BOUND["port"], "configured_port": desired,
            "bound_bind": BOUND["bind"], "configured_bind": cfg_bind,
            "needs_restart": (BOUND["port"] != desired or BOUND["bind"] != cfg_bind),
            "scheme": "https" if tls else "http",
        }

    def _embed_status(self):
        cur_vt = VALUES.get("embedding_vector_type", "f16")
        des_vt = VALUES.get("desired_embedding_vector_type", "f16")
        cur_model = VALUES.get("embedding_model", "all-MiniLM-L6-v2")
        des_model = VALUES.get("desired_embedding_model", cur_model)
        cur_dim = int(VALUES.get("embedding_dimensions", "384"))
        des_dim = int(VALUES.get("desired_embedding_dimensions", "384"))
        needs = (cur_vt != des_vt or cur_model != des_model or cur_dim != des_dim)
        return {
            "current": {"model": cur_model, "vector_type": cur_vt, "dimensions": cur_dim},
            "desired": {"model": des_model, "vector_type": des_vt, "dimensions": des_dim},
            "needs_update": needs and not EMBED["reembedding"],
            "reembedding": EMBED["reembedding"],
        }

    def _do_reembed(self):
        # Simulate a re-embed: hold the flag ~4s, then promote current:=desired.
        EMBED["reembedding"] = True
        time.sleep(4)
        VALUES["embedding_model"] = VALUES.get("desired_embedding_model", VALUES["embedding_model"])
        VALUES["embedding_vector_type"] = VALUES.get("desired_embedding_vector_type", "f16")
        VALUES["embedding_dimensions"] = VALUES.get("desired_embedding_dimensions", "384")
        EMBED["reembedding"] = False

    def do_PUT(self):
        path = self.path.split("?", 1)[0]
        m = re.fullmatch(r"/config/([A-Za-z0-9_]+)", path)
        if not m:
            self._send(404, json.dumps({"error": "not found"}))
            return
        key = m.group(1)
        row = SCHEMA_BY_KEY.get(key)
        if not row:
            self._send(404, json.dumps({"error": "unknown key"}))
            return
        if row[4] == "locked":
            self._send(403, json.dumps({"error": "key is locked"}))
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        try:
            value = json.loads(body).get("value", "")
        except json.JSONDecodeError:
            value = ""
        if not validate(row, value):
            self._send(400, json.dumps({"error": "invalid value"}))
            return
        # Empty -> revert to default.
        VALUES[key] = row[5] if value == "" else value
        self._send(200, json.dumps({
            "key": key, "value": VALUES[key],
            "restart_required": row[4] == "restart",
            "rebuild_required": row[4] == "rebuild",
        }))

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/restart":
            # Preview can't re-exec; simulate the startup rectify: adopt
            # desired_port into the bound/committed port after a short delay.
            dp = VALUES.get("desired_port", "")
            new_port = int(dp) if dp else BOUND["port"]
            cfg_bind = VALUES.get("bind", "127.0.0.1") or "127.0.0.1"
            tls = bool(VALUES.get("cert")) and bool(VALUES.get("key"))
            scheme = "https" if tls else "http"
            url = "{}://{}:{}/dashboard".format(scheme, cfg_bind, new_port)
            def promote():
                time.sleep(2)
                BOUND["port"] = new_port
                BOUND["bind"] = cfg_bind
                VALUES["port"] = str(new_port)      # committed port follows
            threading.Thread(target=promote, daemon=True).start()
            self._send(200, json.dumps({"status": "restarting",
                       "reconnect_url": url, "port": new_port}))
            return
        if path == "/embedding/update":
            if EMBED["reembedding"]:
                self._send(409, json.dumps({"error": "already in progress"}))
                return
            if not self._embed_status()["needs_update"]:
                self._send(200, json.dumps({"status": "up-to-date", "reembedding": False}))
                return
            threading.Thread(target=self._do_reembed, daemon=True).start()
            self._send(200, json.dumps({"status": "started", "reembedding": True}))
            return
        # Search short-circuit demo while re-embedding.
        if path == "/search":
            if EMBED["reembedding"]:
                self._send(200, json.dumps({"results": [{"id": 0,
                    "text": "Search not available. Re-embedding in progress.",
                    "score": 0.0, "metadata": {}, "timestamp": ""}]}))
            else:
                self._send(200, json.dumps({"results": []}))
            return
        self._send(404, json.dumps({"error": "not found"}))

    def _stats(self):
        return {
            "status": "running", "version": "dev-preview", "memories": 8763,
            "tables": {"turns": 8763, "summaries": 8120, "turn_summaries": 8120,
                       "sessions": 542, "documents": 34, "decisions": 61,
                       "models": 3},
        }

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        i = 0
        try:
            while True:
                frame = "event: stats\ndata: " + json.dumps(self._stats()) + "\n\n"
                self.wfile.write(frame.encode())
                if i % 2 == 0:
                    line = FAKE_ACTIVITY[i // 2 % len(FAKE_ACTIVITY)]
                    a = "event: activity\ndata: " + json.dumps({"line": line}) + "\n\n"
                    self.wfile.write(a.encode())
                self.wfile.flush()
                i += 1
                time.sleep(2)
        except (BrokenPipeError, ConnectionResetError):
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--open", action="store_true", help="open in browser")
    args = ap.parse_args()

    if not os.path.exists(HTML_PATH):
        print(f"WARNING: {HTML_PATH} does not exist yet — build/generate it "
              f"before previewing.", file=sys.stderr)

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/dashboard"
    print(f"Ragger dashboard preview (MOCK backend) → {url}")
    print("Ctrl+C to stop.")
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
