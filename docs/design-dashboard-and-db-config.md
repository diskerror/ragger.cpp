# Design: DB-backed config + web dashboard

Status: proposal / pre-implementation. No code yet.

## Goal

Retire `settings.ini` as the config source. Config lives in the DB `settings`
table. Two faces read/write it: the **CLI** and a **web dashboard**. One
metadata table (name, type, allowed values, validation, default, help text)
feeds both, maintained in one place.

---

## 1. Config storage — INI out, `settings` table in

The `settings` table already exists (`key TEXT PRIMARY KEY, value TEXT`).

**Load order:** compiled-in defaults → overlay rows present in `settings`.
No file parsing. A missing/blank row means "use the default" — which gives us
the dashboard's "blank box = default value" behavior automatically.

**Server-locked keys** (port, embedding model/dimensions/vector_type, socket
path, system ceilings) stay locked — writable only where they're currently
allowed, not freely editable at runtime. (Embedding settings have a known
wrinkle to discuss separately.)

**Migration:** one-time import of an existing `settings.ini` into the table,
then the file is ignored. `default-settings.txt` stops being the runtime seed.

---

## 2. Single metadata source — `include/lang/en.h`

A compile-time table. Each config key carries:

| field | purpose |
|---|---|
| key | DB row key (`bm25_weight`) |
| pretty name | UI label ("BM25 Weight") |
| type | int / float / bool / enum / string / path |
| allowed values | enum → popup menu options |
| validation | e.g. digits-only for int boxes |
| default | fallback when row blank/missing |
| help text | the explanations now living in INI comments |

- **CLI** iterates this for help output and validates `set` against it.
- **HTML/JS** gets the same help strings injected at compile time via `{}`
  placeholders (same pattern as `embed_ini.cmake`).

One edit updates both faces. Replaces today's INI comments, which no code
can read.

---

## 3. CLI — standard get/set

```
ragger config                      → help text (all keys + explanations)
ragger config get -a [-j]          → all values (‑j = JSON)
ragger config get <name> [-j]      → one value
ragger config set <name> <value>   → validate, write row, signal daemon reload
```

`get` with neither `-a` nor a name → help text.

---

## 4. Dashboard

**Launch:**
```
ragger dashboard        → prints http://127.0.0.1:<port>/dashboard?token=<token>
ragger dashboard -b     → also opens it in the default browser
```
Same port as normal Ragger access. HTML/JS is **compiled into the binary**
(like the INI today) and served from memory — self-contained, no missing-asset
failures.

**Access control:** the token gates the URL. Only a user with RW access to
`~/.ragger` can read the token file, so possessing the token *is* proof of
authorization. No separate auth layer.

**Transport (decided): HTTP + SSE**
- **Config reads/writes** — plain HTTP. `GET /config` (all), `PUT /config/<key>`
  (one). Every committed change is one PUT.
- **Live push** — SSE. Browser opens `EventSource('/events?token=…')`; server
  holds it open and pushes stats + activity. One-directional, runs on the
  existing httplib server, no new library, auto-reconnects. WebSockets rejected
  — bidirectional capability Ragger never needs.

**Behavior:**
- Change a value (Return / Tab / mouseup) → immediate `PUT` → stored in
  `settings` → live in the program.
- Enum keys → popup menus. Bool keys → checkboxes. Int/float boxes reject
  invalid characters at input (e.g. `[0-9]` only).
- Clearing a box → key reverts to default; daemon pushes the resolved default
  value back to the box.
- Stats + recent-activity panes refresh every ~5s over the SSE stream.

**Layout:** GitHub-style three-pane — top status/stats bar, left section menu
(tabs), main settings panel. Activity log docked at the bottom. (Prototype
exists: `index.html`.)

---

## Open items

- **Embedding settings wrinkle** — RESOLVED, see below.
- Which keys are server-locked vs freely editable in the final list.

## Port / network change: desired_port + startup rectify (implemented)

Network settings (`port`, `bind`, `socket_enable`, TLS cert/key) can't hot-apply
— httplib listeners bind once at startup and SIGHUP reload explicitly refuses
them. `port` uses the SAME current-vs-desired split as embeddings:

- **`port`** = the committed/current port the daemon binds. **Locked** (read-only)
  in the UI. Persisted.
- **`desired_port`** = the editable target the dashboard writes. Persisted.
  0/blank = "same as port" (seeded from `port` on load).
- **needs_restart** = bound port != desired_port.

**Startup rectify (the key behavior).** In the `serve` path, unless `--port` was
given on the CLI, the daemon adopts `desired_port` into `port` before binding:
if they differ it sets `port := desired_port`, persists it, and binds it. This
makes ANY restart path — dashboard button, `systemctl`/launchd, crash respawn,
or a bare `ragger serve` — pick up a UI port change, not just the dashboard's
Restart button. Because `port` is Locked, the rectify write uses the internal
`allow_locked=true` overload of `set_config_persisted`/`apply_config_value`.

**CLI `--port X` is a transient override.** It binds X and the live config's
`port` is set to X in memory (so the read-only "TCP Port" field is honest), but
`desired_port` is left untouched and nothing is persisted — so the next plain
restart falls back to `desired_port`. `/restart/status` then shows
current=X (bound) / desired=`desired_port`, needs_restart lit.

Verified scenarios:
- Set desired_port in UI, restart via bare `ragger serve` → rectify adopts it. ✅
- Launch `--port 19002` with DB desired=19001 → shows 19002/19001, no rectify. ✅
- Plain restart after that override → goes back to DB desired (19001). ✅

**Routes:** `GET /restart/status` (bound vs desired + needs_restart + scheme),
`POST /restart` (returns `reconnect_url` on the desired port, then flags a
restart).

**Restart mechanism = self re-exec.** The daemon captures full argv at startup
(`set_full_argv`); on restart the run loop stops the listeners, `run()` returns
true, and `main()` `execv`s its own image — same PID, works whether launched by
hand or a service manager, can't brick. The replayed argv is filtered to strip
`--host`/`--port` so the new image reads the (rectified) port from the DB.

**Dashboard reconnect.** The Server tab has a restart banner (bound-vs-desired +
"Restart now"). On click it POSTs `/restart`, then polls the new listener's
`/health` — cross-origin, so a `no-cors` fetch is used (resolve = up, throw =
down) — and redirects the browser to `reconnect_url` once it answers.

**Pitfall fixed:** httplib's `Server::stop()` asserts if called when
`is_running_` is true but the socket is already invalidated. The listener stop
must happen once (run loop), and the post-listen cleanup must only `join()` the
TCP thread, never `stop()` again.



## Legacy settings.ini → DB migration (implemented)

On `serve`, after the DB exists, `migrate_ini_to_db(db_path)` runs once:

1. If the DB `settings` table has an `ini_migrated` marker, no-op.
2. Otherwise, if `settings.ini` exists, parse it (`load_config`), and for every
   schema key whose INI value differs from the compiled default, write a DB row
   — **unless a row already exists** (existing DB values always win).
3. Rename `settings.ini` → `settings.ini.migrated`.
4. Set the DB `ini_migrated=true` marker.

The marker is essential because `bootstrap_user_config()` recreates a *default*
`settings.ini` on any later launch where the file is missing — "file exists"
can't gate the migration, but the DB marker can. `ini_migrated` lives OUTSIDE
the config schema, so `overlay_settings_from_db` ignores it.

After a migration with imports, `main` re-runs `overlay_settings_from_db` so the
freshly-imported values take effect in the same run (no extra restart needed).

Note: the shipped `DEFAULT_CONFIG` INI template carries real values that differ
from the C++ struct defaults (summarizer model/url/prompt, etc.), so a fresh
install still imports those into the DB on first serve — correct and intended.

### Pitfall fixed: DB overlay must apply Locked keys

Making `port` `CfgEdit::Locked` silently broke `overlay_settings_from_db`,
which routes through `apply_config_value` — that rejected Locked keys, so a
DB-persisted `port` never applied on startup (every plain restart fell back to
the INI/default port). Fix: the DB→live overlay and the startup rectify pass
`allow_locked=true`. Locked gates USER writes (CLI/dashboard) only, never the
authoritative DB→live overlay.


Embedding is the one setting whose change can't hot-apply — the stored vectors
must be re-encoded. So it gets a two-slot model instead of a single editable
value.

**Two namespaces in the `settings` table:**

| Concept | Keys | Written by | Editable |
|---|---|---|---|
| Current (what the vectors ARE) | `embedding_model`, `vector_type`, `dimensions` | drift guard + rebuild | no (read-only display) |
| Desired (the target) | `desired_embedding_model`, `desired_embedding_vector_type`, `desired_embedding_dimensions` | dashboard / CLI | yes |

The embedder always loads the CURRENT model, so vectors stay valid. Desired is
a staged wish until an explicit update.

**State (settings table):**
- `needs_update` = computed (`current != desired`).
- `reembedding` flag: `true` while an update runs, else `false`/absent.
- While `reembedding` is true, any embedding-requiring search short-circuits
  and returns ONE record: text = "Search not available. Re-embedding in
  progress."

**Update-now** (dashboard button / `ragger rebuild-embeddings`): set
`reembedding=true`, construct a desired-model Embedder, re-encode all rows on
the existing backend (`rebuild_embeddings(Embedder&)` — no second backend),
promote current := desired in settings, swap the daemon's live embedder, clear
`reembedding`.

**Choices:** model list = contents of `~/.ragger/models/` (via `GET /models`).
Bit-depth = f16 or f32 (only supported depths now). Vector length = model-
determined. Downloading new models comes later.

