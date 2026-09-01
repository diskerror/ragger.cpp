# Loose ends

Known dead code and half-finished mechanisms, found while debugging the stuck
re-embed on 2026-08-29. None of these are bugs today; they are traps for
whoever reads the code next and believes it. Recorded rather than fixed, so
each can be a deliberate decision.

Resolved in the same pass, for context: `max_search_limit` was deleted (it was
only ever enforced by `apply_user_overrides()`, which went away with the
two-layer system/user config), and `server_name` was documented as reserved.
`inject_data` is left alone — it is unrelated to `build_context` and will be
addressed separately. The `scripts/import/` scripts were switched off
settings.ini and onto the settings table in the DB (2026-08-30).

---

## 1. 34 unchecked `while (s.step())` loops in `src/sqlite_backend.cpp`

**Highest value item here.** `Stmt::step()` returns
`sqlite3_step(...) == SQLITE_ROW`, so `SQLITE_ERROR`, `SQLITE_BUSY` and
`SQLITE_ABORT` are indistinguishable from "no more rows". Every one of these
loops therefore ends *cleanly* when the read fails, and its caller reports
success on partial data.

This is not hypothetical. It hid a partial re-embed for three days: a rebuild
stopped at 1632 of 5654 documents and reported success, leaving two different
vector spaces in one table with no way to tell them apart. Fixed in
`embed_tables()` only.

`Stmt::step_checked()` (`include/util/sqlite.h`) already exists and throws on a
real error. Convert the loops where a short read is a correctness bug rather
than a cosmetic one — **export and import first**, where a truncated result
becomes a silently incomplete file on disk.

Not a blanket find-and-replace: each site needs a judgement about whether
throwing is the right response. A UI count that comes up short is not the same
as an export that loses rows.

## 2. `Config::resolve_model()` is a no-op

`src/config.cpp`. Returns its argument unchanged; the `[models]` aliasing it
used to wrap was removed, and the pass-through was kept in case aliasing came
back.

Every call site reads as though it canonicalises a model name. It does not.
That misreading cost real time during the 2026-08-29 debugging: the canonical
form is `provider/model`, and `resolve_model()` looks like the thing that
guarantees it.

Delete it, or rename it to something honest (`model_name_as_configured()`).
Deleting is a mechanical substitution of the argument at each call site.

## 3. ~~Three lang constants exist but the call sites hardcode the literal~~ — resolved 2026-09-01

All three call sites now use their `en.h` constants instead of the raw
literal: `HTTP_UNAUTHORIZED` (`src/server.cpp` 401 handler),
`HTTP_NO_TOKEN_FILE`, and `HTTP_SYSTEM_USER_NOT_FOUND` (both in the
`/user/token` handler). Folded into the #4 pass below since fixing one
required auditing the whole file for the same disease.

## 4. ~~~33 unreferenced lang constants~~ — resolved 2026-09-01

Audited every constant in `include/lang/en.h` against `src/` + `include/`
call sites using the regenerate-the-list script below. Two kinds of drift,
fixed differently:

- **Constant exists, call site hardcodes the literal instead** (this was
  item #3, and turned up more of the same pattern while auditing #4):
  wired 5 call sites to their constants — `HTTP_UNAUTHORIZED`,
  `HTTP_JSON_ERROR`, `HTTP_SYSTEM_USER_NOT_FOUND`, `HTTP_NO_TOKEN_FILE`
  (all `src/server.cpp`), and `MSG_SUMMARIZER_L3`
  (`src/summarizer_service.cpp`, replacing an ad-hoc `[summarizer] session
  boundary closed for {} ({} -> {})` log line) and `ERR_ROUTE_FAILED`
  (`src/server.cpp`'s catch-all handler, replacing an inline `"{} {}
  failed: {}"` format string).
- **Constant describes a feature that doesn't exist in the code** (login
  endpoint, session/turn-storage-specific error wrapping, a payload-dump
  directory, low-level socket client errors, a `--dump-payloads` CLI
  flag, a generic `MSG_WARNING`, an `ERR_LOG_OPEN` string already owned by
  c_lib's `Logger.cp` and never duplicated here): deleted 27 dead
  constants outright rather than wire them to nothing. This is more than
  the ~33 estimated in the original note because the estimate lumped in
  constants already deleted in the 2026-08-29 file-config removal (see
  the "Resolved in the same pass" note at the top of this file) — the
  actual live-but-dead count in this pass was 27.

Verified with the regenerate script (0 unreferenced afterward) plus a
full build + `ctest` run (15/15 passing).

Regenerate the list with:

```
python3 - <<'EOF'
import re, glob
names = re.findall(r'constexpr const char\* (\w+)', open('include/lang/en.h').read())
src = "".join(open(f).read() for f in
              glob.glob('src/**/*.cpp', recursive=True)
            + glob.glob('include/**/*.h', recursive=True)
            + glob.glob('tests/*.cpp') if 'lang/en.h' not in f)
print("\n".join(n for n in names if n not in src))
EOF
```

## 5. ~~`normalize_whitespace_and_cap_dashes()` is never called~~ — resolved 2026-09-01

Correction: it was declared in `include/import.h` but never defined anywhere,
not just unreferenced. The actual logic lives as two separate static helpers
in `src/import.cpp` — `normalize_whitespace()` and `cap_dashes()` — each
called exactly once, inline, from `clean_document_text()`. That's the right
shape for single-use helpers; there was no duplication to consolidate.
Deleted the dead declaration from `import.h`.

## 6. `recipe_cli`'s `default` sentinel points at a file that no longer exists

`src/recipe_cli.cpp:50` and `include/recipe_cli.h` describe the `default`
recipe as "track `[server] default_recipe` in settings.ini". Ragger stopped
reading or writing `settings.ini` on 2026-08-29 — defaults are compiled in and
the settings table is the only store — so that sentinel now tracks nothing.

Left alone on purpose: context-building recipes are unfinished and being
reworked. Revisit with that work, not before.
