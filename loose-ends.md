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

## 3. Three lang constants exist but the call sites hardcode the literal

Extracted into `include/lang/en.h`, never wired up — so translating them
changes nothing.

| Constant | Literal | Site |
|---|---|---|
| `HTTP_UNAUTHORIZED` | `"Unauthorized"` | `src/server.cpp:651` |
| `HTTP_NO_TOKEN_FILE` | `"no token file"` | `src/server.cpp:689` |
| `HTTP_SYSTEM_USER_NOT_FOUND` | `"system user not found"` | `src/server.cpp` |

## 4. ~33 unreferenced lang constants

Nothing reads them. Two groups worth separating:

- **Orphaned 2026-08-29** by removing the file-based config path — already
  deleted in the same commit, listed only so the count below reconciles:
  `ERR_CONFIG_OPEN`, `ERR_CONFIG_SYSTEM_NOT_FOUND`, `ERR_CONFIG_SYSTEM_LOAD`,
  `ERR_CONFIG_SYSTEM_PARSE`, `ERR_CONFIG_SYSTEM_UNKNOWN`, `MSG_CONFIG_CREATED`,
  `MSG_CONFIG_LOADED`.
- **Older, still present**: the whole `HTTP_*` block, all `ERR_CLIENT_*`, `ERR_LOG_OPEN`,
  `MSG_WARNING`, `CLI_DUMP_PAYLOADS`, `MSG_SUMMARIZER_L3`,
  `MSG_SUMMARIZED_SESSION`, and others. Check whether each names a feature
  that was removed or one that was never finished before deleting.

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
