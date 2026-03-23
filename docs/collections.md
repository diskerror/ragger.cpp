# Collections

Memories are organized into **collections** — logical groups that let you
separate reference material from conversation memories and search the
right pool for the right question.

## What Are Collections?

Collections are the top-level partition in Ragger's data model. Every
memory belongs to exactly one collection. When you search, you specify
which collections to include (or search all of them).

Think of collections as separate bookshelves:

- `memory` — Your personal notes, decisions, session summaries
- `docs` — Project documentation, API references
- `reference` — Technical manuals, specifications
- `notes` — Meeting notes, research, bookmarks

You pull from the right bookshelf based on what you're looking for.

## Built-in Collections

| Collection | Purpose |
|------------|---------|
| `memory` | Agent-stored memories: facts, decisions, preferences, session summaries (default) |

When you use `ragger store` without `--collection`, memories go into
the default collection (configurable via `default_collection` in the
config file, usually `memory`).

## Custom Collections

Any string can be a collection name. Create them by importing or storing
with `--collection`:

```bash
# Import reference docs into "docs" collection
ragger import api-guide.md --collection docs

# Store a note in "work" collection
ragger store "Deploy to staging every Friday" --collection work
```

Collections are created on first use — no setup needed.

### Example Custom Collections

| Collection | Purpose |
|------------|---------|
| `docs` | Project documentation, API references |
| `reference` | Technical manuals, specifications |
| `orchestration` | Orchestration guides, instrument ranges |
| `notes` | Meeting notes, research, bookmarks |
| `work` | Work-specific context, procedures |

## Collections vs Categories

**Collections** are the top-level partition — they control which pool of
documents gets searched. Every memory belongs to exactly one collection.

**Categories** are metadata *within* a collection — they describe what
kind of memory it is (fact, decision, preference, session-summary, etc.).
Categories are stored in `metadata.category` and are useful for filtering
or organizing, but don't affect which documents are included in search.

```
memory (collection)
├── fact (category)
├── decision (category)
├── preference (category)
├── lesson (category)
└── session-summary (category)

docs (collection) — imported reference material
orchestration (collection) — imported reference material
```

Use collections to partition your data. Use categories to label what's
inside each partition.

## Searching Collections

Specify which collections to search via the `--collections` flag:

```bash
# Search all collections (default)
ragger search "API authentication"

# Search specific collections
ragger search "API authentication" --collections docs reference

# Search only one collection
ragger search "my notes about deployment" --collections memory
```

**In the HTTP API:**

```json
{
  "query": "API authentication",
  "collections": ["docs", "reference"]
}
```

Omit `collections` to search all collections. Pass `["*"]` for explicit
"search everything."

## Tagging at Import

Use `--collection` when importing to organize reference material:

```bash
# Import a guide into "docs"
ragger import api-guide.md --collection docs

# Import meeting notes into "notes"
ragger import weekly-standup-2024-03-20.md --collection notes

# Import without --collection → uses default (usually "memory")
ragger import meeting-notes.md
```

Imported files are split into paragraph-sized chunks (unless you disable
chunking). All chunks from the same file get the same collection tag.

## Working with AI Agents

The real power of collections shows up when an AI agent uses Ragger as
its memory backend. The agent learns what you're working on from context
and searches the right collections automatically:

- Working on your API? The agent includes `docs` in its search.
- Asking about deployment procedures? It searches `reference`.
- General conversation? Just `memory` — no noise from reference docs.

You don't need to tell the agent which collection to use. It figures it
out the same way a good assistant would — by paying attention.

### How Agents Pick Collections

Agents infer collection choice from:

1. **Project context** — What project are you working on? What files are
   open in your editor?
2. **Query semantics** — "How do I deploy this?" → `docs`, `reference`.
   "What did we decide last week?" → `memory`.
3. **Recent searches** — If you just searched `docs`, the agent assumes
   you're still in that context.

The agent can also ask if it's uncertain: "Should I search your reference
docs or just your notes?"

### Setting Up Collections for Your Workflow

Work with your AI to define collections that match your workflow:

1. **Import your reference materials** with descriptive collection names:
   ```bash
   ragger import orchestration-guide.md --collection orchestration
   ragger import python-api-ref.md --collection docs
   ```

2. **Let the agent know what's available.** Tell it once:
   > "I have three collections: `memory` for my notes, `docs` for API
   > references, and `orchestration` for music notation guides."

3. **The agent learns over time.** After a few searches, it knows which
   collections to use for which questions.

This is a collaborative process. Your AI gets better at knowing when to
reach for reference material versus conversation history.

## Best Practices

### Keep Collections Focused

Don't dump everything into one collection. Narrow collections produce
better search results because the semantic space is more coherent.

**Good:**
- `docs` — API references, project documentation
- `notes` — Meeting notes, research
- `memory` — Personal facts, decisions, preferences

**Less good:**
- `stuff` — Everything mixed together

### Use Descriptive Names

`orchestration` is better than `music` (too vague) or `instrument-ranges`
(too specific — what if you add notation rules?).

### Separate Conversation from Reference

Store conversation turns in `memory` (or a dedicated `conversation`
collection) and imported docs in `docs`, `reference`, etc. This lets you:

- Search your notes without noise from reference material
- Search reference material without your ephemeral chat history
- Tune retention/expiration separately (e.g., delete old turns but keep
  imported docs forever)

### Don't Over-organize

Resist the urge to create dozens of micro-collections. Start with 3-5
broad categories. You can always split later if search results feel
muddled.

## Exporting Collections

Export all documents in a collection to disk:

```bash
# Export "docs" collection
ragger export docs ./exported/

# Export "memory" collection
ragger export memories ./exported/

# Export all collections
ragger export all ./exported/
```

Each document is saved as a separate file with its metadata in a JSON
sidecar.

## Related

- [Search & RAG](search-and-rag.md) — How hybrid search works
- [Getting Started](getting-started.md) — Importing your first docs
- [Configuration](configuration.md) — Setting `default_collection`
