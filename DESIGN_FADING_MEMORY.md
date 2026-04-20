# Design: Full Chat Manager with Fading Memory

**Date:** 2026-04-13
**Status:** Accepted — ready for implementation

---

## The Idea

Ragger started as document storage for RAG. But conversations have
structure that documents don't — they progress, they build context,
they fade. Most AI memory systems fight forgetting. Ragger should
embrace it.

**The context window should behave like human working memory.** Recent
exchanges are vivid and verbatim. Older exchanges compress into gist.
Old conversations fade entirely — unless you need them, and then the
database is there to look them up.

The inference engine is stateless. Every request is a fresh payload.
Ragger is the one that decides what goes in that payload. That makes
Ragger the natural place to implement memory that fades.

**There are no sessions from the user's perspective.** There's just
a conversation. It might span hours or days. The user never picks a
thread, scrolls back through a session list, or wonders "which
conversation had that thing I need." The computer manages topic
boundaries, summarization, and context assembly. The user just talks.
Every other chat system — ChatGPT, Claude.ai, OpenClaw — makes the
user manage sessions manually. Ragger eliminates that entirely.

It should feel like talking to a human assistant — one who fades on
the details of last week's conversation just like you do, but unlike
a human, can instantly pull up the exact words when needed. The
fading is natural. The recall is superhuman.

---

## Ragger's Three Roles

Ragger serves three distinct purposes. The boundary between them is
the key architectural decision.

### Role 1: Ragger Memory (Storage API)

Three operations: store, search, count. The external agent builds its
own turn payloads, manages its own context window, and decides when
to search. Ragger is a filing cabinet.

This role stays as-is. It's useful and complete. But it's the least
interesting mode — the agent has to do all the memory work itself.

### Role 2: Ragger Chat (Full Chat Manager)

When a client connects via `POST /chat` or `ragger chat`, Ragger
assembles the **entire** inference payload:

1. Persona files (SOUL.md, USER.md, AGENTS.md, TOOLS.md)
2. Memory search results (RAG context relevant to current query)
3. Conversation history (tiered by recency, with fading)
4. The new user message

Ragger sends this to the inference engine, streams the response back,
stores the turn, and manages summarization. The client — whether it's
the web UI, a mobile app, or a terminal — is just a viewport.

**This is the recommended way to use Ragger** for direct conversation.

### Role 3: Ragger Proxy (Augmented Pass-Through)

When an agent framework like OpenClaw needs both its own tool/skill
orchestration **and** Ragger's memory + persona management, Ragger
acts as an inference proxy. The agent points its inference URL at
Ragger instead of the LLM provider directly.

See **Proxy Mode** section below for full design.

### The Responsibility Split

Ragger and agent frameworks like OpenClaw have a clean division:

| Concern                         | Ragger | OpenClaw |
|---------------------------------|--------|----------|
| Persona (SOUL.md, USER.md)      | ✓      | —        |
| Memory (store, search, fading)  | ✓      | —        |
| Conversation history management | ✓      | —        |
| Context window budgeting        | ✓      | —        |
| Summarization                   | ✓      | —        |
| Tool execution                  | —      | ✓        |
| Skill management                | —      | ✓        |
| Agent orchestration             | —      | ✓        |
| Function calling protocol       | —      | ✓        |
| Inference routing               | ✓      | —        |

Ragger owns **who the assistant is** (persona) and **what it
remembers** (memory). OpenClaw owns **what it can do** (tools)
and **how it coordinates** (orchestration). Neither needs to know
the other's internals.

### Same Assistant, Different Capabilities

From the user's perspective, Roles 2 and 3 are the same assistant
with the same memory. The difference is what it can do:

- **Role 2 (Ragger Chat — web UI, CLI):** The assistant knows who
  it is, remembers past conversations, and talks to you. No tools.
- **Role 3 (Proxy — e.g. Telegram via OpenClaw):** The same
  assistant, same persona, same memory — but now it can also search
  the web, execute code, manage files, or whatever tools OC provides.

The user can chat via the web UI in the morning, switch to Telegram
with OpenClaw in the afternoon, and the assistant remembers both
conversations. The only difference is that the Telegram session can
*do things*. The memory is shared. The persona is shared. The
identity is continuous.

---

## Storage Hierarchy

Ragger stores six distinct types of information. They differ in who
can write them, how long they're retained, and how they enter the
inference payload.

### Level 1: Raw Turns

Verbatim user/assistant exchanges. Every word, as it happened.

- **Written by:** Ragger (automatic)
- **Retention:** Configurable. A solo user might keep a day. A
  business might keep forever (compliance). Config decides.
- **Payload role:** Tier 1 — recent raw turns enter the payload
  verbatim within the configured time window.
- **Analogy:** A recording of the conversation.

### Level 2: Per-Turn Summaries

Each turn is individually summarized by the local memory model,
immediately or near-immediately after the exchange.

- **Written by:** Ragger via local memory model (automatic)
- **Retention:** Configurable, typically longer than raw turns.
  Even after raw turns are deleted, per-turn summaries persist as
  granular, independently searchable records.
- **Payload role:** Available to RAG search. If a query matches a
  specific past turn, the per-turn summary surfaces.
- **Analogy:** Margin notes on the recording transcript.

### Level 3: Session Summaries (Incremental)

A running summary of the current conversation, updated after each
turn. The local memory model receives the previous session summary +
the new per-turn summary and produces an updated session summary.

- **Written by:** Ragger via local memory model (automatic, rolling)
- **Retention:** Configurable, typically longer than per-turn summaries.
- **Payload role:** Tier 2 — recent session summaries enter the
  payload within the configured time window.
- **Topic-shift detection:** If a new turn doesn't fit the running
  summary, the model starts a new session summary. The inability to
  coherently merge IS the topic-shift signal — no explicit classifier
  needed. The old summary is marked complete.
- **Timing:** Updated after each turn (input is two short summaries,
  so this is fast and cheap). Also updated on pause and quit.
- **Analogy:** "What are we talking about right now?"

### Level 4: Project/Subject Summaries

Summary of summaries. When multiple session summaries cluster around
the same topic, they get distilled into a higher-level summary.

- **Written by:** Ragger via local memory model (scheduled)
- **Retention:** Long-lived. These are the durable "what happened
  over the past week/month" records.
- **Payload role:** Available to RAG search. Provides broad context
  when a query relates to a topic discussed across multiple sessions.
- **Schedule:** Runs periodically (configurable, e.g. daily or
  every N hours). Scans recent completed session summaries, groups
  by embedding similarity, and produces or updates project summaries.
- **Incremental:** Like Level 3, project summaries are running
  summaries — each scheduled pass merges new session summaries into
  existing project summaries or creates new ones. Can also reach
  backwards to reorganize: if a session summary from last week
  fits better under a different project, the model can reassign it.
- **Analogy:** "What have we been working on this month?"

### Level 5: RAG Entries (User-Controlled Documents)

Imported documents, reference material, curated knowledge.

- **Written by:** User only. The agent CANNOT create, edit, or
  delete RAG entries. This is the user's curated knowledge base.
- **Retention:** Permanent until the user removes them.
- **Payload role:** Available to RAG search. Relevance-gated, not
  recency-gated — a document imported months ago surfaces if it
  matches the current query.
- **Why user-only:** RAG entries are authoritative reference material.
  If the agent could modify them, the user couldn't trust the
  contents. The agent can *read* them (via search) but never *write*.
- **Analogy:** Your reference library. You shelve the books; the
  assistant reads them.

### Level 6: Decisions

Explicit records of choices, conclusions, and rules that emerged
from conversations.

- **Written by:** Agent or user. The agent can create decisions
  based on conversation content. The user can create them manually.
- **Retention:** Nearly permanent. Decisions persist until superseded.
- **Mutability:** The agent CANNOT delete or edit a decision's
  content. Instead, the agent can **mark** a decision with a status:
  `current` (default), `superseded`, `revisit`, `deprecated`. The
  original text is always preserved. When a decision is superseded,
  the agent creates a new decision and marks the old one.
- **Payload role:** Active decisions (`current` status) are available
  to RAG search. Superseded/deprecated decisions are excluded from
  default search but remain in the database and can be found with
  explicit queries.
- **Why not deletable:** Decisions are history. "We chose SQLite
  over Postgres" is useful even after you later switch to Postgres —
  it explains the codebase's evolution. Marking as superseded
  preserves the record while keeping it out of the active context.
- **Analogy:** Meeting minutes. You can note that a decision was
  reversed, but you don't tear the page out.

### Summary Table

| Level | Type               | Written by               | Retention               | Agent can modify?         | Payload role             |
|-------|--------------------|--------------------------|-------------------------|---------------------------|--------------------------|
| 1     | Raw turns          | Ragger (auto)            | Config (1 day–forever)  | —                         | Verbatim window          |
| 2     | Per-turn summaries | Memory model (auto)      | Config (longer than L1) | —                         | RAG search               |
| 3     | Session summaries  | Memory model (rolling)   | Config (longer than L2) | —                         | Recent summary window    |
| 4     | Project summaries  | Memory model (scheduled) | Long-lived              | —                         | RAG search               |
| 5     | RAG entries        | User only                | Permanent               | No (read-only)            | RAG search               |
| 6     | Decisions          | Agent or user            | Permanent               | Status only (not content) | RAG search (active only) |

### When Summarization Happens

| Event                      | What runs                                             | Storage levels affected |
|----------------------------|-------------------------------------------------------|-------------------------|
| After each turn            | Per-turn summary (L2) + session summary update (L3)   | 2, 3                    |
| Topic shift detected       | New session summary started, old one completed        | 3                       |
| Session pause (N min idle) | Final session summary update                          | 3                       |
| Session quit               | Final session summary update                          | 3                       |
| Scheduled (e.g. daily)     | Project summary scan + merge                          | 4                       |
| Turn ages past retention   | Raw turn eligible for deletion (per retention config) | 1                       |

---

## Payload Assembly

**Storage and payload are independent systems.** Storage defines what
exists in the database and how long it's kept. Payload assembly
defines what gets sent to the inference engine on each turn. A
storage level can exist without ever appearing in a payload, and
the payload rules can change without affecting what's stored.

### Default Payload Contents

For a normal conversational turn, the payload contains:

```
1. Persona files (SOUL.md, USER.md, AGENTS.md, TOOLS.md)
2. Current running session summary (Level 3 — the active one)
3. Current project summary (Level 4 — if one matches the topic)
4. Active decisions matching the query (Level 6)
5. RAG entries matching the query (Level 5)
6. Per-turn summaries matching the query (Level 2)
7. Recent per-turn summaries (Level 2 — last 20 count or 30 min)
8. Previous raw turn (Level 1 — just the last exchange, verbatim)
9. New user message
```

Note: only **one** raw turn is included by default — the immediately
preceding exchange. The user just said something and the assistant
just replied; that's the live context. Everything older is already
captured as per-turn summaries (Level 2) and folded into the running
session summary (Level 3).

This is lean by design. It works on an 8K local model. Users with
larger context windows widen the windows via config.

### The Fading Effect

As a conversation ages, its representation in the payload compresses:

```
Just now     → Raw turn (verbatim, Level 1)
5 min ago    → Per-turn summary (one sentence, Level 2)
1 hour ago   → Part of session summary (one paragraph, Level 3)
Last week    → Part of project summary (a few sentences, Level 4)
Any time     → RAG hit or decision (only if query-relevant, L5/L6)
```

The database keeps everything. The payload shows a natural gradient
from vivid to compressed to absent. The full detail is always one
search away.

### Subject Switching

When the user changes topic ("on the subject of X"), the payload
needs to pivot — drop old topic context, pull in X's context.

**Phase 1a: Explicit command.** The user prompts "new subject" (or
"new topic", "change subject" — a small literal match set). Ragger
handles this directly:

1. Finalize and complete the current session summary (Level 3)
2. Clear the session's verbatim turns and per-turn summaries
   from the payload
3. Return a clean payload: persona + a handful of the most recent
   project summaries (Level 4)

The agent sees a fresh context with broad orientation — recent
project summaries remind it what topics are active. The user's
next message provides the specific direction. No extra round trip,
no throwaway payload.

This is a command, not interception. Literal string matching. Easy.

**Phase 1b: Agent-driven.** For less explicit switches ("on the
subject of X"), the conversation agent detects the subject change
and makes a fresh request to Ragger with X as the query. Ragger
returns X's support payload (project summary, decisions, RAG hits).
The agent discards most of the previous payload and builds a new one.

This costs an extra round trip and one throwaway payload, but it
works with no changes to Ragger's payload assembly logic. The agent
is the one that understands conversational intent.

**Phase 2 (future optimization): Ragger-intercepted.** Ragger
detects subject-switch patterns in the prompt (explicit cues like
"on the subject of," "let's talk about," "switching to") and
handles the pivot internally:

1. Finalize current session summary (Level 3) for the old topic
2. Start new session summary for topic X
3. Pivot all searches to X
4. Pull in X's project summary (Level 4) if one exists
5. Drop old topic's verbatim turns and per-turn summaries
6. Return the pivoted payload directly — no extra round trip

This eliminates the throwaway payload and the extra request. But
it requires Ragger to understand conversational intent, which is
harder to get right and should only be attempted once the payload
defaults are tuned from real usage.

**The session summary system already handles implicit topic shifts.**
If the memory model can't coherently merge a new turn into the
running session summary, it starts a new one. This is the organic
signal. Explicit interception is the optimization layer on top.

### The Shrinking Algorithm

The context window is fixed. The payload must fit. When it doesn't,
content is shed in a defined priority order — **lowest priority
cut first:**

**Priority order (highest = cut last):**

| Priority    | Content                          | Why it's protected                 |
|-------------|----------------------------------|------------------------------------|
| 1 (highest) | New user message                 | Without this, nothing works        |
| 2           | Persona files                    | Identity — inconsistent without it |
| 3           | Previous raw turn                | Immediate context for the reply    |
| 4           | Running session summary          | "What are we talking about?"       |
| 5           | Active decisions (query-matched) | Established facts                  |
| 6           | RAG entries (query-matched)      | Reference material                 |
| 7           | Recent per-turn summaries        | Recent conversation gist           |
| 8           | Project summary                  | Broad context                      |
| 9 (lowest)  | Older per-turn summaries         | Least urgent                       |

**Shedding rules:**

- Cut from priority 9 upward. Within a priority level, cut the
  oldest or lowest-relevance items first.
- Persona files (priority 2) use paragraph-aware truncation — keep
  complete paragraphs from the highest-priority file (SOUL.md)
  before moving to lower-priority files. Already implemented in
  `chat.cpp`.
- Never cut the new user message or the previous raw turn — if
  even those don't fit, the context window is too small to be useful.
  Return an error.

**How it works in practice:**

| Context Size     | What fits                                                                          |
|------------------|------------------------------------------------------------------------------------|
| 4K (tiny)        | Persona (truncated) + prev turn + new message. That's it.                          |
| 8K (small local) | Persona + session summary + prev turn + 2–3 search hits                            |
| 32K (typical)    | Everything at default settings                                                     |
| 128K+ (large)    | Everything with widened windows (more raw turns, more summaries, more search hits) |

### Widening for Large Contexts

Users with large context windows can include more raw history:

```ini
[chat]
verbatim_turns = 1          # default: just the previous turn
verbatim_minutes = 0        # default: no time-based window

# User with 128K context overrides:
verbatim_turns = 20
verbatim_minutes = 30

# User with 1M context:
verbatim_turns = 100
verbatim_minutes = 120
```

The defaults are conservative (one raw turn). Users explicitly
widen based on their budget. The shrinking algorithm still applies
if the total exceeds the context window — widened settings are
a request, not a guarantee.

### Context Window Detection

Context size varies wildly — 4K for tiny local models, 1M for Claude.
There is no universal API to query it.

**Strategy (in priority order):**

1. **Config is authoritative.** `max_context` per inference endpoint.
   The user sets it, Ragger trusts it. This always works.
2. **Auto-detect when possible.** On startup, probe provider-specific
   endpoints (Ollama `/api/show`, llama.cpp `/props`, vLLM `/v1/models`).
   If detected, populate config and log it.
3. **Known-model table as fallback.** Ship a lookup of common models
   and their context sizes. Stale eventually, but better than unknown.
4. **Unknown = conservative.** If context size is 0 (unknown), use
   conservative defaults and let the user override.

---

## Composable Payload Builder

The inference engine is stateless. Ragger owns the entire payload.
That means the payload isn't a fixed template — it's a **composition
of named sources**, assembled per turn by rules the user (and the
agent) can influence.

Every piece of content that can enter a payload is a **source**.
Sources are enumerated, typed, independently retrievable, and
independently budgetable. `build_messages()` is not a hard-coded
sequence of `load_persona(); load_summary(); load_turns();` — it's
a pipeline that asks each configured source for its contribution,
budgets the result, and emits the final message array.

### Source Catalog

| Source                   | Level | Output shape                       | Gating                    |
|--------------------------|-------|------------------------------------|---------------------------|
| `persona`                | —     | Concatenated workspace files       | Always on                 |
| `dynamic_system`         | —     | Computed string (time, state, …)   | Hook-provided             |
| `raw_turns`              | L1    | `[{role:user}, {role:assistant}]…` | `verbatim_turns/minutes`  |
| `turn_summaries_recent`  | L2    | Bulleted list, newest first        | `summary_count/minutes`   |
| `turn_summaries_search`  | L2    | Relevance-ranked list              | Query, score threshold    |
| `session_summary`        | L3    | Single paragraph                   | Current running summary   |
| `project_summary`        | L4    | Single paragraph                   | Topic match to query      |
| `rag`                    | L5    | Relevance-ranked list              | Query, score threshold    |
| `decisions`              | L6    | Bulleted list                      | `current` status + query  |
| `tools`                  | —     | OpenAI/Anthropic tools array       | Supplied by caller (R3)   |
| `new_user_message`       | —     | `{role:user, content}`             | Always last               |

A source is a function: `(context) → { content, token_cost, priority }`.
The context bundles the current query, session id, user id, config,
and an embedder/search handle. Sources are pure w.r.t. the database
(no writes), so they parallelise freely.

### Dynamic System Messages

`dynamic_system` is a first-class source, not a special case of
persona. It produces text that depends on *runtime state* — things
that are true right now and may be false next turn:

- Current local time and timezone (so the model stops guessing)
- Active project / working directory / git branch (if IDE-integrated)
- Active tool availability (e.g. "network is offline")
- User mood / status hints ("deep-work mode — be terse")
- Scheduled reminders firing in this turn
- Mode flags ("subject just switched — treat prior context as stale")

Dynamic content goes in its own segment of `messages[0]`, clearly
delimited, so a future turn's version can replace it without
disturbing persona or memory blocks.

### Recipes

A **recipe** is a named, ordered list of sources with per-source
budget caps. Ragger ships defaults; users override in config; agents
can request a specific recipe per request via `X-Ragger-Recipe` or
a body field.

```ini
[recipe.default]
# The standard conversational payload (Role 2)
sources = persona, dynamic_system, session_summary, project_summary,
          decisions, rag, turn_summaries_search,
          turn_summaries_recent, raw_turns, new_user_message

[recipe.lean]
# For tiny (4K) models: drop everything but the essentials
sources = persona, session_summary, raw_turns, new_user_message

[recipe.research]
# Favor RAG breadth over recent chat continuity
sources = persona, rag, decisions, project_summary,
          new_user_message
rag.max_results = 20
turn_summaries_recent.count = 0

[recipe.fresh_subject]
# After "new subject" command: drop session context
sources = persona, dynamic_system, project_summary,
          new_user_message
```

A recipe is just a declarative config over the same underlying
sources. No new machinery — it's how the existing priorities,
budgets, and windows get packaged.

### Builder API (internal)

```cpp
// include/ragger/payload_builder.h
class PayloadSource {
public:
    virtual ~PayloadSource() = default;
    virtual std::string name() const = 0;
    virtual int priority() const = 0;            // shedding order
    virtual std::vector<Message> produce(const BuildContext&) = 0;
    virtual int estimate_tokens(const BuildContext&) = 0;
};

class PayloadBuilder {
public:
    void add(std::unique_ptr<PayloadSource>);
    BuildResult build(const BuildContext&);      // applies budget + priority
};
```

`ChatSession::build_messages()` becomes a thin wrapper:

1. Pick the recipe (default, per-user override, or per-request)
2. Instantiate the sources the recipe names
3. Run them (parallel where safe)
4. Hand results to `PayloadBuilder` which applies the shrinking
   algorithm (see "The Shrinking Algorithm" above)
5. Emit the final `messages` array

Every current payload item becomes a `PayloadSource` subclass.
Adding a new memory type in the future (e.g. "calendar events",
"open file buffer") is a new source class and a recipe entry —
no changes to the builder.

### Exposing It to Agents

Agents that want full control (Role 3, OpenClaw, MCP clients) can:

1. **Pick a recipe** via request header/field
2. **Override per-source params** via request body (e.g. raise
   `rag.max_results`, narrow `verbatim_minutes`)
3. **Inject an extra `dynamic_system` block** via a request field
   (e.g. OC injects current tool state)
4. **Skip sources** entirely (e.g. `skip=persona` for a diagnostic
   probe)

The HTTP contract for this is TBD — likely `X-Ragger-Recipe` +
a `ragger_overrides` body field — but the internal builder is
recipe-agnostic from day one.

### Implementation Note

This is **not a new phase** — it's a refactor of how Phase 3
(Tiered Payload Assembly) is implemented. Instead of hard-coding
the assembly sequence in `build_messages()`, build it as a source
pipeline from the start. Same behavior, same budget, same priority
rules — just composable.

---

## The Router: A Small Local Model That Decides

A composable builder is only as good as the thing choosing the
composition. Deterministic rules ("include last 20 turns") get you
80% of the way. The remaining 20% — *which* recipe, *what* query to
run against RAG, *is this a subject switch*, *does the session
summary still cover what's being said* — needs judgment.

That judgment comes from a **router model**: a small local LLM whose
job is not to talk to the user but to make structural decisions
about the payload before it's sent.

### What the Router Decides

| Decision                           | Input                                   | Output                                 |
|------------------------------------|-----------------------------------------|----------------------------------------|
| Recipe selection                   | New user message + running summary      | Recipe name (`default`/`research`/…)   |
| RAG query rewrite                  | New user message + last turn            | Search query string (may differ from user wording) |
| Subject-switch detection           | New user message + running summary      | `continue` / `switch` / `branch`       |
| Summary-merge feasibility          | Running summary + new per-turn summary  | `merge_ok` / `start_new`               |
| Per-source budget nudges           | Estimated total + content profile       | Which sources to widen/narrow          |
| Capture-worthiness                 | Completed turn                          | `store` / `skip`                       |

All of these are short-input, short-output, structural calls. None
of them require reasoning quality. A 3–8B model running locally is
overqualified.

### Same Model as the Memory Model

This is the same model already specified under `[inference.memory]`.
Summarization and routing are the same skill: read a small amount
of text, return a small structured answer. One endpoint, one model,
two uses.

The task table in **Model Separation** expands:

| Task                            | Model                      |
|---------------------------------|----------------------------|
| Conversation with user          | `[inference] model`        |
| Turn/session/project summaries  | `[inference.memory] model` |
| Auto-capture classification     | `[inference.memory] model` |
| **Recipe selection**            | `[inference.memory] model` |
| **RAG query rewrite**           | `[inference.memory] model` |
| **Subject-switch detection**    | `[inference.memory] model` |
| **Summary-merge feasibility**   | `[inference.memory] model` |

### External vs. Embedded

The router must be **local** — it runs before every turn, so cloud
latency would ruin the experience, and routing a user's raw prompt
to an external service for *planning* duplicates the privacy surface
we're trying to contain. Two ways to get there:

**Option A — External local engine (default).** Point
`[inference.memory]` at an already-running Ollama or LM Studio. User
picks the model, swaps it freely, shares GPU with other tools. Zero
added binary weight. Requires the user to have a local engine
installed — currently a safe assumption for our audience, but a
friction point for the GUI-onboarding path (see GUI priority memo).

**Option B — Embedded engine (optional, shipped).** Compile
`llama.cpp` (or a minimal GGUF runtime) into the Ragger binary and
ship a small default model (e.g. Qwen2.5-3B-Instruct, ~2 GB GGUF) as
an optional download on first run. The router Just Works out of the
box. Cost: +5–20 MB binary for the runtime, one more thing to keep
building against, and a model-download step during onboarding.

**Recommendation:** Ship A as the default path and add B specifically
for the zero-config onboarding flow — the non-technical user who
installs Ragger and expects it to work without also installing
Ollama. The config surface stays the same (`[inference.memory]`);
the embedded engine is just another endpoint backend, selected by
URL scheme (e.g. `embedded://qwen2.5-3b`). Existing code paths don't
change.

### Fallback Behavior

If no router is configured and no embedded model is available,
recipes fall back to **`default`** unconditionally and deterministic
rules handle everything the router would have. The system still
works — it just loses the 20% of polish the router provides.
Explicit user commands ("new subject") still work because they don't
need the router.

### Open Questions (Router)

- Does the router run *before* the conversation model (blocking the
  turn) or *in parallel* (eating its first few hundred ms by
  speculatively building the default recipe, then swapping if the
  router disagrees)? Parallel is faster but wastes compute on wrong
  guesses.
- Can router decisions be cached per session? Recipe selection
  probably shifts less than once per 10 turns; caching could avoid
  90% of calls.
- Structured output: JSON mode / grammar-constrained decoding to
  guarantee the router returns a valid recipe name, not prose.

---

## Model Separation: Conversation vs. Memory Management

A critical design principle: **the model that talks to the user is
not the model that manages memory.**

Conversation requires the best available model — reasoning, nuance,
creativity. That's Claude, GPT-4, or a large local model. It costs
real money per token (cloud) or real GPU time (local).

Memory management is mechanical work:

- Summarize these 10 turns into key facts
- Classify whether this exchange is worth storing
- Decide which memories are relevant to this query

A small local model handles all of this. A 3–8B parameter model
running on Ollama costs nothing per call, runs on modest hardware,
and produces perfectly adequate summaries.

### Why This Matters

1. **Cost.** Summarizing 50 turns with Claude costs tokens. Doing it
   with a local Qwen 7B costs electricity. Over thousands of sessions,
   this is the difference between viable and expensive.

2. **Latency.** Background summarization shouldn't compete with the
   conversation for inference capacity. A separate local model runs
   independently.

3. **Privacy.** Memory management can stay entirely local even when
   the conversation model is cloud-hosted. Summaries, capture
   decisions, and memory classification never leave the machine.

4. **User control.** The user picks the conversation model based on
   quality needs. They pick the memory model based on cost tolerance.
   Bigger history window + cheap summarizer = more context for less
   money.

### Configuration

```ini
[inference]
model = claude-sonnet-4-5           # conversation model (best available)

[inference.memory]
api_url = http://localhost:11434/v1  # local Ollama
model = qwen2.5:7b                  # cheap, fast, adequate for summaries
```

If `[inference.memory]` is not configured, Ragger falls back to the
main conversation model. This keeps single-endpoint setups simple
while allowing cost-conscious users to split the workload.

### What Uses the Memory Model

| Task                            | Model                      | Why                         |
|---------------------------------|----------------------------|-----------------------------|
| Conversation with user          | `[inference] model`        | Quality matters             |
| Turn summarization              | `[inference.memory] model` | Compression, not creativity |
| Session summarization           | `[inference.memory] model` | Same                        |
| Auto-capture classification     | `[inference.memory] model` | Binary decision             |
| Summary splitting (multi-topic) | `[inference.memory] model` | Structural, not creative    |

### The Cost/Accuracy Trade-Off

The user controls this trade-off through two independent knobs:

1. **History window size** (tier1_minutes, tier2_hours) — How much
   context enters the payload. Wider windows = more tokens = higher
   cost with cloud models, but better continuity.

2. **Summarization model** — How aggressively old turns are compressed.
   A better summarizer preserves more nuance in Tier 2. A worse but
   cheaper one loses some detail but costs nothing.

These are independent. A user on a budget might use:

- Claude for conversation (quality)
- Tiny local model for summarization (free)
- Narrow Tier 1 window (fewer cloud tokens per turn)
- Wide Tier 2 window (summaries are small, more of them is fine)

A user with a large context window and budget might use:

- Claude 1M context for everything
- Wide Tier 1 window (verbatim history is cheap when context is huge)
- Tier 2 almost unnecessary (raw turns fit)

The system adapts to both.

---

## Configuration

New config keys:

```ini
[memory]
# Storage retention (how long data is kept in the database)
raw_turn_retention = 24h        # Level 1 (e.g. 1h, 24h, 7d, forever)
turn_summary_retention = 7d     # Level 2
session_summary_retention = 30d # Level 3
project_summary_interval = 24h  # Level 4: how often to run the summarizer
project_summary_lookback = 7d   # Level 4: how far back to scan

[chat]
# Payload: raw turn inclusion (Level 1)
verbatim_turns = 1              # number of recent raw turns to include (default: 1)
verbatim_minutes = 0            # time-based window, 0 = use count only

# Payload: per-turn summary inclusion (Level 2)
summary_count = 20              # recent per-turn summaries to include
summary_minutes = 30            # time-based window (whichever is smaller)

# Payload: search results (Levels 2, 4, 5, 6)
search_max_results = 5          # RAG + decision + project summary hits
search_min_score = 0.3          # relevance threshold

# Payload: context budget (percentage of context window)
budget_persona_pct = 20         # SOUL.md, USER.md, workspace files
budget_session_pct = 15         # running session + project summaries
budget_summaries_pct = 25       # per-turn summaries + raw turns
budget_search_pct = 15          # RAG, decisions, project summary hits
budget_reserve_pct = 25         # response tokens + safety margin
```

The defaults are conservative — they work on an 8K local model.
Users with larger context windows widen `verbatim_turns`,
`summary_count`, and `search_max_results` to use more of their
budget. The shrinking algorithm still applies if the total exceeds
the context window.

### System Ceilings (Multi-User)

```ini
[memory]
raw_turn_retention_limit = 7d       # admin cap on retention
turn_summary_retention_limit = 30d
session_summary_retention_limit = 90d

[chat]
verbatim_turns_limit = 50      # admin cap on payload inclusion
summary_count_limit = 100
search_max_results_limit = 20
```

---

## Implementation Plan

### Phase 1: Storage Hierarchy + Memory Model Separation

Foundation: establish the six storage levels in the database and
split inference into conversation model and memory model.

**Changes:**

- **Database schema:** Add `level` column to memories table (or use
  `category` values): `raw-turn`, `turn-summary`, `session-summary`,
  `project-summary`, `rag`, `decision`. Add `status` column for
  decisions (`current`, `superseded`, `revisit`, `deprecated`).
- **Write permissions:** Enforce at the API layer — agent requests
  cannot create/modify Level 5 (RAG) entries. Agent can create
  Level 6 (decisions) and update their status but not their content.
- **Memory model config:** New `[inference.memory]` section with its
  own endpoint. `InferenceClient` gains a `memory_client()` accessor
  that resolves to the memory endpoint (falls back to main model).
- **Retention config:** New `[memory]` section with per-level
  retention settings.

**Files:** `sqlite_backend.h`, `sqlite_backend.cpp`, `config.h`,
`config.cpp`, `inference.h`, `inference.cpp`, `server.cpp`

### Phase 2: Per-Turn + Incremental Session Summaries

The core summarization pipeline: Level 2 and Level 3 production.

**Changes:**

- **After each turn:** Memory model produces a Level 2 (per-turn
  summary) from the raw exchange.
- **After each Level 2:** Memory model receives the current Level 3
  (session summary) + the new Level 2 → produces updated Level 3.
- **Topic-shift detection:** If the model can't coherently merge the
  new turn into the running session summary, it starts a new Level 3
  and marks the old one complete.
- **On pause/quit:** Final Level 3 update and mark complete.
- All summarization routes through the memory model, not the
  conversation model.

**Files:** `chat_sessions.h`, `chat_sessions.cpp`, `chat.cpp`,
`server.cpp`

### Phase 3: Tiered Payload Assembly + Context Budget

Build payloads from the storage hierarchy instead of the current
flat "last N turns."

**Changes:**

- `build_messages()` assembles from multiple levels:
	- Level 1 (raw turns) within `verbatim_minutes` window
	- Level 3 (session summaries) within `summary_hours` window
	- Levels 2, 4, 5, 6 via memory search (relevance-gated)
	- Level 6 decisions filtered to `current` status only
- Context budget: token estimation per section, percentage-based
  allocation from `max_context`
- Graceful degradation: if total exceeds budget, shed from oldest
  summaries first, then lowest-relevance search hits, then oldest
  verbatim turns

**Files:** `chat_sessions.h`, `chat_sessions.cpp`, `config.h`,
`config.cpp`

### Phase 4: Scheduled Project Summaries

Level 4 production: periodic scan and merge of session summaries.

**Changes:**

- Scheduled task (configurable interval, e.g. daily): scan completed
  Level 3 session summaries from the lookback window
- Group by embedding similarity — sessions about similar topics
  cluster together
- For each cluster: if an existing Level 4 project summary exists,
  merge new session summaries into it (incremental update). If not,
  create a new one.
- Reorganization: if a session summary fits better under a different
  project summary than where it was originally grouped, reassign it.
- All work done by memory model.

**Files:** `server.cpp` (scheduled task), `sqlite_backend.cpp`

### Phase 5: Proxy Mode

Enhance `/v1/chat/completions` to be a memory-augmented proxy.

**Changes:**

- Parse incoming messages array; find system message and last user
  message
- Load persona files (reuse `load_workspace_files()`)
- Search memory: Tier 2 (recent summaries by timestamp) + Tier 3
  (RAG hits for last user message)
- Enrich system message: prepend persona, append memories
- Forward enriched request to upstream; stream response back
- Auto-capture: store the user/assistant exchange post-response
- Session tracking: identify sessions for summarization
- Config: new `[proxy]` section with augmentation controls

**Files:** `server.cpp`, `config.h`, `config.cpp`

### Phase 6: Context Window Auto-Detection

Convenience feature for the setup experience.

**Changes:**

- On startup or first `/chat` request, probe the inference endpoint
  for context size using provider-specific APIs
- Cache the result in config (or in-memory)
- Log detected size for user visibility
- Ship known-model lookup table as fallback

**Files:** `inference.cpp`, `config.cpp`

---

## What This Enables

- **Conversations that feel continuous** across hours and sessions.
  The LLM doesn't lose track of what was decided — it fades naturally
  from verbatim to summary to "I'd need to look that up."

- **Works with any model size.** A 8K local model gets aggressive
  compression. A 1M cloud model gets hours of rich context. Same
  mechanism, different budget.

- **The database is the safety net.** Nothing is lost. The user (or
  the LLM, if given a search tool) can always retrieve the full detail
  of any past conversation. Fading is a payload strategy, not a
  deletion strategy.

- **Ragger Chat as the recommended interface.** Users connect via
  web UI, CLI, or any HTTP client. Ragger handles the complexity of
  memory, context, and conversation management. The client is simple.

- **Unified memory across interfaces.** Conversations through Ragger
  Chat, OpenClaw proxy, or any `/v1/chat/completions` client all feed
  the same memory pool. A decision made in an OC tool session is
  remembered in a Ragger Chat conversation the next day.

- **Clean separation of concerns.** Ragger owns persona and memory.
  Agent frameworks own tools and orchestration. Neither reimplements
  the other's job. The proxy makes them composable without coupling.

- **Storage API unchanged.** Agents that prefer direct control can
  still use store/search/count. No breaking changes.

---

## Proxy Mode: Ragger + OpenClaw Together

### The Problem

OpenClaw (and similar agent frameworks) manage tools, skills, and
orchestration — but they have no persistent memory across sessions
and no persona management. Today, the OpenClaw plugin uses Ragger as
dumb storage: OC calls `search` before each turn and `store` after.
OC decides what to search for, what to inject, and when to capture.

This means OC reimplements (badly) the memory logic that Ragger does
well. And the persona files (SOUL.md, USER.md) that define the
assistant's personality? OC doesn't know about them at all. Those
original OC persona files are ignored.

### The Solution

Ragger becomes an OpenAI-compatible inference proxy. OpenClaw points
its API base URL at Ragger instead of Claude/OpenAI directly:

```
Before:  OpenClaw → Claude API
After:   OpenClaw → Ragger → Claude API
```

OpenClaw doesn't know the difference. It sends standard
`/v1/chat/completions` requests. Ragger intercepts, enriches with
persona and memory, forwards to the real inference engine, captures
the exchange, and returns the response.

### How Proxy Mode Works

Ragger does not enrich OC's payload — it **replaces** it. OC sends
a standard `/v1/chat/completions` request with its full message
history. Ragger extracts only what it needs and builds the entire
payload from its own storage, exactly as it does in Role 2.

**What Ragger extracts from OC's request:**

- The latest user message (the new prompt)
- The `tools` array (OC's tool definitions — passed through)
- Any tool call/result messages from the current turn's execution
  chain (if OC is mid-tool-loop, those messages are part of the
  active turn and must be preserved)
- The model name

**What Ragger discards:**

- OC's conversation history (Ragger has its own — better organized,
  with fading, summaries, and budget control)
- OC's system prompt (Ragger builds its own from persona files +
  memory)

**What Ragger builds:**

The same payload as Role 2: persona, running session summary,
per-turn summaries, RAG hits, active decisions, previous raw turn,
and the new user message. The only addition is OC's tool
definitions, appended so the model knows what tools are available.

**Why replacement, not enrichment:** If Ragger enriched OC's
payload (adding memory on top of OC's history), you'd get
duplication — OC's 50 turns of history PLUS Ragger's memory of
those same turns. Token budget impossible to control. By replacing
the payload entirely, Ragger controls the budget. The fading memory
model applies consistently regardless of which interface the user
is talking through.

### Payload Assembly (Proxy Mode)

OC sends a standard `/v1/chat/completions` request:

```json
{
  "model": "claude-sonnet-4-5",
  "messages": [
    {"role": "system", "content": "[OC's system prompt]"},
    {"role": "user", "content": "What database should I use?"},
    {"role": "assistant", "content": "For your use case I'd suggest..."},
    {"role": "user", "content": "What about the schema we discussed yesterday?"}
  ],
  "tools": [...]
}
```

Ragger extracts the last user message and builds its own payload:

```json
{
  "model": "claude-sonnet-4-5",
  "messages": [
    {"role": "system", "content": "[SOUL.md persona]\n[USER.md]\n\n---\n\n## Session context:\nDatabase design discussion. Settled on SQLite WAL...\n\n## Relevant memories:\n1. Decision: Use SQLite with WAL mode (2026-03-15)\n2. RAG: SQLite schema best practices\n\n## Recent:\n- Discussed migration strategy, designed tags table..."},
    {"role": "user", "content": "For your use case I'd suggest..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "What about the schema we discussed yesterday?"}
  ],
  "tools": [...]
}
```

The payload is identical to what Role 2 (web UI) would produce for
the same conversation state — plus OC's `tools` array. Same memory,
same persona, same fading model, same budget. The only difference
is the model can call tools.

### Auto-Capture

After the response streams back, Ragger stores the exchange:

```json
{
  "text": "User: What about the schema we discussed yesterday?\n\nAssistant: Based on our earlier discussion...",
  "metadata": {
    "collection": "conversation",
    "category": "chat-turn",
    "source": "proxy-openclaw"
  }
}
```

This means:

- OC conversations feed the same memory pool as Ragger Chat sessions
- Summaries generated from OC conversations are available in future
  Ragger Chat sessions, and vice versa
- The user has one unified memory regardless of which interface they use
- Memory captured from tool-augmented OC sessions enriches future
  conversations that don't have tools

### Summarization in Proxy Mode

Ragger tracks proxy sessions the same way it tracks chat sessions:

- **Session tracking:** Identify sessions by a header
  (`X-Session-Id`) or by username + time proximity
- **Turn accumulation:** Buffer captured turns
- **Pause summarization:** When the session goes idle (no requests
  for `pause_minutes`), summarize buffered turns
- **Rolling summarization:** If configured, summarize turns as they
  age past `tier1_minutes` even during active use

The summarization infrastructure is identical to Ragger Chat. The
only difference is that Tier 1 (verbatim turns) lives in OC's
message array, not Ragger's — so Ragger's summarization works from
the auto-captured turns.

### Configuration

```ini
[proxy]
augment = true              # enrich /v1/chat/completions with memory + persona
augment_persona = true      # inject SOUL.md, USER.md into system prompt
augment_memory = true       # inject memory search results
augment_max_results = 5     # max RAG hits to inject
augment_min_score = 0.3     # relevance threshold
auto_capture = true         # store user/assistant exchanges as memories
capture_summarize = true    # summarize captured turns on session idle
```

### OpenClaw Configuration Change

```json
{
  "inference": {
    "apiBase": "http://localhost:8432/v1"
  },
  "plugins": {
    "slots": {
      "memory": null
    }
  }
}
```

Two changes: point inference at Ragger, and **remove the memory
plugin**. The plugin's store/search/auto-recall/auto-capture is
now redundant — Ragger's proxy does all of it, better, because
it has access to the full fading memory model instead of a simple
"search and prepend" heuristic.

### Why This Is Better Than the Plugin

The current OpenClaw plugin (`memory-ragger`) does:

1. `before_agent_start` → search Ragger → prepend results to prompt
2. `agent_end` → regex-match "important" messages → store a few

This is crude:

- **No persona.** OC doesn't load SOUL.md.
- **No tiered memory.** Just "top 3 hits."
- **No summaries.** Only raw RAG search.
- **Naive capture.** Regex heuristics decide what's "important."
- **No context budgeting.** Memories are prepended without regard
  for context window limits.

Proxy mode replaces all of this with Ragger's full memory system:
persona files, tiered fading memory, context budgeting, reliable
auto-capture, and background summarization. The plugin becomes
unnecessary.

More importantly: because Ragger **replaces** the payload rather
than enriching it, there's no duplication. OC's history is
redundant — Ragger has the same information, better organized.
The token budget is fully under Ragger's control.

### Implementation

The existing `/v1/chat/completions` proxy endpoint in `server.cpp`
is a pure pass-through. Enhancing it:

1. **Parse incoming request** — Extract the last user message, the
   `tools` array, and any mid-turn tool call/result messages
2. **Discard OC's history and system prompt** — Ragger builds its own
3. **Build payload** — Same assembly as Role 2: persona files,
   session summary, per-turn summaries, RAG hits, decisions,
   previous raw turn, new user message
4. **Append tools** — Add OC's `tools` array to the request
5. **Forward** — Send to upstream inference (existing proxy logic)
6. **Capture** — After response, store the exchange as a memory turn
7. **Session tracking** — Associate with a session for summarization

Estimated: 200–400 lines in `server.cpp`. No new files needed. The
persona loading, memory search, session management, payload
assembly, and proxy forwarding all exist — this connects them in
a new configuration.

### Compatibility

- **Transparent to OC.** Standard OpenAI-compatible API. Any agent
  framework that speaks `/v1/chat/completions` works.
- **Opt-in.** `augment = false` (default for backward compatibility)
  keeps the existing dumb-pipe behavior.
- **Coexists with Ragger Chat.** Same server, same memory database,
  same persona files. Users can use Ragger Chat directly AND through
  OC with proxy mode.
- **Works with any upstream.** Claude, OpenAI, Ollama, LM Studio —
  Ragger already routes to multiple inference endpoints.

---

## Open Questions

1. **Summary granularity.** Should mid-session summaries cover
   fixed time windows (every 30 minutes) or natural conversation
   breaks (topic shifts)? Topic detection is harder but produces
   better summaries.

2. **Multi-topic splitting.** A single conversation might cover
   three unrelated subjects. Should summaries be split by topic
   (better search) or kept as one block (simpler)?

3. **LLM self-search.** Should the LLM be able to explicitly search
   Tier 4 during conversation? ("Let me look up what we decided
   about X.") This would require tool-use support in the chat
   endpoint. In proxy mode this is a non-issue — OC can provide a
   `memory_search` tool. In direct Ragger Chat mode, it would mean
   adding Level 1 tool calling (Ragger's own tools only). Moderate
   effort, high value.

4. **Summary quality.** The summarizer needs to know what matters.
   "We discussed databases" is useless. "Chose SQLite over Postgres
   for single-file deployment" is valuable. The current summarization
   prompt is decent but may need refinement as real usage data comes in.

5. **Overlap between Tier 2 and Tier 3.** A summary from 2 hours ago
   might also be a RAG hit for the current query. Should it appear
   twice? Probably deduplicate by memory ID.

6. **Proxy session identification.** How to associate multiple
   `/v1/chat/completions` requests into a single session for
   summarization? Options: `X-Session-Id` header (explicit, requires
   OC support), username + time proximity (implicit, may mis-group),
   or content hashing (fragile). Explicit header is cleanest but
   requires OC to send one.

7. **Deprecating the OC memory plugin.** Once proxy mode is stable,
   the `openclaw-plugin/` directory becomes unnecessary. Transition
   plan: ship proxy mode, document it, keep the plugin for backward
   compatibility for one release, then remove.

---

## Related

- [Configuration](docs/configuration.md) — Existing config reference
- [Chat Persistence](docs/chat-persistence.md) — Turn storage and summarization
- [Agent Integration](docs/agent-integration.md) — Best practices for external agents
- [FEATURE_SUGGESTIONS.md](FEATURE_SUGGESTIONS.md) — GUI and UX improvements
