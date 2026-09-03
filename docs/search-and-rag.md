# Search & RAG

Ragger uses **hybrid RAG with BM25 + dense retrieval** — combining keyword
matching with semantic vector search for better recall.

## How It Works

### 1. Indexing

Documents are split into paragraph-sized chunks (short paragraphs merged
to a minimum size, never split mid-sentence). Each chunk is:

1. Embedded into a 384-dimensional vector using a local sentence-transformer
   model (`all-MiniLM-L6-v2`)
2. Indexed for BM25 keyword search via SQLite's built-in FTS5 (no external services)
3. Stored alongside the original text and metadata

Imported document chunks include heading context prepended to the text
(full heading chain from the document hierarchy) and a `section` breadcrumb
using `»` separators for deeper nesting.

### 2. Retrieval

At query time, the query is embedded with the same model. Two scores are
computed for each document:

- **Vector score:** Cosine similarity computed with Eigen3 (semantic meaning)
- **BM25 score:** Okapi BM25 keyword relevance (exact term matching)

Both scores are min-max normalized to [0,1], then blended with configurable
weights (default: vector 8 / BM25 4 / phonetic 1 — meaning-first, with
keyword and a gentle "sounds-like" nudge). Top-k results are returned
ranked by the blended score; the reported score remains raw cosine
similarity for consistency.

### 3. Generation

Retrieved results are injected into the LLM's context window as reference
material. The LLM generates its response grounded in the retrieved text.

This is the standard RAG pattern. Ragger handles retrieval; your LLM
handles generation.

## Hybrid Search Scoring

Hybrid search blends two retrieval methods:

1. **Dense retrieval (vector search):** Finds documents with similar
   *meaning* to the query, even if they don't share keywords. Good for
   paraphrasing, conceptual questions, cross-domain search.

2. **Sparse retrieval (BM25):** Finds documents with exact keyword matches,
   weighted by term frequency and document length. Good for technical terms,
   proper nouns, acronyms.

**Why blend?**

Pure vector search can miss exact-match terms (e.g., "OAuth2" vs "authentication").
Pure keyword search misses paraphrases and synonyms (e.g., "deploy" vs "ship to production").
Hybrid search gets the best of both.

### Score Normalization

Raw vector scores (cosine similarity) are already in [0,1]. Raw BM25 scores
are unbounded. To blend them fairly, Ragger applies min-max normalization
to BM25 scores:

```
normalized_score = (score - min) / (max - min)
```

After normalization, both scores are in [0,1] and can be blended with
configurable weights.

### Blending Weights

Default blend: **vector 8 / BM25 4 / phonetic 1** ("sounds-like" via
Double Metaphone, off by default weight 0 in older versions — now on
lightly by default).

Configure via `vector_weight`, `bm25_weight`, and `phon_weight` in `[search]`:

```ini
[search]
vector_weight = 8
bm25_weight   = 4
phon_weight   = 1
```

Weights are ratios (not percentages). They're normalized internally.
Using integers avoids floating-point config parsing issues.

**To disable BM25 or the phonetic signal:**

```ini
[search]
bm25_enabled = false
phon_weight  = 0
```

## BM25 Tuning

Keyword relevance is computed by SQLite's built-in FTS5 `bm25()` ranking
function, which uses fixed internal parameters (`k1` and `b`). These are not
exposed as config — the only keyword knob is `bm25_weight`, which sets how
much the keyword score contributes to the hybrid blend (see above).

## Chunking Strategy

Ragger splits documents at paragraph boundaries (`\n\n`), never mid-sentence.

### Minimum Chunk Size

Short paragraphs are merged until they reach a minimum size (default: 300
characters). This ensures each chunk contains enough context for meaningful
embeddings.

```bash
# Import with custom minimum chunk size
ragger import-docs doc.md --min-chunk-size 500
```

**Why merge short paragraphs?**

A single-sentence paragraph like "See section 3.2 for details." lacks
context. Merged with the previous paragraph, the embedding captures the
full idea.

**Why not split mid-sentence?**

Splitting mid-sentence breaks grammatical structure and produces low-quality
embeddings. Sentence-level chunking is supported by splitting on `. ` or
`. \n`, but paragraph-level is generally better for technical docs.

### Heading Context

When importing Markdown files, Ragger prepends the full heading chain to
each chunk:

```
# API Reference
## Authentication
### OAuth2 Flow

The client must request a token from the authorization endpoint.
```

Becomes:

```
API Reference » Authentication » OAuth2 Flow

The client must request a token from the authorization endpoint.
```

This ensures search results include navigational context, making it easier
to locate the source material.

## File Import

### Supported Formats

Any text file works: `.md`, `.txt`, `.log`, `.csv`, etc.

For binary formats (PDF, DOCX, etc.), convert to text first with a tool like:

- [docling](https://github.com/DS4SD/docling) — PDF, DOCX, PPTX, HTML
- [pandoc](https://pandoc.org/) — Universal document converter
- `pdftotext` — Simple PDF → text

### File Size Limits

No practical limit when using `--min-chunk-size` — files are split into
chunks and stored as separate documents.

**Without chunking**, each file becomes one document. For anything longer
than a page or two, use `--min-chunk-size`.

### Import Examples

```bash
# Import a Markdown file with default chunking (300 chars)
ragger import-docs notes.md --tags docs

# Import with custom chunk size
ragger import-docs large-doc.md --min-chunk-size 500

# Import multiple files with a shared tag
ragger import-docs doc1.md doc2.md doc3.md --tags reference

# Import a converted PDF
docling myfile.pdf -o myfile.md
ragger import-docs myfile.md --tags docs
```

> **Important:** Imported document chunks (L5) are **not** returned by
> `ragger search` or the `/search` HTTP endpoint. The `search()` method
> queries the `summaries` table only; the `documents` table has its own
> FTS5 index but is not yet merged into general search. Document search
> via a recipe layer is planned for a future release. Use `store` (not
> `import-docs`) if you need your content to appear in regular `search` results.

## Performance Characteristics

- **Vector search:** ~10-50ms for 50K documents on Apple Silicon
- **Hybrid search:** ~15-60ms (BM25 adds ~5-10ms overhead)
- **Embedding time:** ~20-50ms per query (model stays loaded in server mode)

**Scaling:**

- **50K docs:** Fast, no index needed (brute-force Eigen cosine is sufficient)
- **100K docs:** Still usable, but slower (~100ms per query)
- **500K+ docs:** Consider an HNSW index or external vector store
  (Qdrant, Pinecone, etc.) for sub-linear search time

The current implementation uses brute-force cosine similarity (all
embeddings loaded into an Eigen matrix and matched in one matmul).
Simple, fast for moderate datasets, no external dependencies.

For larger deployments, the pluggable `StorageBackend` interface
(`include/storage_backend.h`) makes it straightforward to swap
in a vector database — `SqliteBackend` is the only concrete
implementation today.

## Retrieval instrumentation (optional, build-time)

Per-query logging is not a runtime setting — there is no
`query.log` and no `[logging] query_log` key. Instead, a compile-time
flag (`RAGGER_STATS`, off by default) logs every search's per-signal
score breakdown to a separate, discardable `~/.ragger/stats.db`. See
[Configuration → retrieval stats](configuration.md#build-time-instrumentation-retrieval-stats-ragger_stats)
for the full schema and example queries.

## Related

- [Configuration](configuration.md) — Tuning search parameters
- [HTTP API](http-api.md) — `/search` endpoint, MCP `search` tool
- [Agent integration](agent-integration.md) — When agents should call `search`
