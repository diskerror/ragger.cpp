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
weights (default: 70% vector, 30% BM25). Top-k results are returned ranked
by the blended score; the reported score remains raw cosine similarity for
consistency.

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

Default blend: **70% vector, 30% BM25**.

Configure via `vector_weight` and `bm25_weight` in `[search]`:

```ini
[search]
vector_weight = 7
bm25_weight = 3
```

Weights are ratios (not percentages). They're normalized to sum to 1.0
internally. Using integers avoids floating-point config parsing issues.

**To disable BM25 entirely:**

```ini
[search]
bm25_enabled = false
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
ragger import doc.md --min-chunk-size 500
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
ragger import notes.md --collection docs

# Import with custom chunk size
ragger import large-doc.md --min-chunk-size 500

# Import multiple files
ragger import doc1.md doc2.md doc3.md --collection reference

# Import a converted PDF
docling myfile.pdf -o myfile.md
ragger import myfile.md --collection docs
```

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
(`include/ragger/storage_backend.h`) makes it straightforward to swap
in a vector database — `SqliteBackend` is the only concrete
implementation today.

## Query Logging

Search queries are logged to `~/.ragger/query.log` as single-line JSON
entries with timing, result scores, and quality metrics.

**Example log entry:**

```json
{
  "timestamp": "2024-03-20T15:23:45",
  "query": "API authentication",
  "limit": 5,
  "min_score": 0.4,
  "results": 3,
  "top_score": 0.82,
  "elapsed_ms": 12.3,
  "total_docs": 10614
}
```

**Fields:**

- `timestamp` — ISO 8601 timestamp
- `query` — The search query
- `limit` — Requested result count
- `min_score` — Minimum score threshold
- `results` — Actual result count (after filtering by score)
- `top_score` — Highest score in results
- `elapsed_ms` — Query time in milliseconds
- `total_docs` — Total documents in the database

**Enable/disable:**

```ini
[search]
query_log = true
```

Logging failures are caught silently — they never break search operations.

## Related

- [Configuration](configuration.md) — Tuning search parameters
- [HTTP API](http-api.md) — `/search` endpoint, MCP `search` tool
- [Agent integration](agent-integration.md) — When agents should call `search`
