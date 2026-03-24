# Verify Your Installation

## What This Is

A simple way to test that your Ragger Memory installation is working
correctly with **your own data**. Your AI agent samples your stored
memories and generates targeted queries — both positive (should find
results) and negative (should return nothing relevant).

This validates the full pipeline: embedding model → vector search →
BM25 → result ranking.

## How To Run

Ask your AI agent:

> Sample my stored memories and create a retrieval quality test.
> Pick 10-15 chunks at random across different collections and sources.
> For each, write a natural-language query that should find it, and one
> that asks about something completely unrelated that should return
> nothing. Run the queries and report pass/fail.

## What the Agent Should Do

1. **Sample chunks**: Pull 10-15 random memories from your database
   using `ragger search` or the API. Vary the collections and sources.

2. **Write positive queries**: For each sampled chunk, write a
   natural-language question that someone might actually ask, where
   that chunk is the correct answer. Don't just copy the text — rephrase it.

3. **Write negative queries**: Write queries about topics that
   definitely aren't in your database. These should return zero
   relevant results (or very low scores).

4. **Run and grade**: Execute each query against your running instance.
   - **Positive**: PASS if the target chunk appears in the top 5 results.
   - **Negative**: PASS if no results score above the minimum threshold
     (default 0.4).

5. **Report**: Summary of pass/fail with any notable failures.

## What Good Looks Like

- All positive queries find their target chunk in the top 5
- All negative queries return empty or below-threshold results
- No false positives (unrelated chunks scoring high)

## What Bad Looks Like

- Positive queries miss their target → possible embedding model issue,
  chunking too aggressive, or BM25 index stale (`ragger rebuild-bm25`)
- Negative queries return high-scoring results → possible over-fitting
  in embeddings or chunk text too generic

## Manual Quick Test

If you prefer to test manually:

```bash
# Should return relevant results
ragger search "your topic here"

# Should return nothing relevant
ragger search "completely unrelated nonsense topic"

# Check total memory count
ragger count
```

## Automated Test Suite

For developers, see `tests/test_retrieval_quality.py` — a pytest suite
that runs parameterized queries against the real database. The user test
described above is the accessible version of the same idea.
