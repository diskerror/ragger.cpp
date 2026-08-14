# Vector storage dtypes (on-disk embedding precision)

Ragger computes and compares embeddings in **float32**. The `embedding.vector_type`
setting only governs how each vector is *stored* on disk — every query decodes
the blob back to f32 before the cosine scan. So picking a smaller dtype trades
disk/cache size (and startup load time, since the whole corpus is held in an
in-memory `Eigen::MatrixXf`) against a small loss of ranking fidelity. It never
changes the math, only the stored bytes.

## Supported types

| dtype  | bytes/dim | layout | notes |
|--------|-----------|--------|-------|
| `f32`  | 4 | IEEE 754 single | Lossless baseline. |
| `f16`  | 2 | IEEE half (1s/5e/10m) | **Default.** More mantissa than bf16; smaller exponent range (max ≈ 65504). Fine for normalized embeddings. |
| `bf16` | 2 | bfloat16 (1s/8e/7m) | Same exponent range as f32, fewer mantissa bits. Round-to-nearest-even, NaN preserved. Useful only if a model produces large-magnitude components that f16 would overflow. |
| `int8` | 1 | symmetric per-vector | `scale = max|x| / 127`, stored per vector in the blob header. 4× smaller than f32, 2× smaller than f16. |

`f16` and `bf16` are **not** interchangeable despite both being 2 bytes/dim —
see the self-describing blob format below.

## On-disk blob format

Blobs written by `vector_codec::encode` carry a 12-byte little-endian header so
the dtype is unambiguous on read (f16 and bf16 are the same size; an int8 blob
of N dims is byte-identical in length to an f16 blob of N/2 dims — size alone
cannot disambiguate them, and the `dimensions` setting isn't enough either):

```
offset  size  field
0       3     magic 'R','V','1'
3       1     dtype (0=f32 1=f16 2=bf16 3=int8)
4       1     flags (reserved, 0)
5       1     reserved (0)
6       2     dims  (uint16)
8       4     scale (float32; int8 dequant multiplier, 0 for float types)
12      ...   payload
```

**Legacy blobs** written before this header existed are raw `dims*2` (f16) or
`dims*4` (f32) bytes with no magic. `decode()` detects those by exact byte
length first, so old databases keep reading correctly until a
`ragger rebuild-embeddings` rewrites them in the headered form. Real embedding
models are ≥16-dim, so the headered vs legacy size ranges never collide.

## Changing the stored precision

`vector_type` is part of the DB's vector identity (model + dims + dtype). To
change it on a DB that already holds data:

1. Set the desired type (dashboard "Desired Vector Type", or
   `desired_embedding_vector_type` in config / the `settings` table).
2. Run `ragger rebuild-embeddings` (or "Update now" in the dashboard) to
   re-encode every stored vector at the new precision.

A drift guard refuses to run a daemon whose stored `vector_type` disagrees with
config until you rebuild, so you can't silently mix precisions in one DB.

## Benchmark: recall vs precision on real data

`scripts/bench_vector_quant.py` measures how each dtype perturbs
nearest-neighbor rankings versus full precision, using the actual embeddings in
a `memories.db`. Recall@k = "of the k nearest neighbors under full precision,
how many still appear in the top-k under the quantized store" (1.0 = identical
ranking).

Run on an 18,623-vector corpus (384-dim `all-MiniLM-L6-v2`, 400 sampled
queries). **Note:** this DB was already stored as f16, so the f32/f16 rows are
trivially 1.0 (upcasting f16→f32 is lossless); the meaningful signal is bf16
and int8 measured relative to that f16 store.

| dtype  | size    | recall@1 | recall@5 | recall@10 | recall@20 | mean cos err |
|--------|---------|----------|----------|-----------|-----------|--------------|
| f32    | 27.9 MB | 1.0000   | 1.0000   | 1.0000    | 1.0000    | 5.3e-08 |
| f16    | 14.0 MB | 1.0000   | 1.0000   | 1.0000    | 1.0000    | 5.3e-08 |
| bf16   | 14.0 MB | 1.0000   | 0.9895   | 0.9830    | 0.9925    | 1.4e-06 |
| int8   |  7.0 MB | 0.9925   | 0.9875   | 0.9785    | 0.9878    | 4.3e-05 |

### Takeaways for `all-MiniLM-L6-v2` (the default model)

- **f16 (default) is the right default.** For this normalized 384-dim model it
  is effectively lossless vs f32 at half the size — no overflow issues, no
  measurable recall loss.
- **bf16 is strictly worse than f16 here.** Same size, lower recall — its extra
  exponent range buys nothing because MiniLM components are already small
  (`|x| < 1`). Only reach for bf16 with a model that emits large-magnitude
  components f16 would overflow (rare for normalized embeddings).
- **int8 gives 4× compression vs f32 (2× vs f16) for ~1–2% recall loss.**
  Attractive when the corpus grows large enough that the in-memory matrix or
  startup load time matters, and a couple points of recall@10 is acceptable.
  At small corpus sizes there's no reason to bother — f16 already fits.
- **Subject matter matters.** These numbers are model- and corpus-specific.
  Re-run `bench_vector_quant.py` against your own DB before switching a large
  or accuracy-critical store to int8; a domain with tightly clustered
  embeddings can lose more recall to quantization than this general corpus did.
