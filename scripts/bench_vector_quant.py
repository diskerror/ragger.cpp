#!/usr/bin/env python3
"""
bench_vector_quant.py — empirical precision/recall tradeoff for on-disk vector
dtypes (f32 / f16 / bf16 / int8), measured on the REAL embeddings in a Ragger
memories.db.

Ragger computes and compares embeddings in float32; the on-disk dtype only
governs how vectors are stored, and every query decodes back to f32 before the
cosine scan. So the question this answers is: if we store at a lower precision,
how much does the decoded-back vector perturb nearest-neighbor rankings versus
storing at full precision?

Method
------
1. Read all non-NULL embeddings from summaries/turns/documents/decisions. The
   stored blobs are raw f16 (768 B = 384*2) or raw f32 (1536 B). Decode to f32;
   this decoded vector is the per-dtype "ground truth" reference (the best we
   can reconstruct from what's on disk).
2. For each candidate dtype, round-trip every vector through that dtype's
   encode/decode (matching the C++ vector_codec exactly: IEEE f16, bfloat16
   round-to-nearest-even, symmetric per-vector int8 with scale=max|x|/127).
3. Sample query vectors from the corpus. For each query, compute the true
   top-k neighbors using the f32 reference vectors, then the top-k using the
   quantized vectors, and report recall@k (overlap) plus mean rank
   displacement and cosine error.

Recall@k here = "of the k nearest neighbors under full precision, how many
still appear in the top-k under the quantized store." 1.0 = identical ranking.

Usage:
  python3 scripts/bench_vector_quant.py [--db PATH] [--k 10] [--queries 300]
"""

import argparse
import sqlite3
import struct
import sys
import numpy as np


def decode_blob(blob: bytes, dims: int) -> np.ndarray:
    """Decode a stored embedding blob (raw f16 or raw f32, headerless)."""
    if len(blob) == dims * 2:
        return np.frombuffer(blob, dtype=np.float16).astype(np.float32)
    if len(blob) == dims * 4:
        return np.frombuffer(blob, dtype=np.float32).copy()
    raise ValueError(f"unexpected blob length {len(blob)} for dims={dims}")


# --- dtype round-trips (must match src/vector_codec.cpp) -------------------

def rt_f32(X):
    return X.copy()

def rt_f16(X):
    return X.astype(np.float16).astype(np.float32)

def rt_bf16(X):
    # bfloat16 = high 16 bits of the f32 pattern, round-to-nearest-even.
    u = X.astype(np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    u = u + 0x7FFF + lsb
    bf = (u >> 16).astype(np.uint16)
    back = (bf.astype(np.uint32) << 16).view(np.float32)
    return back

def rt_int8(X):
    # symmetric per-vector int8: scale = max|x|/127
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        v = X[i]
        maxabs = np.max(np.abs(v))
        scale = (maxabs / 127.0) if maxabs > 0 else 1.0
        q = np.clip(np.round(v / scale), -127, 127).astype(np.int8)
        out[i] = q.astype(np.float32) * scale
    return out


ROUNDTRIPS = {"f32": rt_f32, "f16": rt_f16, "bf16": rt_bf16, "int8": rt_int8}
BYTES_PER_DIM = {"f32": 4, "f16": 2, "bf16": 2, "int8": 1}


def normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def main():
    ap = argparse.ArgumentParser()
    import os
    ap.add_argument("--db", default=os.path.expanduser("~/.ragger/memories.db"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--dims", type=int, default=384)
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    rows = []
    for tbl in ("summaries", "turns", "documents", "decisions"):
        try:
            cur = con.execute(
                f"SELECT embedding FROM {tbl} WHERE embedding IS NOT NULL")
        except sqlite3.OperationalError:
            continue
        for (blob,) in cur:
            try:
                rows.append(decode_blob(blob, args.dims))
            except ValueError:
                pass
    con.close()

    if len(rows) < 50:
        print(f"Not enough embeddings ({len(rows)}) to benchmark.", file=sys.stderr)
        sys.exit(1)

    X = np.vstack(rows).astype(np.float32)
    n = X.shape[0]
    print(f"corpus: {n} vectors x {X.shape[1]} dims  (from {args.db})")
    print(f"recall@{args.k} over {min(args.queries, n)} sampled queries\n")

    # f32-reference (decoded from disk) is the ranking ground truth.
    ref = normalize(X)

    rng = np.random.default_rng(42)
    qidx = rng.choice(n, size=min(args.queries, n), replace=False)

    # Precompute reference top-k for each query (exclude self).
    ref_topk = {}
    for qi in qidx:
        sims = ref @ ref[qi]
        sims[qi] = -np.inf
        ref_topk[qi] = set(np.argpartition(-sims, args.k)[: args.k])

    print(f"{'dtype':>6} {'B/dim':>6} {'size':>8} {'recall@k':>10} "
          f"{'mean cos err':>13} {'mean |Δrank|':>13}")
    print("-" * 62)
    for name, fn in ROUNDTRIPS.items():
        Q = normalize(fn(X))
        recalls, coserrs, rankshift = [], [], []
        # cosine error vs reference, averaged over all vectors
        coserr_all = float(np.mean(np.abs(np.sum(ref * Q, axis=1) - 1.0)))
        for qi in qidx:
            sims = Q @ Q[qi]
            sims[qi] = -np.inf
            order = np.argpartition(-sims, args.k)[: args.k]
            got = set(order)
            true = ref_topk[qi]
            recalls.append(len(got & true) / args.k)
            # rank displacement of the true #1 neighbor
            true1 = max(true, key=lambda j: ref[j] @ ref[qi])
            full_order = np.argsort(-sims)
            pos = int(np.where(full_order == true1)[0][0])
            rankshift.append(abs(pos - 0))
        size_kb = n * args.dims * BYTES_PER_DIM[name] / 1024.0
        print(f"{name:>6} {BYTES_PER_DIM[name]:>6} {size_kb:>7.0f}K "
              f"{np.mean(recalls):>10.4f} {coserr_all:>13.2e} "
              f"{np.mean(rankshift):>13.3f}")


if __name__ == "__main__":
    main()
