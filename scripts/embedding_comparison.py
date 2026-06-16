#!/usr/bin/env python3
"""
Compare each raw turn (L1) to its corresponding turn summary (L2).
Joined on timestamp — L1 and L2 share the exact same timestamp.

Goal: high similarity + small reduction% = good summary quality.

Output TSV columns: date, similar, reduction, model

  date      — turn timestamp
  similar   — cosine similarity between L1 turn embedding and L2 summary embedding (0–1, 2 dp)
  reduction — (L2 text size / L1 text size) * 100, whole number percent
  model     — summarizer model name from the summaries table

Usage: python embedding_comparison.py <output_file.tsv>
"""

import sys
import sqlite3
import struct
import math
from pathlib import Path


def load_embedding(blob: bytes) -> list:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def cosine_similarity(a: list, b: list) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    raw = dot / (mag_a * mag_b)
    return max(0.0, min(1.0, (raw + 1.0) / 2.0))


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output_file.tsv>", file=sys.stderr)
        sys.exit(1)

    output_path = sys.argv[1]
    db_path = Path.home() / ".ragger" / "memories.db"

    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    rows = con.execute(
        "SELECT t.timestamp, "
        "       t.embedding AS t_emb, "
        "       length(t.user_text) + coalesce(length(t.assistant_text), 0) AS t_size, "
        "       s.s_emb, s.s_size, s.model_name "
        "FROM turns t "
        "LEFT JOIN ( "
        "    SELECT s.timestamp, s.embedding AS s_emb, length(s.text) AS s_size, "
        "           m.name AS model_name "
        "    FROM summaries s LEFT JOIN models m ON m.model_id = s.model_id "
        "    WHERE s.level = 'turn' AND s.embedding IS NOT NULL "
        "      AND s.model_id IS NOT NULL "  # only actually-summarized rows; NULL = still raw
        "    GROUP BY s.timestamp HAVING s.summary_id = MAX(s.summary_id) "
        ") s USING (timestamp) "
        "WHERE t.embedding IS NOT NULL "
        "ORDER BY t.timestamp"
    ).fetchall()

    con.close()

    rows_written = 0
    skipped = 0
    with open(output_path, "w") as f:
        f.write("date\tsimilar\treduction\tmodel\n")
        for row in rows:
            if row["s_emb"] is None:
                skipped += 1
                continue
            t_vec = load_embedding(row["t_emb"])
            s_vec = load_embedding(row["s_emb"])
            if len(t_vec) != len(s_vec):
                skipped += 1
                continue
            sim = cosine_similarity(t_vec, s_vec)
            t_size = row["t_size"] or 0
            s_size = row["s_size"] or 0
            reduction = round((s_size / t_size) * 100) if t_size > 0 else 0
            f.write(f"{row['timestamp']}\t{sim:.2f}\t{reduction}\t{row['model_name'] or ''}\n")
            rows_written += 1

    print(f"Wrote {rows_written} rows to {output_path} ({skipped} turns skipped — no matching L2)", file=sys.stderr)


if __name__ == "__main__":
    main()
