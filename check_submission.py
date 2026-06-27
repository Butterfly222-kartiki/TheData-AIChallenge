import csv, sys

with open("submission.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Rows: {len(rows)}")

violations = []
prev = None
for r in rows:
    s = float(r["score"])
    if prev is not None and s > prev + 1e-9:
        violations.append("  rank %s: %.6f -> %.6f (UP)" % (r["rank"], prev, s))
    prev = s

if violations:
    print("FAIL: %d monotonicity violations:" % len(violations))
    for v in violations[:5]:
        print(v)
    sys.exit(1)
else:
    print("OK: score non-increasing across all %d rows" % len(rows))
    print("Score range: %.6f (rank 1) -> %.6f (rank 100)" % (
        float(rows[0]["score"]), float(rows[-1]["score"])))

print("Top-5 candidates:")
for r in rows[:5]:
    print("  rank %s: %s  score=%.6f" % (r["rank"], r["candidate_id"], float(r["score"])))

print("Ranks 8-12:")
for r in rows[7:12]:
    print("  rank %s: %s  score=%.6f" % (r["rank"], r["candidate_id"], float(r["score"])))
