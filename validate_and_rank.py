"""
validate_and_rank.py - runs full rank + validates output
"""
import csv, subprocess, sys, os

CANDIDATES = "India_runs_data_and_ai_challenge/candidates.jsonl"
CONFIG     = "config/jd_config.yaml"
ARTIFACTS  = "artifacts"
OUT        = "submission.csv"
VALIDATOR  = "India_runs_data_and_ai_challenge/validate_submission.py"

print("=" * 60)
print("STEP 1: Running rank.py ...")
print("=" * 60)
ret = subprocess.run([sys.executable, "rank.py",
     "--candidates", CANDIDATES, "--config", CONFIG,
     "--artifacts-dir", ARTIFACTS, "--out", OUT], check=False)
if ret.returncode != 0:
    print(f"[FAIL] rank.py exited with code {ret.returncode}")
    sys.exit(ret.returncode)

print("\n" + "=" * 60)
print("STEP 2: Checking score monotonicity ...")
print("=" * 60)
with open(OUT, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

violations = []
prev_score = None
for row in rows:
    score = float(row["score"])
    if prev_score is not None and score > prev_score + 1e-9:
        violations.append(f"  rank {row['rank']}: score rose {prev_score:.6f} -> {score:.6f}")
    prev_score = score

if violations:
    print(f"[FAIL] {len(violations)} monotonicity violation(s):")
    for v in violations[:10]: print(v)
    sys.exit(1)
else:
    print(f"[OK] score non-increasing across all {len(rows)} rows.")
    print(f"     range: {float(rows[0]['score']):.6f} (rank 1) -> {float(rows[-1]['score']):.6f} (rank 100)")

print("\n" + "=" * 60)
print("STEP 3: Official validate_submission.py ...")
print("=" * 60)
if os.path.exists(VALIDATOR):
    ret2 = subprocess.run([sys.executable, VALIDATOR, OUT], check=False)
    if ret2.returncode != 0:
        print("[FAIL] Official validator rejected."); sys.exit(ret2.returncode)
    else:
        print("[OK] Official validator passed.")
else:
    print(f"[SKIP] Validator not found at {VALIDATOR}")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED - submission.csv is ready.")
print("=" * 60)
