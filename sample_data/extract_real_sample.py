"""
sample_data/extract_real_sample.py

Pulls a representative SAMPLE of REAL candidates out of the full
candidates.jsonl for fast local development/testing, instead of running
against the full 100K pool every time. This is real data, not synthetic —
it includes a deliberately-curated set of known trap-type candidate IDs
(found by direct inspection of the dataset) plus a random cross-section,
so the pipeline can be sanity-checked end-to-end quickly.

Usage:
    python sample_data/extract_real_sample.py \
        --candidates /path/to/candidates.jsonl \
        --out sample_data/candidates_sample.jsonl \
        --n 3000
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine.data import iter_candidates, cid

# IDs identified by direct inspection of the real dataset during
# calibration — known examples of each trap type, for fast spot-checking.
KNOWN_INTERESTING_IDS = {
    "CAND_0007353", "CAND_0008960", "CAND_0010294",  # timeline/role-duration honeypots
    "CAND_0016000", "CAND_0046649", "CAND_0060642",   # zero-duration-expert-cluster honeypots
    "CAND_0000121", "CAND_0000212", "CAND_0000312",   # keyword stuffers (support/content roles, AI skills)
    "CAND_0001610", "CAND_0002025", "CAND_0005260",   # genuine elite candidates
    "CAND_0005538",                                    # plain-language tier-5 (rare skill-name variants)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="sample_data/candidates_sample.jsonl")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    known, others = [], []
    for c in iter_candidates(args.candidates):
        if cid(c) in KNOWN_INTERESTING_IDS:
            known.append(c)
        else:
            others.append(c)

    n_random = max(0, args.n - len(known))
    sampled = known + rng.sample(others, min(n_random, len(others)))
    rng.shuffle(sampled)

    with open(args.out, "w", encoding="utf-8") as f:
        for c in sampled:
            f.write(json.dumps(c) + "\n")

    print(f"[extract_real_sample] wrote {len(sampled)} real candidates to {args.out} "
          f"({len(known)} known-interesting + {len(sampled)-len(known)} random)")


if __name__ == "__main__":
    main()
