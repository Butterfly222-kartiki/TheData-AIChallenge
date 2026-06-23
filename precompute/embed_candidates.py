"""
precompute/embed_candidates.py

Generates artifacts/career_embeddings.npz and artifacts/skill_embeddings.npz
for the full candidate pool. JD-agnostic — compute once per dataset, reuse
across any number of JD configs.

Streams the input file (engine.data.iter_candidates) so the full 487MB /
100K-candidate real dataset doesn't need to fit in memory all at once
during text extraction; embeddings are still batch-encoded.

Usage:
    python precompute/embed_candidates.py \
        --candidates path/to/candidates.jsonl \
        --artifacts-dir artifacts \
        --model all-MiniLM-L6-v2
"""

from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine.data import iter_candidates, cid, career_text, skill_text
from engine.embedder import get_embedder, embed_texts, is_using_fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--dimension", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="optional cap for quick testing")
    args = ap.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)

    print(f"[embed_candidates] streaming candidates from {args.candidates} ...")
    ids, careers, skill_texts = [], [], []
    t0 = time.time()
    for i, c in enumerate(iter_candidates(args.candidates)):
        if args.limit is not None and i >= args.limit:
            break
        ids.append(cid(c))
        careers.append(career_text(c))
        skill_texts.append(skill_text(c))
    print(f"[embed_candidates] loaded {len(ids)} candidates in {time.time()-t0:.1f}s")

    # Item 5: pass artifacts_dir so local vendored model weights are found first
    get_embedder(args.model, args.dimension, artifacts_dir=args.artifacts_dir)
    from engine.embedder import ensure_corpus_fitted
    if is_using_fallback():
        ensure_corpus_fitted(careers + skill_texts, artifacts_dir=args.artifacts_dir)
        print(f"[embed_candidates] WARNING: not using the real sentence-transformers model. "
              f"Run scripts/download_model.py for zero-network Tier 1 setup. "
              f"Active fallback tier after fitting: see warnings above.")

    t0 = time.time()
    print(f"[embed_candidates] embedding {len(ids)} career texts ...")
    career_embs = embed_texts(careers, batch_size=args.batch_size, show_progress=True)
    print(f"[embed_candidates] embedding {len(ids)} skill texts ...")
    skill_embs = embed_texts(skill_texts, batch_size=args.batch_size, show_progress=True)
    print(f"[embed_candidates] embedding done in {time.time()-t0:.1f}s")

    career_path = os.path.join(args.artifacts_dir, "career_embeddings.npz")
    skill_path = os.path.join(args.artifacts_dir, "skill_embeddings.npz")

    np.savez_compressed(career_path, ids=np.array(ids), embeddings=career_embs.astype(np.float16))
    np.savez_compressed(skill_path, ids=np.array(ids), embeddings=skill_embs.astype(np.float16))

    print(f"[embed_candidates] wrote {career_path} ({os.path.getsize(career_path)/1e6:.1f} MB)")
    print(f"[embed_candidates] wrote {skill_path} ({os.path.getsize(skill_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
