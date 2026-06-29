"""
scripts/download_model.py — ONE-TIME MODEL DOWNLOAD

Downloads the sentence-transformers/all-MiniLM-L6-v2 model weights (~90MB)
to artifacts/models/all-MiniLM-L6-v2/ so that rank.py and the precompute
scripts can load from a local path with zero network access.

Run this ONCE on any machine with internet access:
    python scripts/download_model.py --artifacts-dir artifacts

Then copy the entire artifacts/models/ directory to your offline machine.
All subsequent runs (precompute + rank) will load from the local path.

Optional: specify a different model name or output path.
"""

from __future__ import annotations
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-dir", default="artifacts",
                    help="Directory where models/ subdirectory will be created")
    ap.add_argument("--model", default="paraphrase-MiniLM-L3-v2",
                    help="HuggingFace model name to download")
    args = ap.parse_args()

    local_path = os.path.join(args.artifacts_dir, "models", args.model)
    os.makedirs(local_path, exist_ok=True)

    print(f"[download_model] downloading {args.model} to {local_path} ...")
    print(f"[download_model] this is a one-time ~90MB download ...")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[download_model] ERROR: sentence-transformers not installed.")
        print("  Run: pip install sentence-transformers")
        sys.exit(1)

    try:
        model = SentenceTransformer(args.model)
        model.save(local_path)
        print(f"[download_model] model saved to {local_path}")
        print(f"[download_model] directory size: "
              f"{sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(local_path) for f in fs) / 1e6:.1f} MB")
        print(f"[download_model] DONE. Future rank.py runs will load from {local_path} (no network needed).")
    except Exception as e:
        print(f"[download_model] ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
