"""
app.py — Redrob Hackathon Sandbox (HuggingFace Spaces)

Accepts a JSONL file of up to 100 candidates, runs the full 5-channel
ranking pipeline end-to-end, and serves the ranked CSV for download.

Satisfies the hackathon's sandbox requirement (Section 10.5):
  - Accepts a small candidate sample (≤100) as input via upload
  - Runs the ranking system end-to-end
  - Produces a ranked CSV output
  - Completes within the compute budget (no GPU, no network calls at rank time)
"""
import os
import sys
import csv
import json
import tempfile
import time
from datetime import date
from pathlib import Path

import gradio as gr
import numpy as np
import yaml

# Make engine importable
sys.path.insert(0, os.path.dirname(__file__))

from engine.data import iter_candidates, cid, career_text, skill_text, current_title, parse_date
from engine.embedder import get_embedder, embed_texts, cosine_sim_matrix, is_using_fallback, active_tier, ensure_corpus_fitted
from engine.channel_semantic import compute_semantic_scores
from engine.channel_skills import compute_skill_score
from engine.channel_career import compute_career_score
from engine.channel_behavioral import compute_behavioral
from engine.channel_integrity import compute_integrity
from engine.fusion import compute_stuffer_penalty, fuse
from engine.reasoning import generate_reasoning
from engine.skill_synonyms import load_overrides

CONFIG_PATH = "config/jd_config.yaml"
ARTIFACTS_DIR = "artifacts"
MAX_CANDIDATES = 100

# --------------------------------------------------------------------------
# Pre-load config and embedding model once at startup
# --------------------------------------------------------------------------
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

load_overrides(CFG.get("skill_overrides"))

# Boot the embedder (Tier 1 if model weights present, else Tier 2 LSA)
get_embedder(
    CFG["embedding"]["model_name"],
    CFG["embedding"]["dimension"],
    artifacts_dir=ARTIFACTS_DIR,
)
if is_using_fallback():
    lsa_path = os.path.join(ARTIFACTS_DIR, "lsa_model.pkl")
    if os.path.exists(lsa_path):
        ensure_corpus_fitted(None, artifacts_dir=ARTIFACTS_DIR)
    # If no LSA model, it'll be fit on-the-fly from the uploaded candidates

print(f"[sandbox] Embedding tier at startup: {active_tier()}")


# --------------------------------------------------------------------------
# Core ranking function (called per upload)
# --------------------------------------------------------------------------
def rank_candidates(upload_file, participant_id: str) -> tuple[str, str, str]:
    """
    Parameters
    ----------
    upload_file : file-like from Gradio upload
    participant_id : str  — used to name the output CSV

    Returns (status_message, log_text, output_csv_path)
    """
    if upload_file is None:
        return "❌ Please upload a JSONL file.", "", None

    t0 = time.time()
    logs = []
    def log(msg):
        print(msg)
        logs.append(msg)

    # --- Read candidates ---
    try:
        file_path = upload_file.name if hasattr(upload_file, "name") else upload_file
        candidates = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                candidates.append(json.loads(line))
                if len(candidates) >= MAX_CANDIDATES:
                    break
    except Exception as e:
        return f"❌ Failed to read JSONL: {e}", "", None

    n = len(candidates)
    if n == 0:
        return "❌ No valid candidates found in file.", "", None

    log(f"[sandbox] Loaded {n} candidates (max {MAX_CANDIDATES})")

    # --- Embedder boot (fit LSA on-the-fly if not pre-fit) ---
    if is_using_fallback():
        corpus = [career_text(c) for c in candidates] + [skill_text(c) for c in candidates]
        ensure_corpus_fitted(corpus, artifacts_dir=ARTIFACTS_DIR)
        log(f"[sandbox] Embedding tier: {active_tier()} (TF-IDF+LSA, fit on uploaded corpus)")
    else:
        log(f"[sandbox] Embedding tier: {active_tier()} (neural model)")

    # --- Embed ---
    ids = [cid(c) for c in candidates]
    log("[sandbox] Embedding career texts ...")
    career_embs = embed_texts([career_text(c) for c in candidates], show_progress=False)
    log("[sandbox] Embedding skill texts ...")
    skill_embs = embed_texts([skill_text(c) for c in candidates], show_progress=False)

    # --- Channel 1: Semantic ---
    log("[sandbox] Channel 1: semantic relevance ...")
    query_weights = _build_query_weights(CFG)
    sd_cfg = CFG.get("stuffer_detection", {})
    blend_alpha = sd_cfg.get("semantic_blend_alpha", 0.5)
    query_embs = embed_texts(CFG["semantic_queries"])
    semantic_scores, _ = compute_semantic_scores(career_embs, query_embs, query_weights=query_weights, blend_alpha=blend_alpha)
    skill_emb_sims = cosine_sim_matrix(skill_embs, query_embs)
    skill_sim_mean = np.clip((skill_emb_sims.mean(axis=1) + 1.0) / 2.0, 0.0, 1.0)
    career_sim_mean = semantic_scores

    # --- Channel 2: Skills ---
    log("[sandbox] Channel 2: skill match ...")
    skill_clusters = CFG["skill_clusters"]
    skill_scores = np.zeros(n, dtype=np.float32)
    matched_skills_list = []
    skill_channel_details = []
    for i, c in enumerate(candidates):
        res = compute_skill_score(c, skill_clusters)
        skill_scores[i] = res["skill_score"]
        matched_skills_list.append(res["matched_skills"])
        skill_channel_details.append(res)

    core_skill_names_lower = {s.lower() for s in skill_clusters.get("core", [])}

    # --- Channel 3: Career ---
    log("[sandbox] Channel 3: career trajectory ...")
    jd_title = CFG["job_title"]
    jd_title_emb = embed_texts([jd_title])
    title_embs_cache = embed_texts([current_title(c) for c in candidates])
    career_scores = np.zeros(n, dtype=np.float32)
    career_details = []
    for i, c in enumerate(candidates):
        res = compute_career_score(c, CFG["constraints"], jd_title,
                                   title_emb=title_embs_cache[i:i+1],
                                   jd_title_emb=jd_title_emb,
                                   core_skill_names_lower=core_skill_names_lower)
        career_scores[i] = res["career_score"]
        career_details.append(res)

    # --- Channel 4: Behavioral ---
    log("[sandbox] Channel 4: behavioral ...")
    reference_date = date.today()
    behavioral_multipliers = np.zeros(n, dtype=np.float32)
    behavioral_details = []
    for i, c in enumerate(candidates):
        res = compute_behavioral(c, CFG.get("behavioral"), reference_date=reference_date)
        behavioral_multipliers[i] = res["multiplier"]
        behavioral_details.append(res)

    # --- Channel 5: Integrity ---
    log("[sandbox] Channel 5: integrity ...")
    integrity_multipliers = np.zeros(n, dtype=np.float32)
    integrity_details = []
    for i, c in enumerate(candidates):
        res = compute_integrity(c, CFG.get("integrity"),
                                title_emb=title_embs_cache[i:i+1],
                                skill_emb=skill_embs[i:i+1],
                                career_emb=career_embs[i:i+1])
        integrity_multipliers[i] = res["multiplier"]
        integrity_details.append(res)

    # --- Stuffer detection ---
    ramp_width = sd_cfg.get("stuffer_ramp_width", 0.15)
    stuffer_penalty, _ = compute_stuffer_penalty(
        skill_sim_mean, career_sim_mean,
        sd_cfg.get("gap_threshold", 0.4),
        sd_cfg.get("penalty_multiplier", 0.3),
        ramp_width=ramp_width,
    )

    # --- Fusion ---
    raw_score, final_score = fuse(semantic_scores, skill_scores, career_scores,
                                   behavioral_multipliers, integrity_multipliers,
                                   stuffer_penalty, CFG["channel_weights"])

    # --- Score floors for reasoning ---
    score_floors = _compute_score_floors(semantic_scores, skill_scores, career_scores, raw_score)
    role_descriptor = CFG.get("role_descriptor")

    # --- Rank (top min(n, 100)) ---
    top_n = min(n, CFG["output"]["top_n"])
    rounded_scores = [round(float(s), 6) for s in final_score]
    rounded_raw = [round(float(s), 6) for s in raw_score]
    honeypot_mask = [integrity_details[i].get("is_honeypot", False) for i in range(n)]
    raw_order = sorted(range(n), key=lambda i: (-rounded_raw[i], ids[i]))
    pool_idx = [i for i in raw_order if not honeypot_mask[i]][:top_n]
    pool_by_final = sorted(pool_idx, key=lambda i: (-rounded_scores[i], -rounded_raw[i], ids[i]))

    TOP_REORDER_N = min(10, len(pool_by_final))
    top_10_reordered = sorted(pool_by_final[:TOP_REORDER_N],
                               key=lambda i: (-rounded_raw[i], -rounded_scores[i], ids[i]))
    top_idx = top_10_reordered + pool_by_final[TOP_REORDER_N:]

    # Running-minimum for monotone scores
    reported_scores = []
    running_min = float(final_score[top_idx[0]])
    for idx in top_idx:
        s = float(final_score[idx])
        running_min = min(running_min, s)
        reported_scores.append(round(running_min, 6))

    # Tie-break: ascending candidate_id within equal-score segments
    top_idx = list(top_idx)
    i = 0
    while i < len(top_idx):
        j = i + 1
        while j < len(top_idx) and reported_scores[j] == reported_scores[i]:
            j += 1
        if j > i + 1:
            top_idx[i:j] = sorted(top_idx[i:j], key=lambda k: ids[k])
        i = j

    # --- Build rows + reasoning ---
    rows = []
    for rank_pos, (idx, score) in enumerate(zip(top_idx, reported_scores), start=1):
        c = candidates[idx]
        channel_results = {
            "semantic_score": float(semantic_scores[idx]),
            "skill_score": float(skill_scores[idx]),
            "career_score": float(career_scores[idx]),
            "matched_skills": matched_skills_list[idx],
            "behavioral": behavioral_details[idx],
            "integrity": integrity_details[idx],
            "disqualifier_flags": career_details[idx].get("disqualifier_flags", {}),
            "experience_fit": career_details[idx].get("subscores", {}).get("experience_fit", 1.0),
            "experience_yoe": float(raw_score[idx]),
        }
        reasoning = generate_reasoning(c, channel_results, rank=rank_pos,
                                        role_descriptor=role_descriptor,
                                        score_floors=score_floors)
        rows.append({"candidate_id": cid(c), "rank": rank_pos, "score": score, "reasoning": reasoning})

    # --- Write CSV ---
    csv_name = f"{participant_id.strip() or 'submission'}.csv"
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False,
                                      newline="", encoding="utf-8", prefix=csv_name + "_")
    writer = csv.DictWriter(tmp, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()
    writer.writerows(rows)
    tmp.close()

    elapsed = time.time() - t0
    log(f"[sandbox] Done in {elapsed:.1f}s — {len(rows)} candidates ranked")

    honeypots_found = sum(1 for idx in top_idx if integrity_details[idx].get("is_honeypot"))
    status = (
        f"✅ Ranked {len(rows)} candidates in {elapsed:.1f}s\n"
        f"Embedding tier: {active_tier()}\n"
        f"Honeypots blocked: {sum(honeypot_mask)} | In top-{top_n}: {honeypots_found}"
    )
    return status, "\n".join(logs), tmp.name


def _build_query_weights(cfg):
    polarities = cfg.get("query_polarities")
    if not polarities:
        return None
    weight_map = {"must_have": 1.0, "nice_to_have": 0.6, "not_wanted": 0.0, "neutral": 0.8}
    return np.array([weight_map.get(p, 0.8) for p in polarities], dtype=np.float32)


def _compute_score_floors(semantic, skill, career, raw, top_n=500, percentile=25.0):
    idx = np.argsort(raw)[::-1][:min(top_n, len(raw))]
    return {
        "semantic": float(np.percentile(semantic[idx], percentile)),
        "skill": float(np.percentile(skill[idx], percentile)),
        "career": float(np.percentile(career[idx], percentile)),
    }


# --------------------------------------------------------------------------
# Gradio UI
# --------------------------------------------------------------------------
DESCRIPTION = """
## Redrob Hackathon — Candidate Ranking Sandbox

Upload a `.jsonl` file of candidates (up to **100** rows, same schema as the official `candidates.jsonl`).  
The system runs the full **5-channel ranking pipeline** end-to-end:

| Channel | What it measures |
|---|---|
| **Semantic** | Career narrative vs JD-intent queries (neural / LSA embeddings) |
| **Skills** | Credibility-weighted skill match (proficiency × log(duration) × log(endorsements)) |
| **Career** | Title/experience/industry fit + 6 explicit JD disqualifier checks |
| **Behavioral** | Availability multiplier (recency, response rate, notice period) |
| **Integrity** | Honeypot/fabrication detection (timeline, role-duration, zero-duration-expert clusters) |

Output: a submission-ready CSV (`candidate_id, rank, score, reasoning`).

> **Compute**: CPU-only · No network calls at rank time · Typically finishes in < 10s for 100 candidates
"""

with gr.Blocks(title="Redrob Ranking Sandbox", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            upload = gr.File(label="Upload candidates.jsonl (≤ 100 rows)", file_types=[".jsonl"])
            participant_id_box = gr.Textbox(
                label="Participant ID (used as CSV filename)",
                placeholder="e.g. team_xyz",
                value="team_submission",
            )
            run_btn = gr.Button("▶  Run Ranking", variant="primary")

        with gr.Column(scale=2):
            status_box = gr.Textbox(label="Status", lines=4, interactive=False)
            log_box = gr.Textbox(label="Run Log", lines=10, interactive=False)
            output_csv = gr.File(label="Download Ranked CSV")

    run_btn.click(
        fn=rank_candidates,
        inputs=[upload, participant_id_box],
        outputs=[status_box, log_box, output_csv],
    )

    gr.Markdown("""
---
**How to use:**
1. Take a slice of `candidates.jsonl` (up to 100 lines) and upload it.
2. Enter your participant ID.
3. Click **Run Ranking** — results download automatically.

This sandbox is for reproducibility verification only. The full 100K ranking uses the same code via `rank.py` CLI.
    """)

if __name__ == "__main__":
    demo.launch()
