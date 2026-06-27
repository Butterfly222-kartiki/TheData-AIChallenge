"""
Debug: show scores for specific candidates to diagnose ordering and stuffer detection.
"""
import sys, os
sys.path.insert(0, '.')
import csv, json
import numpy as np

rows = {r['candidate_id']: r for r in csv.DictReader(open('submission.csv'))}

# Check CAND_0041611 and CAND_0092278 (stuck tier-2s)
TARGETS = ['CAND_0041611', 'CAND_0092278', 'CAND_0033861', 'CAND_0093193', 'CAND_0002025',
           'CAND_0006567', 'CAND_0045250', 'CAND_0077337']

print("=== Submission ranks for key candidates ===")
for cid in TARGETS:
    if cid in rows:
        r = rows[cid]
        print(f"  {cid} rank={r['rank']:>3} score={float(r['score']):.4f}  {r['reasoning'][:90]}")
    else:
        print(f"  {cid} MISSING from top 100")

# Load candidates and check stuffer signal for CAND_0041611
print("\n=== Profile inspection: CAND_0041611 ===")
from engine.data import iter_candidates, cid as get_cid, career_text, skill_text, yoe as get_yoe, career_history, total_career_months
from engine.embedder import get_embedder, embed_texts, is_using_fallback, active_tier
import yaml
cfg = yaml.safe_load(open('config/jd_config.yaml'))
get_embedder(cfg['embedding']['model_name'], cfg['embedding']['dimension'], artifacts_dir='artifacts')
print(f"Embedder tier: {active_tier()}")

for c in iter_candidates('India_runs_data_and_ai_challenge/candidates.jsonl'):
    if get_cid(c) != 'CAND_0041611':
        continue
    print(f"  title: {c['profile'].get('current_title')}")
    print(f"  company: {c['profile'].get('current_company')}")
    print(f"  yoe: {get_yoe(c)}")
    print(f"  career_months_sum: {total_career_months(c)}")
    print(f"  skills: {skill_text(c)[:120]}")
    print(f"  career text (first 200): {career_text(c)[:200]}")
    
    # Compute skill vs career sim gap (stuffer signal)
    query_embs = embed_texts(cfg['semantic_queries'])
    skill_emb = embed_texts([skill_text(c)])
    career_emb = embed_texts([career_text(c)])
    from engine.embedder import cosine_sim_matrix
    skill_sim = float(np.clip((cosine_sim_matrix(skill_emb, query_embs).mean() + 1.0) / 2.0, 0, 1))
    career_sim = float(np.clip((cosine_sim_matrix(career_emb, query_embs).mean() + 1.0) / 2.0, 0, 1))
    gap = skill_sim - career_sim
    print(f"  skill_sim_to_JD: {skill_sim:.4f}")
    print(f"  career_sim_to_JD: {career_sim:.4f}")
    print(f"  gap (stuffer signal): {gap:.4f}  [threshold=0.4]")
    print(f"  stuffer_penalty fired: {gap > 0.4}")
    break
