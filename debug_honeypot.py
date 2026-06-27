import sys, json
sys.path.insert(0, '.')

from engine.data import iter_candidates, cid, total_career_months, yoe as get_yoe, career_history
from engine.channel_integrity import compute_integrity, check_timeline_consistency, DEFAULT_PARAMS

TARGET = 'CAND_0039754'

for c in iter_candidates('India_runs_data_and_ai_challenge/candidates.jsonl'):
    if cid(c) != TARGET:
        continue

    yoe = get_yoe(c)
    cm = total_career_months(c)
    claimed = yoe * 12
    diff = abs(cm - claimed)

    print(f"=== {TARGET} ===")
    print(f"  yoe (from profile): {yoe}")
    print(f"  career_months (sum of duration_months): {cm}")
    print(f"  claimed_months (yoe * 12): {claimed:.1f}")
    print(f"  abs diff: {diff:.1f} months")
    print(f"  tolerance: {DEFAULT_PARAMS['timeline_consistency_tolerance_months']}")
    print(f"  check_timeline_consistency PASSES: {check_timeline_consistency(c, DEFAULT_PARAMS['timeline_consistency_tolerance_months'])}")
    print()

    print("  Career history roles:")
    for r in career_history(c):
        print(f"    {r.get('title','?')} @ {r.get('company','?')} — {r.get('duration_months','?')} months | is_current={r.get('is_current')}")
    print()

    # Full integrity check (no embeddings - semantic checks skip gracefully)
    result = compute_integrity(c)
    print(f"  integrity result: n_failed={result['n_failed']}, is_honeypot={result['is_honeypot']}")
    print(f"  checks: {result['checks']}")
    print(f"  multiplier: {result['multiplier']}")
    break
