import csv

rows = list(csv.DictReader(open('submission.csv')))

print('=== TOP 15 ===')
for r in rows[:15]:
    print(f"Rank {r['rank']:>3} | {r['candidate_id']} | score={float(r['score']):.4f}")
    print(f"         {r['reasoning'][:140]}")
    print()

print('=== BOTTOM 10 (ranks 91-100) ===')
for r in rows[90:]:
    print(f"Rank {r['rank']:>3} | {r['candidate_id']} | score={float(r['score']):.4f}")
    print(f"         {r['reasoning'][:140]}")
    print()

# --- Quality checks ---
reasonings = [r['reasoning'] for r in rows]
scores = [float(r['score']) for r in rows]

print('=== QUALITY CHECKS ===')
print(f"Unique reasoning strings: {len(set(reasonings))} / {len(reasonings)}")
print(f"Old 'closely aligns with' template hits: {sum(1 for r in reasonings if 'closely aligns with' in r)}")
print(f"Concern clauses present: {sum(1 for r in reasonings if 'Note:' in r)}")
print(f"Experience-band concerns: {sum(1 for r in reasonings if 'seniority band' in r or 'seniority threshold' in r)}")
print(f"Score non-increasing violations: {sum(1 for i in range(len(scores)-1) if scores[i] < scores[i+1])}")

# Check the specific candidates called out in feedback
print()
print('=== SPECIFIC CANDIDATES FROM FEEDBACK ===')
ids_to_find = {
    'CAND_0007411': 'Amazon SML 8yr (was evicted by low resp rate)',
    'CAND_0018549': 'Uber RecSys (was rank 84 before)',
    'CAND_0000031': 'Swiggy RecSys (was rank 100 before)',
    'CAND_0009024': 'Google Search Engineer (was missing)',
    'CAND_0039754': 'Honeypot (should NOT be in top 100)',
    'CAND_0046064': 'Salesforce NLP 8.9yr (rank 84 last time)',
}
found = {r['candidate_id']: r for r in rows}
for cid, desc in ids_to_find.items():
    if cid in found:
        r = found[cid]
        print(f"  FOUND rank {r['rank']:>3} | {cid} ({desc})")
        print(f"           {r['reasoning'][:130]}")
    else:
        print(f"  MISSING       | {cid} ({desc})")
