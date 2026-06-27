import sys, csv
sys.stdout.reconfigure(encoding='utf-8')
with open('submission.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
print(f'Total rows: {len(rows)}')
print(f'Score range: {rows[-1]["score"]} (rank 100) --> {rows[0]["score"]} (rank 1)')
print()
print('=== TOP 10 ===')
for r in rows[:10]:
    print(f"Rank {r['rank']:>3}  {r['candidate_id']}  score={r['score']}")
    print(f"     {r['reasoning'][:110]}")
    print()
print('=== RANKS 98-100 ===')
for r in rows[-3:]:
    print(f"Rank {r['rank']:>3}  {r['candidate_id']}  score={r['score']}")
    print(f"     {r['reasoning'][:110]}")
