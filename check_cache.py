import json
with open('artifacts/features.json', 'r') as f:
    data = json.load(f)
candidates = data.get('candidates', data)
has_integrity = sum(1 for v in candidates.values() if 'integrity' in v)
print(f'Candidates with cached integrity: {has_integrity}')
hp = candidates.get('CAND_0039754', {})
print(f'CAND_0039754 cached keys: {list(hp.keys())}')
print(f'CAND_0039754 integrity cached: {"integrity" in hp}')
