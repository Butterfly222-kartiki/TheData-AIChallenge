---
title: Redrob Ranking Sandbox
emoji: 🏆
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
python_version: "3.11"
pinned: false
license: mit
short_description: "5-channel candidate ranking pipeline for Redrob Hackathon v4"
---

# Redrob Hackathon — Senior AI Engineer Ranking System

A JD-agnostic, five-channel candidate ranking engine, built and calibrated
against the **real** `candidates.jsonl` (100,000 candidates) and the real
`job_description.docx` / `redrob_signals_doc.docx` / `submission_spec.docx`
for this hackathon. `config/jd_config.yaml` is the only file that knows
anything about this specific job — every scoring function in `engine/` is
universal.

**This has been run end-to-end against the real 100K-candidate file and
validated with the organizers' own `validate_submission.py` (passes).**
See "What we actually verified" below for the numbers.

---

## Quickstart

```bash
pip install -r requirements.txt

python precompute/embed_candidates.py --candidates /path/to/candidates.jsonl --artifacts-dir artifacts
python precompute/extract_features.py --candidates /path/to/candidates.jsonl --artifacts-dir artifacts --integrity-config config/jd_config.yaml
python rank.py --candidates /path/to/candidates.jsonl --config config/jd_config.yaml --artifacts-dir artifacts --out submission.csv

python /path/to/validate_submission.py submission.csv   # the organizers' own checker
python evaluate.py --submission submission.csv --candidates /path/to/candidates.jsonl --config config/jd_config.yaml
```

`submission.csv` in this repo is real output from a full run against the
actual 100,000-candidate file (see below for caveats on embedding quality).

---

## Project structure

```
project/
├── config/jd_config.yaml        ← ONLY job-specific file, grounded in the real JD + real skill vocabulary
├── engine/
│   ├── data.py                   ← real schema field accessors (profile.*, career_history[], redrob_signals.*)
│   ├── skill_synonyms.py         ← rare plain-language skill-name variants found in the real data
│   ├── embedder.py               ← 3-tier embedding backend (real model / TF-IDF+LSA / hash fallback)
│   ├── channel_semantic.py       ← Channel 1: career-text vs JD-intent relevance
│   ├── channel_skills.py         ← Channel 2: credibility-weighted skill match
│   ├── channel_career.py         ← Channel 3: trajectory fit + 6 explicit JD disqualifier checks
│   ├── channel_behavioral.py     ← Channel 4: availability (multiplicative)
│   ├── channel_integrity.py      ← Channel 5: honeypot detection (multiplicative)
│   ├── fusion.py                 ← combines channels + stuffer detection
│   └── reasoning.py              ← grounded, hallucination-free per-candidate justification
├── precompute/
│   ├── embed_candidates.py       ← career/skill embeddings, streamed (handles the full 487MB file)
│   └── extract_features.py       ← skill credibility, integrity, behavioral — with ADAPTIVE coherence thresholds
├── sample_data/
│   ├── extract_real_sample.py    ← pulls a real-data sample (not synthetic) for fast local testing
│   ├── candidates_sample.jsonl   ← a 3,000-candidate real sample, including known trap-type IDs
│   └── submission_sample3000.csv ← output of a full run against that sample
├── rank.py                       ← MAIN ENTRY POINT
├── evaluate.py                   ← self-evaluation framework (no leaderboard during the competition)
├── tests/test_smoke.py           ← unit tests using real-data-calibrated fixtures
├── submission.csv                ← REAL output: full run against all 100,000 actual candidates
├── submission_metadata.yaml      ← filled out against the official template (fill in TODOs before submitting)
└── requirements.txt
```

---

## Embedding tiers — read this before you submit

`engine/embedder.py` has three tiers, tried in order:

1. **Real model** (`sentence-transformers/all-MiniLM-L6-v2`). True neural
   semantic embeddings. Needs `pip install sentence-transformers` and a
   one-time internet connection to download model weights — this happens
   during pre-computation, not during `rank.py` itself, so it doesn't
   violate the "no network during ranking" rule (see submission_spec).
2. **TF-IDF + LSA fallback** (`scikit-learn`, fit on the candidate corpus
   itself). Activates automatically if Tier 1 is unavailable. This is a
   legitimate, CPU-only, no-internet-ever-required semantic-lite technique
   (latent semantic analysis) — not a toy. **All the results documented
   below were produced with this tier**, because the sandbox this was
   built in could not reach huggingface.co. It is meaningfully better than
   keyword matching but weaker than the real model, especially for very
   short text comparisons.
3. **Hash-projection fallback** (last resort, no real semantics at all).
   Only activates if scikit-learn itself is unavailable.

**Before a real submission**, install `sentence-transformers`, make sure
you have internet access during the precompute step (or pre-download the
model on a machine with internet and copy your `~/.cache/huggingface`
folder over), delete `artifacts/lsa_model.pkl` if present, and re-run the
two precompute scripts. Quality with the real model should meet or exceed
everything documented below.

---

## What we actually verified (real data, real validator)

Everything in this section was measured, not estimated, by running this
code against the real `candidates.jsonl` and the organizers'
`validate_submission.py`, using the **Tier 2 (TF-IDF+LSA)** embedding
backend (Tier 1 unavailable in the build environment — see above).

**Format correctness:**
```
$ python validate_submission.py submission.csv
Submission is valid.
```

**Performance at full 100K scale:**

| Step | Time |
|---|---|
| `embed_candidates.py` (100,000 candidates) | ~53s |
| `extract_features.py` (100,000 candidates, incl. adaptive threshold calibration) | ~17s |
| `rank.py` (100,000 candidates to top 100) | ~27s |

All well inside the 5-minute ranking budget, with room to spare even if
Tier 1 (the real model) is several times slower than the LSA fallback.

**Quality at full 100K scale (top 100 of 100,000):**
- **0 honeypots** — every candidate in the top 100 passed all 5 integrity
  checks (verified against the dataset's actual honeypot population, not
  a guess — see "Calibration notes" below).
- **0 obvious keyword-stuffers** — zero candidates with a non-technical
  title (HR Manager, Content Writer, Sales Executive, etc.) made the top 100.
- **15 of the dataset's 25 most senior AI-titled candidates** (Senior/Lead/
  Staff AI/ML/NLP Engineer) appear in the top 100, several in the top 10.
- Top-10 reasoning reads like genuinely strong matches: "Lead AI Engineer
  at Razorpay," "Owned the end-to-end ranking pipeline at Meta," "RAG-based
  ranking pipeline serving 50M+ queries/month" — not generic noise.
- Score gradient from rank 1 (0.48) to rank 100 (0.35) is smooth, and even
  rank 90-100 profiles are thematically on-topic (ranking/RAG/recommendation
  work), not noise that snuck in.

**On a 3,000-candidate real-data sample** (`sample_data/candidates_sample.jsonl`,
deliberately seeded with known stuffer/honeypot/elite IDs found by direct
inspection): all 3 of the genuinely elite/plain-language-tier-5 seeded IDs
made the top 100; the known stuffer IDs did not.

---

## Calibration notes (why the thresholds are what they are)

Numbers below come from **directly inspecting** the real `candidates.jsonl`
— not the architecture document's illustrative examples.

**Skill vocabulary (133 unique skill names total).** There's a clear
three-tier structure:
- A ~5,000-candidate "AI buzzword" pool (Pinecone, FAISS, RAG, Embeddings,
  but also Computer Vision/Speech/GANs — a mix of genuinely-relevant and
  domain-mismatched terms). This is the keyword-stuffer bait pool.
- A ~1,350-candidate "genuine ML practitioner" pool (Python, PyTorch,
  QLoRA, Weaviate, BM25, Elasticsearch, etc.) — deeper, rarer terms.
- A handful of **single-digit-frequency plain-language variants**
  ("Vector Representations", "Information Retrieval Systems", "Content
  Matching") — the literal Tier-5 trap. Handled via `engine/skill_synonyms.py`.

**Skill credibility formula validated directly:** real keyword-stuffer
profiles (e.g. an HR Manager with 6+ core AI skills) have those skills at
**0-4 endorsements and 6-17 months duration**. Real genuine experts have
**16-40 endorsements and 35-95 months duration** on the same skill names.
`credibility = proficiency_weight * log(1+duration) * log(1+endorsements)`
spreads these two populations by roughly two orders of magnitude per skill
— confirmed empirically, not assumed.

**Integrity thresholds calibrated against the actual honeypot population:**
- `timeline_consistency` (career_months minus yoe times 12 > 60): cleanly
  isolates ~14 candidates with gaps of 88-185 months; everyone else
  clusters near 0.
- `role_duration` (single role > yoe times 12 plus 12): isolates ~6 more
  candidates with single roles of 100-228 months against a much lower
  claimed yoe.
- **Zero-duration "expert" skills**: exactly 21 individual occurrences in
  the real data, several clustered as **exactly 5 such skills on one
  profile** — clearly a deliberately injected pattern, not noise. A
  dedicated `zero_duration_expert_cluster` check (3+ on one profile) was
  added specifically because this pattern is much stronger evidence than
  a single zero-duration skill, which alone is more ambiguous.
- `skill_duration` (skill duration vs yoe) was found to be **mostly noise**
  in the real data — even ordinary, non-honeypot profiles routinely have
  skill durations exceeding total claimed experience (e.g. picking up a
  skill at a previous job before a later career restart). This check is
  kept generous (48-month buffer) specifically to avoid false positives;
  it's not a strong signal in this dataset.
- The two **semantic** coherence checks (title-skill, skill-career) use an
  **adaptive percentile threshold** computed from the empirical similarity
  distribution across the whole pool (default: 5th percentile), not a
  fixed absolute number. We found a fixed 0.25 threshold (reasonable for
  real neural embeddings) badly misfires under the LSA fallback, where
  short-text similarity sits on a different scale — flagging genuinely
  strong candidates as incoherent. The adaptive threshold self-calibrates
  to whichever embedding backend is actually active.

**JD disqualifiers** (`engine/channel_career.py`) are lifted directly from
job_description.docx's "things we explicitly do NOT want" section: title-
chasing (3+ short stints with escalating seniority titles), tech-lead
drift (architecture/management title held 18+ months), pure-research-with-
no-production-language, shallow/recent-only AI skill exposure, an entirely
consulting-firm career (career-wide check, not current-role-only — someone
*currently* at a consulting firm but *previously* at a product company is
NOT penalized), and a soft penalty for senior candidates with zero visible
GitHub/open-source signal.

---

## Re-running on a different JD

1. Edit `config/jd_config.yaml` — semantic_queries, skill_clusters
   (re-derive these from your own dataset's actual skill vocabulary rather
   than guessing — see `sample_data/extract_real_sample.py` for an example
   of how we did this kind of inspection), constraints, disqualifiers.
2. Re-run the three commands in Quickstart.
3. Run `evaluate.py` — its checks call out stuffer leakage, dead/inactive
   profiles, and integrity failures so you can iterate on the config
   without a leaderboard.

---

## Tie-break / format correctness

`validate_submission.py` requires: exactly 100 rows, header
`candidate_id,rank,score,reasoning`, ranks 1-100 unique, score
non-increasing by rank, and **ties broken by candidate_id ascending**.
`rank.py` sorts directly on the *rounded* score (not full precision, then
rounded after) specifically because that ordering is what mathematically
guarantees both the monotonicity and tie-break rules simultaneously — an
earlier version of this code rounded after sorting and occasionally
violated the tie-break rule on rounding-induced ties. `evaluate.py`'s
Check 0 mirrors the validator's exact rules if you want to sanity-check
before running the official script.

---

## Reasoning generator

`engine/reasoning.py` only fills templates with fields read directly from
the candidate's own record (title, company, career description snippet,
matched skill names) — it never invents facts, and surfaces real concerns
(notice period, low responsiveness, disqualifier flags) when they're
present. Preserve this constraint if you adapt the templates; hallucinated
reasoning is explicitly checked in the organizers' review stages.
