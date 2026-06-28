"""
tests/test_smoke.py

Lightweight smoke tests for the core formulas against the REAL Redrob
candidate schema. Run directly with: python tests/test_smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.channel_skills import skill_credibility, compute_skill_score
from engine.channel_behavioral import compute_behavioral
from engine.channel_integrity import (
    check_timeline_consistency, check_role_duration,
    check_proficiency_duration_coherence, check_zero_duration_expert_cluster,
    compute_integrity,
)
from engine.channel_career import experience_fit_score, notice_fit_score, check_pure_consulting_career
from engine.skill_synonyms import canonicalize
from datetime import date


def _candidate(**overrides):
    base = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "current_title": "ML Engineer", "current_industry": "Product",
            "current_company_size": "51-200", "years_of_experience": 5.0,
            "current_company": "TestCo", "headline": "", "summary": "",
            "location": "Pune, MH", "country": "India",
        },
        "career_history": [],
        "skills": [],
        "redrob_signals": {
            "last_active_date": "2026-06-01", "open_to_work_flag": True,
            "notice_period_days": 30, "recruiter_response_rate": 0.5,
            "saved_by_recruiters_30d": 10, "interview_completion_rate": 0.8,
            "verified_email": True, "verified_phone": True, "linkedin_connected": True,
            "profile_completeness_score": 80, "willing_to_relocate": True,
            "github_activity_score": 50,
        },
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and k in base:
            base[k].update(v)
        else:
            base[k] = v
    return base


def test_credibility_zeroes_out_stuffer_pattern():
    # advanced proficiency, low duration, ZERO endorsements (real stuffer pattern found in data)
    cred = skill_credibility({"proficiency": "advanced", "duration_months": 11, "endorsements": 0})
    assert cred == 0.0, f"expected 0.0, got {cred}"
    print("PASS: zero-endorsement stuffer-pattern skill has 0 credibility")


def test_credibility_rewards_genuine_depth():
    # real genuine-expert pattern found in data: expert, 72-92mo, 16-18 endorsements
    cred_genuine = skill_credibility({"proficiency": "expert", "duration_months": 72, "endorsements": 16})
    cred_stuffer = skill_credibility({"proficiency": "advanced", "duration_months": 11, "endorsements": 0})
    assert cred_genuine > cred_stuffer
    print(f"PASS: genuine ({cred_genuine:.2f}) > stuffer ({cred_stuffer:.2f})")


def test_skill_synonym_canonicalization():
    assert canonicalize("Vector Representations") == "Embeddings"
    assert canonicalize("Information Retrieval Systems") == "Information Retrieval"
    assert canonicalize("Python") == "Python"  # unchanged passthrough
    print("PASS: rare plain-language skill variants canonicalize correctly")


def test_timeline_consistency_catches_real_honeypot_pattern():
    # real pattern found in dataset (CAND_0007353-style): yoe=9.9, career
    # history totalling 251 months (diff=132, far past the 60mo tolerance)
    c = _candidate(profile={"years_of_experience": 9.9},
                    career_history=[{"duration_months": 251, "company": "X", "industry": "Product"}])
    assert check_timeline_consistency(c, 60) is False
    assert check_role_duration(c, 12) is False
    print("PASS: real honeypot timeline pattern (9.9 yoe, 251mo career) flagged on both checks")


def test_zero_duration_expert_cluster_catches_real_pattern():
    # real pattern found in dataset: exactly 5 expert skills with duration_months=0
    skills = [{"name": f"Skill{i}", "proficiency": "expert", "duration_months": 0, "endorsements": 0}
              for i in range(5)]
    c = _candidate(skills=skills)
    assert check_proficiency_duration_coherence(c) is False
    assert check_zero_duration_expert_cluster(c, 3) is False
    print("PASS: zero-duration expert skill cluster (5 skills) fails BOTH integrity checks")


def test_zero_duration_expert_cluster_does_not_overfire_on_single_skill():
    skills = [{"name": "Skill0", "proficiency": "expert", "duration_months": 0, "endorsements": 0}]
    c = _candidate(skills=skills)
    assert check_proficiency_duration_coherence(c) is False     # fails the basic check
    assert check_zero_duration_expert_cluster(c, 3) is True      # but NOT the cluster check
    print("PASS: a single zero-duration expert skill does not trigger the stronger cluster check")


def test_behavioral_multiplier_floor_and_ceiling():
    good_sig = {"last_active_date": "2026-06-20", "open_to_work_flag": True, "notice_period_days": 0,
                "recruiter_response_rate": 0.9, "saved_by_recruiters_30d": 80, "interview_completion_rate": 1.0,
                "verified_email": True, "verified_phone": True, "linkedin_connected": True,
                "profile_completeness_score": 100}
    bad_sig = {"last_active_date": "2025-08-01", "open_to_work_flag": False, "notice_period_days": 150,
               "recruiter_response_rate": 0.05, "saved_by_recruiters_30d": 0, "interview_completion_rate": 0.1,
               "verified_email": False, "verified_phone": False, "linkedin_connected": False,
               "profile_completeness_score": 10}
    good = _candidate(redrob_signals=good_sig)
    bad = _candidate(redrob_signals=bad_sig)
    ref = date(2026, 6, 22)
    good_mult = compute_behavioral(good, reference_date=ref)["multiplier"]
    bad_mult = compute_behavioral(bad, reference_date=ref)["multiplier"]
    assert good_mult > 0.9, good_mult
    # Floor is now 0.75 (raised from 0.3): behavioral nudges max 25%, never evicts tier-5s
    assert bad_mult >= 0.75, f"floor=0.75 should keep even bad profiles >= 0.75, got {bad_mult}"
    assert bad_mult < good_mult, f"good ({good_mult}) should beat bad ({bad_mult})"
    print(f"PASS: good behavioral multiplier={good_mult:.3f}, bad={bad_mult:.3f} (floor=0.75)")


def test_experience_fit_peaks_at_center():
    peak = experience_fit_score(7, 7, 5, 9, 3, 14)
    off = experience_fit_score(15, 7, 5, 9, 3, 14)
    assert peak > off and peak > 0.99
    print(f"PASS: experience fit peak={peak:.3f} > off-center={off:.4f}")


def test_notice_fit_decays():
    near = notice_fit_score(5, 30)
    far = notice_fit_score(150, 30)
    assert near > far
    print(f"PASS: notice fit near={near:.3f} > far={far:.3f}")


def test_pure_consulting_career_penalty():
    career = [
        {"company": "TCS", "industry": "IT Services", "duration_months": 36},
        {"company": "Infosys", "industry": "IT Services", "duration_months": 24},
    ]
    flagged = check_pure_consulting_career(
        career, "IT Services", "TCS", ["IT Services", "Consulting"], ["TCS", "Infosys", "Wipro"])
    assert flagged is True
    not_flagged = check_pure_consulting_career(
        [{"company": "TCS", "industry": "IT Services", "duration_months": 12},
         {"company": "Razorpay", "industry": "Fintech", "duration_months": 36}],
        "Fintech", "Razorpay", ["IT Services", "Consulting"], ["TCS"])
    assert not_flagged is False
    print("PASS: pure-consulting-career flagged only when ENTIRE career is consulting")


def test_full_skill_channel_integration():
    skill_clusters = {
        "core": ["Information Retrieval", "Embeddings"],
        "secondary": ["Python"], "nice_to_have": ["AWS"],
        "domain_mismatch_anti_skills": ["Computer Vision"],
        "business_mismatch_anti_skills": ["Sales"],
    }
    c = _candidate(skills=[
        {"name": "Vector Representations", "proficiency": "expert", "duration_months": 60, "endorsements": 30},
        {"name": "Python", "proficiency": "advanced", "duration_months": 40, "endorsements": 10},
    ])
    res = compute_skill_score(c, skill_clusters)
    assert "Vector Representations" in res["matched_skills"]["core"], \
        "synonym 'Vector Representations' should match core 'Embeddings'"
    assert res["skill_score"] > 0
    print(f"PASS: synonym-aware skill matching works end-to-end, skill_score={res['skill_score']}")


def test_honeypot_CAND0039754_caught():
    """CAND_0039754: claims 16.2 yoe but career history sums to 98 months (8.2 yr).
    Gap = 96.4 months. Both timeline_consistency (>30) and severe_timeline_gap
    (>90) must fire => n_failed>=2 => is_honeypot=True.
    This test was added after the candidate appeared at rank 55 because the
    config still had tolerance=60, which let severe_gap pass (96<180)."""
    c = _candidate(profile={"years_of_experience": 16.2})
    c["career_history"] = [
        {"title": "Sr Applied Scientist", "company": "Meta",
         "duration_months": 37, "is_current": True},
        {"title": "Sr ML Engineer", "company": "Apple",
         "duration_months": 40, "is_current": False},
        {"title": "Sr Applied Scientist", "company": "Observe.AI",
         "duration_months": 21, "is_current": False},
    ]
    # Use corrected config params (as they now appear in jd_config.yaml)
    params = {
        "timeline_consistency_tolerance_months": 30,
        "severe_timeline_gap_multiplier": 3,
        "honeypot_threshold": 2,
        "failure_multipliers": {"0": 1.0, "1": 0.7, "2+": 0.0},
    }
    res = compute_integrity(c, params)
    assert not res["checks"]["timeline_consistency"], \
        f"timeline_consistency should FAIL: gap=96.4m > 30m tolerance"
    assert not res["checks"]["severe_timeline_gap"], \
        f"severe_timeline_gap should FAIL: gap=96.4m > 90m severe threshold"
    assert res["n_failed"] >= 2, f"expected n_failed>=2, got {res['n_failed']}"
    assert res["is_honeypot"], f"expected is_honeypot=True, got False"
    assert res["multiplier"] == 0.0, f"expected multiplier=0.0, got {res['multiplier']}"
    print(f"PASS: CAND_0039754 correctly flagged as honeypot (n_failed={res['n_failed']})")


def test_honeypot_blocked_by_old_tolerance():
    """Regression: with the OLD tolerance=60, the 96-month gap only tripped
    timeline_consistency (n_failed=1) and severe_gap used 60*3=180 threshold
    (96<180 => pass). The candidate got multiplier=0.7, not 0.0.
    This test asserts the OLD behavior no longer exists."""
    c = _candidate(profile={"years_of_experience": 16.2})
    c["career_history"] = [
        {"title": "Sr Applied Scientist", "company": "Meta",
         "duration_months": 37, "is_current": True},
        {"title": "Sr ML Engineer", "company": "Apple",
         "duration_months": 40, "is_current": False},
        {"title": "Sr Applied Scientist", "company": "Observe.AI",
         "duration_months": 21, "is_current": False},
    ]
    from engine.channel_integrity import DEFAULT_PARAMS
    # Verify the live DEFAULT_PARAMS has the fixed tolerance
    assert DEFAULT_PARAMS["timeline_consistency_tolerance_months"] == 30, \
        (f"DEFAULT_PARAMS tolerance is {DEFAULT_PARAMS['timeline_consistency_tolerance_months']}, "
         f"expected 30. The config override bug: jd_config.yaml must also say 30.")
    assert DEFAULT_PARAMS.get("severe_timeline_gap_multiplier") == 3, \
        "severe_timeline_gap_multiplier must be 3 in DEFAULT_PARAMS"
    print("PASS: DEFAULT_PARAMS has corrected tolerance=30 and severe_multiplier=3")


def test_clean_profile_not_honeypot():
    """A real profile with a small legitimate gap (~5 months) must NOT be flagged."""
    c = _candidate(profile={"years_of_experience": 5.0})
    c["career_history"] = [
        {"title": "ML Engineer", "company": "Startup",
         "duration_months": 55, "is_current": True},  # 5yr - 5 months = fine
    ]
    params = {
        "timeline_consistency_tolerance_months": 30,
        "severe_timeline_gap_multiplier": 3,
        "honeypot_threshold": 2,
        "failure_multipliers": {"0": 1.0, "1": 0.7, "2+": 0.0},
    }
    res = compute_integrity(c, params)
    assert res["checks"]["timeline_consistency"], \
        f"Clean profile should PASS timeline_consistency (gap=5m < 30m)"
    assert res["checks"]["severe_timeline_gap"], \
        f"Clean profile should PASS severe_timeline_gap (gap=5m < 90m)"
    assert not res["is_honeypot"], \
        f"Clean profile should NOT be honeypot (n_failed={res['n_failed']})"
    print(f"PASS: clean profile with 5-month gap correctly NOT flagged (n_failed={res['n_failed']})")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"FAIL: {t.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR: {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    sys.exit(1 if failures else 0)
