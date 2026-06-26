"""
jd_compiler.py — JOB DESCRIPTION COMPILER

Reads raw JD text and emits a complete jd_config.yaml automatically.
Point it at any JD text file and it compiles a ranking spec.

Algorithm:
  1. Section splitting — splits JD on header patterns, assigns each
     section a polarity: must_have | nice_to_have | not_wanted | neutral.
  2. Sentence polarity classification — within each section, each sentence
     gets a polarity from (a) its section header and (b) modal verb
     override patterns (so a "must" sentence inside a neutral section
     still becomes must_have).
  3. Semantic queries — must_have sentences used verbatim as Channel 1 queries.
  4. Structured constraint extraction — regex patterns for experience
     ranges, cities, notice periods, job title.
  5. Disqualifier enablement — each of the 6 known disqualifiers has an
     anchor phrase; if the not_wanted sentences are semantically close to
     that anchor (cosine > 0.45 at compile time), the disqualifier gets
     enabled: true. Otherwise enabled: false.
  6. Role descriptor — first must_have sentence trimmed to ≤80 chars,
     written as role_descriptor in config (reasoning.py uses it instead
     of the old hardcoded phrase).
  7. Skill clusters — emitted as empty lists with a comment; the separate
     precompute/derive_skill_clusters.py step fills them via embeddings.

Usage:
    python jd_compiler.py --jd /path/to/job_description.txt \\
                          --out config/jd_config.yaml \\
                          [--candidates /path/to/candidates.jsonl] \\
                          [--base-config config/jd_config.yaml]

    The --candidates flag lets the compiler verify city names against the
    pool's actual location values. Without it, city extraction still works
    but skips pool-membership validation.

    The --base-config flag lets you start from an existing config and only
    override the JD-derived sections, preserving manually tuned weights.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
import textwrap
from typing import Literal

import yaml

# ---------------------------------------------------------------------------
# Polarity types
# ---------------------------------------------------------------------------
Polarity = Literal["must_have", "nice_to_have", "not_wanted", "neutral"]

# ---------------------------------------------------------------------------
# Section header → polarity mapping
# Headers are matched case-insensitively anywhere in the line.
# ---------------------------------------------------------------------------
_SECTION_HEADERS: list[tuple[re.Pattern, Polarity]] = [
    # must_have patterns
    (re.compile(r"absolutely\s+need|must\s+have|required|non.negotiable|hard\s+require", re.I), "must_have"),
    (re.compile(r"what\s+you('ll|'ll)?\s+(need|bring|have)|key\s+require|core\s+require", re.I), "must_have"),
    (re.compile(r"things?\s+we\s+(need|look\s+for|require)|you\s+must", re.I), "must_have"),
    # nice_to_have patterns
    (re.compile(r"nice\s+to\s+have|won'?t?\s+reject|bonus|would\s+love|prefer(red)?|good\s+to\s+have", re.I), "nice_to_have"),
    (re.compile(r"ideal\s+but\s+not|plus\s+but\s+not\s+required|desirable", re.I), "nice_to_have"),
    # not_wanted patterns
    (re.compile(r"do\s+not\s+want|don'?t?\s+want|things?\s+we\s+(do\s+not|don'?t?)\s+want", re.I), "not_wanted"),
    (re.compile(r"explicitly\s+do\s+not|not\s+looking\s+for|we\s+will\s+not\s+consider", re.I), "not_wanted"),
    (re.compile(r"disqualif|red\s+flag|avoid|NOT\s+a\s+fit", re.I), "not_wanted"),
]

# ---------------------------------------------------------------------------
# Sentence-level modal verb overrides
# These can flip a sentence's polarity regardless of section.
# ---------------------------------------------------------------------------
_MODAL_MUST: re.Pattern = re.compile(
    r"\b(must|require[sd]?|essential|need to|mandatory|critical|non.negotiable)\b", re.I
)
_MODAL_NICE: re.Pattern = re.compile(
    r"\b(prefer|nice to have|bonus|ideally|would love|plus|desirable|good to have)\b", re.I
)
_MODAL_NOT: re.Pattern = re.compile(
    r"\b(do not want|don'?t want|avoid|will not|won'?t consider|not a fit|red flag)\b", re.I
)

# ---------------------------------------------------------------------------
# Known disqualifier anchor phrases (used for compile-time enablement)
# ---------------------------------------------------------------------------
_DISQUALIFIER_ANCHORS: dict[str, str] = {
    "title_chase": "switching jobs every year chasing seniority titles without substance",
    "tech_lead_drift": "moved to architecture or management and hasn't written production code recently",
    "pure_research_no_production": "purely academic or research background without shipping to real users",
    "shallow_ai_recent_only": "recent AI experience only from LangChain or OpenAI wrappers without depth",
    "pure_consulting_career": "entire career at IT services or consulting firms without product company work",
    "closed_source_no_validation": "senior engineer with no open source or GitHub presence to validate claims",
}

# ---------------------------------------------------------------------------
# Constraint extraction patterns
# ---------------------------------------------------------------------------
_EXP_RANGE_RE = re.compile(
    r"(\d+)\s*[-–−to]+\s*(\d+)\s*(?:years?|yrs?|yoe)", re.I
)
_EXP_SINGLE_RE = re.compile(r"(\d+)\+?\s*(?:years?|yrs?|yoe)\s+(?:of\s+)?(?:experience|exp)", re.I)
_NOTICE_RE = re.compile(r"(?:sub[- ]?|under\s+|within\s+|(?:less|no more) than\s+)(\d+)[- ]?day", re.I)
_NOTICE_ALT_RE = re.compile(r"(\d+)[- ]?day\s+notice", re.I)

# City names we know about — validated against pool if --candidates provided
_KNOWN_CITIES = [
    "Pune", "Noida", "Hyderabad", "Mumbai", "Delhi", "Bangalore", "Bengaluru",
    "Chennai", "Kolkata", "Ahmedabad", "Jaipur", "Surat", "Lucknow", "Chandigarh",
    "Gurgaon", "Gurugram", "Indore", "Bhopal", "Vadodara", "Coimbatore",
    "Kochi", "Thiruvananthapuram", "Nagpur", "Visakhapatnam", "Patna",
]
_WELCOME_CITIES_DEFAULT = ["Hyderabad", "Mumbai", "Delhi", "Bangalore"]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving reasonable chunks."""
    # Split on sentence-ending punctuation followed by whitespace + capital
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\-•])", text)
    # Also split on bullet-point lines
    result = []
    for p in parts:
        for line in p.split("\n"):
            line = re.sub(r"^\s*[-•*–]\s*", "", line).strip()
            if len(line) > 20:
                result.append(line)
    return result


def _section_polarity(header: str) -> Polarity:
    """Determine polarity from a section header string."""
    for pattern, polarity in _SECTION_HEADERS:
        if pattern.search(header):
            return polarity
    return "neutral"


def _sentence_polarity(sentence: str, section_polarity: Polarity) -> Polarity:
    """Refine sentence polarity using modal verb overrides."""
    if _MODAL_NOT.search(sentence):
        return "not_wanted"
    if _MODAL_MUST.search(sentence) and section_polarity not in ("not_wanted", "nice_to_have"):
        return "must_have"
    if _MODAL_NICE.search(sentence) and section_polarity == "neutral":
        return "nice_to_have"
    return section_polarity if section_polarity != "neutral" else "must_have"


def _is_header_line(line: str) -> bool:
    """True if the line looks like a section header (short, no period)."""
    stripped = line.strip()
    if not stripped:
        return False
    # Markdown headers
    if stripped.startswith("#"):
        return True
    # All caps short line
    if stripped.isupper() and len(stripped) < 80:
        return True
    # Ends with colon (section label)
    if stripped.endswith(":") and len(stripped) < 80:
        return True
    # Bold markdown
    if stripped.startswith("**") and stripped.endswith("**") and len(stripped) < 80:
        return True
    return False


def parse_jd(jd_text: str) -> dict[str, list[str]]:
    """
    Parse JD text into bucketed sentences by polarity.
    Returns: {"must_have": [...], "nice_to_have": [...], "not_wanted": [...]}
    """
    lines = jd_text.split("\n")
    current_section_polarity: Polarity = "neutral"
    sentences_by_polarity: dict[str, list[str]] = {
        "must_have": [], "nice_to_have": [], "not_wanted": []
    }

    current_para: list[str] = []

    def flush_para():
        nonlocal current_para
        para_text = " ".join(current_para).strip()
        current_para = []
        if not para_text:
            return
        for sent in _split_sentences(para_text):
            sent = sent.strip()
            if len(sent) < 20:
                continue
            pol = _sentence_polarity(sent, current_section_polarity)
            if pol in sentences_by_polarity:
                sentences_by_polarity[pol].append(sent)

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_para()
            continue
        if _is_header_line(stripped):
            flush_para()
            # Clean markdown/punctuation from header for matching
            clean = re.sub(r"[#*:_]", " ", stripped).strip()
            current_section_polarity = _section_polarity(clean)
        else:
            current_para.append(stripped)
    flush_para()

    return sentences_by_polarity


def extract_semantic_queries(must_have_sentences: list[str], max_queries: int = 10) -> list[str]:
    """
    Use must_have sentences directly as semantic queries.
    Filter by length and deduplicate semantically similar ones with a naive prefix check.
    """
    queries = []
    seen_prefixes: set[str] = set()
    for s in must_have_sentences:
        # Normalize: lowercase, strip trailing period
        norm = re.sub(r"\s+", " ", s).strip().rstrip(".")
        # Length filter: useful queries are 30–300 chars
        if not (30 <= len(norm) <= 300):
            continue
        # Skip pure logistics sentences (notice, salary, location) for semantic queries
        if re.search(r"\b(salary|compensation|notice period|location|relocation|visa|onsite|office)\b", norm, re.I):
            continue
        prefix = norm[:40].lower()
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        # Lowercase for embedding (sentence-transformers handles casing fine,
        # but lowercase makes the queries more robust across model versions)
        queries.append(norm)
        if len(queries) >= max_queries:
            break
    return queries


def extract_experience_constraints(all_sentences: list[str]) -> dict:
    """Extract experience range from sentences."""
    for sent in all_sentences:
        m = _EXP_RANGE_RE.search(sent)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            center = round((lo + hi) / 2.0)
            return {
                "ideal_min": lo,
                "ideal_max": hi,
                "ideal_center": center,
                "hard_min": max(0, lo - 2),
                "hard_max": hi + 5,
            }
    # Fallback: single-number mention
    for sent in all_sentences:
        m = _EXP_SINGLE_RE.search(sent)
        if m:
            n = int(m.group(1))
            return {
                "ideal_min": max(0, n - 2),
                "ideal_max": n + 2,
                "ideal_center": n,
                "hard_min": max(0, n - 4),
                "hard_max": n + 7,
            }
    return {}


def extract_notice_constraint(all_sentences: list[str]) -> dict:
    """Extract notice period constraint."""
    for sent in all_sentences:
        m = _NOTICE_RE.search(sent) or _NOTICE_ALT_RE.search(sent)
        if m:
            days = int(m.group(1))
            return {
                "ideal_max_days": days,
                "higher_bar_days": days,
                "hard_max_days": days * 5,
            }
    return {}


def extract_location_constraint(all_sentences: list[str],
                                 pool_cities: set[str] | None = None) -> dict:
    """Extract city/location mentions from all sentences."""
    city_re = re.compile(
        r"\b(" + "|".join(re.escape(c) for c in _KNOWN_CITIES) + r")\b", re.I
    )
    found_cities: list[str] = []
    for sent in all_sentences:
        for m in city_re.finditer(sent):
            city = m.group(1).title()
            if city not in found_cities:
                found_cities.append(city)

    # Validate against pool if available
    if pool_cities:
        found_cities = [c for c in found_cities if c in pool_cities]

    if not found_cities:
        return {}

    # First city is "preferred", rest are "welcome"
    preferred = found_cities[:2]
    welcome = found_cities[2:] if len(found_cities) > 2 else _WELCOME_CITIES_DEFAULT

    # Detect country requirement
    required_country = ""
    if re.search(r"\bindia\b|\bindian\b", " ".join(all_sentences), re.I):
        required_country = "India"

    result: dict = {
        "preferred_cities": preferred,
        "welcome_cities": welcome,
        "required_country": required_country,
    }
    if re.search(r"\brelocat", " ".join(all_sentences), re.I):
        result["relocation_credit"] = 0.6
    if re.search(r"\bno visa\b|\bvisa sponsorship\b", " ".join(all_sentences), re.I):
        result["no_visa_sponsorship"] = True
    return result


def extract_job_title(jd_text: str) -> str:
    """Best-effort extraction of job title from the JD's first few lines."""
    for line in jd_text.split("\n")[:15]:
        line = line.strip()
        if not line:
            continue
        # Skip company name lines and very long lines
        if 5 < len(line) < 80 and not line.endswith(":"):
            clean = re.sub(r"[#*_]", "", line).strip()
            if clean:
                return clean
    return "Unknown Role"


def _extract_role_descriptor(must_have_sentences: list[str]) -> str:
    """Derive a short role descriptor from the first must_have sentence."""
    for s in must_have_sentences:
        clean = re.sub(r"\s+", " ", s).strip().rstrip(".")
        if 20 <= len(clean) <= 200:
            # Trim to 80 chars at a word boundary
            if len(clean) > 80:
                clean = clean[:77].rsplit(" ", 1)[0] + "..."
            return clean
    return "this role's requirements"


def _pool_cities(candidates_path: str) -> set[str]:
    """Scan candidate pool for unique city values."""
    cities: set[str] = set()
    try:
        with open(candidates_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 5000:  # scan first 5k for speed
                    break
                try:
                    c = json.loads(line)
                    city = c.get("profile", {}).get("location", {}).get("city", "")
                    if city:
                        cities.add(city.strip())
                except (json.JSONDecodeError, AttributeError):
                    continue
    except FileNotFoundError:
        pass
    return cities


def _enable_disqualifiers(not_wanted_sentences: list[str]) -> dict[str, dict]:
    """
    For each known disqualifier, decide enabled=True/False by checking
    whether any not_wanted sentence is semantically close to its anchor.
    Uses simple token-overlap (Jaccard on bigrams) so no model is needed at
    compile time (avoids circular dependency — model may not be loaded yet).
    """
    def bigrams(text: str) -> set[str]:
        tokens = re.findall(r"[a-z]+", text.lower())
        return {f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)}

    results = {}
    nw_bigrams = set()
    for sent in not_wanted_sentences:
        nw_bigrams |= bigrams(sent)

    for dq_name, anchor in _DISQUALIFIER_ANCHORS.items():
        anchor_bigrams = bigrams(anchor)
        if not anchor_bigrams:
            enabled = False
        else:
            overlap = len(anchor_bigrams & nw_bigrams)
            jaccard = overlap / len(anchor_bigrams | nw_bigrams) if (anchor_bigrams | nw_bigrams) else 0.0
            # Also do simple keyword check as fallback
            keyword_hit = any(
                kw in " ".join(not_wanted_sentences).lower()
                for kw in anchor.lower().split()[:3]  # first 3 words of anchor
            )
            enabled = jaccard > 0.05 or keyword_hit
        results[dq_name] = {"enabled": enabled}

    return results


# ---------------------------------------------------------------------------
# Default parameter blocks (used when JD doesn't mention them)
# ---------------------------------------------------------------------------
_DEFAULT_CHANNEL_WEIGHTS = {
    "semantic": 0.32,
    "skills": 0.22,
    "career": 0.26,
    "behavioral": 0.12,
    "integrity": 0.08,
}

_DEFAULT_STUFFER_DETECTION = {
    "gap_threshold": 0.4,
    "stuffer_ramp_width": 0.15,
    "penalty_multiplier": 0.3,
    "apply_to_top_n": 300,
}

_DEFAULT_INTEGRITY = {
    "timeline_consistency_tolerance_months": 60,
    "role_duration_buffer_months": 12,
    "skill_duration_buffer_months": 48,
    "zero_duration_expert_cluster_min": 3,
    "title_skill_coherence_min": 0.25,
    "skill_career_coherence_min": 0.25,
    "failure_multipliers": {"0": 1.0, "1": 0.7, "2": 0.3, "3+": 0.0},
}

_DEFAULT_BEHAVIORAL = {
    "recency_midpoint_days": 90,
    "recency_scale_days": 30,
    "availability_open_to_work_weight": 0.6,
    "availability_notice_weight": 0.4,
    "availability_notice_cap_days": 180,
    "market_validation_max_saves_30d": 80,
    "rescale_floor": 0.3,
    "rescale_ceiling": 1.0,
    "weights": {
        "recency": 0.25,
        "responsiveness": 0.25,
        "availability": 0.25,
        "market_validation": 0.10,
        "reliability": 0.10,
        "verification": 0.025,
        "profile_investment": 0.025,
    },
}

_DEFAULT_DISQUALIFIER_PARAMS = {
    "title_chase": {
        "enabled": False,
        "max_avg_tenure_months": 18,
        "min_roles_to_flag": 3,
        "seniority_words": ["senior", "staff", "principal", "lead"],
    },
    "tech_lead_drift": {
        "enabled": False,
        "drift_title_words": ["architect", "engineering manager", "director", "head of", "vp "],
        "min_current_role_months": 18,
    },
    "pure_research_no_production": {
        "enabled": False,
        "research_industries": ["Academia", "Research"],
        "production_keywords": ["production", "shipped", "deployed", "real users", "scale", "live"],
    },
    "shallow_ai_recent_only": {
        "enabled": False,
        "max_core_skill_duration_months": 12,
        "min_yoe_to_flag": 2,
    },
    "pure_consulting_career_penalty_multiplier": 0.5,
    "pure_consulting_career_enabled": False,
    "closed_source_no_validation": {
        "enabled": False,
        "min_yoe_to_flag": 5,
        "github_activity_threshold": 0,
        "penalty_multiplier": 0.92,
    },
}


def compile_jd(jd_text: str,
               candidates_path: str | None = None,
               base_config: dict | None = None) -> dict:
    """
    Main compilation function. Returns a complete jd_config dict.
    """
    base = base_config or {}

    # Pool cities for location validation
    pool_cities = _pool_cities(candidates_path) if candidates_path else None

    # Parse JD into polarized sentences
    buckets = parse_jd(jd_text)
    all_sentences = (
        buckets["must_have"] + buckets["nice_to_have"] + buckets["not_wanted"]
    )

    # Job title
    job_title = extract_job_title(jd_text)
    company = base.get("company", "")
    # Try to extract company from JD
    company_re = re.compile(r"(?:at|join|for)\s+([A-Z][A-Za-z0-9\s]+?)(?:\s*[,.]|\s+as\b|\s+—)", re.I)
    cm = company_re.search(jd_text[:500])
    if cm and not company:
        company = cm.group(1).strip()

    # Role descriptor
    role_descriptor = _extract_role_descriptor(buckets["must_have"])

    # Semantic queries from must_have sentences
    semantic_queries = extract_semantic_queries(buckets["must_have"])
    if not semantic_queries and buckets["nice_to_have"]:
        # Fallback: use nice_to_have if must_have is empty
        semantic_queries = extract_semantic_queries(buckets["nice_to_have"])

    # Constraints
    experience = extract_experience_constraints(all_sentences)
    location = extract_location_constraint(all_sentences, pool_cities)
    notice = extract_notice_constraint(all_sentences)

    # Company constraints from base config or empty (let scorer be neutral)
    company_constraints = base.get("constraints", {}).get("company", {})

    # Disqualifier enablement
    dq_enabled = _enable_disqualifiers(buckets["not_wanted"])
    dq_params: dict = {}
    # Merge enabled flags into default params
    for dq_name, params in _DEFAULT_DISQUALIFIER_PARAMS.items():
        if dq_name in dq_enabled:
            merged = dict(params)
            merged["enabled"] = dq_enabled[dq_name]["enabled"]
            dq_params[dq_name] = merged
        else:
            dq_params[dq_name] = params

    # Build constraints block — omit sub-blocks the JD doesn't mention
    constraints: dict = {}
    if experience:
        constraints["experience"] = experience
    if location:
        constraints["location"] = location
    if company_constraints:
        constraints["company"] = company_constraints
    if notice:
        constraints["notice_period"] = notice
    constraints["work_mode"] = base.get("constraints", {}).get("work_mode", {})
    constraints["disqualifiers"] = dq_params

    config = {
        "job_title": job_title,
        "company": company,
        "role_descriptor": role_descriptor,

        # Semantic queries — compiler-generated from must_have sentences
        "semantic_queries": semantic_queries,

        # Skill clusters — INTENTIONALLY EMPTY; run precompute/derive_skill_clusters.py
        # to fill these automatically via embedding similarity.
        # Manual overrides can be added to skill_overrides below.
        "skill_clusters": {
            "core": [],
            "secondary": [],
            "nice_to_have": [],
            "domain_mismatch_anti_skills": [],
            "business_mismatch_anti_skills": [],
        },
        "skill_overrides": base.get("skill_overrides", {}),

        "constraints": constraints,
        "channel_weights": base.get("channel_weights", _DEFAULT_CHANNEL_WEIGHTS),
        "stuffer_detection": base.get("stuffer_detection", _DEFAULT_STUFFER_DETECTION),
        "integrity": base.get("integrity", _DEFAULT_INTEGRITY),
        "behavioral": base.get("behavioral", _DEFAULT_BEHAVIORAL),
        "embedding": base.get("embedding", {"model_name": "all-MiniLM-L6-v2", "dimension": 384}),
        "output": base.get("output", {"top_n": 100}),

        # Compiler metadata — useful for debugging and Stage-5 story
        "_compiler_meta": {
            "must_have_sentences": len(buckets["must_have"]),
            "nice_to_have_sentences": len(buckets["nice_to_have"]),
            "not_wanted_sentences": len(buckets["not_wanted"]),
            "queries_generated": len(semantic_queries),
            "disqualifiers_enabled": [k for k, v in dq_enabled.items() if v["enabled"]],
        },
    }

    return config


def _yaml_comment_header(job_title: str) -> str:
    return textwrap.dedent(f"""\
        # =============================================================================
        # jd_config.yaml  —  AUTO-GENERATED by jd_compiler.py
        #
        # Job: {job_title}
        #
        # SKILL CLUSTERS ARE INTENTIONALLY EMPTY HERE.
        # Run the following to fill them via embedding similarity:
        #
        #   python precompute/derive_skill_clusters.py \\
        #       --candidates /path/to/candidates.jsonl \\
        #       --config config/jd_config.yaml \\
        #       --artifacts-dir artifacts
        #
        # That script writes skill_clusters directly into this file's skill_clusters
        # section, or into artifacts/skill_clusters.yaml for manual review.
        #
        # Manual overrides: add entries to skill_overrides below; they take
        # precedence over embedding-derived assignments.
        # =============================================================================

    """)


def main():
    ap = argparse.ArgumentParser(description="Compile a raw JD text file into jd_config.yaml")
    ap.add_argument("--jd", required=True, help="Path to raw JD text file")
    ap.add_argument("--out", default="config/jd_config.yaml", help="Output YAML path")
    ap.add_argument("--candidates", default=None,
                    help="Optional path to candidates.jsonl for city validation")
    ap.add_argument("--base-config", default=None,
                    help="Optional existing config to inherit weights/params from")
    ap.add_argument("--print", dest="print_only", action="store_true",
                    help="Print compiled config to stdout instead of writing file")
    args = ap.parse_args()

    with open(args.jd, "r", encoding="utf-8") as f:
        jd_text = f.read()

    base_config = None
    if args.base_config and os.path.exists(args.base_config):
        with open(args.base_config, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)
        print(f"[jd_compiler] inheriting weights/params from {args.base_config}")

    print(f"[jd_compiler] compiling {args.jd} ...")
    config = compile_jd(jd_text, candidates_path=args.candidates, base_config=base_config)

    meta = config.get("_compiler_meta", {})
    print(f"[jd_compiler] parsed  {meta.get('must_have_sentences')} must-have / "
          f"{meta.get('nice_to_have_sentences')} nice-to-have / "
          f"{meta.get('not_wanted_sentences')} not-wanted sentences")
    print(f"[jd_compiler] generated {meta.get('queries_generated')} semantic queries")
    print(f"[jd_compiler] disqualifiers enabled: {meta.get('disqualifiers_enabled', [])}")
    print(f"[jd_compiler] role_descriptor: \"{config['role_descriptor']}\"")

    yaml_str = _yaml_comment_header(config["job_title"]) + yaml.dump(
        config, default_flow_style=False, allow_unicode=True, sort_keys=False
    )

    if args.print_only:
        print(yaml_str)
    else:
        os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(yaml_str)
        print(f"[jd_compiler] wrote {args.out}")
        print(f"[jd_compiler] NEXT STEP: run precompute/derive_skill_clusters.py to fill skill_clusters")


if __name__ == "__main__":
    main()
