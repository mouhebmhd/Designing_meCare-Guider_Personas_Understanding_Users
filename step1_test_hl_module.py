"""
Step 1: Validate and Test the Health Literacy Module
Verifies the HL module works and documents the exact output structure.
"""

import json
import sys
from pprint import pprint

from health_literacy_evaluation_and_persona_creation.legacy.hl_analysis_module import analyze_health_literacy


# ── Test sentences spanning all three HL profiles ──────────────────────────
TEST_CASES = {
    "specialized": "The patient presented with myocardial infarction requiring emergent percutaneous coronary intervention.",
    "transitional": "High blood pressure can cause serious heart problems if left untreated by a doctor.",
    "balanced":     "I have a headache. Should I take medicine?",
}

REQUIRED_TOP_KEYS = {"profile", "suitability", "clarity_consensus", "annotations"}
REQUIRED_CLARITY  = {"jargon", "explanation", "coherence", "fre", "fluency", "n_terms", "n_explained"}
REQUIRED_ANNOT    = {"clarity_agents", "hl_dimensions"}


def validate_result(label: str, result: dict) -> bool:
    ok = True

    # ── Top-level keys ──────────────────────────────────────────────────────
    missing_top = REQUIRED_TOP_KEYS - result.keys()
    if missing_top:
        print(f"  [WARN] Missing top-level keys: {missing_top}")
        ok = False

    # ── clarity_consensus ───────────────────────────────────────────────────
    cc = result.get("clarity_consensus", {})
    missing_cc = REQUIRED_CLARITY - cc.keys()
    if missing_cc:
        print(f"  [WARN] Missing clarity_consensus keys: {missing_cc}")
        ok = False

    # ── annotations ─────────────────────────────────────────────────────────
    ann = result.get("annotations", {})
    missing_ann = REQUIRED_ANNOT - ann.keys()
    if missing_ann:
        print(f"  [WARN] Missing annotations keys: {missing_ann}")
        ok = False

    # ── clarity_agents format: list of [token, label|None] ──────────────────
    agents = ann.get("clarity_agents", [])
    if agents:
        sample = agents[0]
        if not (isinstance(sample, (list, tuple)) and len(sample) == 2):
            print(f"  [WARN] clarity_agents items should be [token, label], got: {sample}")
            ok = False

    # ── suitability should have the three profile keys ───────────────────────
    suit = result.get("suitability", {})
    expected_profiles = {"Balanced", "Transitional", "Specialized"}
    missing_prof = expected_profiles - suit.keys()
    if missing_prof:
        print(f"  [WARN] Missing suitability profiles: {missing_prof}")
        ok = False

    return ok


def summarise_result(label: str, result: dict):
    """Print a compact, human-readable summary of one HL result."""
    cc  = result.get("clarity_consensus", {})
    ann = result.get("annotations", {})
    agents = ann.get("clarity_agents", [])

    jargon_tokens = [tok for tok, lbl in agents if lbl == "Jargon"]
    explained     = [tok for tok, lbl in agents if lbl == "Explanation"]

    print(f"\n{'─'*60}")
    print(f"  Label      : {label}")
    print(f"  Profile    : {result.get('profile')}  (hl_level={result.get('hl_level')})")
    print(f"  Suitability: {result.get('suitability')}")
    print(f"  Clarity    : jargon={cc.get('jargon'):.3f}  coherence={cc.get('coherence'):.3f}"
          f"  fre={cc.get('fre')}  n_terms={cc.get('n_terms')}")
    print(f"  Jargon tokens ({len(jargon_tokens)}): {jargon_tokens[:8]}")
    print(f"  Explained abbrevs: {explained[:8]}")


def main():
    all_passed = True
    results = {}

    for label, text in TEST_CASES.items():
        print(f"\n{'═'*60}")
        print(f"Testing [{label}]:\n  \"{text[:80]}\"")
        try:
            result = analyze_health_literacy(text)
        except Exception as exc:
            print(f"  [ERROR] analyze_health_literacy raised: {exc}")
            all_passed = False
            continue

        passed = validate_result(label, result)
        if passed:
            print("  [OK] All required fields present.")
        else:
            all_passed = False

        summarise_result(label, result)
        results[label] = result

    # ── Dump full JSON of the first result for reference ────────────────────
    first_key = next(iter(results))
    with open("hl_sample_output.json", "w") as f:
        json.dump(results[first_key], f, indent=2)
    print(f"\n[Saved] Full output of '{first_key}' → hl_sample_output.json")

    if all_passed:
        print("\n✅  Module validation PASSED – all required fields present.\n")
    else:
        print("\n⚠️  Module validation finished with warnings – check output above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
