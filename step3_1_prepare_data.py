"""
Step 3.1 – Prepare Training Data Format
Converts data/marked/{split}_marked.jsonl into model-ready records:
  - input_text  : complex_with_markers  (encoder input)
  - target_text : simple                (decoder target, plain text)
  - hl_profile  : Transitional | Balanced | Specialized
  - suitability : {Balanced: float, Transitional: float, Specialized: float}
  - weight      : suitability score for the TARGET profile (used in weighted loss)

Output:
  data/model_ready/train.jsonl
  data/model_ready/val.jsonl
  data/model_ready/test.jsonl
"""

import json
from pathlib import Path
from statistics import mean

MARKED_DIR     = Path("data/marked")
MODEL_READY_DIR = Path("data/model_ready")
SPLITS         = ["train", "val", "test"]

# ── Profile label → integer index (for classifier head) ──────────────────────
PROFILE_TO_IDX = {"Balanced": 0, "Transitional": 1, "Specialized": 2}
IDX_TO_PROFILE = {v: k for k, v in PROFILE_TO_IDX.items()}


def compute_weight(hl_complex: dict, hl_simple: dict) -> float:
    """
    Weighted loss scalar for this training example.

    Strategy: use the suitability score of the SIMPLE side's profile,
    drawn from the COMPLEX side's suitability dict.
    Rationale: if the complex text is poorly suited to the target profile,
    the model needs to work harder → higher weight.

    Normalised to [0.1, 1.0] to avoid zero-weight examples.
    """
    target_profile = hl_simple.get("profile", "Transitional")
    suit_dict      = hl_complex.get("suitability", {})
    raw_score      = suit_dict.get(target_profile, 50.0)   # 0–100 scale
    # Invert: low suitability → high weight
    inverted = max(0.0, 100.0 - raw_score)
    # Normalise to [0.1, 1.0]
    weight = 0.1 + 0.9 * (inverted / 100.0)
    return round(weight, 4)


def profile_label(hl: dict) -> str:
    return hl.get("profile", "Transitional")


def format_example(rec: dict) -> dict | None:
    """
    Convert one marked record to a model-ready dict.
    Returns None if required fields are missing / errored.
    """
    hl_complex = rec.get("hl_complex", {})
    hl_simple  = rec.get("hl_simple",  {})

    # Skip HL-error records
    if "hl_error" in hl_complex or "hl_error" in hl_simple:
        return None

    input_text  = rec.get("complex_with_markers", rec.get("complex", "")).strip()
    target_text = rec.get("simple", "").strip()

    if not input_text or not target_text:
        return None

    src_profile = profile_label(hl_complex)
    tgt_profile = profile_label(hl_simple)
    weight      = compute_weight(hl_complex, hl_simple)

    # Clarity scores of the simple output (used as aux supervision signal)
    simple_clarity = hl_simple.get("clarity", {})

    return {
        # ── Core seq2seq fields ───────────────────────────────────────────────
        "input_text":    input_text,
        "target_text":   target_text,

        # ── HL metadata ───────────────────────────────────────────────────────
        "src_profile":   src_profile,
        "tgt_profile":   tgt_profile,
        "src_profile_idx": PROFILE_TO_IDX.get(src_profile, 1),
        "tgt_profile_idx": PROFILE_TO_IDX.get(tgt_profile, 1),
        "suitability":   hl_complex.get("suitability", {}),
        "weight":        weight,

        # ── Clarity of the TARGET (for auxiliary classifier supervision) ───────
        "target_fre":    simple_clarity.get("fre", 0.0),
        "target_jargon": simple_clarity.get("jargon", 0.0),

        # ── Marker info (for analysis / debugging) ────────────────────────────
        "marker_map":    rec.get("marker_map", []),
        "n_replace":     sum(1 for m in rec.get("marker_map", []) if m.get("marker_type") == "REPLACE"),
        "n_explain":     sum(1 for m in rec.get("marker_map", []) if m.get("marker_type") == "EXPLAIN"),

        # ── Provenance ────────────────────────────────────────────────────────
        "source_corpus": rec.get("source_corpus", "unknown"),
    }


def process_split(split: str) -> dict:
    src = MARKED_DIR    / f"{split}_marked.jsonl"
    dst = MODEL_READY_DIR / f"{split}.jsonl"

    if not src.exists():
        print(f"  [{split}] Source not found: {src} – skipping.")
        return {}

    with open(src, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    formatted, skipped = [], 0
    for rec in records:
        ex = format_example(rec)
        if ex:
            formatted.append(ex)
        else:
            skipped += 1

    MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for ex in formatted:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # ── Stats ──────────────────────────────────────────────────────────────
    weights    = [ex["weight"] for ex in formatted]
    avg_weight = round(mean(weights), 4) if weights else 0
    profiles   = {}
    for ex in formatted:
        profiles[ex["tgt_profile"]] = profiles.get(ex["tgt_profile"], 0) + 1

    print(f"  [{split}] {len(formatted):,} examples  |  skipped: {skipped}"
          f"  |  avg_weight: {avg_weight}  |  profiles: {profiles}")
    return {"n": len(formatted), "avg_weight": avg_weight, "profiles": profiles}


def print_sample(split: str = "train"):
    path = MODEL_READY_DIR / f"{split}.jsonl"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        ex = json.loads(f.readline())
    print(f"\n{'─'*60}  SAMPLE ({split})")
    print(f"  INPUT : {ex['input_text'][:200]}")
    print(f"  TARGET: {ex['target_text'][:120]}")
    print(f"  PROFILE src→tgt: {ex['src_profile']} → {ex['tgt_profile']}")
    print(f"  WEIGHT: {ex['weight']}  |  n_replace: {ex['n_replace']}  n_explain: {ex['n_explain']}")


def main():
    MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)
    print("Preparing model-ready training data…\n")
    for split in SPLITS:
        process_split(split)
    print_sample("train")
    print("\n✅  Data preparation complete.\n")


if __name__ == "__main__":
    main()
