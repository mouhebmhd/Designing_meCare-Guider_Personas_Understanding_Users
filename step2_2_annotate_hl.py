"""
Step 2.2: Run the HL Module on Every (complex, simple) Pair
Adds hl_complex and hl_simple fields to each example, writing:
  data/annotated/train_hl.jsonl
  data/annotated/val_hl.jsonl
  data/annotated/test_hl.jsonl

Design notes:
  - Resumable: already-written lines are skipped on re-run.
  - Robust: per-example errors are caught; the record is written with
    hl_error fields so the pipeline can filter/handle them later.
  - Parallelism: ThreadPoolExecutor speeds up I/O-bound HL calls.
"""

import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from tqdm import tqdm

from health_literacy_evaluation_and_persona_creation.legacy.hl_analysis_module import analyze_health_literacy


# ── Paths ────────────────────────────────────────────────────────────────────
MERGED_DIR    = Path("data/merged")
ANNOTATED_DIR = Path("data/annotated")

SPLITS = ["train", "val", "test"]

# ── Tuning ───────────────────────────────────────────────────────────────────
MAX_WORKERS   = 4     # threads for parallel HL calls
RETRY_LIMIT   = 2     # retries on transient errors
RETRY_DELAY   = 1.0   # seconds between retries


# ── HL field extractor ───────────────────────────────────────────────────────

def extract_hl_fields(result: dict) -> dict:
    """
    Return a compact, serialisable dict of the fields we need downstream.
    This avoids storing the full (sometimes large) raw result.
    """
    cc  = result.get("clarity_consensus", {})
    ann = result.get("annotations", {})

    return {
        "profile":    result.get("profile"),
        "hl_level":   result.get("hl_level"),
        "sigma":      result.get("sigma"),
        "suitability": result.get("suitability", {}),
        "raw_scores":  result.get("raw_scores", {}),
        "clarity": {
            "final":      cc.get("final"),
            "jargon":     cc.get("jargon"),
            "explanation": cc.get("explanation"),
            "fluency":    cc.get("fluency"),
            "coherence":  cc.get("coherence"),
            "fre":        cc.get("fre"),
            "n_terms":    cc.get("n_terms"),
            "n_explained": cc.get("n_explained"),
        },
        # Full token-level annotations – needed for marker insertion in Step 2.3
        "clarity_agents": ann.get("clarity_agents", []),
        "hl_dimensions":  ann.get("hl_dimensions", []),
        "flagged_transitions": cc.get("flagged_transitions", []),
    }


# ── Safe HL call with retry ───────────────────────────────────────────────────

def safe_analyze(text: str, label: str) -> dict:
    """
    Calls analyze_health_literacy with retries.
    Returns a compact HL dict or an error dict.
    """
    for attempt in range(RETRY_LIMIT + 1):
        try:
            raw = analyze_health_literacy(text)
            return extract_hl_fields(raw)
        except Exception as exc:
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)
            else:
                return {
                    "hl_error": str(exc),
                    "hl_traceback": traceback.format_exc()[-400:],
                }
    return {"hl_error": "Unknown error after retries"}   # unreachable, but safe


# ── Annotate one example ──────────────────────────────────────────────────────

def annotate_example(example: dict) -> dict:
    hl_complex = safe_analyze(example["complex"], "complex")
    hl_simple  = safe_analyze(example["simple"],  "simple")
    return {**example, "hl_complex": hl_complex, "hl_simple": hl_simple}


# ── Process one split ─────────────────────────────────────────────────────────

def process_split(split: str):
    src_path = MERGED_DIR    / f"{split}.jsonl"
    dst_path = ANNOTATED_DIR / f"{split}_hl.jsonl"

    if not src_path.exists():
        print(f"[{split}] Source file not found: {src_path} – skipping.")
        return

    # ── Load source ──────────────────────────────────────────────────────────
    with open(src_path, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    # ── Resumption: skip already-written lines ────────────────────────────────
    n_done = 0
    if dst_path.exists():
        with open(dst_path, encoding="utf-8") as f:
            n_done = sum(1 for line in f if line.strip())
        print(f"[{split}] Resuming from line {n_done} / {len(examples)}")
        examples = examples[n_done:]

    if not examples:
        print(f"[{split}] Already fully annotated.")
        return

    # ── Thread-safe append ────────────────────────────────────────────────────
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = Lock()
    error_count = 0

    with open(dst_path, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(annotate_example, ex): i for i, ex in enumerate(examples)}

            with tqdm(total=len(examples), desc=f"  [{split}]", unit="pairs") as pbar:
                for future in as_completed(futures):
                    try:
                        annotated = future.result()
                        if "hl_error" in annotated.get("hl_complex", {}):
                            error_count += 1
                    except Exception as exc:
                        annotated = {**examples[futures[future]], "hl_error": str(exc)}
                        error_count += 1

                    with write_lock:
                        out_f.write(json.dumps(annotated, ensure_ascii=False) + "\n")
                        out_f.flush()

                    pbar.update(1)

    total_written = n_done + len(examples)
    print(f"  [{split}] Done: {total_written:,} pairs written  |  HL errors: {error_count}")


# ── Summary stats ─────────────────────────────────────────────────────────────

def print_summary():
    print(f"\n{'═'*60}")
    print("  ANNOTATION SUMMARY")
    for split in SPLITS:
        path = ANNOTATED_DIR / f"{split}_hl.jsonl"
        if not path.exists():
            print(f"  {split:5s}: NOT FOUND")
            continue
        with open(path) as f:
            records = [json.loads(l) for l in f if l.strip()]
        errors = sum(1 for r in records if "hl_error" in r.get("hl_complex", {}))
        print(f"  {split:5s}: {len(records):>8,} pairs  |  HL errors: {errors}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        print(f"\n{'─'*60}")
        print(f"  Processing split: {split}")
        process_split(split)
    print_summary()


if __name__ == "__main__":
    main()
