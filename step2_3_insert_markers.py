"""
Step 2.3: Create Symbolic Marker Mappings (complex_with_markers)
For each training pair, aligns jargon tokens in the complex text with their
simplified counterparts in the simple text and inserts neuro-symbolic markers:

  [REPLACE:jargon→simplified]jargon   – when a simplified form was found
  [EXPLAIN:jargon]jargon              – fallback when no alignment found

Alignment strategy (three-tier, in order):
  1. Exact or case-insensitive substring match in the simple sentence.
  2. Longest Common Substring (LCS) via difflib – ratio > threshold.
  3. Token overlap heuristic: simple tokens that partially cover the jargon.

Output:
  data/marked/train_marked.jsonl
  data/marked/val_marked.jsonl
  data/marked/test_marked.jsonl

Each output record adds:
  "complex_with_markers" : str   ← input for the seq2seq model encoder
  "marker_map"           : list  ← [{jargon, marker_type, simplified}, …]
"""

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from tqdm import tqdm


# ── Paths ─────────────────────────────────────────────────────────────────────
ANNOTATED_DIR = Path("data/annotated")
MARKED_DIR    = Path("data/marked")
SPLITS        = ["train", "val", "test"]

# ── Alignment hyper-parameters ────────────────────────────────────────────────
LCS_THRESHOLD    = 0.60   # minimum SequenceMatcher ratio for fuzzy match
MIN_TOKEN_OVERLAP = 0.50  # fraction of jargon chars that must appear in a simple token


# ── Text helpers ──────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.lower().strip())


def tokenise_simple(simple: str) -> list[str]:
    """
    Return word-level tokens from the simple sentence.
    We keep multi-word chunks by also adding bigrams and trigrams.
    """
    words = re.findall(r"[\w\-']+", simple)
    tokens = list(words)
    # bigrams
    tokens += [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    # trigrams
    tokens += [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words) - 2)]
    return tokens


# ── Alignment tiers ───────────────────────────────────────────────────────────

def tier1_exact(jargon: str, simple: str) -> str | None:
    """Direct substring match (case-insensitive)."""
    jn = normalise(jargon)
    sn = normalise(simple)
    if jn in sn:
        # Return the original-case version from simple
        pat = re.compile(re.escape(jn), re.IGNORECASE)
        m   = pat.search(simple)
        return m.group(0) if m else None
    return None


def tier2_lcs(jargon: str, simple_tokens: list[str]) -> str | None:
    """Longest Common Substring via SequenceMatcher."""
    jn = normalise(jargon)
    best_ratio = 0.0
    best_token = None
    for tok in simple_tokens:
        ratio = SequenceMatcher(None, jn, normalise(tok)).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_token = tok
    if best_ratio >= LCS_THRESHOLD and best_token is not None:
        return best_token
    return None


def tier3_overlap(jargon: str, simple_tokens: list[str]) -> str | None:
    """Token overlap: fraction of jargon chars appearing in any simple token."""
    jn_chars = set(normalise(jargon).replace(" ", ""))
    for tok in simple_tokens:
        tok_chars = set(normalise(tok).replace(" ", ""))
        if not jn_chars:
            continue
        overlap = len(jn_chars & tok_chars) / len(jn_chars)
        if overlap >= MIN_TOKEN_OVERLAP:
            return tok
    return None


def find_simplified_form(jargon: str, simple: str) -> tuple[str | None, str]:
    """
    Try all three tiers in order.
    Returns (simplified_form | None, tier_used).
    """
    stokens = tokenise_simple(simple)

    result = tier1_exact(jargon, simple)
    if result:
        return result, "exact"

    result = tier2_lcs(jargon, stokens)
    if result:
        return result, "lcs"

    result = tier3_overlap(jargon, stokens)
    if result:
        return result, "overlap"

    return None, "none"


# ── Marker insertion ──────────────────────────────────────────────────────────

def extract_jargon_tokens(clarity_agents: list) -> list[str]:
    """Extract all tokens labelled 'Jargon' (preserving order, dedup'd)."""
    seen: set[str] = set()
    jargon_list: list[str] = []
    for item in clarity_agents:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            token, label = item
            if label == "Jargon" and token.strip() and token not in seen:
                seen.add(token)
                jargon_list.append(token.strip())
    return jargon_list


def insert_markers(complex_text: str, jargon_tokens: list[str], simple: str) -> tuple[str, list[dict]]:
    """
    Insert symbolic markers before each jargon token in the complex text.
    Returns (complex_with_markers, marker_map).
    """
    marker_map: list[dict] = []
    result = complex_text

    # Sort by length descending to avoid partial replacements of substrings
    jargon_sorted = sorted(jargon_tokens, key=len, reverse=True)

    for jargon in jargon_sorted:
        simplified, tier = find_simplified_form(jargon, simple)

        if simplified and normalise(simplified) != normalise(jargon):
            marker     = f"[REPLACE:{jargon}→{simplified}]"
            entry_type = "REPLACE"
        else:
            marker     = f"[EXPLAIN:{jargon}]"
            entry_type = "EXPLAIN"
            simplified = None

        marker_map.append({
            "jargon":      jargon,
            "marker_type": entry_type,
            "simplified":  simplified,
            "align_tier":  tier,
        })

        # Insert marker immediately before the jargon token (whole-word aware)
        pattern = re.compile(r"(?<!\[)" + re.escape(jargon), re.IGNORECASE)
        result  = pattern.sub(lambda m: marker + m.group(0), result, count=1)

    return result, marker_map


# ── Process one split ─────────────────────────────────────────────────────────

def process_split(split: str):
    src = ANNOTATED_DIR / f"{split}_hl.jsonl"
    dst = MARKED_DIR    / f"{split}_marked.jsonl"

    if not src.exists():
        print(f"[{split}] Source not found: {src} – skipping.")
        return

    with open(src, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    dst.parent.mkdir(parents=True, exist_ok=True)

    replace_count = explain_count = skip_count = 0

    with open(dst, "w", encoding="utf-8") as out_f:
        for rec in tqdm(records, desc=f"  [{split}]", unit="pairs"):
            hl_complex = rec.get("hl_complex", {})

            # Skip records where HL annotation failed
            if "hl_error" in hl_complex:
                out_f.write(json.dumps({**rec, "complex_with_markers": rec["complex"], "marker_map": []},
                                       ensure_ascii=False) + "\n")
                skip_count += 1
                continue

            jargon_tokens = extract_jargon_tokens(hl_complex.get("clarity_agents", []))

            if not jargon_tokens:
                out_f.write(json.dumps({**rec, "complex_with_markers": rec["complex"], "marker_map": []},
                                       ensure_ascii=False) + "\n")
                skip_count += 1
                continue

            complex_with_markers, marker_map = insert_markers(
                rec["complex"], jargon_tokens, rec["simple"]
            )

            for m in marker_map:
                if m["marker_type"] == "REPLACE":
                    replace_count += 1
                else:
                    explain_count += 1

            out_f.write(json.dumps(
                {**rec, "complex_with_markers": complex_with_markers, "marker_map": marker_map},
                ensure_ascii=False
            ) + "\n")

    total = len(records)
    print(f"  [{split}] {total:,} pairs  |  "
          f"REPLACE: {replace_count:,}  EXPLAIN: {explain_count:,}  no-jargon/error: {skip_count:,}")


# ── Spot-check helper ─────────────────────────────────────────────────────────

def spot_check(split: str = "train", n: int = 3):
    path = MARKED_DIR / f"{split}_marked.jsonl"
    if not path.exists():
        return
    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    # Show examples that have at least one REPLACE marker
    examples = [r for r in records if any(m["marker_type"] == "REPLACE"
                                          for m in r.get("marker_map", []))][:n]

    print(f"\n{'─'*60}")
    print(f"  SPOT CHECK ({split}, {len(examples)} REPLACE examples)")
    for ex in examples:
        print(f"\n  Complex  : {ex['complex'][:120]}")
        print(f"  Marked   : {ex['complex_with_markers'][:200]}")
        print(f"  Simple   : {ex['simple'][:120]}")
        print(f"  Markers  : {ex['marker_map']}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    MARKED_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        print(f"\n{'─'*60}")
        print(f"  Inserting markers for split: {split}")
        process_split(split)

    spot_check("train", n=3)
    print("\n✅  Marker insertion complete.\n")


if __name__ == "__main__":
    main()
