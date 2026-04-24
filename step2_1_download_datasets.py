"""
Step 2.1 (v2): Download datasets via direct Parquet HTTP — no `datasets` lib needed.
Uses HuggingFace's public Parquet export endpoint (works for any public dataset).

Output:
  data/raw/{corpus}/{split}.jsonl
  data/merged/{split}.jsonl
"""

import json
import io
import random
from pathlib import Path

import requests
import pyarrow.parquet as pq
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DIR    = Path("data/raw")
MERGED_DIR = Path("data/merged")
SEED       = 42
HF_API     = "https://datasets-server.huggingface.co/parquet?dataset={repo}"

DATASETS = {
    "cochrane": {
        "repo":          "GEM/cochrane-simplification",
        "complex_field": "source",
        "simple_field":  "target",
        "splits": {"train": "train", "validation": "val", "test": "test"},
    },
    "plaba": {
        "repo":          "drAbreu/PLABA-2023",
        "complex_field": "sentence",
        "simple_field":  "simplification",
        "splits":        {"train": "train"},
        "auto_split":    True,
    },
    "medeasi": {
        "repo":          "cbasu/MedEasi",
        "complex_field": "Complex",
        "simple_field":  "Simple",
        "splits": {"train": "train", "validation": "val", "test": "test"},
    },
}


def get_parquet_urls(repo: str) -> dict:
    url = HF_API.format(repo=repo)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    result = {}
    for entry in data.get("parquet_files", []):
        split = entry["split"]
        result.setdefault(split, []).append(entry["url"])
    return result


def download_parquet_shard(url: str) -> list:
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    buf = io.BytesIO()
    total = int(resp.headers.get("content-length", 0))
    with tqdm(total=total, unit="B", unit_scale=True,
              desc=f"    {url.split('/')[-1][:40]}", leave=False) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            buf.write(chunk)
            bar.update(len(chunk))
    buf.seek(0)
    table = pq.read_table(buf)
    return table.to_pylist()


def normalise_rows(rows, complex_field, simple_field):
    out, seen = [], set()
    for row in rows:
        c = str(row.get(complex_field, "") or "").strip()
        s = str(row.get(simple_field,  "") or "").strip()
        if c and s and (c, s) not in seen:
            seen.add((c, s))
            out.append({"complex": c, "simple": s})
    return out


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):>7,} pairs -> {path}")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def auto_split(rows):
    random.seed(SEED)
    random.shuffle(rows)
    n = len(rows)
    t, v = int(n * 0.8), int(n * 0.1)
    return {"train": rows[:t], "val": rows[t:t+v], "test": rows[t+v:]}


def download_dataset(name, cfg):
    repo = cfg["repo"]
    print(f"\n{'─'*60}")
    print(f"  Downloading: {name}  ({repo})")

    try:
        shard_urls = get_parquet_urls(repo)
    except Exception as exc:
        print(f"  [ERROR] Could not fetch Parquet index: {exc}")
        return {}

    print(f"  Available HF splits: {list(shard_urls.keys())}")
    result = {}
    all_rows = []

    fetch_splits = (list(shard_urls.keys())[:1]
                    if cfg.get("auto_split") else list(cfg["splits"].keys()))

    for hf_split in fetch_splits:
        urls = shard_urls.get(hf_split, [])
        if not urls:
            print(f"  [WARN] No shards for '{hf_split}'")
            continue

        raw_rows = []
        for url in urls:
            try:
                raw_rows.extend(download_parquet_shard(url))
            except Exception as exc:
                print(f"  [WARN] Shard failed: {exc}")

        rows = normalise_rows(raw_rows, cfg["complex_field"], cfg["simple_field"])
        print(f"  [{hf_split}] {len(rows):,} pairs from {len(urls)} shard(s)")

        if cfg.get("auto_split"):
            all_rows.extend(rows)
        else:
            local = cfg["splits"][hf_split]
            write_jsonl(rows, RAW_DIR / name / f"{local}.jsonl")
            result[local] = rows

    if cfg.get("auto_split") and all_rows:
        for local, rows in auto_split(all_rows).items():
            write_jsonl(rows, RAW_DIR / name / f"{local}.jsonl")
            result[local] = rows

    return result


def merge_all(all_splits):
    print(f"\n{'─'*60}")
    print("  Merging all corpora...")
    random.seed(SEED)
    for split in ("train", "val", "test"):
        combined = []
        for corpus, splits in all_splits.items():
            for rec in splits.get(split, []):
                combined.append({**rec, "source_corpus": corpus})
        random.shuffle(combined)
        write_jsonl(combined, MERGED_DIR / f"{split}.jsonl")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    # Auto-clean empty stubs
    for stub in list(RAW_DIR.rglob("*.jsonl")) + list(MERGED_DIR.glob("*.jsonl")):
        if stub.exists() and stub.stat().st_size == 0:
            stub.unlink()
            print(f"  [clean] Removed empty stub: {stub}")

    all_splits = {}
    for name, cfg in DATASETS.items():
        all_splits[name] = download_dataset(name, cfg)

    merge_all(all_splits)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    for split in ("train", "val", "test"):
        path = MERGED_DIR / f"{split}.jsonl"
        n = sum(1 for _ in open(path, encoding="utf-8")) if path.exists() else 0
        print(f"  {split:5s}: {n:>8,} pairs  ->  {path}")
    print()


if __name__ == "__main__":
    main()