# HyLitS Pipeline – Steps 1 & 2

## File Overview

```
hylits/
├── step1_test_hl_module.py       ← Step 1  – module validation
├── step2_1_download_datasets.py  ← Step 2.1 – download corpora
├── step2_2_annotate_hl.py        ← Step 2.2 – run HL on every pair
└── step2_3_insert_markers.py     ← Step 2.3 – insert symbolic markers
```

## Data Flow

```
data/merged/{split}.jsonl            (complex, simple, source_corpus)
         ↓  step2_2_annotate_hl.py
data/annotated/{split}_hl.jsonl     (+hl_complex, +hl_simple)
         ↓  step2_3_insert_markers.py
data/marked/{split}_marked.jsonl    (+complex_with_markers, +marker_map)
```

## Running in Order

```bash
# 1. Validate the HL module (fast)
python step1_test_hl_module.py

# 2. Download corpora (requires `pip install datasets requests tqdm`)
python step2_1_download_datasets.py

# 3. Annotate with HL (slow – set MAX_WORKERS in the script)
python step2_2_annotate_hl.py

# 4. Insert symbolic markers
python step2_3_insert_markers.py
```

## Marker Format

Each jargon token in the complex sentence gets a prefix marker:

| Situation                              | Marker inserted                          |
|----------------------------------------|------------------------------------------|
| Simplified form found in simple text   | `[REPLACE:myocardial infarction→heart attack]myocardial infarction` |
| No simplified form found               | `[EXPLAIN:eosinophil-dependent]eosinophil-dependent`                |

Alignment uses three tiers (in order): **exact substring → LCS (difflib) → token overlap**.
The `align_tier` field in `marker_map` records which tier succeeded.

## Key Fields After Step 2.3

```json
{
  "complex": "...",
  "simple":  "...",
  "source_corpus": "cochrane",
  "hl_complex": { "profile": "Specialized", "clarity": {...}, "clarity_agents": [...] },
  "hl_simple":  { "profile": "Balanced",    "clarity": {...}, "clarity_agents": [...] },
  "complex_with_markers": "[REPLACE:myocardial infarction→heart attack]myocardial infarction ...",
  "marker_map": [
    { "jargon": "myocardial infarction", "marker_type": "REPLACE",
      "simplified": "heart attack", "align_tier": "exact" }
  ]
}
```
