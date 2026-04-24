"""
=============================================================================
  Health Literacy — Persona Development for the Three Profile Types
=============================================================================
  Inputs  : hl_profiling_results.csv   (produced by the main notebook)
  Outputs :
      • persona_report.txt             — full text persona cards
      • persona_factor_space.png       — factor-space scatter coloured by profile
      • persona_radar.png              — radar / spider chart per persona
      • persona_hl_distribution.png    — HL-level breakdown per profile
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

# ─── 0. Config ────────────────────────────────────────────────────────────────
RESULTS_CSV   = "./hl_profiling_results.csv"
OUT_REPORT    = "./persona_report.txt"
OUT_SCATTER   = "./persona_factor_space.png"
OUT_RADAR     = "./persona_radar.png"
OUT_HL_DIST   = "./persona_hl_distribution.png"
N_EXEMPLARS   = 5       # representative posts extracted per persona
RANDOM_SEED   = 42

np.random.seed(RANDOM_SEED)

PROFILE_COLOURS = {
    "Balanced":     "#4CAF50",   # green
    "Transitional": "#FF9800",   # orange
    "Specialized":  "#9C27B0",   # purple
}

HL_COLOURS = {
    "Low":          "#F44336",
    "Basic":        "#FF9800",
    "Intermediate": "#2196F3",
    "High":         "#4CAF50",
}

# ─── 1. Load Results ──────────────────────────────────────────────────────────
print(f"Loading {RESULTS_CSV} …")
df = pd.read_csv(RESULTS_CSV)
print(f"  Loaded {len(df):,} rows | columns: {list(df.columns)}")

# Normalise profile labels (strip whitespace, title-case)
df["profile_type"] = df["profile_type"].str.strip().str.title()
df["sub_type"]     = df["sub_type"].fillna("").str.strip()
df["hl_level"]     = df["hl_level"].str.strip()

# ─── 2. Per-profile statistics ────────────────────────────────────────────────
print("\nComputing per-profile statistics …")

factor_cols = ["factor1_core", "factor2_digital", "factor3_applied"]
stat_rows   = []

for profile, grp in df.groupby("profile_type"):
    n     = len(grp)
    frac  = n / len(df) * 100
    stats = {
        "profile":  profile,
        "n":        n,
        "pct":      frac,
        "f1_mean":  grp["factor1_core"].mean(),
        "f1_std":   grp["factor1_core"].std(),
        "f2_mean":  grp["factor2_digital"].mean(),
        "f2_std":   grp["factor2_digital"].std(),
        "f3_mean":  grp["factor3_applied"].mean(),
        "f3_std":   grp["factor3_applied"].std(),
        "sigma_mean": grp["sigma_unc"].mean(),
        "flagged_pct": grp["flagged"].mean() * 100,
    }
    for lvl in ["Low", "Basic", "Intermediate", "High"]:
        stats[f"hl_{lvl}_pct"] = (grp["hl_level"] == lvl).mean() * 100
    # dominant HL level
    stats["dominant_hl"] = grp["hl_level"].value_counts().idxmax()
    stat_rows.append(stats)

stats_df = pd.DataFrame(stat_rows).set_index("profile")
print(stats_df[["n", "pct", "f1_mean", "f2_mean", "f3_mean",
                "sigma_mean", "dominant_hl"]].round(3))

# ─── 3. Extract Centroid Exemplars ───────────────────────────────────────────
print("\nFinding centroid exemplars …")

exemplars = {}
for profile, grp in df.groupby("profile_type"):
    fmat   = grp[factor_cols].values
    center = fmat.mean(axis=0, keepdims=True)
    dists  = cdist(fmat, center, metric="euclidean").ravel()
    top_idx = np.argsort(dists)[:N_EXEMPLARS]
    exemplars[profile] = grp.iloc[top_idx][
        ["fullText", "factor1_core", "factor2_digital",
         "factor3_applied", "sigma_unc", "hl_level", "sub_type"]
    ].reset_index(drop=True)

# ─── 4. Build Persona Cards (text) ───────────────────────────────────────────

PERSONA_SPECS = {
    "Balanced": {
        "archetype":    "The Health-Savvy Self-Advocate",
        "tagline":      "Reads widely, questions carefully, acts with confidence.",
        "description": (
            "This user demonstrates well-rounded health literacy across all five "
            "dimensions. They can comprehend medical text (FHL), ask contextualised "
            "questions (CHL), reason about causality and evidence (CRHL), use digital "
            "health tools appropriately (DHL), and express their situation articulately "
            "(EHL). They typically understand treatment options, weigh tradeoffs, and "
            "engage productively with both clinicians and peer communities."
        ),
        "strengths": [
            "Integrates information from multiple credible sources",
            "Frames questions with relevant personal context",
            "Comfortable using medical terminology",
            "Recognises hedging language and acknowledges uncertainty",
        ],
        "challenges": [
            "May over-research and experience information overload",
            "Can sometimes conflate authoritative sources with anecdotal ones",
        ],
        "communication_tips": [
            "Provide evidence-based rationale; they respond well to citations",
            "Encourage shared-decision-making framing",
            "Avoid over-simplifying — they prefer complete information",
        ],
        "design_tips": [
            "Dense but well-structured content is appropriate",
            "Link to primary sources (guidelines, trials)",
            "Decision-support tools (pros/cons tables) are well received",
        ],
    },

    "Transitional": {
        "archetype":    "The Curious Newcomer",
        "tagline":      "Learning the language of health, one question at a time.",
        "description": (
            "This user is at an earlier stage of health literacy development. Their "
            "core integrated proficiency (F1) is below median, and neither digital "
            "nor applied literacy is particularly pronounced. They often seek basic "
            "explanations, struggle with medical jargon, and rely heavily on plain-"
            "language community advice. Their posts tend to be shorter, emotionally "
            "loaded, and question-driven, reflecting genuine uncertainty about their "
            "health situation."
        ),
        "strengths": [
            "High motivation to learn and improve",
            "Asks direct, honest questions without assumed background",
            "Receptive to accessible explanations",
            "Community-oriented and likely to share useful answers with peers",
        ],
        "challenges": [
            "Difficulty distinguishing reliable from unreliable sources",
            "Limited ability to critically evaluate treatment claims",
            "May misinterpret complex or technical responses",
            "Higher risk of acting on misinformation",
        ],
        "communication_tips": [
            "Use plain language and define any clinical term used",
            "Break down advice into small, numbered steps",
            "Validate their concern before delivering information",
            "Recommend trusted lay resources (patient leaflets, NHS, Mayo Clinic)",
        ],
        "design_tips": [
            "Short paragraphs, large font, iconography",
            "Avoid acronyms or spell them out on first use",
            "Progress indicators and checklists support engagement",
        ],
    },

    "Specialized": {
        "archetype":    "The Domain Expert (Niche Proficiency)",
        "tagline":      "Deep knowledge in one lane; gaps in the others.",
        "description": (
            "This user shows an atypical literacy profile — either unusually strong "
            "or weak on the digital (F2) or applied/functional (F3) dimension while "
            "their core proficiency (F1) may be moderate. Two sub-types emerge:\n\n"
            "  • Digitally-Specialized (|F2| > 1.0): Highly proficient at finding, "
            "interpreting and citing digital health resources — PubMed, clinical "
            "databases, online symptom checkers — but may lack applied communication "
            "skills or struggle to translate findings into personal decisions.\n\n"
            "  • Functionally-Specialized (|F3| > 0.8): Highly capable of expressing "
            "personal health narratives and applying information to their own context, "
            "but may not actively seek out or critically evaluate external evidence. "
            "Often seen in long-term condition patients who have accumulated lived "
            "expertise."
        ),
        "strengths": [
            "Digitally-Specialized: skilled at evidence synthesis and source evaluation",
            "Functionally-Specialized: rich personal context, applies info to real life",
            "Both sub-types contribute high-quality signal to peer communities",
        ],
        "challenges": [
            "Digitally-Specialized: may over-index on research, under-personalise care",
            "Functionally-Specialized: anecdotal reasoning, limited evidence appraisal",
            "Profile mismatch can produce high model uncertainty (flagged)",
        ],
        "communication_tips": [
            "For Digitally-Specialized: bridge evidence to personal decision context",
            "For Functionally-Specialized: affirm lived experience, gently introduce evidence",
            "Tailor depth of clinical detail to the detected sub-type",
        ],
        "design_tips": [
            "Digitally-Specialized: advanced filters, export to reference manager, DOI links",
            "Functionally-Specialized: narrative/story formats, peer testimonials",
            "Adaptive content depth based on sub-type detection",
        ],
    },
}

def build_persona_card(profile: str, spec: dict, row: pd.Series) -> str:
    """Format a full persona card as plain text."""
    sep  = "═" * 70
    sep2 = "─" * 70
    lines = [
        sep,
        f"  PERSONA: {profile.upper()}",
        f"  Archetype : {spec['archetype']}",
        f"  Tagline   : \"{spec['tagline']}\"",
        sep,
        "",
        "  POPULATION STATISTICS",
        sep2,
        f"  Users in dataset : {row['n']:,} ({row['pct']:.1f}% of total)",
        f"  Dominant HL level: {row['dominant_hl']}",
        f"  HL breakdown     : "
            f"Low {row['hl_Low_pct']:.1f}% | "
            f"Basic {row['hl_Basic_pct']:.1f}% | "
            f"Intermediate {row['hl_Intermediate_pct']:.1f}% | "
            f"High {row['hl_High_pct']:.1f}%",
        "",
        "  FACTOR PROFILE  (mean ± std)",
        sep2,
        f"  F1 Core Integrated Proficiency : {row['f1_mean']:+.3f} ± {row['f1_std']:.3f}",
        f"  F2 Digital Proficiency         : {row['f2_mean']:+.3f} ± {row['f2_std']:.3f}",
        f"  F3 Applied Functional Literacy : {row['f3_mean']:+.3f} ± {row['f3_std']:.3f}",
        f"  Avg epistemic uncertainty (σ)  : {row['sigma_mean']:.4f}",
        f"  Flagged for expert review      : {row['flagged_pct']:.1f}%",
        "",
        "  DESCRIPTION",
        sep2,
    ]
    for para in spec["description"].split("\n\n"):
        lines.append("  " + para.strip())
        lines.append("")

    lines += ["  STRENGTHS", sep2]
    for s in spec["strengths"]:
        lines.append(f"  ✓  {s}")
    lines.append("")

    lines += ["  CHALLENGES", sep2]
    for c in spec["challenges"]:
        lines.append(f"  ✗  {c}")
    lines.append("")

    lines += ["  COMMUNICATION TIPS  (for clinicians / content designers)", sep2]
    for t in spec["communication_tips"]:
        lines.append(f"  →  {t}")
    lines.append("")

    lines += ["  UX / DESIGN TIPS", sep2]
    for d in spec["design_tips"]:
        lines.append(f"  →  {d}")
    lines.append("")

    return "\n".join(lines)


print("\nGenerating persona cards …")
report_lines = [
    "=" * 70,
    "  HEALTH LITERACY — PERSONA REPORT",
    "  Generated from hl_profiling_results.csv",
    "=" * 70,
    "",
]

for profile, spec in PERSONA_SPECS.items():
    if profile not in stats_df.index:
        continue
    row  = stats_df.loc[profile]
    card = build_persona_card(profile, spec, row)
    report_lines.append(card)

    # Append exemplar posts
    report_lines.append("  CENTROID EXEMPLAR POSTS")
    report_lines.append("─" * 70)
    exs = exemplars.get(profile, pd.DataFrame())
    for i, (_, ex) in enumerate(exs.iterrows(), 1):
        snippet = str(ex.get("fullText", ""))[:300].replace("\n", " ")
        report_lines += [
            f"  [{i}] HL={ex['hl_level']}  "
            f"F1={ex['factor1_core']:+.3f}  "
            f"F2={ex['factor2_digital']:+.3f}  "
            f"F3={ex['factor3_applied']:+.3f}  "
            f"σ={ex['sigma_unc']:.3f}",
            f"      {snippet}…",
            "",
        ]
    report_lines.append("")

report_text = "\n".join(report_lines)
with open(OUT_REPORT, "w", encoding="utf-8") as fh:
    fh.write(report_text)
print(f"  Saved → {OUT_REPORT}")

# ─── 5. Figure A — Factor-Space Scatter ──────────────────────────────────────
print("\nPlotting factor-space scatter …")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sample = df.sample(min(30_000, len(df)), random_state=RANDOM_SEED)
colours_mapped = sample["profile_type"].map(PROFILE_COLOURS).fillna("#888888")

# F1 vs F2
ax = axes[0]
ax.scatter(sample["factor1_core"], sample["factor2_digital"],
           c=colours_mapped, alpha=0.25, s=8, linewidths=0)
for p, col in PROFILE_COLOURS.items():
    grp = df[df["profile_type"] == p]
    ax.scatter(grp["factor1_core"].mean(), grp["factor2_digital"].mean(),
               c=col, s=200, marker="*", edgecolors="black", linewidths=0.8,
               zorder=5, label=p)
ax.axhline(1.0,  color="grey", ls=":", lw=0.8, alpha=0.7)
ax.axhline(-1.0, color="grey", ls=":", lw=0.8, alpha=0.7)
ax.set_xlabel("F1  Core Integrated Proficiency", fontsize=11)
ax.set_ylabel("F2  Digital Proficiency", fontsize=11)
ax.set_title("Profile Space: F1 × F2", fontweight="bold")
ax.legend(fontsize=9)

# F1 vs F3
ax = axes[1]
ax.scatter(sample["factor1_core"], sample["factor3_applied"],
           c=colours_mapped, alpha=0.25, s=8, linewidths=0)
for p, col in PROFILE_COLOURS.items():
    grp = df[df["profile_type"] == p]
    ax.scatter(grp["factor1_core"].mean(), grp["factor3_applied"].mean(),
               c=col, s=200, marker="*", edgecolors="black", linewidths=0.8,
               zorder=5, label=p)
ax.axhline(0.8,  color="grey", ls=":", lw=0.8, alpha=0.7)
ax.axhline(-0.8, color="grey", ls=":", lw=0.8, alpha=0.7)
ax.set_xlabel("F1  Core Integrated Proficiency", fontsize=11)
ax.set_ylabel("F3  Applied Functional Literacy", fontsize=11)
ax.set_title("Profile Space: F1 × F3", fontweight="bold")
ax.legend(fontsize=9)

for ax in axes:
    ax.set_facecolor("#F8F8F8")
    ax.grid(True, lw=0.4, alpha=0.5)

fig.suptitle("Health Literacy Profiles — Factor Space", fontweight="bold",
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(OUT_SCATTER, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {OUT_SCATTER}")

# ─── 6. Figure B — Radar Chart ───────────────────────────────────────────────
print("Plotting radar chart …")

# Normalise factor means to [0, 1] across profiles for radar display
radar_labels = ["F1 Core", "F2 Digital", "F3 Applied",
                "HL Score\n(norm)", "Certainty\n(1-σ)"]
N_axes = len(radar_labels)
angles = np.linspace(0, 2 * np.pi, N_axes, endpoint=False).tolist()
angles += angles[:1]   # close the loop

# Build normalised values per profile
all_f1 = stats_df["f1_mean"].values
all_f2 = stats_df["f2_mean"].values
all_f3 = stats_df["f3_mean"].values

HL_MAP = {"Low": 0, "Basic": 1, "Intermediate": 2, "High": 3}
stats_df["hl_score_num"] = stats_df["dominant_hl"].map(HL_MAP).astype(float)

def minmax(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / max(hi - lo, 1e-9)

f1_norm  = minmax(all_f1)
f2_norm  = minmax(all_f2)
f3_norm  = minmax(all_f3)
hl_norm  = minmax(stats_df["hl_score_num"].values)
cer_norm = 1 - minmax(stats_df["sigma_mean"].values)   # certainty = 1 - uncertainty

fig, ax = plt.subplots(figsize=(8, 8),
                        subplot_kw=dict(polar=True))

for i, profile in enumerate(stats_df.index):
    values = [f1_norm[i], f2_norm[i], f3_norm[i], hl_norm[i], cer_norm[i]]
    values += values[:1]
    col = PROFILE_COLOURS.get(profile, "#999999")
    ax.plot(angles, values, "o-", color=col, linewidth=2.5, label=profile)
    ax.fill(angles, values, alpha=0.15, color=col)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75])
ax.set_yticklabels(["25%", "50%", "75%"], fontsize=8, color="grey")
ax.grid(color="grey", linewidth=0.5, alpha=0.5)
ax.set_facecolor("#F9F9F9")
ax.set_title("Persona Radar — Normalised Factor & Quality Scores",
             fontweight="bold", fontsize=13, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=11)

plt.tight_layout()
plt.savefig(OUT_RADAR, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {OUT_RADAR}")

# ─── 7. Figure C — HL Level Distribution per Profile ─────────────────────────
print("Plotting HL level distribution …")

hl_order   = ["Low", "Basic", "Intermediate", "High"]
profiles   = list(stats_df.index)
x          = np.arange(len(profiles))
bar_width  = 0.2

fig, ax = plt.subplots(figsize=(11, 5))

for j, lvl in enumerate(hl_order):
    col = HL_COLOURS[lvl]
    heights = [stats_df.loc[p, f"hl_{lvl}_pct"] for p in profiles]
    bars = ax.bar(x + j * bar_width, heights, bar_width,
                  label=lvl, color=col, edgecolor="white", alpha=0.9)
    for bar, h in zip(bars, heights):
        if h > 2:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x + bar_width * 1.5)
ax.set_xticklabels(profiles, fontsize=12, fontweight="bold")
ax.set_ylabel("% of profile users", fontsize=11)
ax.set_title("HL Level Distribution by Profile Type", fontweight="bold", fontsize=13)
ax.legend(title="HL Level", fontsize=10, title_fontsize=10)
ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
ax.set_facecolor("#F8F8F8")
ax.grid(axis="y", lw=0.5, alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)

for p_idx, profile in enumerate(profiles):
    n = int(stats_df.loc[profile, "n"])
    ax.text(x[p_idx] + bar_width * 1.5, -5.5,
            f"n={n:,}", ha="center", fontsize=9, color="#555555")

plt.tight_layout()
plt.savefig(OUT_HL_DIST, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {OUT_HL_DIST}")

# ─── 8. Console summary ──────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("  PERSONA SUMMARY")
print("═" * 70)
for profile, spec in PERSONA_SPECS.items():
    if profile not in stats_df.index:
        continue
    row = stats_df.loc[profile]
    print(f"\n  {profile.upper()} — {spec['archetype']}")
    print(f"    n={row['n']:,} ({row['pct']:.1f}%) | dominant HL: {row['dominant_hl']}")
    print(f"    F1={row['f1_mean']:+.3f}  F2={row['f2_mean']:+.3f}  "
          f"F3={row['f3_mean']:+.3f}  σ={row['sigma_mean']:.4f}")
    print(f"    Tagline: \"{spec['tagline']}\"")

print("\n" + "═" * 70)
print("  OUTPUT FILES")
print("─" * 70)
for path in [OUT_REPORT, OUT_SCATTER, OUT_RADAR, OUT_HL_DIST]:
    size_kb = os.path.getsize(path) // 1024 if os.path.exists(path) else 0
    print(f"  {path:<40}  ({size_kb} KB)")
print("═" * 70)
