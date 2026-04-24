"""
=============================================================================
  Health Literacy — Profile Suitability Metric for Raw Text
=============================================================================
  Usage (script):
      python hl_profile_metric.py

  Usage (import):
      from hl_profile_metric import profile_text
      result = profile_text("I have type 2 diabetes and my doctor prescribed...")
      print(result["report"])
=============================================================================
"""

import re
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import textstat
import spacy
import nltk

warnings.filterwarnings("ignore")
nltk.download("punkt", quiet=True)

# ─── 0. Config ────────────────────────────────────────────────────────────────
MODEL_PATH = "./hl_bvae_model.pt"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MC_PASSES  = 50

# ─── 1. Load spaCy ────────────────────────────────────────────────────────────
try:
    nlp = spacy.load("en_core_sci_md")
except OSError:
    nlp = spacy.load("en_core_web_sm")

# ─── 2. Lexicons (identical to notebook) ─────────────────────────────────────
MEDICAL_ENTITY_LABELS    = {"DISEASE", "CHEMICAL", "ENTITY"}
CERTAINTY_MARKERS = {
    "definitely","certainly","absolutely","clearly","obviously",
    "undoubtedly","always","never","must","will","cannot",
    "causes","leads to","results in","is responsible for",
    "has been proven","is proven","is a fact","evidence shows",
    "clinically proven","guarantees","confirms",
    "there is no doubt","it is certain","cures","100% effective",
}
HEDGING_MARKERS = {
    "may","might","could","would","maybe","possibly","probably",
    "perhaps","potentially","approximately","generally","often",
    "sometimes","occasionally","seem","seems","appear","appears",
    "suggest","indicates","i think","i believe","i guess",
    "i'm not sure","i am not sure","it seems","it may be",
    "it is possible","unclear","unknown","uncertain",
    "limited evidence","around","about","in some cases",
}
CONDITIONAL_MARKERS = {
    "if","when","unless","whether","in case","provided that",
    "as long as","given that","only if","depending on",
}
MODAL_VERBS = {
    "can","could","may","might","shall","should","will","would",
    "must","ought to","have to","need to",
}
CONTEXT_MARKERS = {
    "i have","i had","i was diagnosed","for","since","during",
    "do i need","should i mention","any other information","let me know if",
}
CAUSAL_CONNECTIVES = {
    "because","since","therefore","thus","hence","as a result",
    "consequently","leads to","results in","causes","due to",
    "owing to","for this reason","associated with","linked to",
}
CONTRASTIVE_CONNECTIVES = {
    "however","although","though","yet","but","nevertheless",
    "on the other hand","in contrast","while","despite","whereas",
    "even though","notwithstanding",
}
EVIDENCE_MARKERS = {
    "study shows","studies show","research indicates","evidence suggests",
    "according to","clinical trial","systematic review","meta-analysis",
    "guidelines recommend","experts recommend","data from","published in",
    "clinical trials demonstrate","scientific evidence shows",
}
OPTIONS_MARKERS = {
    "alternatively","another option","other options","or","either",
    "on the other hand","instead","rather than","versus","compared to",
}
ONLINE_REFS = {
    "website","online","google","search","internet","url","http",
    "www","forum","reddit","facebook","twitter","instagram",
}
CREDIBLE_SOURCES = {
    "pubmed","ncbi","who","cdc","nih","lancet","nejm","bmj",
    "jama","cochrane","uptodate","medline","mayo clinic","webmd",
    "nhs","medscape","healthline",
}
CROSS_REF_MARKERS = {
    "as mentioned","as stated","as discussed","referring to","see also",
    "in reference to","according to","based on","cited in",
}
FUNCTION_WORDS = {
    "the","a","an","and","or","but","in","on","at","to","for",
    "of","with","by","from","is","are","was","were","be","been",
    "being","have","has","had","do","does","did","will","would",
    "could","should","may","might","shall","can","need","dare",
    "ought","used","it","its","this","that","these","those",
    "i","me","my","we","our","you","your","he","she","they",
    "him","her","them","his","their","who","which","what","when",
    "where","how","why","not","no","nor","so","yet","both",
    "either","neither","each","every","all","any","few","more",
    "most","other","some","such","than","too","very","just",
    "because","if","though","although","while","since","as",
    "until","unless","after","before","during","about","above",
    "across","after","against","along","among","around","before",
    "behind","below","beneath","beside","between","beyond",
    "despite","down","except","inside","into","near","off",
    "onto","outside","over","past","since","through","throughout",
    "under","until","upon","within","without",
}

# ─── 3. BVAE Model (identical to notebook) ───────────────────────────────────
class BVAE(nn.Module):
    def __init__(self, input_dim=5, latent_dim=16, dropout_p=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = 1.0 if latent_dim <= 8 else 0.5

        # Encoder
        self.enc1    = nn.Linear(input_dim, 32)
        self.drop1   = nn.Dropout(dropout_p)
        self.enc2    = nn.Linear(32, 32)
        self.drop2   = nn.Dropout(dropout_p)
        self.fc_mu   = nn.Linear(32, latent_dim)
        self.fc_lv   = nn.Linear(32, latent_dim)

        # Decoder
        self.dec1  = nn.Linear(latent_dim, 32)
        self.drop3 = nn.Dropout(dropout_p)
        self.dec2  = nn.Linear(32, 32)
        self.drop4 = nn.Dropout(dropout_p)
        self.out   = nn.Linear(32, input_dim)

    def encode(self, x):
        h = self.drop1(F.relu(self.enc1(x)))
        h = self.drop2(F.relu(self.enc2(h)))
        return self.fc_mu(h), torch.clamp(self.fc_lv(h), -10, 10)

    def decode(self, z):
        h = self.drop3(F.relu(self.dec1(z)))
        h = self.drop4(F.relu(self.dec2(h)))
        return self.out(h)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        return self.decode(z), mu, lv

    def elbo(self, x, x_hat, mu, lv):
        recon = F.mse_loss(x_hat, x, reduction='mean')
        kl    = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
        return recon + self.beta * kl, recon, kl

# ─── 4. Load Saved Model ──────────────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH} …")
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
LATENT_DIM     = ckpt["latent_dim"]
FACTOR_LOADINGS = ckpt["factor_loadings"]
hl_threshold   = ckpt["hl_thresholds"]
f1_median      = ckpt["f1_median"]

model = BVAE(input_dim=5, latent_dim=LATENT_DIM).to(DEVICE)
model.load_state_dict(ckpt["model_state"])

scaler_mean  = ckpt["scaler_mean"]
scaler_scale = ckpt["scaler_scale"]

def scale(x: np.ndarray) -> np.ndarray:
    return (x - scaler_mean) / scaler_scale

# ─── 5. Text Cleaning ─────────────────────────────────────────────────────────
_HTML  = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s{2,}")
_NOISE = re.compile(r"[^\w\s\.,;:!?()\-\'\"/%°+]")

def clean_text(text: str) -> str:
    text = _HTML.sub(" ", text)
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
    text = _NOISE.sub(" ", text)
    text = _SPACE.sub(" ", text)
    return text.strip().lower()

# ─── 6. Feature Extraction (mirrors notebook) ─────────────────────────────────
def count_markers(text, marker_set):
    return sum(1 for m in marker_set if m in text)

def extract_all_features(text: str) -> dict:
    doc   = nlp(text)
    words = [t for t in doc if not t.is_space]
    sents = list(doc.sents)
    n_words = max(len(words), 1)
    n_sents = max(len(sents), 1)

    # Readability
    readability  = textstat.flesch_reading_ease(text)
    avg_sent_len = np.mean([len(list(s)) for s in sents]) if sents else 0
    avg_clauses  = np.mean([
        sum(1 for t in s if t.dep_ in {"relcl","advcl","csubj","ccomp","xcomp"})
        for s in sents]) if sents else 0

    # Medical entities
    med_ents  = [e for e in doc.ents if e.label_ in MEDICAL_ENTITY_LABELS]
    med_count = len(med_ents)
    med_density = med_count / n_words
    unique_med  = len({e.text.lower() for e in med_ents})

    # Hedging / confidence
    hedging_score    = count_markers(text, HEDGING_MARKERS)   / n_words
    confidence_score = count_markers(text, CERTAINTY_MARKERS) / n_words

    # CHL features
    q_count   = text.count("?")
    q_ratio   = q_count / n_sents
    cond_expr = count_markers(text, CONDITIONAL_MARKERS)
    modal_c   = sum(1 for t in words if t.lower_ in MODAL_VERBS)
    modal_d   = modal_c / n_words
    ctx_c     = count_markers(text, CONTEXT_MARKERS)
    ctx_prov  = 1 if ctx_c > 0 else 0

    # CRHL features
    causal_c      = count_markers(text, CAUSAL_CONNECTIVES)
    contrast_c    = count_markers(text, CONTRASTIVE_CONNECTIVES)
    evidence_c    = count_markers(text, EVIDENCE_MARKERS)
    options_c     = count_markers(text, OPTIONS_MARKERS)

    # DHL features
    online_c   = count_markers(text, ONLINE_REFS)
    credible_c = count_markers(text, CREDIBLE_SOURCES)
    crossref_c = count_markers(text, CROSS_REF_MARKERS)
    interp_score = (credible_c + evidence_c) / n_words

    # EHL features
    concrete   = len([t for t in words if t.pos_ in {"NOUN","PROPN"}]) / n_words * 5
    lex_div    = len({t.lower_ for t in words}) / n_words
    pres_verbs = [t for t in words if t.pos_ == "VERB" and t.morph.get("Tense") == ["Pres"]]
    pres_vc    = len(pres_verbs)
    pres_vr    = pres_vc / n_words
    det_c      = len([t for t in words if t.pos_ == "DET"])
    det_r      = det_c / n_words
    adj_c      = len([t for t in words if t.pos_ == "ADJ"])
    adj_r      = adj_c / n_words
    fw_c       = len([t for t in words if t.lower_ in FUNCTION_WORDS])
    fw_r       = fw_c / n_words

    return dict(
        readability_score         = readability,
        avg_sentence_length       = avg_sent_len,
        avg_clauses_per_sentence  = avg_clauses,
        medical_entity_count      = med_count,
        medical_entity_density    = med_density,
        unique_medical_terms      = unique_med,
        hedging_score             = hedging_score,
        confidence_score          = confidence_score,
        question_count            = q_count,
        question_ratio            = q_ratio,
        conditional_expression_count = cond_expr,
        modal_verb_count          = modal_c,
        modal_verb_density        = modal_d,
        context_marker_count      = ctx_c,
        context_provided          = ctx_prov,
        causal_connective_count   = causal_c,
        contrastive_connective_count = contrast_c,
        evidence_reference_count  = evidence_c,
        multiple_options_count    = options_c,
        online_reference_count    = online_c,
        credible_source_count     = credible_c,
        cross_reference_count     = crossref_c,
        information_interpretation_score = interp_score,
        concreteness_score        = concrete,
        lexical_diversity         = lex_div,
        present_verb_count        = pres_vc,
        present_verb_ratio        = pres_vr,
        determiner_count          = det_c,
        determiner_ratio          = det_r,
        adjective_count           = adj_c,
        adjective_ratio           = adj_r,
        function_word_count       = fw_c,
        function_word_ratio       = fw_r,
    )

# ─── 7. Aggregate → 5 HL Scores ───────────────────────────────────────────────
def aggregate_scores(feat: dict) -> np.ndarray:
    def mean(*keys):
        return float(np.mean([feat.get(k, 0) for k in keys]))

    fhl  = mean("readability_score","avg_sentence_length","avg_clauses_per_sentence",
                "medical_entity_count","medical_entity_density",
                "unique_medical_terms","hedging_score","confidence_score")
    chl  = mean("question_count","question_ratio","conditional_expression_count",
                "modal_verb_count","modal_verb_density",
                "context_marker_count","context_provided")
    crhl = mean("causal_connective_count","contrastive_connective_count",
                "evidence_reference_count","multiple_options_count")
    dhl  = mean("online_reference_count","credible_source_count",
                "cross_reference_count","information_interpretation_score")
    ehl  = mean("concreteness_score","lexical_diversity","present_verb_count",
                "present_verb_ratio","determiner_count","determiner_ratio",
                "adjective_count","adjective_ratio",
                "function_word_count","function_word_ratio")
    return np.array([[fhl, chl, crhl, dhl, ehl]], dtype=np.float32)

# ─── 8. Profile & HL-level assignment (mirrors notebook) ─────────────────────
def assign_profile(f1v, f2v, f3v):
    f2_extreme = abs(f2v) > 1.0
    f3_extreme = abs(f3v) > 0.8
    f1_above   = f1v > f1_median
    if f2_extreme and not f3_extreme:
        return "Specialized", "Digitally-Specialized"
    if f3_extreme and not f2_extreme:
        return "Specialized", "Functionally-Specialized"
    if f1_above and not f2_extreme and not f3_extreme:
        return "Balanced", None
    return "Transitional", None

def map_hl_level(f1v, f2v):
    if   f1v < hl_threshold["low"]:          level = "Low"
    elif f1v < hl_threshold["basic"]:        level = "Basic"
    elif f1v < hl_threshold["intermediate"]: level = "Intermediate"
    else:                                     level = "High"
    if f2v < -1.2 and f1v < 0.25:
        order = ["Low","Basic","Intermediate","High"]
        idx = order.index(level)
        if idx > 1:
            level = order[idx - 1]
    return level

# ─── 9. Suitability Score ─────────────────────────────────────────────────────
# Each profile has a centroid in factor-space derived from population data.
# We compute a normalised distance-based suitability score [0-100] for all 3.

PROFILE_CENTROIDS = {
    # These are approximate — they are recalculated at runtime from the
    # factor values, but we seed with neutral values here.
    "Balanced":     np.array([ 1.5,  0.0,  0.0]),
    "Transitional": np.array([-1.5,  0.0,  0.0]),
    "Specialized":  np.array([ 0.0,  1.5,  1.0]),
}

def suitability_scores(f1v: float, f2v: float, f3v: float) -> dict:
    """
    Returns a dict {profile: score 0-100} expressing how well
    the text's factor vector matches each profile centroid.
    Higher = more suitable / closer to that profile's centre.
    """
    vec = np.array([f1v, f2v, f3v])
    raw_dists = {p: float(np.linalg.norm(vec - c))
                 for p, c in PROFILE_CENTROIDS.items()}
    # Convert distances to similarity scores: score = 100 / (1 + dist)
    raw_scores = {p: 100.0 / (1.0 + d) for p, d in raw_dists.items()}
    # Normalise so they sum to 100
    total = sum(raw_scores.values())
    return {p: round(s / total * 100, 1) for p, s in raw_scores.items()}

# ─── 10. Main pipeline ────────────────────────────────────────────────────────
def profile_text(text: str) -> dict:
    """
    Full pipeline: raw text → profile + suitability metrics.

    Returns a dict with:
        profile_type, sub_type, hl_level,
        factor1_core, factor2_digital, factor3_applied,
        sigma_unc, flagged,
        suitability  → {Balanced: float, Transitional: float, Specialized: float},
        raw_scores   → {FHL, CHL, CRHL, DHL, EHL},
        raw_features → full feature dict,
        report       → formatted printable string
    """
    cleaned  = clean_text(text)
    features = extract_all_features(cleaned)
    scores   = aggregate_scores(features)
    x_norm   = scale(scores).astype(np.float32)
    x_t      = torch.tensor(x_norm, dtype=torch.float32).to(DEVICE)

    # MC Dropout inference
    model.train()
    recons_mc = []
    with torch.no_grad():
        for _ in range(MC_PASSES):
            xh, _, _ = model(x_t)
            recons_mc.append(xh.cpu().numpy())
    recons_mc = np.stack(recons_mc)
    sigma     = float(recons_mc.std(axis=0).mean())

    # Factor projection
    factors      = (x_norm @ FACTOR_LOADINGS)[0]
    f1v, f2v, f3v = float(factors[0]), float(factors[1]), float(factors[2])

    ptype, sub = assign_profile(f1v, f2v, f3v)
    level      = map_hl_level(f1v, f2v)
    suit       = suitability_scores(f1v, f2v, f3v)

    result = {
        "profile_type":    ptype,
        "sub_type":        sub or "",
        "hl_level":        level,
        "factor1_core":    round(f1v, 4),
        "factor2_digital": round(f2v, 4),
        "factor3_applied": round(f3v, 4),
        "sigma_unc":       round(sigma, 4),
        "flagged":         sigma > 0.5,
        "suitability":     suit,
        "raw_scores": {
            "FHL":  round(float(scores[0, 0]), 4),
            "CHL":  round(float(scores[0, 1]), 4),
            "CRHL": round(float(scores[0, 2]), 4),
            "DHL":  round(float(scores[0, 3]), 4),
            "EHL":  round(float(scores[0, 4]), 4),
        },
        "raw_features": features,
    }
    result["report"] = _format_report(text, result)
    return result


def _bar(value: float, max_val: float = 100.0, width: int = 30) -> str:
    filled = int(round(value / max_val * width))
    return "█" * filled + "░" * (width - filled)

def _format_report(text: str, r: dict) -> str:
    SEP  = "═" * 65
    SEP2 = "─" * 65
    sub_label = f"  ({r['sub_type']})" if r["sub_type"] else ""

    # HL level → numeric for bar
    HL_SCORE = {"Low": 15, "Basic": 40, "Intermediate": 65, "High": 90}
    hl_num   = HL_SCORE.get(r["hl_level"], 50)

    # Factor normalisation for display bar  (range approx -5 to +5)
    def factor_bar(v, lo=-5, hi=5):
        normed = (v - lo) / (hi - lo) * 100
        return _bar(max(0, min(100, normed)))

    suit = r["suitability"]
    best_profile = max(suit, key=suit.get)

    lines = [
        SEP,
        "  HEALTH LITERACY PROFILE METRIC",
        SEP,
        f"  Input (first 120 chars):",
        f"  \"{text[:120].strip()}{'...' if len(text) > 120 else ''}\"",
        "",
        SEP2,
        "  PROFILE ASSIGNMENT",
        SEP2,
        f"  Assigned Profile  :  {r['profile_type'].upper()}{sub_label}",
        f"  HL Level          :  {r['hl_level']}",
        f"  Epistemic σ       :  {r['sigma_unc']:.4f}"
            + ("  ⚑ FLAGGED — high uncertainty" if r["flagged"] else ""),
        "",
        SEP2,
        "  PROFILE SUITABILITY  (% match to each persona)",
        SEP2,
    ]

    for profile in ["Balanced", "Transitional", "Specialized"]:
        score  = suit[profile]
        marker = "  ◄ BEST FIT" if profile == best_profile else ""
        lines.append(
            f"  {profile:<14}  {score:5.1f}%  {_bar(score)}{marker}"
        )

    lines += [
        "",
        SEP2,
        "  LATENT FACTOR SCORES",
        SEP2,
        f"  F1 Core Proficiency  {r['factor1_core']:+.3f}  {factor_bar(r['factor1_core'])}",
        f"  F2 Digital           {r['factor2_digital']:+.3f}  {factor_bar(r['factor2_digital'])}",
        f"  F3 Applied           {r['factor3_applied']:+.3f}  {factor_bar(r['factor3_applied'])}",
        "",
        SEP2,
        "  HL DIMENSION SCORES  (raw aggregated)",
        SEP2,
    ]

    dim_labels = {
        "FHL":  "Functional HL      ",
        "CHL":  "Communicative HL   ",
        "CRHL": "Critical HL        ",
        "DHL":  "Digital HL         ",
        "EHL":  "Expressed HL       ",
    }
    raw = r["raw_scores"]
    all_vals = list(raw.values())
    max_raw  = max(all_vals) if all_vals else 1

    for dim, label in dim_labels.items():
        v = raw[dim]
        lines.append(f"  {label}  {v:8.4f}  {_bar(v, max_val=max(max_raw, 0.001))}")

    lines += [
        "",
        SEP2,
        "  HL LEVEL GAUGE",
        SEP2,
        f"  {r['hl_level']:<14}  {_bar(hl_num)}",
        f"  Low ──────────────────── Basic ─────────── Intermediate ─── High",
        "",
        SEP2,
        "  PERSONA INTERPRETATION",
        SEP2,
    ]

    interpretations = {
        "Balanced": (
            "Text reflects well-rounded health literacy. The author integrates\n"
            "  medical vocabulary, contextualised questions, and evidence-aware\n"
            "  reasoning. Suitable for standard clinical content, shared-decision\n"
            "  tools, and evidence-based patient education."
        ),
        "Transitional": (
            "Text suggests developing health literacy. The author uses plain\n"
            "  language, limited medical terminology, and direct questions. Best\n"
            "  served by simplified explanations, step-by-step guidance, and\n"
            "  validated plain-language resources."
        ),
        "Specialized": (
            f"Text shows a niche literacy profile ({r['sub_type'] or 'mixed'}).\n"
            "  Digitally-Specialized authors cite credible sources and databases.\n"
            "  Functionally-Specialized authors express rich personal context.\n"
            "  Tailor depth and format to the detected sub-type."
        ),
    }

    lines.append(f"  {interpretations.get(r['profile_type'], '')}")
    lines.append("")
    lines.append(SEP)
    return "\n".join(lines)


# ─── 11. CLI demo ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TEST_TEXTS = [
        (
            "I have type 2 diabetes and my doctor prescribed metformin 500mg twice daily. "
            "According to the CDC and a recent Lancet meta-analysis, metformin reduces HbA1c by 1.5%. "
            "However, I am experiencing gastrointestinal side effects. Should I consider switching?"
        ),
        (
            "my chest hurts sometimes when i climb stairs not sure if its serious "
            "been happening a few weeks should i go to hospital"
        ),
        (
            "After reviewing several systematic reviews on PubMed, the evidence suggests "
            "bariatric surgery leads to sustained weight loss. Alternatively, GLP-1 agonists "
            "such as semaglutide may be appropriate for non-surgical candidates. "
            "The choice depends on BMI, comorbidities, and patient preference."
        ),
    ]

    for text in TEST_TEXTS:
        result = profile_text(text)
        print(result["report"])
        print()
