"""
=============================================================================
  Health Literacy — Profile Dashboard  (Python / Dash)
=============================================================================
  Requirements:
      pip install dash dash-bootstrap-components plotly

  Run:
      python hl_dashboard.py
  Then open  http://127.0.0.1:8050
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import textstat
import spacy
import nltk

from health_literacy_regex_patterns import (
    extract_fhl_features,
    extract_chl_features,
    extract_crhl_features,
    extract_dhl_features,
    extract_ehl_features,
    extract_all_features,
    HealthLiteracyPatterns,
)

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

nltk.download("punkt", quiet=True)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH = "./hl_bvae_model.pt"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MC_PASSES  = 50

PROFILE_COLOURS = {
    "Balanced":     "#22c55e",
    "Transitional": "#f97316",
    "Specialized":  "#a855f7",
}
HL_COLOURS = {
    "Low":          "#ef4444",
    "Basic":        "#f97316",
    "Intermediate": "#3b82f6",
    "High":         "#22c55e",
}
AGENT_COLOURS = {
    "Jargon":      "#f43f5e",   # rose
    "Explanation": "#8b5cf6",   # violet
    "Fluency":     "#06b6d4",   # cyan
    "Coherence":   "#f59e0b",   # amber
}
CLARITY_ANNOTATION_COLOURS = {
    "Jargon":      {"color": "#f43f5e", "label": "Jargon — unexplained medical term",    "bg": "rgba(244,63,94,0.15)"},
    "Explanation": {"color": "#8b5cf6", "label": "Explanation — inline definition",       "bg": "rgba(139,92,246,0.15)"},
    "Coherence":   {"color": "#f59e0b", "label": "Coherence — discourse connective",      "bg": "rgba(245,158,11,0.15)"},
    "Fluency":     {"color": "#06b6d4", "label": "Fluency — complex/long word",           "bg": "rgba(6,182,212,0.15)"},
}
DIMENSION_COLOURS = {
    "FHL":  {"color": "#3b82f6", "label": "Functional HL",    "bg": "rgba(59,130,246,0.15)"},
    "CHL":  {"color": "#eab308", "label": "Communicative HL", "bg": "rgba(234,179,8,0.15)"},
    "CRHL": {"color": "#f97316", "label": "Critical HL",      "bg": "rgba(249,115,22,0.15)"},
    "DHL":  {"color": "#a855f7", "label": "Digital HL",       "bg": "rgba(168,85,247,0.15)"},
    "EHL":  {"color": "#06b6d4", "label": "Expressed HL",     "bg": "rgba(6,182,212,0.15)"},
}
# ─── Color helper — defined first, used everywhere below ──────────────────────
def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    try:
        hx = str(hex_color).strip().lstrip("#").ljust(6, "0")[:6]
        r  = int(hx[0:2], 16)
        g  = int(hx[2:4], 16)
        b  = int(hx[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    except Exception:
        return f"rgba(99,102,241,{alpha})"

# ─── spaCy ────────────────────────────────────────────────────────────────────
try:
    nlp = spacy.load("en_core_sci_md")
except OSError:
    nlp = spacy.load("en_core_web_sm")

# Used by clarity_consensus and annotate_clarity (spaCy NER-based functions)
MEDICAL_ENTITY_LABELS = {"DISEASE", "CHEMICAL", "ENTITY"}

# ─── BVAE ─────────────────────────────────────────────────────────────────────
class BVAE(nn.Module):
    def __init__(self, input_dim=5, latent_dim=16, dropout_p=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = 1.0 if latent_dim <= 8 else 0.5
        self.enc1  = nn.Linear(input_dim, 32)
        self.drop1 = nn.Dropout(dropout_p)
        self.enc2  = nn.Linear(32, 32)
        self.drop2 = nn.Dropout(dropout_p)
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_lv = nn.Linear(32, latent_dim)
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

# ─── Load model ───────────────────────────────────────────────────────────────
print("Loading model …")
ckpt            = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
LATENT_DIM      = ckpt["latent_dim"]
FACTOR_LOADINGS = ckpt["factor_loadings"]
hl_threshold    = ckpt["hl_thresholds"]
f1_median       = ckpt["f1_median"]
scaler_mean     = ckpt["scaler_mean"]
scaler_scale    = ckpt["scaler_scale"]

model = BVAE(input_dim=5, latent_dim=LATENT_DIM).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
print("Model loaded ✓")

def scale(x): return (x - scaler_mean) / scaler_scale

# ─── Text pipeline ────────────────────────────────────────────────────────────
_HTML  = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s{2,}")
_NOISE = re.compile(r"[^\w\s\.,;:!?()\-\'\"/%°+]")

def clean_text(text):
    text = _HTML.sub(" ", text)
    text = text.replace("\r\n"," ").replace("\n"," ").replace("\t"," ")
    text = _NOISE.sub(" ", text)
    return _SPACE.sub(" ", text).strip().lower()

def extract_features(text):
    """
    Extract feature dictionary using regex + POS patterns.
    Delegates to health_literacy_regex_patterns.extract_all_features().
    Output keys are identical to the previous implementation so
    aggregate_scores() and all downstream code remain unchanged.
    """
    return extract_all_features(text)

def aggregate_scores(feat):
    def m(*keys): return float(np.mean([feat.get(k, 0) for k in keys]))
    fhl  = m("readability_score","avg_sentence_length","avg_clauses_per_sentence","medical_entity_count","medical_entity_density","unique_medical_terms","hedging_score","confidence_score")
    chl  = m("question_count","question_ratio","conditional_expression_count","modal_verb_count","modal_verb_density","context_marker_count","context_provided")
    crhl = m("causal_connective_count","contrastive_connective_count","evidence_reference_count","multiple_options_count")
    dhl  = m("online_reference_count","credible_source_count","cross_reference_count","information_interpretation_score")
    ehl  = m("concreteness_score","lexical_diversity","present_verb_count","present_verb_ratio","determiner_count","determiner_ratio","adjective_count","adjective_ratio","function_word_count","function_word_ratio")
    return np.array([[fhl, chl, crhl, dhl, ehl]], dtype=np.float32)

def assign_profile(f1v, f2v, f3v):
    if abs(f2v) > 1.0 and abs(f3v) <= 0.8: return "Specialized", "Digitally-Specialized"
    if abs(f3v) > 0.8 and abs(f2v) <= 1.0: return "Specialized", "Functionally-Specialized"
    if f1v > f1_median:                      return "Balanced", ""
    return "Transitional", ""

def map_hl_level(f1v, f2v):
    if   f1v < hl_threshold["low"]:          level = "Low"
    elif f1v < hl_threshold["basic"]:        level = "Basic"
    elif f1v < hl_threshold["intermediate"]: level = "Intermediate"
    else:                                     level = "High"
    if f2v < -1.2 and f1v < 0.25:
        order = ["Low","Basic","Intermediate","High"]
        idx = order.index(level)
        if idx > 1: level = order[idx - 1]
    return level

def suitability(f1v, f2v, f3v):
    centroids = {
        "Balanced":     np.array([ 1.5, 0.0, 0.0]),
        "Transitional": np.array([-1.5, 0.0, 0.0]),
        "Specialized":  np.array([ 0.0, 1.5, 1.0]),
    }
    vec = np.array([f1v, f2v, f3v])
    raw = {p: 100.0 / (1.0 + float(np.linalg.norm(vec - c))) for p, c in centroids.items()}
    tot = sum(raw.values())
    return {p: round(s / tot * 100, 1) for p, s in raw.items()}

def analyse(text):
    cleaned  = clean_text(text)
    feat     = extract_features(cleaned)
    scores   = aggregate_scores(feat)
    x_norm   = scale(scores).astype(np.float32)
    x_t      = torch.tensor(x_norm, dtype=torch.float32).to(DEVICE)
    model.train()
    recons = []
    with torch.no_grad():
        for _ in range(MC_PASSES):
            xh, _, _ = model(x_t)
            recons.append(xh.cpu().numpy())
    sigma    = float(np.stack(recons).std(axis=0).mean())
    factors  = (x_norm @ FACTOR_LOADINGS)[0]
    f1v, f2v, f3v = float(factors[0]), float(factors[1]), float(factors[2])
    ptype, sub = assign_profile(f1v, f2v, f3v)
    level      = map_hl_level(f1v, f2v)
    suit       = suitability(f1v, f2v, f3v)
    raw_scores = {
        "FHL":  round(float(scores[0, 0]), 4),
        "CHL":  round(float(scores[0, 1]), 4),
        "CRHL": round(float(scores[0, 2]), 4),
        "DHL":  round(float(scores[0, 3]), 4),
        "EHL":  round(float(scores[0, 4]), 4),
    }
    return dict(profile=ptype, sub_type=sub, hl_level=level, flagged=sigma > 0.5,
                f1=round(f1v, 4), f2=round(f2v, 4), f3=round(f3v, 4),
                sigma=round(sigma, 4), suitability=suit, raw_scores=raw_scores)

# ─── Guidance ─────────────────────────────────────────────────────────────────
GUIDANCE = {
    ("Transitional","Balanced"): {
        "elevate": ["Use more medical terminology and disease/drug names","Add causal reasoning (because, therefore, as a result)","Include personal health context (diagnosis, duration, treatment)","Reference credible sources (CDC, NHS, clinical guidelines)","Use contrastive connectives (however, although, despite)"],
        "reduce":  ["Avoid very short, fragmented sentences","Reduce over-reliance on questions without context","Limit informal/colloquial phrasing"],
    },
    ("Transitional","Specialized"): {
        "elevate": ["Cite specific databases or journals (PubMed, Lancet, JAMA) for Digital sub-type","Describe your personal condition in rich detail for Functional sub-type","Reference multiple treatment options and compare them","Use evidence markers (systematic review, meta-analysis, clinical trial)"],
        "reduce":  ["Reduce vague or unspecified claims","Avoid generic statements without data or context"],
    },
    ("Balanced","Specialized"): {
        "elevate": ["Cite specific studies, DOIs, or clinical databases (for Digital sub-type)","Add detailed first-person narrative of lived experience (for Functional sub-type)","Push F2 digital factor above 1.0 by referencing credible online sources","Push F3 applied factor above 0.8 with concrete application of knowledge"],
        "reduce":  ["Balance — being Specialized is niche, not always better","Reduce generic breadth if targeting a specific expertise axis"],
    },
    ("Specialized","Balanced"): {
        "elevate": ["Broaden across all 5 HL dimensions","Add communicative context (questions, modal verbs, conditionals)","Include both evidence references AND personal application","Use hedging language to signal nuanced reasoning"],
        "reduce":  ["Reduce over-reliance on a single dimension (digital or applied)","Avoid purely academic or purely personal framing"],
    },
    ("Balanced","Transitional"): {
        "elevate": [],
        "reduce":  ["Note: Transitional represents lower overall literacy — moving here is a downgrade","Simplify vocabulary, reduce sentence complexity, remove evidence references if targeting a lay audience"],
    },
    ("Specialized","Transitional"): {
        "elevate": [],
        "reduce":  ["Note: Transitional represents lower overall literacy","Remove technical references, simplify to plain language if targeting a general audience"],
    },
}

def get_guidance(current, target):
    return GUIDANCE.get((current, target), {
        "elevate": ["Improve overall medical vocabulary and sentence structure","Add more contextual reasoning and evidence references"],
        "reduce":  ["Reduce elements that push toward the current profile"],
    })
# ─── ClarityConsensus-Agents (lightweight, reuses spaCy + textstat) ───────────
# Mirrors the MedEval-Agents framework:
# Jargon(0.40) + Explanation(0.30) + Fluency(0.15) + Coherence(0.15)

def clarity_consensus(text: str) -> dict:
    """
    Lightweight ClarityConsensus score replicating the multi-agent framework.

    Agents:
      JargonAgent      — ratio of medical terms that lack a nearby explanation
      ExplanationAgent — coverage of parenthetical / appositive definitions
      FluencyAgent     — Flesch Reading Ease normalised to [0,1]
      CoherenceAgent   — mean cosine similarity between consecutive spaCy sentence vectors
    """
    doc   = nlp(text)
    sents = list(doc.sents)

    # ── Jargon helpers ────────────────────────────────────────────────────────

    import re

    JARGON_MORPHOLOGY = re.compile(
        r'\b\w*(?:'
        
        # --- COMMON MEDICAL SUFFIXES ---
        r'ology|onomy|ography|ometry|'
        r'itis|osis|iasis|asis|oma|omas|omata|'
        r'emia|uria|algia|dynia|pathy|plegia|paresis|'
        r'trophy|genesis|lysis|rrhea|rrhage|rrhagia|'
        r'sclerosis|stenosis|malacia|megaly|'
        r'penia|cytosis|phasia|phagia|phonia|'
        r'blastoma|carcinoma|sarcoma|adenoma|fibroma|'
        
        # --- SURGICAL / PROCEDURAL ---
        r'ectomy|ostomy|otomy|plasty|scopy|oscopy|'
        r'graphy|gram|meter|metry|therapy|desis|pexy|'
        r'centesis|tripsy|rrhaphy|lysis|'
        
        # --- CELL / BIOLOGY ---
        r'cyte|blast|clast|plasm|some|gen|genesis|'
        r'phage|phil|phobe|kine|taxis|'
        
        # --- PREFIXES (CONDITIONS / QUANTITIES) ---
        r'hyper|hypo|normo|dys|eu|'
        r'brady|tachy|poly|oligo|pan|'
        r'neo|pseudo|auto|hetero|homo|'
        r'inter|intra|extra|sub|super|'
        r'pre|post|peri|endo|ecto|'
        r'anti|pro|contra|'
        
        # --- ORGANS / SYSTEM ROOTS ---
        r'cardio|neuro|hepato|nephro|gastro|dermato|'
        r'osteo|myo|angio|vasculo|pulmo|pneumo|'
        r'encephalo|ophthalmo|otol|laryngo|'
        r'rhino|stomato|procto|colo|entero|'
        r'spleno|thyro|adreno|hystero|orchido|'
        
        # --- BIOCHEMISTRY / SUBSTANCES ---
        r'lipid|glyco|gluco|proteo|nucleo|'
        r'enzym|hormon|toxin|acid|base|'
        
        # --- MICROBIOLOGY / PATHOGENS ---
        r'bacter|virus|viral|fung|myco|parasite|helminth|'
        
        # --- GENERAL SCIENTIFIC ---
        r'logy|nomy|ics|iatry|iatric|genic|genic|genous|'
        
        r')\w*\b',
        re.IGNORECASE
    )

    import re

    EXPLANATION_CUES = re.compile(
        r'(?:'
        
        # --- PARENTHESIS (explanations, clarifications) ---
        r'\(([^)]{5,})\)'
        
        r'|'
        
        # --- CLASSIC DEFINITION PHRASES ---
        r',?\s*(?:'
        r'also known as|aka|'
        r'defined as|is defined as|can be defined as|'
        r'refers to|is referring to|'
        r'means|meaning|'
        r'i\.e\.|e\.g\.|'
        r'in other words|that is|that is to say|'
        r'which is|which means|which refers to|'
        r'what we call|known as|called|termed|'
        r')\s'
        
        r'|'
        
        # --- APPOSITIONS (noun phrase explanations) ---
        r',\s*(?:a|an|the)\s+[^,]{3,},'
        
        r'|'
        
        # --- DASH / COLON EXPLANATIONS ---
        r'\s*[-–—:]\s*(?:'
        r'a|an|the|'
        r'meaning|defined as|refers to|'
        r')?\s*[^.,;]{3,}'
        
        r'|'
        
        # --- VERB-BASED EXPLANATIONS ---
        r'\b(?:'
        r'is|are|was|were|'
        r'means|denotes|indicates|describes|represents|'
        r'consists of|comprises|involves|'
        r')\s+(?:a|an|the)?\s*[^.,;]{3,}'
        
        r'|'
        
        # --- SIMPLIFICATION / PATIENT-FRIENDLY ---
        r',?\s*(?:'
        r'simply put|put simply|basically|'
        r'in simple terms|to put it simply|'
        r')\s'
        
        r')',
        re.IGNORECASE
    )

    def _is_rare_long(token) -> bool:
        return (
            token.is_alpha
            and not token.is_stop
            and len(token.text) >= 8
            and token.prob < -13.0
        )

    def _detect_jargon_tokens(doc, text: str) -> list:
        jargon_spans = {}

        for m in JARGON_MORPHOLOGY.finditer(text):
            word = m.group(0)
            if len(word) < 6:
                continue
            jargon_spans[m.start()] = (word, m.start(), m.end())

        for token in doc:
            if _is_rare_long(token):
                if token.idx not in jargon_spans:
                    jargon_spans[token.idx] = (token.text, token.idx, token.idx + len(token.text))

        return list(jargon_spans.values())

    # ── Agent 1: Jargon ───────────────────────────────────────────────────────
    jargon_hits = _detect_jargon_tokens(doc, text)
    n_terms     = len(jargon_hits)

    explained = 0
    for (word, start_char, end_char) in jargon_hits:
        window = text[end_char: end_char + 120]
        if EXPLANATION_CUES.search(window):
            explained += 1

    unexplained  = max(n_terms - explained, 0)
    jargon_score = 1.0 - min(1.0, unexplained / max(n_terms, 1))
    jargon_raw   = n_terms / max(len([t for t in doc if not t.is_space]), 1)

    # ── Agent 2: Explanation coverage ─────────────────────────────────────────
    explanation_score = explained / max(n_terms, 1) if n_terms > 0 else 1.0

    # ── Agent 3: Fluency — Flesch Reading Ease → [0,1] ────────────────────────
    fre           = textstat.flesch_reading_ease(text)
    fluency_score = float(min(1.0, max(0.0, fre / 100.0)))

    # ── Agent 4: Coherence — mean pairwise cosine between consecutive sentences
    coherence_score = 1.0
    flagged_transitions = []
    if len(sents) >= 2:
        sims = []
        for s1, s2 in zip(sents[:-1], sents[1:]):
            v1 = s1.vector; v2 = s2.vector
            n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                sim = float(np.dot(v1, v2) / (n1 * n2))
                sims.append(sim)
                if sim < 0.55:
                    flagged_transitions.append((str(s1)[:60], str(s2)[:60], round(sim, 3)))
        coherence_score = float(np.mean(sims)) if sims else 1.0

    # ── Weighted final score ───────────────────────────────────────────────────
    W_J, W_E, W_F, W_C = 0.40, 0.30, 0.15, 0.15
    final = W_J * jargon_score + W_E * explanation_score + W_F * fluency_score + W_C * coherence_score

    return {
        "final":               round(final, 4),
        "jargon":              round(jargon_score, 4),
        "explanation":         round(explanation_score, 4),
        "fluency":             round(fluency_score, 4),
        "coherence":           round(coherence_score, 4),
        "n_terms":             n_terms,
        "n_explained":         explained,
        "fre":                 round(fre, 2),
        "flagged_transitions": flagged_transitions,
    }
def annotate_text(text: str):
    """
    Scans the text for HL dimension markers and returns a list of
    (segment, dim_or_None) tuples for rendering as coloured spans.
    Priority: CRHL > DHL > CHL > FHL  (most specific first).
    Uses regex patterns from HealthLiteracyPatterns for improved coverage.
    """
    lower  = text.lower()
    n      = len(text)
    labels = [None] * n   # character-level dimension label

    P = HealthLiteracyPatterns

    dim_patterns = [
        ("CRHL", P.CAUSAL_PATTERNS + P.CONTRASTIVE_PATTERNS
                 + P.EVIDENCE_PATTERNS + P.OPTIONS_PATTERNS),
        ("DHL",  P.TRUSTED_SOURCES_PATTERNS + [P.URL_PATTERN]
                 + P.CROSS_REF_PATTERNS),
        ("CHL",  P.CONTEXT_PATTERNS + P.CONDITIONAL_PATTERNS),
        ("FHL",  P.HEDGING_PATTERNS + P.CERTAINTY_PATTERNS),
    ]

    for dim, patterns in dim_patterns:
        for pat_str in patterns:
            try:
                pattern = re.compile(pat_str, re.IGNORECASE)
                for m in pattern.finditer(lower):
                    s, e = m.start(), m.end()
                    if all(labels[i] is None for i in range(s, e)):
                        for i in range(s, e):
                            labels[i] = dim
            except re.error:
                continue

    # group consecutive chars with the same label into segments
    segments, i = [], 0
    while i < n:
        label = labels[i]
        j = i + 1
        while j < n and labels[j] == label:
            j += 1
        segments.append((text[i:j], label))
        i = j

    return segments
def annotate_clarity(text: str) -> list:
    doc    = nlp(text)
    n      = len(text)
    labels = [None] * n

    # ── Fluency ──────────────────────────────────────────────────────────────
    for token in doc:
        if token.is_alpha and len(token.text) >= 7:
            if textstat.syllable_count(token.text) >= 3:
                for i in range(token.idx, min(token.idx + len(token.text), n)):
                    labels[i] = "Fluency"

    # ── Coherence ─────────────────────────────────────────────────────────────
    COHERENCE_MARKERS = {
        "however","although","though","yet","nevertheless","furthermore",
        "moreover","therefore","thus","hence","consequently","in contrast",
        "on the other hand","in addition","additionally","despite","whereas",
        "as a result","in conclusion","in summary","for example","for instance",
        "that is","in other words","specifically","notably","importantly",
    }
    lower = text.lower()
    for marker in sorted(COHERENCE_MARKERS, key=len, reverse=True):
        pattern = re.compile(r'\b' + re.escape(marker) + r'\b')
        for m in pattern.finditer(lower):
            for i in range(m.start(), min(m.end(), n)):
                labels[i] = "Coherence"

    # ── Jargon ────────────────────────────────────────────────────────────────
    for ent in doc.ents:
        if ent.label_ in MEDICAL_ENTITY_LABELS:
            for i in range(ent.start_char, min(ent.end_char, n)):
                labels[i] = "Jargon"

    # ── Explanation ───────────────────────────────────────────────────────────
    EXPLANATION_CUES = re.compile(
        r'\(([^)]{5,})\)'
        r'|,?\s*(also known as|defined as|refers to|meaning|i\.e\.)[^,\.;]{0,80}',
        re.IGNORECASE
    )
    for m in EXPLANATION_CUES.finditer(text):
        for i in range(m.start(), min(m.end(), n)):
            labels[i] = "Explanation"

    # ── Segment ───────────────────────────────────────────────────────────────
    segments, i = [], 0
    while i < n:
        label = labels[i]
        j = i + 1
        while j < n and labels[j] == label:
            j += 1
        segments.append((text[i:j], label))
        i = j

    return segments   # ← this line must be present and at function level

# ─── DASH APP ─────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
                title="HL Profile Dashboard")

app.layout = dbc.Container(fluid=True, style={"background":"#0f172a","minHeight":"100vh","padding":"0"}, children=[

    html.Div(style={"background":"linear-gradient(135deg,#1e293b 0%,#0f172a 100%)",
                    "borderBottom":"1px solid #334155","padding":"24px 40px","marginBottom":"28px"}, children=[
        html.H1("Health Literacy Profile Dashboard",
                style={"color":"#f1f5f9","fontFamily":"Georgia,serif","fontWeight":"700",
                       "fontSize":"1.9rem","margin":"0","letterSpacing":"-0.5px"}),
        html.P("Submit any health-related text to receive a full profile report and upgrade guidance.",
               style={"color":"#94a3b8","margin":"6px 0 0","fontSize":"0.9rem"}),
    ]),

    dbc.Container(fluid=False, style={"maxWidth":"1300px","padding":"0 24px"}, children=[
        dbc.Card(style={"background":"#1e293b","border":"1px solid #334155",
                        "borderRadius":"12px","marginBottom":"24px"}, children=[
            dbc.CardBody([
                html.Label("Paste your health text below",
                           style={"color":"#94a3b8","fontSize":"0.85rem","fontWeight":"600",
                                  "letterSpacing":"0.05em","textTransform":"uppercase",
                                  "marginBottom":"10px","display":"block"}),
                dcc.Textarea(id="input-text",
                             placeholder="e.g.  I have type 2 diabetes and my doctor prescribed metformin 500mg twice daily…",
                             style={"width":"100%","height":"130px","background":"#0f172a","color":"#f1f5f9",
                                    "border":"1px solid #475569","borderRadius":"8px","padding":"14px",
                                    "fontSize":"0.95rem","fontFamily":"'Courier New',monospace","resize":"vertical"}),
                html.Div(style={"display":"flex","justifyContent":"space-between",
                                "alignItems":"center","marginTop":"14px"}, children=[
                    html.Span(id="word-count", style={"color":"#64748b","fontSize":"0.82rem"}),
                    dbc.Button("Analyse Text →", id="analyse-btn", n_clicks=0,
                               style={"background":"linear-gradient(135deg,#6366f1,#8b5cf6)",
                                      "border":"none","borderRadius":"8px","padding":"10px 28px",
                                      "fontWeight":"600","fontSize":"0.95rem"}),
                ]),
            ])
        ]),
        dcc.Loading(id="loading", type="circle", color="#6366f1", children=[
            html.Div(id="results-area"),
        ]),
    ]),
])

# ── Word count ────────────────────────────────────────────────────────────────
@app.callback(Output("word-count","children"), Input("input-text","value"))
def update_wc(text):
    if not text: return ""
    return f"{len(text.split())} words"

# ── Main analysis callback ─────────────────────────────────────────────────────
@app.callback(
    Output("results-area", "children"),
    Input("analyse-btn", "n_clicks"),
    State("input-text", "value"),
    prevent_initial_call=True,
)
def run_analysis(n, text):
    if not text or len(text.strip()) < 20:
        return dbc.Alert("Please enter at least 20 characters of text.", color="warning")

    try:
        r = analyse(text)
        clarity = clarity_consensus(text)
    except Exception as e:
        return dbc.Alert(f"Analysis error: {e}", color="danger")

    profile    = r["profile"]
    sub        = r["sub_type"]
    hl         = r["hl_level"]
    suit       = r["suitability"]
    raw        = r["raw_scores"]
    f1, f2, f3 = r["f1"], r["f2"], r["f3"]
    sigma      = r["sigma"]
    flagged    = r["flagged"]
    pcol       = PROFILE_COLOURS.get(str(profile).strip(), "#6366f1")
    hlcol      = HL_COLOURS.get(hl, "#6366f1")
    label      = profile + (f" · {sub}" if sub else "")

    # ── Hero ──
    hero = dbc.Row(style={"marginBottom":"20px"}, children=[
        dbc.Col(md=4, children=[
            html.Div(style={"background":"#1e293b",
                            "border":f"2px solid {pcol}",
                            "borderRadius":"12px","padding":"22px","height":"100%"}, children=[
                html.P("ASSIGNED PROFILE",
                       style={"color":"#64748b","fontSize":"0.75rem","fontWeight":"700",
                              "letterSpacing":"0.1em","margin":"0 0 8px"}),
                html.H2(label,
                        style={"color":pcol,"fontFamily":"Georgia,serif","fontSize":"1.6rem",
                               "fontWeight":"700","margin":"0 0 6px"}),
                html.Div(style={"display":"flex","gap":"10px","alignItems":"center","marginTop":"10px"}, children=[
                    html.Span(hl, style={"background":hlcol,"color":"#fff","borderRadius":"20px",
                                         "padding":"4px 14px","fontSize":"0.8rem","fontWeight":"700"}),
                    html.Span(f"σ = {sigma:.4f}", style={"color":"#94a3b8","fontSize":"0.82rem"}),
                    html.Span("⚑ FLAGGED",
                              style={"color":"#ef4444","fontSize":"0.8rem","fontWeight":"700"})
                    if flagged else html.Span(),
                ]),
            ])
        ]),
        dbc.Col(md=8, children=[
            html.Div(style={"background":"#1e293b","border":"1px solid #334155",
                            "borderRadius":"12px","padding":"22px","height":"100%"}, children=[
                html.P("PROFILE SUITABILITY",
                       style={"color":"#64748b","fontSize":"0.75rem","fontWeight":"700",
                              "letterSpacing":"0.1em","margin":"0 0 14px"}),
                *[html.Div(style={"marginBottom":"10px"}, children=[
                    html.Div(style={"display":"flex","justifyContent":"space-between","marginBottom":"4px"}, children=[
                        html.Span(p, style={"color":"#e2e8f0","fontSize":"0.88rem","fontWeight":"600"}),
                        html.Span(f"{suit[p]}%",
                                  style={"color":PROFILE_COLOURS[p],"fontSize":"0.88rem","fontWeight":"700"}),
                    ]),
                    html.Div(style={"background":"#0f172a","borderRadius":"4px","height":"8px","overflow":"hidden"}, children=[
                        html.Div(style={"width":f"{suit[p]}%","height":"100%",
                                        "background":PROFILE_COLOURS[p],"borderRadius":"4px",
                                        "transition":"width 0.8s ease"}),
                    ]),
                ]) for p in ["Balanced","Transitional","Specialized"]],
            ])
        ]),
    ])
    # ── ClarityConsensus section ───────────────────────────────────────────────
    def score_colour(v: float) -> str:
        if v >= 0.80: return "#22c55e"
        if v >= 0.60: return "#f97316"
        return "#ef4444"

    def score_bar(v: float, width: int = 28) -> str:
        filled = int(round(v * width))
        return "█" * filled + "░" * (width - filled)

    agent_rows = [
    ("Jargon",      clarity["jargon"],      "0.40",
     f"{clarity['n_terms']} medical terms, {clarity['n_explained']} explained",
     "Increase explained jargon via parentheticals e.g. 'metformin (a biguanide antidiabetic)'"),
    ("Explanation", clarity["explanation"], "0.30",
     f"{clarity['n_explained']}/{clarity['n_terms']} terms have inline definitions",
     "Add definitions for unexplained medical terms"),
    ("Fluency",     clarity["fluency"],     "0.15",
     f"Flesch Reading Ease = {clarity['fre']}",
     "Shorten sentences, reduce syllable density"),
    ("Coherence",   clarity["coherence"],   "0.15",
     f"{len(clarity['flagged_transitions'])} low-similarity transitions detected",
     "Add linking phrases between abrupt topic changes"),
]
    final_col = score_colour(clarity["final"])

    clarity_section = html.Div(
        style={"background":"#1e293b","border":f"1px solid {hex_to_rgba(final_col, 0.4)}",
               "borderRadius":"12px","padding":"20px","marginBottom":"20px"},
        children=[

            # Header row
            html.Div(style={"display":"flex","justifyContent":"space-between",
                            "alignItems":"center","marginBottom":"18px"}, children=[
                html.Div([
                    html.P("CLARITYCONSENSUS SCORE",
                           style={"color":"#64748b","fontSize":"0.75rem","fontWeight":"700",
                                  "letterSpacing":"0.1em","margin":"0 0 4px"}),
                    html.P("Multi-agent comprehensibility evaluation · Jargon × Explanation × Fluency × Coherence",
                           style={"color":"#475569","fontSize":"0.76rem","margin":"0"}),
                ]),
                html.Div(style={"textAlign":"right"}, children=[
                    html.Span(f"{clarity['final']:.3f}",
                              style={"color":final_col,"fontSize":"2rem","fontWeight":"800",
                                     "fontFamily":"Courier New","lineHeight":"1"}),
                    html.Span(" / 1.000",
                              style={"color":"#475569","fontSize":"0.85rem"}),
                ]),
            ]),

            # Agent breakdown rows
            *[html.Div(style={
    "display":"flex","gap":"14px","alignItems":"flex-start",
    "marginBottom":"12px","paddingBottom":"12px",
    "borderBottom":f"1px solid {hex_to_rgba(AGENT_COLOURS[name], 0.2)}",
    "borderLeft":  f"3px solid {AGENT_COLOURS[name]}",
    "paddingLeft": "12px",
}, children=[
    # Agent name + weight
    html.Div(style={"minWidth":"110px"}, children=[
        html.Span(name, style={
            "color":      AGENT_COLOURS[name],
            "fontSize":   "0.85rem",
            "fontWeight": "700",
        }),
        html.Br(),
        html.Span(f"w = {weight}", style={"color":"#475569","fontSize":"0.75rem"}),
    ]),
    # Score bar + detail
    html.Div(style={"flex":"1"}, children=[
        html.Div(style={"display":"flex","justifyContent":"space-between",
                        "marginBottom":"4px"}, children=[
            html.Span(detail, style={"color":"#94a3b8","fontSize":"0.78rem"}),
            html.Span(f"{score:.3f}", style={
                "color":      AGENT_COLOURS[name],
                "fontWeight": "700",
                "fontSize":   "0.85rem",
                "fontFamily": "Courier New",
            }),
        ]),
        # Background track
        html.Div(style={"background":"#0f172a","borderRadius":"4px",
                        "height":"6px","overflow":"hidden"}, children=[
            html.Div(style={
                "width":        f"{score*100:.1f}%",
                "height":       "100%",
                "background":   f"linear-gradient(90deg, {hex_to_rgba(AGENT_COLOURS[name], 0.5)}, {AGENT_COLOURS[name]})",
                "borderRadius": "4px",
            }),
        ]),
        html.Span(f"▲ {tip}", style={
            "color":"#475569","fontSize":"0.75rem",
            "marginTop":"4px","display":"block",
        }) if score < 0.75 else html.Span(),
    ]),
]) for name, score, weight, detail, tip in agent_rows],
            # Flagged coherence transitions
            *([
                html.Hr(style={"borderColor":"#1e3a5f","margin":"8px 0"}),
                html.P("LOW-COHERENCE TRANSITIONS",
                       style={"color":"#64748b","fontSize":"0.72rem","fontWeight":"700",
                              "letterSpacing":"0.08em","margin":"0 0 8px"}),
                *[html.Div(style={"background":"#0f172a","borderRadius":"6px",
                                  "padding":"8px 12px","marginBottom":"6px",
                                  "borderLeft":"3px solid #f97316"}, children=[
                    html.Span(f"sim={t[2]}  ",
                              style={"color":"#f97316","fontFamily":"Courier New",
                                     "fontSize":"0.78rem","fontWeight":"700"}),
                    html.Span(f'"{t[0]}..." -> "{t[1]}..."',
                              style={"color":"#94a3b8","fontSize":"0.78rem"}),
                ]) for t in clarity["flagged_transitions"][:4]],
            ] if clarity["flagged_transitions"] else []),
        ]
    )
    # ── Radar chart ──
    dims       = ["FHL","CHL","CRHL","DHL","EHL"]
    dim_labels = ["Functional","Communicative","Critical","Digital","Expressed"]
    vals       = [raw[d] for d in dims]
    mx         = max(vals) if max(vals) > 0 else 1
    norm_vals  = [v / mx * 100 for v in vals]

    radar_fig = go.Figure(go.Scatterpolar(
        r=norm_vals + [norm_vals[0]],
        theta=dim_labels + [dim_labels[0]],
        fill="toself",
        fillcolor=hex_to_rgba(pcol, 0.2),
        line=dict(color=pcol, width=2.5),
        marker=dict(size=6, color=pcol),
    ))
    radar_fig.update_layout(
        polar=dict(
            bgcolor="#0f172a",
            radialaxis=dict(visible=True, range=[0,100], gridcolor="#334155",
                            tickfont=dict(color="#64748b", size=9)),
            angularaxis=dict(gridcolor="#334155", tickfont=dict(color="#cbd5e1", size=11)),
        ),
        paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
        margin=dict(l=40, r=40, t=30, b=30), height=290,
        showlegend=False,
    )

    # ── Factor bars ──
    factor_fig = go.Figure()
    for fname, fval, colour in [("F1  Core", f1, "#6366f1"),
                                  ("F2  Digital", f2, "#06b6d4"),
                                  ("F3  Applied", f3, "#f59e0b")]:
        normed = (fval + 5) / 10 * 100
        factor_fig.add_trace(go.Bar(
            x=[normed], y=[fname], orientation="h",
            marker=dict(color=colour, line=dict(width=0)),
            text=f"{fval:+.3f}", textposition="inside",
            textfont=dict(color="#fff", size=11, family="Courier New"),
            width=0.5,
        ))
    factor_fig.add_vline(x=50, line_dash="dash", line_color="#475569", line_width=1)
    factor_fig.update_layout(
        barmode="overlay", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
        xaxis=dict(range=[0,100], showticklabels=False, gridcolor="#1e293b"),
        yaxis=dict(tickfont=dict(color="#cbd5e1", size=11)),
        margin=dict(l=10, r=10, t=10, b=10), height=160,
        showlegend=False,
    )

    charts = dbc.Row(style={"marginBottom":"20px"}, children=[
        dbc.Col(md=5, children=[
            html.Div(style={"background":"#1e293b","border":"1px solid #334155",
                            "borderRadius":"12px","padding":"18px"}, children=[
                html.P("HL DIMENSION RADAR",
                       style={"color":"#64748b","fontSize":"0.75rem","fontWeight":"700",
                              "letterSpacing":"0.1em","margin":"0 0 4px"}),
                dcc.Graph(figure=radar_fig, config={"displayModeBar":False}),
            ])
        ]),
        dbc.Col(md=7, children=[
            html.Div(style={"background":"#1e293b","border":"1px solid #334155",
                            "borderRadius":"12px","padding":"18px","height":"100%"}, children=[
                html.P("LATENT FACTOR SCORES  (range −5 to +5, midline = 0)",
                       style={"color":"#64748b","fontSize":"0.75rem","fontWeight":"700",
                              "letterSpacing":"0.1em","margin":"0 0 4px"}),
                dcc.Graph(figure=factor_fig, config={"displayModeBar":False}),
                html.Hr(style={"borderColor":"#334155","margin":"12px 0"}),
                html.P("RAW HL SCORES",
                       style={"color":"#64748b","fontSize":"0.75rem","fontWeight":"700",
                              "letterSpacing":"0.1em","margin":"0 0 10px"}),
                html.Div(style={"display":"flex","gap":"10px","flexWrap":"wrap"}, children=[
                    html.Div(style={"background":"#0f172a","borderRadius":"8px",
                                    "padding":"10px 16px","textAlign":"center",
                                    "flex":"1","minWidth":"70px"}, children=[
                        html.P(d, style={"color":"#64748b","fontSize":"0.72rem",
                                         "fontWeight":"700","margin":"0"}),
                        html.P(f"{raw[d]:.3f}",
                               style={"color":"#f1f5f9","fontSize":"1.05rem",
                                      "fontWeight":"700","margin":"4px 0 0",
                                      "fontFamily":"Courier New"}),
                    ]) for d in dims
                ]),
            ])
        ]),
    ])

    # ── Guidance cards ──
    other_profiles = [p for p in ["Balanced","Transitional","Specialized"] if p != profile]
    guidance_cards = []
    for target in other_profiles:
        g    = get_guidance(profile, target)
        tcol = PROFILE_COLOURS[target]
        guidance_cards.append(dbc.Col(md=6, children=[
            html.Div(style={"background":"#1e293b",
                            "border":f"1px solid {hex_to_rgba(tcol, 0.3)}",
                            "borderRadius":"12px","padding":"20px","height":"100%"}, children=[
                html.Div(style={"display":"flex","alignItems":"center",
                                "gap":"10px","marginBottom":"14px"}, children=[
                    html.Div(style={"width":"10px","height":"10px","borderRadius":"50%",
                                    "background":tcol,"flexShrink":"0"}),
                    html.P(f"To reach  {target.upper()}",
                           style={"color":tcol,"fontWeight":"700","fontSize":"0.9rem","margin":"0"}),
                ]),
                html.Div(style={"marginBottom":"12px"}, children=[
                    html.P("▲  ELEVATE",
                           style={"color":"#22c55e","fontSize":"0.75rem","fontWeight":"700",
                                  "letterSpacing":"0.08em","margin":"0 0 8px"}),
                    html.Ul(style={"margin":"0","paddingLeft":"18px"}, children=[
                        html.Li(tip, style={"color":"#cbd5e1","fontSize":"0.85rem",
                                             "marginBottom":"5px","lineHeight":"1.4"})
                        for tip in g.get("elevate",[])
                    ] if g.get("elevate") else [
                        html.Li("No specific elevation needed.",
                                style={"color":"#64748b","fontSize":"0.85rem"})
                    ]),
                ]),
                html.Div(children=[
                    html.P("▼  REDUCE",
                           style={"color":"#ef4444","fontSize":"0.75rem","fontWeight":"700",
                                  "letterSpacing":"0.08em","margin":"0 0 8px"}),
                    html.Ul(style={"margin":"0","paddingLeft":"18px"}, children=[
                        html.Li(tip, style={"color":"#cbd5e1","fontSize":"0.85rem",
                                             "marginBottom":"5px","lineHeight":"1.4"})
                        for tip in g.get("reduce",[])
                    ] if g.get("reduce") else [
                        html.Li("No specific reduction needed.",
                                style={"color":"#64748b","fontSize":"0.85rem"})
                    ]),
                ]),
            ])
        ]))

    guidance = html.Div(style={"marginBottom":"24px"}, children=[
        html.P("UPGRADE / TRANSITION GUIDANCE",
               style={"color":"#64748b","fontSize":"0.75rem","fontWeight":"700",
                      "letterSpacing":"0.1em","marginBottom":"14px"}),
        dbc.Row(guidance_cards),
    ])

    # ── Interpretation ──
    interpretations = {
        "Balanced":     "Well-rounded health literacy across all dimensions. This author integrates medical vocabulary, contextualised questions, causal reasoning, and evidence awareness. Suitable for clinical content, shared-decision tools, and evidence-based patient education.",
        "Transitional": "Developing health literacy. The author uses plain language, limited medical terminology, and direct questions reflecting genuine uncertainty. Best served by simplified explanations, step-by-step guidance, and plain-language resources.",
        "Specialized":  f"Niche literacy profile ({sub or 'mixed'}). Digitally-Specialized authors cite credible databases and studies. Functionally-Specialized authors express rich personal health narratives. Content should be tailored to the detected sub-type.",
    }

    interp = html.Div(style={"background":"#1e293b",
                              "border":f"1px solid {hex_to_rgba(pcol, 0.35)}",
                              "borderRadius":"12px","padding":"20px","marginBottom":"28px"}, children=[
        html.P("PROFILE INTERPRETATION",
               style={"color":"#64748b","fontSize":"0.75rem","fontWeight":"700",
                      "letterSpacing":"0.1em","margin":"0 0 10px"}),
        html.P(interpretations.get(profile, ""),
               style={"color":"#cbd5e1","fontSize":"0.92rem","lineHeight":"1.6","margin":"0"}),
    ])
    # ── Text annotation ──────────────────────────────────────────────────────
    raw_spans = annotate_text(text)

    span_elements = []
    for segment, dim in raw_spans:
        if dim and dim in DIMENSION_COLOURS:
            dc = DIMENSION_COLOURS[dim]
            span_elements.append(html.Span(
                segment,
                title=dc["label"],    # tooltip on hover
                style={
                    "background":    dc["bg"],
                    "color":         dc["color"],
                    "borderBottom":  f"2px solid {dc['color']}",
                    "borderRadius":  "3px",
                    "padding":       "1px 3px",
                    "fontWeight":    "600",
                    "cursor":        "help",
                }
            ))
        else:
            span_elements.append(html.Span(segment, style={"color": "#cbd5e1"}))

    dim_legend = html.Div(
        style={"display":"flex","gap":"16px","flexWrap":"wrap","marginBottom":"14px"},
        children=[
            html.Div(style={"display":"flex","alignItems":"center","gap":"6px"}, children=[
                html.Div(style={"width":"11px","height":"11px","borderRadius":"2px",
                                "background": DIMENSION_COLOURS[dim]["color"]}),
                html.Span(
                    f"{dim} — {DIMENSION_COLOURS[dim]['label']}",
                    style={"color":"#94a3b8","fontSize":"0.78rem"}
                ),
            ]) for dim in ["FHL","CHL","CRHL","DHL","EHL"]
        ]
    )

    annotation_section = html.Div(
        style={"background":"#1e293b","border":"1px solid #334155",
               "borderRadius":"12px","padding":"20px","marginBottom":"20px"},
        children=[
            html.P("TEXT ANNOTATION — Highlighted by HL Dimension",
                   style={"color":"#64748b","fontSize":"0.75rem","fontWeight":"700",
                          "letterSpacing":"0.1em","margin":"0 0 12px"}),
            html.P("Hover over a highlighted word to see its dimension. "
                   "Unhighlighted text does not strongly signal any dimension.",
                   style={"color":"#475569","fontSize":"0.78rem","margin":"0 0 12px"}),
            dim_legend,
            html.Hr(style={"borderColor":"#334155","margin":"0 0 14px"}),
            html.Div(
                span_elements,
                style={"fontSize":"0.95rem","lineHeight":"2.0",
                       "fontFamily":"'Courier New',monospace",
                       "whiteSpace":"pre-wrap","wordBreak":"break-word"}
            ),
        ]
    )
    # ── Clarity annotation section ────────────────────────────────────────────
    clarity_spans = annotate_clarity(text) or []

    clarity_elements  = []
    for segment, agent in clarity_spans:
        if agent and agent in CLARITY_ANNOTATION_COLOURS:
            ac = CLARITY_ANNOTATION_COLOURS[agent]
            clarity_elements.append(html.Span(
                segment,
                title=ac["label"],
                style={
                    "background":   ac["bg"],
                    "color":        ac["color"],
                    "borderBottom": f"2px solid {ac['color']}",
                    "borderRadius": "3px",
                    "padding":      "1px 3px",
                    "fontWeight":   "600",
                    "cursor":       "help",
                }
            ))
        else:
            clarity_elements.append(html.Span(segment, style={"color": "#cbd5e1"}))

    clarity_legend = html.Div(
        style={"display":"flex","gap":"16px","flexWrap":"wrap","marginBottom":"14px"},
        children=[
            html.Div(style={"display":"flex","alignItems":"center","gap":"6px"}, children=[
                html.Div(style={"width":"11px","height":"11px","borderRadius":"2px",
                                "background": CLARITY_ANNOTATION_COLOURS[ag]["color"]}),
                html.Span(
                    CLARITY_ANNOTATION_COLOURS[ag]["label"],
                    style={"color":"#94a3b8","fontSize":"0.78rem"}
                ),
            ]) for ag in ["Jargon","Explanation","Coherence","Fluency"]
        ]
    )

    clarity_annotation_section = html.Div(
        style={"background":"#1e293b","border":"1px solid #334155",
               "borderRadius":"12px","padding":"20px","marginBottom":"20px"},
        children=[
            html.P("TEXT ANNOTATION — Highlighted by ClarityConsensus Agent",
                   style={"color":"#64748b","fontSize":"0.75rem","fontWeight":"700",
                          "letterSpacing":"0.1em","margin":"0 0 12px"}),
            html.P("Hover over a highlighted word to see which agent it triggers. "
                   "Unhighlighted text does not strongly signal any agent.",
                   style={"color":"#475569","fontSize":"0.78rem","margin":"0 0 12px"}),
            clarity_legend,
            html.Hr(style={"borderColor":"#334155","margin":"0 0 14px"}),
            html.Div(
                clarity_elements,
                style={"fontSize":"0.95rem","lineHeight":"2.0",
                       "fontFamily":"'Courier New',monospace",
                       "whiteSpace":"pre-wrap","wordBreak":"break-word"}
            ),
        ]
    )
    return html.Div([hero, clarity_section, annotation_section, clarity_annotation_section, charts, guidance, interp])




if __name__ == "__main__":
    app.run(debug=False, port=8050)