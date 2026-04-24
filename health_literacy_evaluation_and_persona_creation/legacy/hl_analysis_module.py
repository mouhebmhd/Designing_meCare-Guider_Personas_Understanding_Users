"""
hl_analyzer.py — Health Literacy Analysis Module

This module provides a function `analyze_health_literacy(text, model_path=None)`
that takes any health‑related text and returns a complete analysis result as a
Python dictionary, ready to be converted to JSON.

Usage:
    from hl_analyzer import analyze_health_literacy
    result = analyze_health_literacy("I have type 2 diabetes...")
    import json
    print(json.dumps(result, indent=2))

Dependencies:
    torch, spacy, textstat, nltk, numpy, scikit-learn (for StandardScaler)
    Also requires a trained model file (default "hl_bvae_model.pt") and
    spaCy model (en_core_sci_md or en_core_web_sm).

 Date: 2026-03-23
"""

import warnings
warnings.filterwarnings("ignore")

import re
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import textstat
import spacy
import nltk

from .health_literacy_regex_patterns  import (
    extract_fhl_features,
    extract_chl_features,
    extract_crhl_features,
    extract_dhl_features,
    extract_ehl_features,
    extract_all_features,
    HealthLiteracyPatterns,
)

# Download required NLTK data if not present
nltk.download("punkt", quiet=True)

# =============================================================================
# Configuration
# =============================================================================
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "..", "assets", "hl_bvae_model.pt")
MODEL_PATH_DEFAULT = os.path.normpath(model_path)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MC_PASSES = 50

# Profile colours (not needed for JSON but kept for reference)
PROFILE_COLOURS = {
    "Balanced":     "#22c55e",
    "Transitional": "#f97316",
    "Specialized":  "#a855f7",
}

# HL level thresholds (loaded from model checkpoint)
HL_THRESHOLDS = None
F1_MEDIAN = None
SCALER_MEAN = None
SCALER_SCALE = None
FACTOR_LOADINGS = None

# Global model variable (lazy loading)
_BVAE_MODEL = None

# =============================================================================
# spaCy model loading (fallback to en_core_web_sm if sci_md not available)
# NOTE: spaCy is also loaded inside health_literacy_regex_patterns; this
# instance is kept for annotate_text / annotate_clarity which need the doc.
# =============================================================================
try:
    nlp = spacy.load("en_core_sci_md")
except OSError:
    nlp = spacy.load("en_core_web_sm")

# Used by clarity_consensus and annotate_clarity (spaCy NER-based functions)
MEDICAL_ENTITY_LABELS = {"DISEASE", "CHEMICAL", "ENTITY"}

# =============================================================================
# Text cleaning and feature extraction
# =============================================================================
_HTML = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s{2,}")
_NOISE = re.compile(r"[^\w\s\.,;:!?()\-\'\"/%°+]")

def clean_text(text):
    text = _HTML.sub(" ", text)
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
    text = _NOISE.sub(" ", text)
    return _SPACE.sub(" ", text).strip().lower()

def extract_features(text: str) -> dict:
    """
    Extract a dictionary of features from text using regex + POS patterns.
    Delegates to health_literacy_regex_patterns.extract_all_features().
    Output keys are identical to the previous keyword-based implementation
    so aggregate_scores() and all downstream code remain unchanged.
    """
    return extract_all_features(text)

def aggregate_scores(feat):
    """Aggregate feature dictionary into the five HL dimensions."""
    def m(*keys):
        values = [feat.get(k, 0) for k in keys]
        return float(np.mean(values)) if values else 0.0

    fhl  = m("readability_score", "avg_sentence_length", "avg_clauses_per_sentence",
             "medical_entity_count", "medical_entity_density", "unique_medical_terms",
             "hedging_score", "confidence_score")
    chl  = m("question_count", "question_ratio", "conditional_expression_count",
             "modal_verb_count", "modal_verb_density", "context_marker_count",
             "context_provided")
    crhl = m("causal_connective_count", "contrastive_connective_count",
             "evidence_reference_count", "multiple_options_count")
    dhl  = m("online_reference_count", "credible_source_count", "cross_reference_count",
             "information_interpretation_score")
    ehl  = m("concreteness_score", "lexical_diversity", "present_verb_count",
             "present_verb_ratio", "determiner_count", "determiner_ratio",
             "adjective_count", "adjective_ratio", "function_word_count",
             "function_word_ratio")

    return np.array([[fhl, chl, crhl, dhl, ehl]], dtype=np.float32)

# =============================================================================
# BVAE model definition
# =============================================================================
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

# =============================================================================
# Model loading and global variables
# =============================================================================
def _load_model(model_path=None):
    """Load the BVAE model and associated metadata. Returns True on success."""
    global _BVAE_MODEL, FACTOR_LOADINGS, HL_THRESHOLDS, F1_MEDIAN, SCALER_MEAN, SCALER_SCALE

    if _BVAE_MODEL is not None:
        return True

    if model_path is None:
        model_path = MODEL_PATH_DEFAULT

    try:
        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
        LATENT_DIM = ckpt["latent_dim"]
        FACTOR_LOADINGS = ckpt["factor_loadings"]
        HL_THRESHOLDS = ckpt["hl_thresholds"]
        F1_MEDIAN = ckpt["f1_median"]
        SCALER_MEAN = ckpt["scaler_mean"]
        SCALER_SCALE = ckpt["scaler_scale"]

        _BVAE_MODEL = BVAE(input_dim=5, latent_dim=LATENT_DIM).to(DEVICE)
        _BVAE_MODEL.load_state_dict(ckpt["model_state"])
        _BVAE_MODEL.eval()
        return True
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

def scale(x):
    """Normalize input array using saved scaler."""
    return (x - SCALER_MEAN) / SCALER_SCALE

# =============================================================================
# Profile and level assignment
# =============================================================================
def assign_profile(f1v, f2v, f3v):
    if abs(f2v) > 1.0 and abs(f3v) <= 0.8:
        return "Specialized", "Digitally-Specialized"
    if abs(f3v) > 0.8 and abs(f2v) <= 1.0:
        return "Specialized", "Functionally-Specialized"
    if f1v > F1_MEDIAN:
        return "Balanced", ""
    return "Transitional", ""

def map_hl_level(f1v, f2v):
    thresholds = HL_THRESHOLDS
    if f1v < thresholds["low"]:
        level = "Low"
    elif f1v < thresholds["basic"]:
        level = "Basic"
    elif f1v < thresholds["intermediate"]:
        level = "Intermediate"
    else:
        level = "High"

    # Adjust for very negative F2 when F1 is low
    if f2v < -1.2 and f1v < 0.25:
        order = ["Low", "Basic", "Intermediate", "High"]
        idx = order.index(level)
        if idx > 1:
            level = order[idx - 1]
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

# =============================================================================
# Main analysis function
# =============================================================================
def analyse(text):
    """Perform the full health literacy analysis."""
    if not text or len(text.strip()) < 20:
        raise ValueError("Text too short (minimum 20 characters).")

    cleaned = clean_text(text)
    feat = extract_features(cleaned)
    scores = aggregate_scores(feat)
    x_norm = scale(scores).astype(np.float32)
    x_t = torch.tensor(x_norm, dtype=torch.float32).to(DEVICE)

    # Monte Carlo passes for uncertainty
    model = _BVAE_MODEL
    model.train()  # to enable dropout for uncertainty estimation
    recons = []
    with torch.no_grad():
        for _ in range(MC_PASSES):
            xh, _, _ = model(x_t)
            recons.append(xh.cpu().numpy())
    sigma = float(np.stack(recons).std(axis=0).mean())

    # Factor scores
    factors = (x_norm @ FACTOR_LOADINGS)[0]
    f1v, f2v, f3v = float(factors[0]), float(factors[1]), float(factors[2])

    profile, sub = assign_profile(f1v, f2v, f3v)
    level = map_hl_level(f1v, f2v)
    suit = suitability(f1v, f2v, f3v)
    raw_scores = {
        "FHL":  round(float(scores[0, 0]), 4),
        "CHL":  round(float(scores[0, 1]), 4),
        "CRHL": round(float(scores[0, 2]), 4),
        "DHL":  round(float(scores[0, 3]), 4),
        "EHL":  round(float(scores[0, 4]), 4),
    }

    return {
        "profile": profile,
        "sub_type": sub,
        "hl_level": level,
        "flagged": sigma > 0.5,
        "sigma": round(sigma, 4),
        "f1": round(f1v, 4),
        "f2": round(f2v, 4),
        "f3": round(f3v, 4),
        "suitability": suit,
        "raw_scores": raw_scores,
    }

# =============================================================================
# ClarityConsensus agents
# =============================================================================
def clarity_consensus(text: str) -> dict:
    """
    Compute the ClarityConsensus score using four agents:
    Jargon, Explanation, Fluency, Coherence.
    """
    doc = nlp(text)
    sents = list(doc.sents)

    # ---- Agent 1: Jargon ---------------------------------------------------
    EXPLANATION_CUES = re.compile(
        r'\(([^)]{5,})\)'
        r'|,?\s*(also known as|defined as|refers to|meaning|i\.e\.)\s',
        re.IGNORECASE
    )
    med_terms = [e for e in doc.ents if e.label_ in MEDICAL_ENTITY_LABELS]
    n_terms = len(med_terms)

    explained = 0
    for ent in med_terms:
        window = text[ent.end_char: ent.end_char + 120]
        if EXPLANATION_CUES.search(window):
            explained += 1

    n_words = max(len([t for t in doc if not t.is_space]), 1)
    jargon_raw = n_terms / n_words
    unexplained = max(n_terms - explained, 0)
    jargon_score = 1.0 - min(1.0, unexplained / max(n_terms, 1))

    # ---- Agent 2: Explanation coverage --------------------------------------
    explanation_score = explained / max(n_terms, 1) if n_terms > 0 else 1.0

    # ---- Agent 3: Fluency ---------------------------------------------------
    fre = textstat.flesch_reading_ease(text)
    fluency_score = float(min(1.0, max(0.0, fre / 100.0)))

    # ---- Agent 4: Coherence -------------------------------------------------
    coherence_score = 1.0
    flagged_transitions = []
    if len(sents) >= 2:
        sims = []
        for s1, s2 in zip(sents[:-1], sents[1:]):
            v1 = s1.vector
            v2 = s2.vector
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                sim = float(np.dot(v1, v2) / (n1 * n2))
                sims.append(sim)
                if sim < 0.55:
                    flagged_transitions.append((str(s1)[:60], str(s2)[:60], round(sim, 3)))
        coherence_score = float(np.mean(sims)) if sims else 1.0

    # ---- Weighted final score ----------------------------------------------
    W_J, W_E, W_F, W_C = 0.40, 0.30, 0.15, 0.15
    final = W_J * jargon_score + W_E * explanation_score + W_F * fluency_score + W_C * coherence_score

    return {
        "final": round(final, 4),
        "jargon": round(jargon_score, 4),
        "explanation": round(explanation_score, 4),
        "fluency": round(fluency_score, 4),
        "coherence": round(coherence_score, 4),
        "n_terms": n_terms,
        "n_explained": explained,
        "fre": round(fre, 2),
        "flagged_transitions": flagged_transitions,
    }

# =============================================================================
# Text annotation functions (for JSON output)
# =============================================================================
def annotate_text(text: str):
    """Return list of (segment, dimension) for HL dimension highlighting."""
    lower = text.lower()
    n = len(text)
    labels = [None] * n

    # Build flat term lists from regex patterns for each dimension.
    # We extract literal terms by compiling each pattern and finding all
    # matches in the lowercased text, then use those spans for labelling.
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

    segments = []
    i = 0
    while i < n:
        label = labels[i]
        j = i + 1
        while j < n and labels[j] == label:
            j += 1
        segments.append((text[i:j], label))
        i = j
    return segments

def annotate_clarity(text: str):
    """Return list of (segment, agent) for ClarityConsensus highlighting."""
    doc = nlp(text)
    n = len(text)
    labels = [None] * n

    # Fluency: long words with ≥3 syllables
    for token in doc:
        if token.is_alpha and len(token.text) >= 7:
            if textstat.syllable_count(token.text) >= 3:
                for i in range(token.idx, min(token.idx + len(token.text), n)):
                    labels[i] = "Fluency"

    # Coherence markers
    COHERENCE_MARKERS = {
        "however", "although", "though", "yet", "nevertheless", "furthermore",
        "moreover", "therefore", "thus", "hence", "consequently", "in contrast",
        "on the other hand", "in addition", "additionally", "despite", "whereas",
        "as a result", "in conclusion", "in summary", "for example", "for instance",
        "that is", "in other words", "specifically", "notably", "importantly",
    }
    lower = text.lower()
    for marker in sorted(COHERENCE_MARKERS, key=len, reverse=True):
        pattern = re.compile(r'\b' + re.escape(marker) + r'\b')
        for m in pattern.finditer(lower):
            for i in range(m.start(), min(m.end(), n)):
                labels[i] = "Coherence"

    # Jargon: medical entities
    for ent in doc.ents:
        if ent.label_ in MEDICAL_ENTITY_LABELS:
            for i in range(ent.start_char, min(ent.end_char, n)):
                labels[i] = "Jargon"

    # Explanation: parenthetical definitions or explanatory phrases
    EXPLANATION_CUES = re.compile(
        r'\(([^)]{5,})\)'
        r'|,?\s*(also known as|defined as|refers to|meaning|i\.e\.)[^,\.;]{0,80}',
        re.IGNORECASE
    )
    for m in EXPLANATION_CUES.finditer(text):
        for i in range(m.start(), min(m.end(), n)):
            labels[i] = "Explanation"

    # Segment building
    segments = []
    i = 0
    while i < n:
        label = labels[i]
        j = i + 1
        while j < n and labels[j] == label:
            j += 1
        segments.append((text[i:j], label))
        i = j
    return segments

# =============================================================================
# Guidance generation
# =============================================================================
GUIDANCE = {
    ("Transitional", "Balanced"): {
        "elevate": [
            "Use more medical terminology and disease/drug names",
            "Add causal reasoning (because, therefore, as a result)",
            "Include personal health context (diagnosis, duration, treatment)",
            "Reference credible sources (CDC, NHS, clinical guidelines)",
            "Use contrastive connectives (however, although, despite)"
        ],
        "reduce": [
            "Avoid very short, fragmented sentences",
            "Reduce over-reliance on questions without context",
            "Limit informal/colloquial phrasing"
        ],
    },
    ("Transitional", "Specialized"): {
        "elevate": [
            "Cite specific databases or journals (PubMed, Lancet, JAMA) for Digital sub-type",
            "Describe your personal condition in rich detail for Functional sub-type",
            "Reference multiple treatment options and compare them",
            "Use evidence markers (systematic review, meta-analysis, clinical trial)"
        ],
        "reduce": [
            "Reduce vague or unspecified claims",
            "Avoid generic statements without data or context"
        ],
    },
    ("Balanced", "Specialized"): {
        "elevate": [
            "Cite specific studies, DOIs, or clinical databases (for Digital sub-type)",
            "Add detailed first-person narrative of lived experience (for Functional sub-type)",
            "Push F2 digital factor above 1.0 by referencing credible online sources",
            "Push F3 applied factor above 0.8 with concrete application of knowledge"
        ],
        "reduce": [
            "Balance — being Specialized is niche, not always better",
            "Reduce generic breadth if targeting a specific expertise axis"
        ],
    },
    ("Specialized", "Balanced"): {
        "elevate": [
            "Broaden across all 5 HL dimensions",
            "Add communicative context (questions, modal verbs, conditionals)",
            "Include both evidence references AND personal application",
            "Use hedging language to signal nuanced reasoning"
        ],
        "reduce": [
            "Reduce over-reliance on a single dimension (digital or applied)",
            "Avoid purely academic or purely personal framing"
        ],
    },
    ("Balanced", "Transitional"): {
        "elevate": [],
        "reduce": [
            "Note: Transitional represents lower overall literacy — moving here is a downgrade",
            "Simplify vocabulary, reduce sentence complexity, remove evidence references if targeting a lay audience"
        ],
    },
    ("Specialized", "Transitional"): {
        "elevate": [],
        "reduce": [
            "Note: Transitional represents lower overall literacy",
            "Remove technical references, simplify to plain language if targeting a general audience"
        ],
    },
}

def get_guidance(current, target):
    """Return guidance dictionary for moving from current to target profile."""
    return GUIDANCE.get((current, target), {
        "elevate": [
            "Improve overall medical vocabulary and sentence structure",
            "Add more contextual reasoning and evidence references"
        ],
        "reduce": [
            "Reduce elements that push toward the current profile"
        ],
    })

# =============================================================================
# Public API
# =============================================================================
def analyze_health_literacy(text, model_path=None):
    """
    Analyze health literacy of a text and return a comprehensive result dictionary.

    Parameters:
        text (str): The health‑related text to analyze.
        model_path (str, optional): Path to the BVAE model checkpoint.
            Defaults to "./hl_bvae_model.pt".

    Returns:
        dict: A dictionary containing all analysis results (profile, scores,
              annotations, guidance). Ready to be serialized to JSON.

    Raises:
        ValueError: If text is too short.
        RuntimeError: If model loading fails.
    """
    # Ensure model is loaded
    _load_model(model_path)

    # Run main analysis
    result = analyse(text)

    # Compute clarity consensus
    clarity = clarity_consensus(text)

    # Generate annotations
    hl_annotations = annotate_text(text)
    clarity_annotations = annotate_clarity(text)

    # Generate guidance for other profiles
    current_profile = result["profile"]
    guidance = {}
    for target in ["Balanced", "Transitional", "Specialized"]:
        if target != current_profile:
            guidance[target] = get_guidance(current_profile, target)

    # Build final output
    output = {
        **result,
        "clarity_consensus": clarity,
        "annotations": {
            "hl_dimensions": hl_annotations,
            "clarity_agents": clarity_annotations,
        },
        "guidance": guidance,
    }
    return output

 