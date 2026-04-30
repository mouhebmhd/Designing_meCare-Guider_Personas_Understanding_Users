"""
=============================================================================
  Health Literacy — Profile Dashboard  (Python / Dash)
=============================================================================
  PATCH NOTES (v3 → v3.1):
  - Removed "ENTITY" from MEDICAL_ENTITY_LABELS.
    en_core_sci_lg uses "ENTITY" as a catch-all that tags almost every noun,
    including plain words like "discharge", "dose", "list", "test", "doctor".
  - Added _NER_PLAIN_ENGLISH_STOPWORDS: a broad set of common words that
    scispaCy incorrectly tags but are NOT medical jargon for lay readers.
  - Added _is_plain_english_ner() guard inside _detect_jargon_spans() so
    NER spans whose text (lowercased) is in the stopword set are skipped.
  - Ordinal / numeric tokens are now explicitly rejected from the NER layer.
  - Multi-token NER spans are only accepted if at least one token survives
    the plain-English filter (prevents "twice daily" type spans).
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
from spacy.matcher import PhraseMatcher
import nltk

try:
    import wordfreq as _wordfreq
    _WORDFREQ_AVAILABLE = True
except ImportError:
    _WORDFREQ_AVAILABLE = False
    print("wordfreq not installed — syllable Zipf filter disabled. Run: pip install wordfreq")

from health_literacy_regex_patterns import (
    extract_all_features,
    HealthLiteracyPatterns,
)

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

nltk.download("punkt", quiet=True)

# =============================================================================
# CONFIG
# =============================================================================
MODEL_PATH = "../../assets/hl_bvae_model.pt"
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
    "Explanation": "#8b5cf6",
    "Fluency":     "#06b6d4",
    "Coherence":   "#f59e0b",
}

JARGON_LAYER_COLOURS = {
    "NER":        {"color": "#f43f5e", "label": "Jargon · NER",        "badge": "NER",   "bg": "rgba(244,63,94,0.15)"},
    "lexicon":    {"color": "#facc15", "label": "Jargon · Lexicon",    "badge": "LEX",   "bg": "rgba(250,204,21,0.15)"},
    "morphology": {"color": "#fb923c", "label": "Jargon · Morphology", "badge": "MORPH", "bg": "rgba(251,146,60,0.15)"},
    "syllable":   {"color": "#a78bfa", "label": "Jargon · Syllable",   "badge": "SYL",   "bg": "rgba(167,139,250,0.15)"},
}

CLARITY_ANNOTATION_COLOURS = {
    **{f"Jargon-{k}": v for k, v in JARGON_LAYER_COLOURS.items()},
    "Explanation": {"color": "#8b5cf6", "label": "Explanation — inline definition",  "bg": "rgba(139,92,246,0.15)"},
    "Coherence":   {"color": "#f59e0b", "label": "Coherence — discourse connective", "bg": "rgba(245,158,11,0.15)"},
    "Fluency":     {"color": "#06b6d4", "label": "Fluency — complex long word",      "bg": "rgba(6,182,212,0.15)"},
    "Quantity": {"color": "#10b981", "label": "Quantity — medical dose / unit", "bg": "rgba(16,185,129,0.15)"},

}
# =============================================================================
# PERSONA MARKER COLOURS
# =============================================================================
BALANCED_MARKER_COLOURS = {
    "EVIDENCE":        {"color": "#3b82f6", "badge": "EVD",  "label": "[EVIDENCE] — cites source / trial",          "bg": "rgba(59,130,246,0.15)"},
    "TRADEOFF":        {"color": "#f97316", "badge": "TRD",  "label": "[TRADEOFF] — pros / cons presented",         "bg": "rgba(249,115,22,0.15)"},
    "UNCERTAINTY":     {"color": "#a855f7", "badge": "UNC",  "label": "[UNCERTAINTY] — ambiguity acknowledged",     "bg": "rgba(168,85,247,0.15)"},
    "SHARED_DECISION": {"color": "#06b6d4", "badge": "SDM",  "label": "[SHARED_DECISION] — participatory framing",  "bg": "rgba(6,182,212,0.15)"},
    "COMPARISON":      {"color": "#eab308", "badge": "CMP",  "label": "[COMPARISON] — side-by-side comparison",     "bg": "rgba(234,179,8,0.15)"},
    "PRIMARY_SOURCE":  {"color": "#22c55e", "badge": "SRC",  "label": "[PRIMARY_SOURCE] — guideline / trial link",  "bg": "rgba(34,197,94,0.15)"},
}

TRANSITIONAL_MARKER_COLOURS = {
    "ACRONYM":          {"color": "#f43f5e", "badge": "ACR",  "label": "[ACRONYM] — abbreviation to spell out",       "bg": "rgba(244,63,94,0.15)"},
    "METAPHOR":         {"color": "#f97316", "badge": "MET",  "label": "[METAPHOR] — analogy / plain comparison",     "bg": "rgba(249,115,22,0.15)"},
    "TEACH_BACK":       {"color": "#8b5cf6", "badge": "TBK",  "label": "[TEACH_BACK] — comprehension check prompt",   "bg": "rgba(139,92,246,0.15)"},
    "EMOTION_VALIDATE": {"color": "#06b6d4", "badge": "EMO",  "label": "[EMOTION_VALIDATE] — acknowledge feeling",    "bg": "rgba(6,182,212,0.15)"},
    "STEP":             {"color": "#22c55e", "badge": "STP",  "label": "[STEP] — numbered single-idea step",          "bg": "rgba(34,197,94,0.15)"},
    "SIMPLE_SENTENCE":  {"color": "#ef4444", "badge": "SPL",  "label": "[SPLIT] — sentence exceeds 20 words",         "bg": "rgba(239,68,68,0.12)"},
    "LAY_LINK":         {"color": "#eab308", "badge": "LLK",  "label": "[LAY_LINK] — vetted plain-language resource", "bg": "rgba(234,179,8,0.15)"},
}

SPECIALIZED_MARKER_COLOURS = {
    "FULL_CITATION":      {"color": "#3b82f6", "badge": "CIT",  "label": "[FULL_CITATION] — DOI / PMID reference",           "bg": "rgba(59,130,246,0.15)"},
    "ADVANCED_FILTER":   {"color": "#a855f7", "badge": "ADV",  "label": "[ADVANCED_FILTER] — PICO / structured search",     "bg": "rgba(168,85,247,0.15)"},
    "BRIDGE_TO_SELF":    {"color": "#06b6d4", "badge": "BTS",  "label": "[BRIDGE_TO_SELF] — evidence → personal context",   "bg": "rgba(6,182,212,0.15)"},
    "VALIDATE_NARRATIVE":{"color": "#f97316", "badge": "VNR",  "label": "[VALIDATE_NARRATIVE] — affirm lived experience",   "bg": "rgba(249,115,22,0.15)"},
    "STORY_FORMAT":      {"color": "#eab308", "badge": "STR",  "label": "[STORY_FORMAT] — narrative framing",               "bg": "rgba(234,179,8,0.15)"},
    "GENTLE_EVIDENCE":   {"color": "#22c55e", "badge": "GEV",  "label": "[GENTLE_EVIDENCE] — soft evidence introduction",   "bg": "rgba(34,197,94,0.15)"},
    "HIGH_UNCERTAINTY":  {"color": "#ef4444", "badge": "HUN",  "label": "[HIGH_UNCERTAINTY] — needs human review",          "bg": "rgba(239,68,68,0.15)"},
}
DIMENSION_COLOURS = {
    "FHL":  {"color": "#3b82f6", "label": "Functional HL",    "bg": "rgba(59,130,246,0.15)"},
    "CHL":  {"color": "#eab308", "label": "Communicative HL", "bg": "rgba(234,179,8,0.15)"},
    "CRHL": {"color": "#f97316", "label": "Critical HL",      "bg": "rgba(249,115,22,0.15)"},
    "DHL":  {"color": "#a855f7", "label": "Digital HL",       "bg": "rgba(168,85,247,0.15)"},
    "EHL":  {"color": "#06b6d4", "label": "Expressed HL",     "bg": "rgba(6,182,212,0.15)"},
}


def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    try:
        hx = str(hex_color).strip().lstrip("#").ljust(6, "0")[:6]
        r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    except Exception:
        return f"rgba(99,102,241,{alpha})"


# =============================================================================
# spaCy — load best available model
# =============================================================================
nlp = None
for _model_name in ("en_core_sci_lg", "en_core_sci_md", "en_core_web_sm"):
    try:
        nlp = spacy.load(_model_name)
        print(f"spaCy model loaded: {_model_name}")
        break
    except OSError:
        continue

if nlp is None:
    raise RuntimeError(
        "No spaCy model found. Install one:\n"
        "  pip install scispacy && pip install https://s3-us-west-2.amazonaws.com/"
        "ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz\n"
        "or: python -m spacy download en_core_web_sm"
    )

# =============================================================================
# NER label set
# =============================================================================
# ── IMPORTANT: "ENTITY" is intentionally excluded. ───────────────────────────
# en_core_sci_lg applies "ENTITY" to almost every noun phrase, including
# completely plain words ("discharge", "dose", "doctor", "list", "test").
# Keeping only the specific, high-precision labels below dramatically reduces
# false positives while retaining true medical terms (diseases, chemicals,
# genes, anatomical structures).
# =============================================================================
MEDICAL_ENTITY_LABELS = {
    # scispaCy BC5CDR model
    "DISEASE", "CHEMICAL",
    # scispaCy CRAFT / JNLPBA model variants
    "Gene_or_gene_product", "Simple_chemical", "Cell",
    "Cellular_component", "Developing_anatomical_structure",
    "Immaterial_anatomical_entity", "Multi_tissue_structure",
    "Organ", "Organism", "Organism_substance", "Pathological_formation",
    "Tissue", "Amino_acid", "Anatomical_system",
    # en_core_web_sm fallback (low precision but zero cost)
    "ORG", "PRODUCT",
}
# NOTE: "ENTITY" (the en_core_sci_lg catch-all) is NOT in this set.

# =============================================================================
# NER PLAIN-ENGLISH STOPWORD FILTER
#
# Words that en_core_sci_lg routinely tags as ENTITY / CHEMICAL / DISEASE
# but which are everyday English and should NOT be shown as jargon to patients.
# This list covers the common false-positive categories:
#   • Time / frequency expressions  (daily, twice, monthly, morning, dose…)
#   • Ordinals / numbers            (first, second, twelve, fourteen…)
#   • Body parts in plain usage     (lips, face, leg, arm, back…)
#   • Clinical-but-plain nouns      (doctor, nurse, patient, hospital…)
#   • Actions / generic verbs       (report, stop, take, call, watch…)
#   • Generic descriptors           (new, early, written, current…)
#   • Common compound fragments     (advice line, patient portal, blood thinners…)
# =============================================================================
_NER_PLAIN_ENGLISH_STOPWORDS: frozenset[str] = frozenset({
    # ── Time / frequency ──────────────────────────────────────────────────────
    "daily", "twice", "twice daily", "once", "once daily", "monthly", "weekly",
    "every day", "every morning", "every night", "every evening",
    "morning", "evening", "night", "bedtime", "at bedtime",
    "dose", "doses", "dosing",
    "twelve months", "twelve", "fourteen", "fourteen days",
    "two-minute", "two minute", "two‐minute",
    "months", "days", "weeks", "hours", "minutes",
    "first", "second", "third", "fourth", "fifth",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    # ── Body parts in plain language ─────────────────────────────────────────
    "lips", "lip", "face", "arm", "leg", "back", "chest", "skin",
    "hand", "hands", "foot", "feet", "eye", "eyes", "ear", "ears",
    "head", "neck", "throat", "mouth", "tongue", "nose",
    "stomach", "belly", "knee", "elbow", "shoulder", "hip",
    # ── Clinical roles / places ───────────────────────────────────────────────
    "doctor", "doctors", "nurse", "nurses", "cardiologist", "patient", "patients",
    "hospital", "clinic", "pharmacy", "emergency room", "emergency",
    "advice line", "nurse advice line", "patient portal",
    # ── Clinical plain nouns ──────────────────────────────────────────────────
    "medication", "medications", "medicine", "medicines", "drug", "drugs",
    "prescription", "supplement", "supplements", "cream", "creams",
    "pill", "pills", "tablet", "tablets", "capsule", "capsules",
    "blood thinners", "blood thinner",
    "discharge", "discharge papers",
    "appointment", "follow-up", "follow-up appointment", "visit",
    "test", "tests", "testing", "result", "results",
    "list", "written list",
    "video", "two-minute video",
    "goal", "goals",
    "number", "numbers",
    # ── Symptoms in plain language ────────────────────────────────────────────
    "swelling", "cough", "dry cough", "pain", "ache",
    "faint", "dizzy", "dizziness", "nausea", "fatigue",
    "symptoms", "symptom",
    # ── Actions / verbs ───────────────────────────────────────────────────────
    "report", "stop", "stopping", "take", "taking", "call", "watch", "check",
    "ask", "explain", "bring", "avoid", "use", "apply",
    "log", "log into",
    "interact", "interacts",
    # ── Qualifiers / generic descriptors ─────────────────────────────────────
    "early", "new", "current", "currently", "written", "online",
    "standing", "missing", "eluting",
    # ── Miscellaneous common fragments ────────────────────────────────────────
    "cardiac rehab", "rehab",
    "own words",
    "next visit",
    "drug-eluting stent", "stent",
    "clot", "clots", "clotting",
    "garlic capsules", "garlic",
    "st. john's wort", "john's wort",
})

# Pre-compiled regex for ordinals and pure-numeric tokens
_ORDINAL_RE = re.compile(r'^\d+(?:st|nd|rd|th)?$', re.IGNORECASE)


def _is_plain_english_ner(span_text: str) -> bool:
    """
    Return True if this NER span is a plain English word/phrase that should
    NOT be presented as medical jargon to a lay reader.

    Checks:
      1. Exact match (case-insensitive) against the stopword set.
      2. All-numeric or ordinal token (e.g. "14", "2nd", "40%").
      3. Very short tokens (≤ 3 chars) — almost always not meaningful jargon.
      4. High wordfreq Zipf score (≥ 5.0) — extremely common English word.
    """
    low = span_text.strip().lower()

    # 1. Exact stopword match
    if low in _NER_PLAIN_ENGLISH_STOPWORDS:
        return True

    # 2. Numeric / ordinal
    # Strip trailing punctuation and % signs before checking
    stripped = low.rstrip(".,;:%").strip()
    if _ORDINAL_RE.match(stripped):
        return True
    if stripped.replace(".", "").replace(",", "").isdigit():
        return True

    # 3. Very short (catches "mg", "dL", single letters, etc.)
    if len(low.replace(" ", "")) <= 3:
        return True

    # 4. High-frequency common word (wordfreq)
    if _WORDFREQ_AVAILABLE:
        try:
            if _wordfreq.zipf_frequency(low, "en") >= 5.0:
                return True
        except Exception:
            pass

    return False


# =============================================================================
# JARGON DETECTION — Layer 3 (Morphology)
# =============================================================================
_MEDICAL_SUFFIX = re.compile(
    r"\b\w+(?:"
    r"itis|osis|emia|uria|algia|trophy|plasia|genesis|lysis|penia"
    r"|megaly|stenosis|sclerosis|spasm|plegia|paresis|philia|phobia"
    r"|ectomy|plasty|otomy|oscopy|ostomy|pexy|rraphy|desis"
    r"|ology|opathy|iatry|iatrics|graphy|gram|meter|scopy"
    r"|mycin|cillin|oxacin|olol|pril|sartan|statin|mab|zumab|ximab|kinib"
    r"|vir|tide|parin|azole|dazole|oxazole|thiazole"
    r"|carcinoma|sarcoma|lymphoma|leukemia|blastoma|cytoma|adenoma"
    r"|cytosis|cytopenia|globin|globulin"
    r")\b",
    re.IGNORECASE,
)

_MEDICAL_PREFIX = re.compile(
    r"\b(?:"
    r"hyper|hypo|tachy|brady|poly|oligo|macro|micro|neo|anti|contra"
    r"|intra|inter|trans|supra|sub|peri|para|endo|exo|hemo|haemo"
    r"|cardio|neuro|hepato|nephro|gastro|entero|dermato|onco|immuno"
    r"|osteo|arthro|myo|angio|lympho|pneumo|pulmo|thrombo|cyto|fibro"
    r")\w{3,}\b",
    re.IGNORECASE,
)

# =============================================================================
# JARGON DETECTION — Layer 2 (Lexicon)
# =============================================================================
MEDICAL_LEXICON = [
    "myocardial infarction", "heart attack", "atrial fibrillation",
    "heart failure", "cardiac arrest", "coronary artery disease",
    "type 1 diabetes", "type 2 diabetes", "diabetes mellitus",
    "hypertension", "hyperlipidemia", "dyslipidemia", "hypercholesterolemia",
    "chronic kidney disease", "acute kidney injury", "renal failure",
    "chronic obstructive pulmonary disease", "COPD", "asthma", "pneumonia",
    "tuberculosis", "sepsis", "septicemia", "bacteremia",
    "stroke", "ischemic stroke", "hemorrhagic stroke", "transient ischemic attack",
    "deep vein thrombosis", "pulmonary embolism",
    "Alzheimer disease", "Parkinson disease", "multiple sclerosis",
    "epilepsy", "seizure disorder",
    "rheumatoid arthritis", "osteoarthritis", "systemic lupus erythematosus",
    "celiac disease", "Crohn disease", "ulcerative colitis",
    "hepatitis A", "hepatitis B", "hepatitis C", "cirrhosis", "liver failure",
    "HIV", "AIDS", "human immunodeficiency virus",
    "breast cancer", "lung cancer", "colorectal cancer", "prostate cancer",
    "melanoma", "leukemia", "lymphoma", "glioblastoma",
    "anemia", "iron deficiency anemia", "sickle cell disease",
    "hyperthyroidism", "hypothyroidism", "Graves disease", "Hashimoto thyroiditis",
    "polycystic ovary syndrome", "PCOS", "endometriosis", "fibromyalgia",
    "anxiety disorder", "major depressive disorder", "bipolar disorder",
    "schizophrenia", "post-traumatic stress disorder", "PTSD",
    "autism spectrum disorder", "attention deficit hyperactivity disorder", "ADHD",
    "myocardium", "pericardium", "endocardium", "epicardium",
    "aorta", "ventricle", "atrium", "mitral valve", "tricuspid valve",
    "alveoli", "bronchiole", "trachea", "pleura", "diaphragm",
    "glomerulus", "nephron", "renal tubule", "ureter", "urethra",
    "synapse", "axon", "dendrite", "neurotransmitter", "cerebellum",
    "hypothalamus", "hippocampus", "amygdala", "prefrontal cortex",
    "pancreas", "liver", "gallbladder", "bile duct", "duodenum",
    "jejunum", "ileum", "colon", "rectum", "peritoneum",
    "femur", "tibia", "fibula", "patella", "humerus", "radius", "ulna",
    "lymph node", "spleen", "thymus", "bone marrow",
    "adrenal gland", "thyroid gland", "parathyroid gland", "pituitary gland",
    "metformin", "insulin", "glipizide", "sitagliptin", "empagliflozin",
    "lisinopril", "ramipril", "enalapril", "amlodipine", "nifedipine",
    "atorvastatin", "rosuvastatin", "simvastatin", "pravastatin",
    "metoprolol", "carvedilol", "bisoprolol", "atenolol",
    "warfarin", "rivaroxaban", "apixaban", "dabigatran", "heparin",
    "aspirin", "clopidogrel", "ticagrelor",
    "amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline",
    "vancomycin", "meropenem", "piperacillin", "tazobactam",
    "prednisone", "dexamethasone", "hydrocortisone", "methylprednisolone",
    "ibuprofen", "naproxen", "diclofenac", "celecoxib",
    "acetaminophen", "paracetamol", "tramadol", "morphine", "oxycodone",
    "levothyroxine", "methimazole", "propylthiouracil",
    "omeprazole", "pantoprazole", "ranitidine", "ondansetron",
    "fluoxetine", "sertraline", "escitalopram", "venlafaxine", "duloxetine",
    "olanzapine", "risperidone", "quetiapine", "haloperidol",
    "lorazepam", "diazepam", "alprazolam", "clonazepam",
    "albuterol", "salbutamol", "salmeterol", "tiotropium", "budesonide",
    "adalimumab", "infliximab", "rituximab", "trastuzumab", "bevacizumab",
    "pembrolizumab", "nivolumab", "ipilimumab",
    "electrocardiogram", "echocardiogram", "angiography", "angioplasty",
    "coronary bypass", "percutaneous coronary intervention",
    "computed tomography", "magnetic resonance imaging", "MRI", "CT scan",
    "positron emission tomography", "PET scan", "ultrasound", "sonography",
    "endoscopy", "colonoscopy", "bronchoscopy", "laparoscopy",
    "biopsy", "fine needle aspiration", "lumbar puncture", "spinal tap",
    "hemodialysis", "peritoneal dialysis",
    "mechanical ventilation", "intubation", "tracheostomy",
    "chemotherapy", "radiotherapy", "immunotherapy", "targeted therapy",
    "bone marrow transplant", "stem cell transplant",
    "complete blood count", "CBC", "metabolic panel", "lipid panel",
    "hemoglobin A1c", "HbA1c", "prothrombin time", "INR",
    "creatinine", "eGFR", "troponin", "BNP", "NT-proBNP",
    "C-reactive protein", "erythrocyte sedimentation rate", "ESR",
    "polymerase chain reaction", "PCR", "enzyme-linked immunosorbent assay", "ELISA",
    "prognosis", "etiology", "pathophysiology", "comorbidity", "contraindication",
    "pharmacokinetics", "pharmacodynamics", "bioavailability", "half-life",
    "adverse event", "adverse effect", "side effect", "iatrogenic",
    "randomized controlled trial", "meta-analysis", "systematic review",
    "confidence interval", "odds ratio", "relative risk", "hazard ratio",
    "placebo", "double-blind", "intention-to-treat", "per-protocol",
    "incidence", "prevalence", "sensitivity", "specificity",
    "positive predictive value", "negative predictive value",
    "number needed to treat", "number needed to harm",
    # ── Extra terms relevant to the example discharge text ────────────────────
    "anterior STEMI", "STEMI", "drug-eluting stent",
    "ejection fraction", "LDL cholesterol", "transthoracic echocardiogram",
    "metoprolol succinate", "pharmacogenomic testing",
    "NSAIDs", "NSAID", "PPIs", "PPI",
    "antiplatelet", "antiplatelet drugs",
    "CYP2C19", "liver enzyme",
]

_PHRASE_MATCHER = PhraseMatcher(nlp.vocab, attr="LOWER")
_PHRASE_MATCHER.add("MEDICAL_JARGON", [nlp.make_doc(t) for t in MEDICAL_LEXICON])

_SYLLABLE_THRESHOLD = 4

# =============================================================================
# SYLLABLE FILTER
# =============================================================================
_SYLLABLE_STOPWORDS = frozenset({
    "immediately", "understand", "understanding", "information",
    "appropriate", "approximately", "community", "particularly",
    "individual", "individuals", "effectively", "available",
    "especially", "environment", "opportunity", "relationship",
    "experience", "important", "associated", "organization",
    "communication", "development", "education", "evaluation",
    "responsibility", "administration", "government", "university",
    "international", "management", "significant", "significantly",
    "consideration", "necessary", "majority", "professional",
    "different", "activity", "activities", "possible", "population",
    "traditional", "additional", "everything", "following",
    "yourself", "although", "however", "whatever", "another",
    "together", "continue", "everything", "everybody", "everywhere",
    "interesting", "understanding", "unfortunately",
    "unbelievable", "uncomfortable", "independent",
    "immediately", "previously", "ultimately", "absolutely",
    "definitely", "completely", "relatively", "apparently",
    "incredibly", "certainly", "generally", "naturally",
    "obviously", "probably", "literally", "basically",
    "actually", "recently", "currently", "typically",
    "regularly", "directly", "perfectly", "properly",
    "quickly", "carefully", "clearly", "simply",
    # Additional plain-language words that are polysyllabic
    "appointment", "medication", "medications", "prescription",
    "supplement", "supplements", "instructions", "information",
    "including", "everything", "something", "anything",
    "important", "carefully", "emergency",
})

_ZIPF_JARGON_THRESHOLD = 4.5


def _is_common_word(token_text: str) -> bool:
    low = token_text.lower()
    if low in _SYLLABLE_STOPWORDS:
        return True
    if _WORDFREQ_AVAILABLE:
        try:
            return _wordfreq.zipf_frequency(low, "en") >= _ZIPF_JARGON_THRESHOLD
        except Exception:
            pass
    return False


# =============================================================================
# SPAN MERGER
# =============================================================================
def _merge_consecutive_spans(
    spans: dict[tuple[int, int], str],
    text: str,
) -> dict[tuple[int, int], str]:
    if not spans:
        return spans
    sorted_spans = sorted(spans.items(), key=lambda kv: kv[0][0])
    merged: list[tuple[tuple[int, int], str]] = []
    cur_start, cur_end = sorted_spans[0][0]
    cur_layer = sorted_spans[0][1]
    for (start, end), layer in sorted_spans[1:]:
        gap = text[cur_end:start]
        if layer == cur_layer and gap.strip() == "":
            cur_end = end
        else:
            merged.append(((cur_start, cur_end), cur_layer))
            cur_start, cur_end, cur_layer = start, end, layer
    merged.append(((cur_start, cur_end), cur_layer))
    return dict(merged)


# =============================================================================
# SHARED JARGON DETECTION ENGINE  (v3.1)
# =============================================================================
def _detect_jargon_spans(doc) -> dict[tuple[int, int], str]:
    n        = len(doc.text)
    assigned = [False] * n
    spans: dict[tuple[int, int], str] = {}

    def _mark(start: int, end: int, layer: str) -> bool:
        if not (0 <= start < end <= n):
            return False
        if any(assigned[i] for i in range(start, end)):
            return False
        spans[(start, end)] = layer
        for i in range(start, end):
            assigned[i] = True
        return True

    # ── Layer 1: NER ──────────────────────────────────────────────────────────
    for ent in doc.ents:
        if ent.label_ not in MEDICAL_ENTITY_LABELS:
            continue
        # ▶ NEW: reject if the span text is plain English
        if _is_plain_english_ner(ent.text):
            continue
        _mark(ent.start_char, ent.end_char, "NER")

    # ── Layer 2: Curated Lexicon ───────────────────────────────────────────────
    for _, s_tok, e_tok in _PHRASE_MATCHER(doc):
        sp = doc[s_tok:e_tok]
        _mark(sp.start_char, sp.end_char, "lexicon")

    # ── Layer 3: Morphology ───────────────────────────────────────────────────
    for pat in (_MEDICAL_SUFFIX, _MEDICAL_PREFIX):
        for m in pat.finditer(doc.text):
            _mark(m.start(), m.end(), "morphology")

    # ── Layer 4: Syllable count ───────────────────────────────────────────────
    for token in doc:
        if (
            token.is_alpha
            and len(token.text) >= 7
            and not token.is_stop
            and not _is_common_word(token.text)
            and textstat.syllable_count(token.text) >= _SYLLABLE_THRESHOLD
        ):
            _mark(token.idx, token.idx + len(token.text), "syllable")
    
    return _merge_consecutive_spans(spans, doc.text)


# =============================================================================
# BVAE
# =============================================================================
class BVAE(nn.Module):
    def __init__(self, input_dim=5, latent_dim=16, dropout_p=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta  = 1.0 if latent_dim <= 8 else 0.5
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
# Load model
# =============================================================================
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


def scale(x):
    return (x - scaler_mean) / scaler_scale


# =============================================================================
# Text pipeline
# =============================================================================
_HTML  = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s{2,}")
_NOISE = re.compile(r"[^\w\s\.,;:!?()\-\'\"/%°+]")


def clean_text(text):
    text = _HTML.sub(" ", text)
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
    text = _NOISE.sub(" ", text)
    return _SPACE.sub(" ", text).strip().lower()


def extract_features(text):
    return extract_all_features(text)


def aggregate_scores(feat):
    def m(*keys):
        return float(np.mean([feat.get(k, 0) for k in keys]))
    fhl  = m("readability_score","avg_sentence_length","avg_clauses_per_sentence",
             "medical_entity_count","medical_entity_density","unique_medical_terms",
             "hedging_score","confidence_score")
    chl  = m("question_count","question_ratio","conditional_expression_count",
             "modal_verb_count","modal_verb_density","context_marker_count","context_provided")
    crhl = m("causal_connective_count","contrastive_connective_count",
             "evidence_reference_count","multiple_options_count")
    dhl  = m("online_reference_count","credible_source_count",
             "cross_reference_count","information_interpretation_score")
    ehl  = m("concreteness_score","lexical_diversity","present_verb_count",
             "present_verb_ratio","determiner_count","determiner_ratio",
             "adjective_count","adjective_ratio","function_word_count","function_word_ratio")
    return np.array([[fhl, chl, crhl, dhl, ehl]], dtype=np.float32)


def assign_profile(f1v, f2v, f3v):
    if abs(f2v) > 1.0 and abs(f3v) <= 0.8: return "Specialized", "Digitally-Specialized"
    if abs(f3v) > 0.8 and abs(f2v) <= 1.0: return "Specialized", "Functionally-Specialized"
    if f1v > f1_median:                      return "Balanced", ""
    return "Transitional", ""


def map_hl_level(f1v, f2v):
    if   f1v < hl_threshold["low"]:           level = "Low"
    elif f1v < hl_threshold["basic"]:         level = "Basic"
    elif f1v < hl_threshold["intermediate"]:  level = "Intermediate"
    else:                                      level = "High"
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


def analyse(text):
    cleaned = clean_text(text)
    feat    = extract_features(cleaned)
    scores  = aggregate_scores(feat)
    x_norm  = scale(scores).astype(np.float32)
    x_t     = torch.tensor(x_norm, dtype=torch.float32).to(DEVICE)
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


# =============================================================================
# Guidance
# =============================================================================
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


# =============================================================================
# ClarityConsensus
# =============================================================================
_EXPLANATION_CUES = re.compile(
    r'\(([^)]{5,})\)'
    r'|,?\s*(also known as|defined as|refers to|meaning|i\.e\.)\s',
    re.IGNORECASE,
)
_EXPLANATION_WINDOW = 120
_MEDICAL_QUANTITY_RE = re.compile(
    r'\b'
    r'(?:'
        # ── numeric value (optionally a range: 10-20 or 10 to 20) ────────────
        r'\d+(?:[.,]\d+)?'
        r'(?:\s*(?:[-–]|to)\s*\d+(?:[.,]\d+)?)?'
    r')'
    r'\s*'
    r'(?:'
        # ── mass / concentration ─────────────────────────────────────────────
        r'mg(?:/(?:dL|mL|kg|L|m²|day|dose))?'
        r'|mcg(?:/(?:kg|mL|day))?'
        r'|[µμ]g(?:/(?:kg|mL|day))?'
        r'|g(?:/(?:dL|L|mL|kg|day))?'
        r'|kg'
        r'|mEq(?:/L)?'
        r'|mmol(?:/L)?'
        r'|mol(?:/L)?'
        r'|nmol(?:/L)?'
        r'|pmol(?:/L)?'
        # ── activity / biological units ───────────────────────────────────────
        r'|IU(?:/(?:mL|L|kg|day))?'
        r'|units?(?:/(?:kg|mL|day))?'
        r'|[Uu]\b'
        # ── volume ────────────────────────────────────────────────────────────
        r'|m[Ll](?:/(?:kg|hr|min|h))?'
        r'|[Ll]\b'
        r'|dL'
        r'|cc'
        # ── pressure / rate / temperature ─────────────────────────────────────
        r'|mmHg'
        r'|cmH2O'
        r'|bpm'
        r'|°[CcFf]'
        # ── dose forms ────────────────────────────────────────────────────────
        r'|tablets?|tabs?'
        r'|capsules?|caps?'
        r'|pills?'
        r'|drops?|gtts?'
        r'|puffs?'
        r'|sprays?'
        r'|patches?'
        r'|suppositories?|suppository'
        r'|teaspoons?|tablespoons?|tsp\.?|tbsp\.?'
        r'|vials?|ampoules?|ampules?'
        # ── time-dose fractions (e.g. "twice", "three times" before "daily") ─
        r'|%'
    r')\b',
    re.IGNORECASE,
)
# =============================================================================
# PERSONA MARKER DETECTION PATTERNS
# =============================================================================

# ── Balanced ──────────────────────────────────────────────────────────────────
_BALANCED_PATTERNS: dict[str, re.Pattern] = {
    "PRIMARY_SOURCE": re.compile(
    r'\b(?:'
    
    # --- Identifiers ---
    r'PMID\s*:?\s*\d{7,9}'
    r'|PMCID\s*:?\s*PMC\d+'
    r'|DOI\s*:?\s*10\.\d{4,9}/[-._;()/:A-Z0-9]+'
    r'|10\.\d{4,9}/[-._;()/:A-Z0-9]+'  # bare DOI
    r'|arXiv\s*:?\s*\d{4}\.\d{4,5}(?:v\d+)?'
    
    # --- Clinical trial registries ---
    r'|NCT\s*\d{8}'
    r'|EudraCT\s*\d{4}-\d{6}-\d{2}'
    r'|ISRCTN\s*\d+'
    r'|clinical\s*trials?\.gov'
    r'|trialregister\.eu'
    r'|isrctn\.com'
    
    # --- Databases / repositories ---
    r'|pubmed(?:\.ncbi)?'
    r'|ncbi\.nlm\.nih\.gov'
    r'|medline'
    r'|embase'
    r'|scopus'
    r'|web\s+of\s+science'
    r'|cochrane(?:\s+library)?'
    
    # --- Preprint servers ---
    r'|medrxiv'
    r'|biorxiv'
    r'|ssrn'
    r'|researchsquare'
    
    # --- Publishers / journals ---
    r'|thelancet'
    r'|new\s+england\s+journal\s+of\s+medicine'
    r'|nejm'
    r'|jama'
    r'|bmj'
    r'|nature'
    r'|science(?:\s+magazine)?'
    r'|plos\s+one'
    r'|elsevier'
    r'|springer'
    r'|wiley'
    r'|taylor\s*&\s*francis'
    
    # --- Health organizations & guidelines ---
    r'|(?:ACC|AHA|ESC|NICE|WHO|CDC|USPSTF|EMA|FDA|NIH|NHS)'
    r'(?:\s+(?:20\d{2}))?\s+guideline[s]?'
    r'|guideline[s]?\s+(?:from|by|per)\s+'
    r'(?:ACC|AHA|ESC|NICE|WHO|CDC|USPSTF|EMA|FDA|NIH|NHS)'
    r'|recommendation[s]?\s+(?:from|by)\s+'
    r'(?:ACC|AHA|ESC|NICE|WHO|CDC|USPSTF|EMA|FDA)'
    r'|consensus\s+(?:statement|report)'
    r'|position\s+statement'
    
    # --- URLs (broad scientific domains) ---
    r'|(?:https?://)?(?:www\.)?'
    r'(?:pubmed\.ncbi\.nlm\.nih\.gov'
    r'|doi\.org'
    r'|cochranelibrary\.com'
    r'|thelancet\.com'
    r'|nejm\.org'
    r'|jamanetwork\.com'
    r'|bmj\.com'
    r'|nature\.com'
    r'|science\.org)'
    
    r')\b',
    re.IGNORECASE,
)
   , 
   "EVIDENCE": re.compile(
    r'\b(?:'
    
    # --- Attribution / evidence phrasing ---
    r'according\s+to'
    r'|per\s+(?:the\s+)?(?:20\d{2}|19\d{2})'
    r'|based\s+on\s+(?:a\s+)?(?:study|trial|review|analysis|data)'
    r'|evidence\s+(?:suggests|shows|indicates|demonstrates)'
    r'|data\s+(?:suggest|show|indicate|demonstrate)'
    r'|findings\s+(?:suggest|show|indicate|demonstrate)'
    
    # --- Publication / journal mentions ---
    r'|published\s+in\s+(?:the\s+)?'
    r'(?:JAMA|NEJM|Lancet|BMJ|Circulation|JACC|N\s*Engl\s*J\s*Med|Nature|Science)'
    r'|(?:JAMA|NEJM|Lancet|BMJ|Circulation|JACC|Nature|Science)\s+(?:20\d{2}|19\d{2})'
    
    # --- Study designs ---
    r'|randomi[sz](?:ed|ation|ing)?'
    r'|RCTs?'
    r'|controlled\s+trial'
    r'|clinical\s+trial'
    r'|phase\s+[I|II|III|IV]+'
    r'|double[-\s]?blind'
    r'|placebo[-\s]?controlled'
    r'|cohort\s+study'
    r'|case[-\s]?control\s+study'
    r'|cross[-\s]?sectional\s+study'
    r'|observational\s+study'
    r'|prospective\s+study'
    r'|retrospective\s+study'
    
    # --- Evidence synthesis ---
    r'|meta[-\s]?analysis'
    r'|systematic\s+review'
    r'|pooled\s+analysis'
    
    # --- Statistical terms ---
    r'|absolute\s+risk\s+reduction'
    r'|relative\s+risk'
    r'|risk\s+ratio'
    r'|hazard\s+ratio'
    r'|odds\s+ratio'
    r'|rate\s+ratio'
    r'|confidence\s+intervals?'
    r'|CI\s*\(?\d{1,2}%\)?'
    r'|p\s*[<=>]\s*0?\.\d+'
    r'|statistically\s+significant'
    r'|effect\s+size'
    
    # --- Clinical metrics ---
    r'|number\s+needed\s+to\s+treat'
    r'|number\s+needed\s+to\s+harm'
    r'|NNT|NNH'
    r'|incidence'
    r'|prevalence'
    r'|mortality'
    r'|morbidity'
    
    # --- Reporting verbs ---
    r'|(?:study|trial|analysis|review)\s+(?:found|showed|demonstrated|reported|concluded|revealed)'
    r'|results\s+(?:showed|demonstrated|indicated|suggested)'
    r'|authors\s+(?:reported|concluded|found)'
    
    r')\b',
    re.IGNORECASE,
),
    "UNCERTAINTY": re.compile(
    r'\b(?:'
    
    # --- Core uncertainty / controversy ---
    r'debat(?:ed|able)'
    r'|controvers(?:ial|y)'
    r'|remains?\s+(?:unclear|uncertain|controversial|unknown)'
    r'|uncertain(?:ty)?'
    r'|unclear'
    r'|unknown'
    r'|inconclusive'
    r'|equivocal'
    
    # --- Evidence quality issues ---
    r'|evidence\s+is\s+(?:limited|mixed|conflicting|emerging|insufficient|weak|scarce)'
    r'|data\s+(?:are|is)\s+(?:limited|scarce|insufficient|inconclusive)'
    r'|lack\s+of\s+(?:evidence|data)'
    r'|paucity\s+of\s+(?:data|evidence)'
    
    # --- Lack of consensus ---
    r'|no\s+(?:clear\s+)?consensus'
    r'|consensus\s+(?:is\s+)?lacking'
    r'|experts?\s+(?:disagree|are\s+divided)'
    r'|opinions?\s+(?:vary|differ)'
    
    # --- Guidelines variability ---
    r'|guidelines?\s+(?:vary|differ|are\s+inconsistent|are\s+conflicting)'
    r'|guidelines?\s+(?:do\s+not\s+specify|provide\s+no\s+clear\s+recommendation)'
    
    # --- Unknown / not established ---
    r'|not\s+(?:yet\s+)?(?:fully\s+)?(?:known|established|proven|confirmed|clear|understood)'
    r'|remains?\s+to\s+be\s+(?:determined|established|clarified)'
    
    # --- Optimal strategy unclear ---
    r'|optimal\s+(?:duration|dose|timing|strategy|approach|management)'
    r'\s+(?:is\s+)?(?:debated|unclear|unknown|not\s+established)'
    
    # --- Study limitations / bias ---
    r'|limited\s+by\s+(?:small\s+sample\s+size|bias|confounding|short\s+follow-?up)'
    r'|subject\s+to\s+(?:bias|confounding)'
    r'|potential\s+(?:bias|confounding|measurement\s+error)'
    r'|heterogene(?:ity|ous)'
    
    # --- Statistical uncertainty ---
    r'|not\s+statistically\s+significant'
    r'|failed\s+to\s+reach\s+statistical\s+significance'
    r'|wide\s+confidence\s+intervals?'
    
    # --- Hedging / cautious language ---
    r'|may\s+(?:suggest|indicate|reflect|be|represent)'
    r'|might\s+(?:suggest|indicate|reflect|be)'
    r'|could\s+(?:suggest|indicate|reflect|be)'
    r'|appears?\s+to\s+(?:suggest|indicate|be)'
    r'|suggests?\s+(?:a\s+)?possible'
    r'|potential(?:ly)?'
    
    # --- Need for more research ---
    r'|further\s+(?:studies|research)\s+(?:are\s+)?needed'
    r'|additional\s+(?:studies|research)\s+(?:are\s+)?required'
    r'|warrant(?:s|ed)?\s+further\s+(?:investigation|study)'
    
    # --- General insufficiency statements ---
    r'|insufficient\s+(?:evidence|data)\s+to'
    r'|cannot\s+(?:be\s+)?(?:determined|concluded|established)'
    
    r')\b',
    re.IGNORECASE,
),
    "TRADEOFF": re.compile(
    r'\b(?:'
    
    # --- Explicit benefit vs risk ---
    r'benefit[s]?\s+(?:and|vs\.?|versus)\s+risk[s]?'
    r'|risk[s]?\s+(?:and|vs\.?|versus)\s+benefit[s]?'
    r'|risk[-\s]?benefit\s+(?:ratio|profile|balance|analysis)'
    
    # --- General tradeoff language ---
    r'|pros?\s+and\s+cons?'
    r'|trade[-\s]?off[s]?'
    r'|cost[-\s]?benefit\s+(?:analysis|ratio|balance)'
    r'|benefit[-\s]?cost\s+(?:ratio|analysis)'
    
    # --- Weighing / balancing ---
    r'|weigh(?:ing)?\s+(?:the\s+)?(?:risks?|benefits?|options|trade[-\s]?offs)'
    r'|balance(?:ing)?\s+(?:the\s+)?(?:risks?|benefits?|harms?)'
    r'|consider(?:ing)?\s+(?:the\s+)?(?:risks?|benefits?)'
    
    # --- Contrast connectors ---
    r'|on\s+the\s+(?:other|flip)\s+(?:hand|side)'
    r'|however'
    r'|nevertheless'
    r'|nonetheless'
    r'|while\s+.*?\b(?:increase|decrease|improve|reduce)'
    
    # --- But / contrast patterns ---
    r'|(?:improves?|reduces?|lowers?|decreases?)\s+[^.]{0,80}?\bbut\b'
    r'|\bbut\b[^.]{0,80}?(?:increases?|raises?|worsens?|reduces?)'
    
    # --- At the expense of ---
    r'|at\s+the\s+expense\s+of'
    r'|comes?\s+with\s+(?:a\s+)?cost'
    r'|associated\s+with\s+(?:increased\s+)?risk'
    
    # --- Adverse effects vs benefit ---
    r'|(?:side\s+effects?|adverse\s+(?:effects?|events)|toxicity)'
    r'\s+(?:may\s+)?(?:increase|worsen|occur)'
    r'|efficacy\s+(?:versus|vs\.?)\s+(?:safety|toxicity)'
    r'|safety\s+(?:versus|vs\.?)\s+efficacy'
    
    # --- Increase one thing, decrease another ---
    r'|(?:increase|improve|reduce|decrease)\s+\w+'
    r'\s+(?:while|but)\s+(?:increasing|reducing|worsening|affecting)\s+\w+'
    
    # --- Outcome contrasts ---
    r'|(?:reduces?\s+mortality\s+but\s+increases?\s+morbidity)'
    r'|(?:improves?\s+survival\s+but\s+worsens?\s+quality\s+of\s+life)'
    
    # --- Double-edged phrasing ---
    r'|double[-\s]?edged'
    r'|mixed\s+(?:benefits?|effects?|outcomes)'
    
    # --- Time / cost / burden tradeoffs ---
    r'|short[-\s]?term\s+(?:benefit|gain).{0,40}long[-\s]?term\s+(?:risk|cost)'
    r'|higher\s+cost\s+but\s+(?:better|improved)\s+outcomes'
    r'|lower\s+cost\s+but\s+(?:reduced|worse)\s+efficacy'
    
    r')\b',
    re.IGNORECASE,
),
    "SHARED_DECISION": re.compile(
    r'\b(?:'
    
    # --- Direct preference questions ---
    r'would\s+you\s+(?:like|prefer|want)'
    r'|do\s+you\s+(?:want|prefer|have\s+a\s+preference)'
    r'|which\s+(?:option|approach|treatment)\s+do\s+you\s+(?:prefer|feel\s+more\s+comfortable\s+with)'
    r'|how\s+do\s+you\s+feel\s+about'
    
    # --- Patient values / preferences ---
    r'|(?:your|the\s+patient[\'’]s?)\s+(?:preference|preferences|values?|goals?)'
    r'|patient\s+(?:choice|preference|values?|goals?)'
    r'|align(?:ing)?\s+with\s+(?:your|patient)\s+(?:values?|goals?)'
    r'|what\s+matters\s+(?:most\s+)?to\s+you'
    r'|what\s+is\s+important\s+to\s+you'
    
    # --- Shared decision making explicit ---
    r'|shared\s+decision(?:[-\s]?making)?'
    r'|decision\s+(?:making|process)\s+(?:together|jointly|collaboratively)'
    r'|collaborative\s+decision(?:[-\s]?making)?'
    
    # --- Collaborative phrasing ---
    r'|together\s+(?:we\s+can\s+)?(?:decide|discuss|choose|weigh|review|consider)'
    r'|we\s+can\s+(?:decide|discuss|review|consider|choose)\s+together'
    r'|let\s+us\s+(?:decide|discuss|review|consider)\s+together'
    
    # --- Offering options / discussion ---
    r'|(?:would|shall)\s+(?:you\s+)?like\s+to\s+'
    r'(?:see|discuss|go\s+over|review|consider|talk\s+about)\s+'
    r'(?:the\s+)?(?:options?|choices?|alternatives?|numbers?|data)'
    r'|here\s+are\s+(?:the\s+)?options'
    r'|there\s+are\s+(?:several|different)\s+(?:options|approaches|treatments)'
    
    # --- Explaining tradeoffs to patient ---
    r'|we\s+can\s+(?:weigh|balance)\s+(?:the\s+)?(?:risks?|benefits?|pros?\s+and\s+cons)'
    r'|let\'?s\s+(?:weigh|review|go\s+over)\s+(?:the\s+)?(?:risks?|benefits?|options)'
    
    # --- Consent / agreement language ---
    r'|do\s+you\s+agree'
    r'|are\s+you\s+comfortable\s+with'
    r'|does\s+that\s+sound\s+acceptable'
    r'|is\s+that\s+okay\s+with\s+you'
    
    # --- Decision support / aids ---
    r'|decision\s+aid'
    r'|we\s+can\s+use\s+(?:a\s+)?decision\s+tool'
    r'|let\s+me\s+explain\s+(?:the\s+)?(?:options?|risks?|benefits?)\s+so\s+you\s+can\s+decide'
    
    r')\b',
    re.IGNORECASE,
),
    "COMPARISON": re.compile(
    r'\b(?:'
    
    # --- Explicit vs / versus ---
    r'\w+(?:\s+\w+){0,3}\s+vs\.?\s+\w+(?:\s+\w+){0,3}'
    r'|\w+(?:\s+\w+){0,3}\s+versus\s+\w+(?:\s+\w+){0,3}'
    
    # --- Compared to / with ---
    r'|compared?\s+(?:to|with)\s+\w+(?:\s+\w+){0,3}'
    r'|\w+(?:\s+\w+){0,3}\s+compared?\s+(?:to|with)\s+\w+(?:\s+\w+){0,3}'
    
    # --- Superiority / inferiority ---
    r'|(?:more|less|as\s+effective\s+as|superior\s+to|inferior\s+to)'
    r'\s+\w+(?:\s+\w+){0,3}'
    r'|\w+(?:\s+\w+){0,3}\s+(?:is|are)\s+(?:more|less|as\s+effective\s+as|superior|inferior)\s+than'
    
    # --- Outcome comparisons ---
    r'|(?:higher|lower|increased|decreased)\s+(?:risk|mortality|morbidity|efficacy|effectiveness|cost|rate)'
    r'\s+than\s+\w+(?:\s+\w+){0,3}'
    
    # --- Effectiveness / safety comparisons ---
    r'|(?:more|less)\s+(?:effective|safe|tolerable|cost[-\s]?effective|convenient)'
    r'\s+than\s+\w+(?:\s+\w+){0,3}'
    
    # --- Side-by-side / direct comparison ---
    r'|side[-\s]?by[-\s]?side'
    r'|head[-\s]?to[-\s]?head'
    
    # --- Alternatives framing ---
    r'|alternative[s]?\s+(?:include|to|for|are|such\s+as)'
    r'|options?\s+(?:include|are|such\s+as)'
    
    # --- Treatment hierarchy ---
    r'|(?:first|second|third)[-\s]?line\s+(?:option|therapy|treatment|agent|therapy)'
    r'|standard\s+of\s+care\s+(?:vs|versus|compared\s+to)'
    
    # --- Differences explicitly stated ---
    r'|difference[s]?\s+between\s+\w+(?:\s+\w+){0,3}\s+and\s+\w+(?:\s+\w+){0,3}'
    r'|distinction\s+between\s+\w+\s+and\s+\w+'
    
    # --- Both / dual comparison structures ---
    r'|both\s+\w+(?:\s+\w+){0,3}\s+and\s+\w+(?:\s+\w+){0,3}\s+(?:have|are|show|reduce|increase)'
    
    # --- Ranking language ---
    r'|(?:better|worse)\s+than'
    r'|most\s+(?:effective|safe|commonly\s+used)'
    
    # --- Statistical comparisons ---
    r'|significantly\s+(?:higher|lower|better|worse)\s+than'
    r'|relative\s+(?:risk|reduction|difference)'
    r'|hazard\s+ratio'
    r'|odds\s+ratio'
    
    # --- Comparative verbs ---
    r'|outperforms?\s+\w+(?:\s+\w+){0,3}'
    r'|\w+(?:\s+\w+){0,3}\s+outperforms?\s+\w+'
    
    r')\b',
    re.IGNORECASE,
),
}
# Process PRIMARY_SOURCE before EVIDENCE (more specific first)
_BALANCED_PATTERN_ORDER = ["PRIMARY_SOURCE", "EVIDENCE", "UNCERTAINTY",
                            "TRADEOFF", "SHARED_DECISION", "COMPARISON"]

# ── Transitional ──────────────────────────────────────────────────────────────
_ACRONYM_RE = re.compile(r'\b[A-Z]{2,5}\b')
# ── Transitional ──────────────────────────────────────────────────────────────
### MARK HERE ###
_TRANSITIONAL_PATTERNS: dict[str, re.Pattern] = {
    "TEACH_BACK": re.compile(
        r'\b(?:tell\s+me\s+(?:back\s+)?in\s+your\s+own\s+words'
        r'|can\s+you\s+(?:explain|describe|tell\s+me)\s+(?:back\s+)?(?:what|how)'
        r'|what\s+(?:would|will)\s+you\s+do\s+if'
        r'|how\s+(?:would|will)\s+you\s+(?:know|remember)\s+(?:when|if|to)'
        r'|in\s+your\s+own\s+words'
        r'|to\s+make\s+sure\s+(?:I\'?ve?\s+)?explained\s+(?:this\s+)?clearly'
        r'|checking?\s+(?:your\s+)?understanding)',
        re.IGNORECASE,
    ),
    "EMOTION_VALIDATE": re.compile(
        r'\b(?:it(?:\'s|\s+is)\s+(?:normal|natural|understandable|okay|ok|common)\s+to\s+(?:feel|worry|be\s+concerned)'
        r'|makes?\s+(?:complete\s+)?sense\s+(?:to\s+feel|that\s+you(?:\s+feel|\s+are)?)'
        r'|(?:many|most)\s+people\s+(?:feel|worry|are)\s+(?:the\s+same|worried|anxious|scared|nervous)'
        r'|your\s+(?:concern|worry|fear|anxiety)\s+is\s+(?:valid|understandable|normal|completely\s+understandable)'
        r'|(?:I\s+)?understand\s+(?:that\s+)?this\s+(?:is|can\s+be)\s+(?:scary|worrying|difficult|overwhelming|a\s+lot\s+to\s+take\s+in)'
        r'|it(?:\'s|\s+is)\s+(?:okay|ok|fine|perfectly\s+(?:okay|normal))\s+to\s+feel)',
        re.IGNORECASE,
    ),
    "METAPHOR": re.compile(
        r'\b(?:like\s+a[n]?\s+\w'
        r'|think\s+of\s+(?:it\s+)?(?:as|like)\s+'
        r'|imagine\s+(?:your|a[n]?\s+|the\s+)'
        r'|just\s+like\s+(?:a[n]?\s+)\w'
        r'|(?:acts?|works?|functions?|behaves?)\s+like\s+a[n]?\s+'
        r'|(?:your|the)\s+\w+\s+is\s+like\s+a[n]?\s+'
        r'|similar\s+to\s+(?:a[n]?\s+|the\s+)\w'
        r'|picture\s+(?:it|your|a[n]?))',
        re.IGNORECASE,
    ),
    "STEP": re.compile(
        r'(?:\bstep\s+\d+\b'
        r'|\bfirst(?:ly)?[,:]?\s+\w'
        r'|\bsecond(?:ly)?[,:]?\s+\w'
        r'|\bthird(?:ly)?[,:]?\s+\w'
        r'|\bfinally[,:]?\s+\w'
        r'|\blastly[,:]?\s+\w'
        r'|^\s*\d+[.)]\s+\w)',
        re.IGNORECASE | re.MULTILINE,
    ),
    "LAY_LINK": re.compile(
        r'\b(?:mayo\s+clinic'
        r'|webmd\b'
        r'|nhs\.uk|nhs\s+website'
        r'|medlineplus'
        r'|healthline\b'
        r'|patient\.org'
        r'|(?:american|british)\s+(?:heart|cancer|diabetes|lung)\s+(?:association|society|foundation)'
        r'|plain.?language\s+(?:resource|guide|version|leaflet)'
        r'|easy.?to.?read\s+(?:version|guide|leaflet)'
        r'|(?:patient|consumer)\s+(?:information\s+)?(?:leaflet|handout|guide))',
        re.IGNORECASE,
    ),
}
_TRANSITIONAL_PATTERN_ORDER = ["TEACH_BACK", "EMOTION_VALIDATE", "METAPHOR",
                                "STEP", "LAY_LINK", "ACRONYM", "SIMPLE_SENTENCE"]

# ── Specialized ───────────────────────────────────────────────────────────────
_SPECIALIZED_PATTERNS: dict[str, re.Pattern] = {
    "FULL_CITATION": re.compile(
        r'\b(?:PMID\s*:?\s*\d{7,9}'
        r'|DOI\s*:?\s*10\.\d{4,}/\S+'
        r'|NCT\s*0*\d{7,8}'
        r'|ISRCTN\s*\d+'
        r'|EudraCT\s*\d{4}-\d{6}-\d{2}'
        r'|(?:N\s*Engl\s*J\s*Med|NEJM|JAMA|Lancet|BMJ|Circulation|JACC|Ann\s+Intern\s+Med|Eur\s+Heart\s+J|Chest|Thorax)\b)',
        re.IGNORECASE,
    ),
    "ADVANCED_FILTER": re.compile(
        r'\b(?:PICO\b'
        r'|(?:population|intervention|comparator?|comparison|outcome)\s*[,:]'
        r'|systematic\s+review'
        r'|meta.?analysis'
        r'|(?:inclusion|exclusion)\s+criteria'
        r'|(?:primary|secondary|composite)\s+endpoint'
        r'|intention.to.treat\s+(?:analysis|population)'
        r'|per.protocol\s+(?:analysis|population)'
        r'|(?:sub)?group\s+analysis'
        r'|sensitivity\s+analysis'
        r'|(?:I²|I-squared|heterogeneity)'
        r'|funnel\s+plot)',
        re.IGNORECASE,
    ),
    "HIGH_UNCERTAINTY": re.compile(
        r'(?:(?:debat|unclear|uncertain|not\s+(?:yet\s+)?(?:known|established|proven)|conflicting\s+evidence|no\s+consensus|limited\s+evidence|insufficient\s+data).{0,120}){2,}',
        re.IGNORECASE | re.DOTALL,
    ),
    "BRIDGE_TO_SELF": re.compile(
        r'\b(?:(?:your|my)\s+(?:specific\s+)?(?:age|weight|bmi|condition|diagnosis|medical\s+history|situation|case)'
        r'|how\s+(?:does\s+)?this\s+(?:apply|relate|translate)\s+to\s+(?:you|your\s+\w+)'
        r'|given\s+(?:your|my)\s+(?:age|history|diagnosis|condition|profile)'
        r'|personalis(?:e|ing|ed)|personaliz(?:e|ing|ed)\s+(?:this\s+)?(?:evidence|recommendation|finding)'
        r'|(?:compare[ds]?|match(?:es)?)\s+(?:your|my)\s+(?:age|profile|condition)\s+(?:to|with|against))',
        re.IGNORECASE,
    ),
    "VALIDATE_NARRATIVE": re.compile(
        r'\b(?:thank\s+you\s+for\s+(?:sharing|telling|opening\s+up)'
        r'|(?:I\s+)?appreciate\s+(?:you\s+)?sharing\s+(?:your\s+)?(?:experience|story|journey|that)'
        r'|your\s+(?:experience|story|journey)\s+(?:is|sounds?|seems?|must\s+(?:have\s+been|be))'
        r'|that\s+(?:must\s+(?:have\s+been|be)|sounds?)\s+(?:difficult|hard|challenging|frightening|overwhelming|a\s+lot)'
        r'|living\s+with\s+\w+(?:\s+\w+)?\s+(?:can\s+be|is)\s+(?:difficult|challenging|hard|complex))',
        re.IGNORECASE,
    ),
    "STORY_FORMAT": re.compile(
        r'\b(?:(?:patient|a\s+\d+.year.old\s+\w+|he|she|they)\s+(?:felt|experienced|noticed|reported|described|presented|complained)'
        r'|(?:after|before|when|while)\s+(?:she|he|they)\s+\w+ed\b'
        r'|one\s+(?:patient|person|day|morning|evening)\b'
        r'|(?:her|his|their)\s+(?:story|experience|journey|account)\b'
        r'|(?:was|were)\s+(?:diagnosed|admitted|treated|referred)\s+(?:when|after|while|following))',
        re.IGNORECASE,
    ),
    "GENTLE_EVIDENCE": re.compile(
        r'\b(?:research\s+(?:also\s+)?(?:shows?|suggests?|indicates?|finds?)'
        r'|studies\s+(?:also\s+)?(?:show|suggest|indicate|find|have\s+(?:shown|found))'
        r'|evidence\s+(?:also\s+)?(?:shows?|suggests?|supports?|points\s+to)'
        r'|adding\s+(?:one\s+)?(?:research|study)\s+finding'
        r'|interestingly[,\s]+(?:research|studies|evidence|data)'
        r'|(?:this\s+)?(?:aligns?\s+with|is\s+supported\s+by)\s+(?:research|evidence|studies)'
        r'|building\s+on\s+(?:your\s+)?(?:experience|story|what\s+you\'ve\s+shared))',
        re.IGNORECASE,
    ),
}
_SPECIALIZED_PATTERN_ORDER = ["FULL_CITATION", "ADVANCED_FILTER", "HIGH_UNCERTAINTY",
                               "BRIDGE_TO_SELF", "VALIDATE_NARRATIVE", "STORY_FORMAT",
                               "GENTLE_EVIDENCE"]
def clarity_consensus(text: str) -> dict:
    doc   = nlp(text)
    sents = list(doc.sents)

    jargon_spans = _detect_jargon_spans(doc)
    n_terms      = len(jargon_spans)

    explained = 0
    for (_, end), _ in jargon_spans.items():
        if _EXPLANATION_CUES.search(text[end: end + _EXPLANATION_WINDOW]):
            explained += 1

    unexplained       = max(n_terms - explained, 0)
    jargon_score      = 1.0 - min(1.0, unexplained / max(n_terms, 1))
    explanation_score = explained / max(n_terms, 1) if n_terms > 0 else 1.0

    layer_counts: dict[str, int] = {}
    for layer in jargon_spans.values():
        layer_counts[layer] = layer_counts.get(layer, 0) + 1

    fre           = textstat.flesch_reading_ease(text)
    fluency_score = float(min(1.0, max(0.0, fre / 100.0)))

    coherence_score     = 1.0
    flagged_transitions = []
    if len(sents) >= 2:
        sims = []
        for s1, s2 in zip(sents[:-1], sents[1:]):
            v1, v2 = s1.vector, s2.vector
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                sim = float(np.dot(v1, v2) / (n1 * n2))
                sims.append(sim)
                if sim < 0.55:
                    flagged_transitions.append((str(s1)[:60], str(s2)[:60], round(sim, 3)))
        coherence_score = float(np.mean(sims)) if sims else 1.0

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
        "layer_counts":        layer_counts,
        "flagged_transitions": flagged_transitions,
    }


# =============================================================================
# Text annotation — HL dimensions
# =============================================================================
def annotate_text(text: str):
    lower  = text.lower()
    n      = len(text)
    labels = [None] * n
    P      = HealthLiteracyPatterns

    dim_patterns = [
        ("CRHL", P.CAUSAL_PATTERNS + P.CONTRASTIVE_PATTERNS + P.EVIDENCE_PATTERNS + P.OPTIONS_PATTERNS),
        ("DHL",  P.TRUSTED_SOURCES_PATTERNS + [P.URL_PATTERN] + P.CROSS_REF_PATTERNS),
        ("CHL",  P.CONTEXT_PATTERNS + P.CONDITIONAL_PATTERNS),
        ("FHL",  P.HEDGING_PATTERNS + P.CERTAINTY_PATTERNS),
    ]
    for dim, patterns in dim_patterns:
        for pat_str in patterns:
            try:
                for m in re.compile(pat_str, re.IGNORECASE).finditer(lower):
                    s, e = m.start(), m.end()
                    if all(labels[i] is None for i in range(s, e)):
                        for i in range(s, e):
                            labels[i] = dim
            except re.error:
                continue

    segments, i = [], 0
    while i < n:
        label = labels[i]
        j = i + 1
        while j < n and labels[j] == label:
            j += 1
        segments.append((text[i:j], label))
        i = j
    return segments


# =============================================================================
# Text annotation — ClarityConsensus
# =============================================================================
def annotate_clarity(text: str) -> list:
    doc    = nlp(text)
    n      = len(text)
    labels = [None] * n

    for token in doc:
        if token.is_alpha and len(token.text) >= 6:
            if textstat.syllable_count(token.text) >= 3:
                for i in range(token.idx, min(token.idx + len(token.text), n)):
                    if labels[i] is None:
                        labels[i] = "Fluency"

    _COHERENCE_MARKERS = {
        "however","although","though","yet","nevertheless","furthermore",
        "moreover","therefore","thus","hence","consequently","in contrast",
        "on the other hand","in addition","additionally","despite","whereas",
        "as a result","in conclusion","in summary","for example","for instance",
        "that is","in other words","specifically","notably","importantly",
    }
    lower = text.lower()
    for marker in sorted(_COHERENCE_MARKERS, key=len, reverse=True):
        for m in re.compile(r'\b' + re.escape(marker) + r'\b').finditer(lower):
            for i in range(m.start(), min(m.end(), n)):
                labels[i] = "Coherence"

    _EXP = re.compile(
        r'\(([^)]{5,})\)'
        r'|,?\s*(also known as|defined as|refers to|meaning|i\.e\.)[^,\.;]{0,80}',
        re.IGNORECASE,
    )
    for m in _EXP.finditer(text):
        for i in range(m.start(), min(m.end(), n)):
            labels[i] = "Explanation"

    jargon_spans = _detect_jargon_spans(doc)
    for (start, end), layer in jargon_spans.items():
        for i in range(start, min(end, n)):
            labels[i] = f"Jargon-{layer}"
    # ── Layer 5: Medical quantities & units (highest priority — overrides all) ──
    for m in _MEDICAL_QUANTITY_RE.finditer(text):
        for i in range(m.start(), min(m.end(), n)):
            labels[i] = "Quantity"
    segments, i = [], 0
    while i < n:
        label = labels[i]
        j = i + 1
        while j < n and labels[j] == label:
            j += 1
        segments.append((text[i:j], label))
        i = j
    return segments

# =============================================================================
# PERSONA ANNOTATION FUNCTIONS
# =============================================================================

def annotate_balanced(text: str) -> list[tuple[str, str | None]]:
    n      = len(text)
    labels = [None] * n
    for key in _BALANCED_PATTERN_ORDER:
        pat = _BALANCED_PATTERNS[key]
        for m in pat.finditer(text):
            for i in range(m.start(), min(m.end(), n)):
                if labels[i] is None:
                    labels[i] = key
    segments, i = [], 0
    while i < n:
        lab = labels[i]; j = i + 1
        while j < n and labels[j] == lab: j += 1
        segments.append((text[i:j], lab)); i = j
    return segments


def annotate_transitional(text: str) -> list[tuple[str, str | None]]:
    doc    = nlp(text)
    n      = len(text)
    labels = [None] * n

    # SIMPLE_SENTENCE — flag whole sentences that exceed 20 alphabetic tokens
    for sent in doc.sents:
        alpha_count = sum(1 for t in sent if t.is_alpha)
        if alpha_count > 20:
            for i in range(sent.start_char, min(sent.end_char, n)):
                labels[i] = "SIMPLE_SENTENCE"

    # Ordered pattern pass (higher-priority patterns first)
    for key in _TRANSITIONAL_PATTERN_ORDER:
        if key == "ACRONYM":
            for m in _ACRONYM_RE.finditer(text):
                for i in range(m.start(), min(m.end(), n)):
                    if labels[i] is None:
                        labels[i] = "ACRONYM"
        elif key == "SIMPLE_SENTENCE":
            continue   # already done above
        else:
            pat = _TRANSITIONAL_PATTERNS[key]
            for m in pat.finditer(text):
                for i in range(m.start(), min(m.end(), n)):
                    if labels[i] is None:
                        labels[i] = key

    segments, i = [], 0
    while i < n:
        lab = labels[i]; j = i + 1
        while j < n and labels[j] == lab: j += 1
        segments.append((text[i:j], lab)); i = j
    return segments


def annotate_specialized(text: str) -> list[tuple[str, str | None]]:
    n      = len(text)
    labels = [None] * n
    for key in _SPECIALIZED_PATTERN_ORDER:
        pat = _SPECIALIZED_PATTERNS[key]
        for m in pat.finditer(text):
            for i in range(m.start(), min(m.end(), n)):
                if labels[i] is None:
                    labels[i] = key
    segments, i = [], 0
    while i < n:
        lab = labels[i]; j = i + 1
        while j < n and labels[j] == lab: j += 1
        segments.append((text[i:j], lab)); i = j
    return segments


def _count_persona_markers(segments: list[tuple[str, str | None]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, lab in segments:
        if lab:
            counts[lab] = counts.get(lab, 0) + 1
    return counts


def _build_persona_span_elements(
    segments: list[tuple[str, str | None]],
    colour_map: dict,
) -> list:
    """Render annotated segments as Dash html.Span elements with badge superscripts."""
    elements = []
    for segment, label in segments:
        if label and label in colour_map:
            cfg = colour_map[label]
            elements.append(
                html.Span(
                    [
                        html.Span(segment),
                        html.Span(
                            cfg["badge"],
                            style={
                                "background":    cfg["color"],
                                "color":         "#0f172a",
                                "borderRadius":  "3px",
                                "padding":       "0 4px",
                                "fontSize":      "0.6rem",
                                "fontWeight":    "800",
                                "verticalAlign": "super",
                                "marginLeft":    "2px",
                                "lineHeight":    "1",
                            },
                        ),
                    ],
                    title=cfg["label"],
                    style={
                        "background":   cfg["bg"],
                        "color":        cfg["color"],
                        "borderBottom": f"2px solid {cfg['color']}",
                        "borderRadius": "3px",
                        "padding":      "1px 3px",
                        "fontWeight":   "600",
                        "cursor":       "help",
                    },
                )
            )
        else:
            elements.append(html.Span(segment, style={"color": "#cbd5e1"}))
    return elements


def _build_persona_legend(colour_map: dict) -> html.Div:
    return html.Div(
        style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "14px"},
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "6px"},
                children=[
                    html.Span(
                        cfg["badge"],
                        style={
                            "background":   cfg["color"],
                            "color":        "#0f172a",
                            "borderRadius": "3px",
                            "padding":      "0 5px",
                            "fontSize":     "0.68rem",
                            "fontWeight":   "800",
                        },
                    ),
                    html.Span(cfg["label"], style={"color": "#94a3b8", "fontSize": "0.78rem"}),
                ],
            )
            for cfg in colour_map.values()
        ],
    )


def _build_persona_summary_pills(counts: dict, colour_map: dict) -> html.Div:
    if not counts:
        return html.Span(
            "No markers detected in this text.",
            style={"color": "#475569", "fontSize": "0.8rem"},
        )
    return html.Div(
        style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
        children=[
            html.Span(
                [
                    html.Span(
                        colour_map[k]["badge"],
                        style={"fontWeight": "800", "marginRight": "5px",
                               "fontSize": "0.68rem", "letterSpacing": "0.04em"},
                    ),
                    html.Span(f"×{v}", style={"fontWeight": "600", "fontSize": "0.8rem"}),
                ],
                style={
                    "background":   colour_map[k]["bg"],
                    "color":        colour_map[k]["color"],
                    "border":       f"1px solid {colour_map[k]['color']}",
                    "borderRadius": "20px",
                    "padding":      "3px 10px",
                    "display":      "inline-flex",
                    "alignItems":   "center",
                },
            )
            for k, v in counts.items()
            if k in colour_map
        ],
    )


def _build_persona_annotation_section(
    persona_name: str,
    accent_color: str,
    description: str,
    segments: list,
    colour_map: dict,
    reading_level_line: html.Span | None = None,
) -> html.Div:
    counts = _count_persona_markers(segments)
    total  = sum(counts.values())
    return html.Div(
        style={
            "background":    "#1e293b",
            "border":        f"1px solid {hex_to_rgba(accent_color, 0.4)}",
            "borderRadius":  "12px",
            "padding":       "20px",
            "marginBottom":  "20px",
        },
        children=[
            # ── Header ────────────────────────────────────────────────────────
            html.Div(
                style={"display": "flex", "justifyContent": "space-between",
                       "alignItems": "flex-start", "marginBottom": "6px"},
                children=[
                    html.Div([
                        html.P(
                            f"TEXT ANNOTATION — {persona_name.upper()} PERSONA MARKERS",
                            style={"color": "#64748b", "fontSize": "0.75rem", "fontWeight": "700",
                                   "letterSpacing": "0.1em", "margin": "0 0 4px"},
                        ),
                        html.P(
                            description,
                            style={"color": "#475569", "fontSize": "0.76rem", "margin": "0"},
                        ),
                    ]),
                    html.Span(
                        f"{total} marker{'s' if total != 1 else ''} detected",
                        style={
                            "background":   hex_to_rgba(accent_color, 0.15),
                            "color":        accent_color,
                            "border":       f"1px solid {accent_color}",
                            "borderRadius": "20px",
                            "padding":      "4px 14px",
                            "fontSize":     "0.78rem",
                            "fontWeight":   "700",
                            "whiteSpace":   "nowrap",
                        },
                    ),
                ],
            ),
            # ── Reading level line (Transitional only) ────────────────────────
            *([reading_level_line] if reading_level_line else []),
            # ── Summary pills ─────────────────────────────────────────────────
            _build_persona_summary_pills(counts, colour_map),
            html.Hr(style={"borderColor": "#334155", "margin": "14px 0"}),
            # ── Legend ────────────────────────────────────────────────────────
            _build_persona_legend(colour_map),
            html.Hr(style={"borderColor": "#334155", "margin": "0 0 14px"}),
            # ── Annotated text ────────────────────────────────────────────────
            html.Div(
                _build_persona_span_elements(segments, colour_map),
                style={
                    "fontSize":      "0.95rem",
                    "lineHeight":    "2.4",
                    "fontFamily":    "'Courier New', monospace",
                    "whiteSpace":    "pre-wrap",
                    "wordBreak":     "break-word",
                },
            ),
        ],
    )
# =============================================================================
# DASH APP  (unchanged from v3)
# =============================================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="HL Profile Dashboard",
)

app.layout = dbc.Container(
    fluid=True,
    style={"background":"#0f172a","minHeight":"100vh","padding":"0"},
    children=[
        html.Div(
            style={"background":"linear-gradient(135deg,#1e293b 0%,#0f172a 100%)",
                   "borderBottom":"1px solid #334155","padding":"24px 40px","marginBottom":"28px"},
            children=[
                html.H1("Health Literacy Profile Dashboard",
                        style={"color":"#f1f5f9","fontFamily":"Georgia,serif","fontWeight":"700",
                               "fontSize":"1.9rem","margin":"0","letterSpacing":"-0.5px"}),
                html.P("Submit any health-related text to receive a full profile report and upgrade guidance.",
                       style={"color":"#94a3b8","margin":"6px 0 0","fontSize":"0.9rem"}),
            ],
        ),
        dbc.Container(
            fluid=False,
            style={"maxWidth":"1300px","padding":"0 24px"},
            children=[
                dbc.Card(
                    style={"background":"#1e293b","border":"1px solid #334155",
                           "borderRadius":"12px","marginBottom":"24px"},
                    children=[dbc.CardBody([
                        html.Label("Paste your health text below",
                                   style={"color":"#94a3b8","fontSize":"0.85rem","fontWeight":"600",
                                          "letterSpacing":"0.05em","textTransform":"uppercase",
                                          "marginBottom":"10px","display":"block"}),
                        dcc.Textarea(
                            id="input-text",
                            placeholder="e.g. I have type 2 diabetes and my doctor prescribed metformin 500 mg twice daily…",
                            style={"width":"100%","height":"130px","background":"#0f172a","color":"#f1f5f9",
                                   "border":"1px solid #475569","borderRadius":"8px","padding":"14px",
                                   "fontSize":"0.95rem","fontFamily":"'Courier New',monospace","resize":"vertical"},
                        ),
                        html.Div(
                            style={"display":"flex","justifyContent":"space-between",
                                   "alignItems":"center","marginTop":"14px"},
                            children=[
                                html.Span(id="word-count", style={"color":"#64748b","fontSize":"0.82rem"}),
                                dbc.Button("Analyse Text →", id="analyse-btn", n_clicks=0,
                                           style={"background":"linear-gradient(135deg,#6366f1,#8b5cf6)",
                                                  "border":"none","borderRadius":"8px","padding":"10px 28px",
                                                  "fontWeight":"600","fontSize":"0.95rem"}),
                            ],
                        ),
                    ])],
                ),
                dcc.Loading(id="loading", type="circle", color="#6366f1",
                            children=[html.Div(id="results-area")]),
            ],
        ),
    ],
)


@app.callback(Output("word-count","children"), Input("input-text","value"))
def update_wc(text):
    if not text:
        return ""
    return f"{len(text.split())} words"


@app.callback(
    Output("results-area","children"),
    Input("analyse-btn","n_clicks"),
    State("input-text","value"),
    prevent_initial_call=True,
)
def run_analysis(n, text):
    if not text or len(text.strip()) < 20:
        return dbc.Alert("Please enter at least 20 characters of text.", color="warning")

    try:
        r       = analyse(text)
        clarity = clarity_consensus(text)
    except Exception as e:
        return dbc.Alert(f"Analysis error: {e}", color="danger")

    profile = r["profile"]
    sub     = r["sub_type"]
    hl      = r["hl_level"]
    suit    = r["suitability"]
    raw     = r["raw_scores"]
    f1, f2, f3 = r["f1"], r["f2"], r["f3"]
    sigma   = r["sigma"]
    flagged = r["flagged"]
    pcol    = PROFILE_COLOURS.get(profile, "#6366f1")
    hlcol   = HL_COLOURS.get(hl, "#6366f1")
    label   = profile + (f" · {sub}" if sub else "")

    hero = dbc.Row(style={"marginBottom":"20px"}, children=[
        dbc.Col(md=4, children=[
            html.Div(style={"background":"#1e293b","border":f"2px solid {pcol}",
                            "borderRadius":"12px","padding":"22px","height":"100%"}, children=[
                html.P("ASSIGNED PROFILE",
                       style={"color":"#64748b","fontSize":"0.75rem","fontWeight":"700",
                              "letterSpacing":"0.1em","margin":"0 0 8px"}),
                html.H2(label, style={"color":pcol,"fontFamily":"Georgia,serif","fontSize":"1.6rem",
                                      "fontWeight":"700","margin":"0 0 6px"}),
                html.Div(style={"display":"flex","gap":"10px","alignItems":"center","marginTop":"10px"}, children=[
                    html.Span(hl, style={"background":hlcol,"color":"#fff","borderRadius":"20px",
                                         "padding":"4px 14px","fontSize":"0.8rem","fontWeight":"700"}),
                    html.Span(f"σ = {sigma:.4f}", style={"color":"#94a3b8","fontSize":"0.82rem"}),
                    html.Span("⚑ FLAGGED", style={"color":"#ef4444","fontSize":"0.8rem","fontWeight":"700"})
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
                        html.Span(f"{suit[p]}%", style={"color":PROFILE_COLOURS[p],"fontSize":"0.88rem","fontWeight":"700"}),
                    ]),
                    html.Div(style={"background":"#0f172a","borderRadius":"4px","height":"8px","overflow":"hidden"}, children=[
                        html.Div(style={"width":f"{suit[p]}%","height":"100%",
                                        "background":PROFILE_COLOURS[p],"borderRadius":"4px"}),
                    ]),
                ]) for p in ["Balanced","Transitional","Specialized"]],
            ])
        ]),
    ])

    def score_colour(v):
        if v >= 0.80: return "#22c55e"
        if v >= 0.60: return "#f97316"
        return "#ef4444"

    final_col    = score_colour(clarity["final"])
    layer_counts = clarity.get("layer_counts", {})

    _LAYER_DISPLAY_ORDER = ["NER", "lexicon", "morphology", "syllable"]
    layer_pill_row = html.Div(
        style={"display":"flex","gap":"8px","flexWrap":"wrap","margin":"10px 0 4px"},
        children=[
            html.Span(
                [
                    html.Span(JARGON_LAYER_COLOURS[lyr]["badge"],
                              style={"fontWeight":"800","marginRight":"4px",
                                     "fontSize":"0.7rem","letterSpacing":"0.05em"}),
                    html.Span(f"{layer_counts[lyr]}",
                              style={"fontWeight":"600","fontSize":"0.8rem"}),
                ],
                style={
                    "background":  JARGON_LAYER_COLOURS[lyr]["bg"],
                    "color":       JARGON_LAYER_COLOURS[lyr]["color"],
                    "border":      f"1px solid {JARGON_LAYER_COLOURS[lyr]['color']}",
                    "borderRadius":"20px","padding":"3px 10px",
                    "fontSize":"0.78rem","display":"inline-flex","alignItems":"center",
                },
            )
            for lyr in _LAYER_DISPLAY_ORDER
            if lyr in layer_counts
        ],
    ) if layer_counts else html.Span()

    agent_rows = [
        ("Jargon",      clarity["jargon"],      "0.40",
         f"{clarity['n_terms']} medical terms, {clarity['n_explained']} explained",
         "Increase explained jargon via parentheticals e.g. 'metformin (a biguanide antidiabetic)'",
         "#f43f5e"),
        ("Explanation", clarity["explanation"], "0.30",
         f"{clarity['n_explained']}/{clarity['n_terms']} terms have inline definitions",
         "Add definitions for unexplained medical terms",
         "#8b5cf6"),
        ("Fluency",     clarity["fluency"],     "0.15",
         f"Flesch Reading Ease = {clarity['fre']}",
         "Shorten sentences, reduce syllable density",
         "#06b6d4"),
        ("Coherence",   clarity["coherence"],   "0.15",
         f"{len(clarity['flagged_transitions'])} low-similarity transitions detected",
         "Add linking phrases between abrupt topic changes",
         "#f59e0b"),
    ]

    clarity_section = html.Div(
        style={"background":"#1e293b","border":f"1px solid {hex_to_rgba(final_col, 0.4)}",
               "borderRadius":"12px","padding":"20px","marginBottom":"20px"},
        children=[
            html.Div(style={"display":"flex","justifyContent":"space-between",
                            "alignItems":"center","marginBottom":"6px"}, children=[
                html.Div([
                    html.P("CLARITYCONSENSUS SCORE",
                           style={"color":"#64748b","fontSize":"0.75rem","fontWeight":"700",
                                  "letterSpacing":"0.1em","margin":"0 0 4px"}),
                    html.P("Multi-agent comprehensibility · Jargon × Explanation × Fluency × Coherence",
                           style={"color":"#475569","fontSize":"0.76rem","margin":"0"}),
                ]),
                html.Div(style={"textAlign":"right"}, children=[
                    html.Span(f"{clarity['final']:.3f}",
                              style={"color":final_col,"fontSize":"2rem","fontWeight":"800",
                                     "fontFamily":"Courier New","lineHeight":"1"}),
                    html.Span(" / 1.000", style={"color":"#475569","fontSize":"0.85rem"}),
                ]),
            ]),
            layer_pill_row,
            html.Hr(style={"borderColor":"#334155","margin":"14px 0"}),
            *[html.Div(style={
                "display":"flex","gap":"14px","alignItems":"flex-start",
                "marginBottom":"12px","paddingBottom":"12px",
                "borderBottom":f"1px solid {hex_to_rgba(col, 0.2)}",
                "borderLeft":  f"3px solid {col}",
                "paddingLeft": "12px",
            }, children=[
                html.Div(style={"minWidth":"110px"}, children=[
                    html.Span(name, style={"color":col,"fontSize":"0.85rem","fontWeight":"700"}),
                    html.Br(),
                    html.Span(f"w = {weight}", style={"color":"#475569","fontSize":"0.75rem"}),
                ]),
                html.Div(style={"flex":"1"}, children=[
                    html.Div(style={"display":"flex","justifyContent":"space-between","marginBottom":"4px"}, children=[
                        html.Span(detail, style={"color":"#94a3b8","fontSize":"0.78rem"}),
                        html.Span(f"{score:.3f}", style={"color":col,"fontWeight":"700",
                                                          "fontSize":"0.85rem","fontFamily":"Courier New"}),
                    ]),
                    html.Div(style={"background":"#0f172a","borderRadius":"4px","height":"6px","overflow":"hidden"}, children=[
                        html.Div(style={"width":f"{score*100:.1f}%","height":"100%",
                                        "background":f"linear-gradient(90deg,{hex_to_rgba(col,0.5)},{col})",
                                        "borderRadius":"4px"}),
                    ]),
                    html.Span(f"▲ {tip}", style={"color":"#475569","fontSize":"0.75rem",
                                                  "marginTop":"4px","display":"block"})
                    if score < 0.75 else html.Span(),
                ]),
            ]) for name, score, weight, detail, tip, col in agent_rows],
            *([
                html.Hr(style={"borderColor":"#1e3a5f","margin":"8px 0"}),
                html.P("LOW-COHERENCE TRANSITIONS",
                       style={"color":"#64748b","fontSize":"0.72rem","fontWeight":"700",
                              "letterSpacing":"0.08em","margin":"0 0 8px"}),
                *[html.Div(style={"background":"#0f172a","borderRadius":"6px",
                                  "padding":"8px 12px","marginBottom":"6px",
                                  "borderLeft":"3px solid #f97316"}, children=[
                    html.Span(f"sim={t[2]}  ", style={"color":"#f97316","fontFamily":"Courier New",
                                                       "fontSize":"0.78rem","fontWeight":"700"}),
                    html.Span(f'"{t[0]}…" → "{t[1]}…"', style={"color":"#94a3b8","fontSize":"0.78rem"}),
                ]) for t in clarity["flagged_transitions"][:4]],
            ] if clarity["flagged_transitions"] else []),
        ]
    )

    dims       = ["FHL","CHL","CRHL","DHL","EHL"]
    dim_labels = ["Functional","Communicative","Critical","Digital","Expressed"]
    vals       = [raw[d] for d in dims]
    mx         = max(vals) if max(vals) > 0 else 1
    norm_vals  = [v / mx * 100 for v in vals]

    radar_fig = go.Figure(go.Scatterpolar(
        r=norm_vals + [norm_vals[0]], theta=dim_labels + [dim_labels[0]],
        fill="toself", fillcolor=hex_to_rgba(pcol, 0.2),
        line=dict(color=pcol, width=2.5), marker=dict(size=6, color=pcol),
    ))
    radar_fig.update_layout(
        polar=dict(bgcolor="#0f172a",
                   radialaxis=dict(visible=True, range=[0,100], gridcolor="#334155",
                                   tickfont=dict(color="#64748b", size=9)),
                   angularaxis=dict(gridcolor="#334155", tickfont=dict(color="#cbd5e1", size=11))),
        paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
        margin=dict(l=40,r=40,t=30,b=30), height=290, showlegend=False,
    )

    factor_fig = go.Figure()
    for fname, fval, colour in [("F1  Core", f1, "#6366f1"),
                                  ("F2  Digital", f2, "#06b6d4"),
                                  ("F3  Applied", f3, "#f59e0b")]:
        factor_fig.add_trace(go.Bar(
            x=[(fval + 5) / 10 * 100], y=[fname], orientation="h",
            marker=dict(color=colour, line=dict(width=0)),
            text=f"{fval:+.3f}", textposition="inside",
            textfont=dict(color="#fff", size=11, family="Courier New"), width=0.5,
        ))
    factor_fig.add_vline(x=50, line_dash="dash", line_color="#475569", line_width=1)
    factor_fig.update_layout(
        barmode="overlay", paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
        xaxis=dict(range=[0,100], showticklabels=False, gridcolor="#1e293b"),
        yaxis=dict(tickfont=dict(color="#cbd5e1", size=11)),
        margin=dict(l=10,r=10,t=10,b=10), height=160, showlegend=False,
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
                        html.P(d, style={"color":"#64748b","fontSize":"0.72rem","fontWeight":"700","margin":"0"}),
                        html.P(f"{raw[d]:.3f}", style={"color":"#f1f5f9","fontSize":"1.05rem",
                                                        "fontWeight":"700","margin":"4px 0 0",
                                                        "fontFamily":"Courier New"}),
                    ]) for d in dims
                ]),
            ])
        ]),
    ])

    other_profiles  = [p for p in ["Balanced","Transitional","Specialized"] if p != profile]
    guidance_cards  = []
    for target in other_profiles:
        g    = get_guidance(profile, target)
        tcol = PROFILE_COLOURS[target]
        guidance_cards.append(dbc.Col(md=6, children=[
            html.Div(style={"background":"#1e293b","border":f"1px solid {hex_to_rgba(tcol,0.3)}",
                            "borderRadius":"12px","padding":"20px","height":"100%"}, children=[
                html.Div(style={"display":"flex","alignItems":"center","gap":"10px","marginBottom":"14px"}, children=[
                    html.Div(style={"width":"10px","height":"10px","borderRadius":"50%",
                                    "background":tcol,"flexShrink":"0"}),
                    html.P(f"To reach  {target.upper()}",
                           style={"color":tcol,"fontWeight":"700","fontSize":"0.9rem","margin":"0"}),
                ]),
                html.Div(style={"marginBottom":"12px"}, children=[
                    html.P("▲  ELEVATE", style={"color":"#22c55e","fontSize":"0.75rem","fontWeight":"700",
                                                 "letterSpacing":"0.08em","margin":"0 0 8px"}),
                    html.Ul(style={"margin":"0","paddingLeft":"18px"}, children=[
                        html.Li(tip, style={"color":"#cbd5e1","fontSize":"0.85rem","marginBottom":"5px","lineHeight":"1.4"})
                        for tip in g.get("elevate",[])
                    ] or [html.Li("No specific elevation needed.", style={"color":"#64748b","fontSize":"0.85rem"})]),
                ]),
                html.Div(children=[
                    html.P("▼  REDUCE", style={"color":"#ef4444","fontSize":"0.75rem","fontWeight":"700",
                                                "letterSpacing":"0.08em","margin":"0 0 8px"}),
                    html.Ul(style={"margin":"0","paddingLeft":"18px"}, children=[
                        html.Li(tip, style={"color":"#cbd5e1","fontSize":"0.85rem","marginBottom":"5px","lineHeight":"1.4"})
                        for tip in g.get("reduce",[])
                    ] or [html.Li("No specific reduction needed.", style={"color":"#64748b","fontSize":"0.85rem"})]),
                ]),
            ])
        ]))

    guidance = html.Div(style={"marginBottom":"24px"}, children=[
        html.P("UPGRADE / TRANSITION GUIDANCE",
               style={"color":"#64748b","fontSize":"0.75rem","fontWeight":"700",
                      "letterSpacing":"0.1em","marginBottom":"14px"}),
        dbc.Row(guidance_cards),
    ])

    interpretations = {
        "Balanced":     "Well-rounded health literacy across all dimensions. This author integrates medical vocabulary, contextualised questions, causal reasoning, and evidence awareness. Suitable for clinical content, shared-decision tools, and evidence-based patient education.",
        "Transitional": "Developing health literacy. The author uses plain language, limited medical terminology, and direct questions reflecting genuine uncertainty. Best served by simplified explanations, step-by-step guidance, and plain-language resources.",
        "Specialized":  f"Niche literacy profile ({sub or 'mixed'}). Digitally-Specialized authors cite credible databases and studies. Functionally-Specialized authors express rich personal health narratives. Content should be tailored to the detected sub-type.",
    }
    interp = html.Div(style={"background":"#1e293b","border":f"1px solid {hex_to_rgba(pcol,0.35)}",
                              "borderRadius":"12px","padding":"20px","marginBottom":"28px"}, children=[
        html.P("PROFILE INTERPRETATION",
               style={"color":"#64748b","fontSize":"0.75rem","fontWeight":"700",
                      "letterSpacing":"0.1em","margin":"0 0 10px"}),
        html.P(interpretations.get(profile,""),
               style={"color":"#cbd5e1","fontSize":"0.92rem","lineHeight":"1.6","margin":"0"}),
    ])

    span_elements = []
    for segment, dim in annotate_text(text):
        if dim and dim in DIMENSION_COLOURS:
            dc = DIMENSION_COLOURS[dim]
            span_elements.append(html.Span(
                segment, title=dc["label"],
                style={"background":dc["bg"],"color":dc["color"],
                       "borderBottom":f"2px solid {dc['color']}","borderRadius":"3px",
                       "padding":"1px 3px","fontWeight":"600","cursor":"help"},
            ))
        else:
            span_elements.append(html.Span(segment, style={"color":"#cbd5e1"}))

    dim_legend = html.Div(
        style={"display":"flex","gap":"16px","flexWrap":"wrap","marginBottom":"14px"},
        children=[
            html.Div(style={"display":"flex","alignItems":"center","gap":"6px"}, children=[
                html.Div(style={"width":"11px","height":"11px","borderRadius":"2px",
                                "background":DIMENSION_COLOURS[dim]["color"]}),
                html.Span(f"{dim} — {DIMENSION_COLOURS[dim]['label']}",
                          style={"color":"#94a3b8","fontSize":"0.78rem"}),
            ]) for dim in ["FHL","CHL","CRHL","DHL","EHL"]
        ],
    )

    annotation_section = html.Div(
        style={"background":"#1e293b","border":"1px solid #334155",
               "borderRadius":"12px","padding":"20px","marginBottom":"20px"},
        children=[
            html.P("TEXT ANNOTATION — Highlighted by HL Dimension",
                   style={"color":"#64748b","fontSize":"0.75rem","fontWeight":"700",
                          "letterSpacing":"0.1em","margin":"0 0 12px"}),
            html.P("Hover over a highlighted word to see its dimension.",
                   style={"color":"#475569","fontSize":"0.78rem","margin":"0 0 12px"}),
            dim_legend,
            html.Hr(style={"borderColor":"#334155","margin":"0 0 14px"}),
            html.Div(span_elements,
                     style={"fontSize":"0.95rem","lineHeight":"2.0",
                            "fontFamily":"'Courier New',monospace",
                            "whiteSpace":"pre-wrap","wordBreak":"break-word"}),
        ],
    )

    clarity_legend_items = []
    for lyr in _LAYER_DISPLAY_ORDER:
        cfg = JARGON_LAYER_COLOURS[lyr]
        clarity_legend_items.append(
            html.Div(style={"display":"flex","alignItems":"center","gap":"6px"}, children=[
                html.Div(style={"width":"11px","height":"11px","borderRadius":"2px","background":cfg["color"]}),
                html.Span(
                    [html.Span(cfg["badge"],
                               style={"background":cfg["color"],"color":"#0f172a","borderRadius":"3px",
                                      "padding":"0 4px","fontSize":"0.65rem","fontWeight":"800",
                                      "marginRight":"4px"}),
                     cfg["label"]],
                    style={"color":"#94a3b8","fontSize":"0.78rem"},
                ),
            ])
        )
    for key in ("Explanation","Coherence","Fluency"):
        cfg = CLARITY_ANNOTATION_COLOURS[key]
        clarity_legend_items.append(
            html.Div(style={"display":"flex","alignItems":"center","gap":"6px"}, children=[
                html.Div(style={"width":"11px","height":"11px","borderRadius":"2px","background":cfg["color"]}),
                html.Span(cfg["label"], style={"color":"#94a3b8","fontSize":"0.78rem"}),
            ])
        )

    clarity_legend = html.Div(
        style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"14px"},
        children=clarity_legend_items,
    )
    cfg = CLARITY_ANNOTATION_COLOURS["Quantity"]
    clarity_legend_items.append(
            html.Div(style={"display":"flex","alignItems":"center","gap":"6px"}, children=[
                html.Div(style={"width":"11px","height":"11px","borderRadius":"2px","background":cfg["color"]}),
                html.Span(cfg["label"], style={"color":"#94a3b8","fontSize":"0.78rem"}),
            ])
        )
    clarity_elements = []
    for segment, agent in annotate_clarity(text):
        if agent and agent in CLARITY_ANNOTATION_COLOURS:
            ac = CLARITY_ANNOTATION_COLOURS[agent]
            if agent.startswith("Jargon-"):
                layer_key = agent.split("-", 1)[1]
                badge_cfg = JARGON_LAYER_COLOURS.get(layer_key, {})
                clarity_elements.append(
                    html.Span(
                        [
                            html.Span(segment),
                            html.Span(
                                badge_cfg.get("badge", layer_key.upper()),
                                style={
                                    "background":    ac["color"],
                                    "color":         "#0f172a",
                                    "borderRadius":  "3px",
                                    "padding":       "0 4px",
                                    "fontSize":      "0.6rem",
                                    "fontWeight":    "800",
                                    "verticalAlign": "super",
                                    "marginLeft":    "2px",
                                    "lineHeight":    "1",
                                },
                            ),
                        ],
                        title=ac["label"],
                        style={
                            "background":    ac["bg"],
                            "color":         ac["color"],
                            "borderBottom":  f"2px solid {ac['color']}",
                            "borderRadius":  "3px",
                            "padding":       "1px 3px",
                            "fontWeight":    "600",
                            "cursor":        "help",
                        },
                    )
                )
            else:
                clarity_elements.append(
                    html.Span(segment, title=ac["label"],
                              style={"background":ac["bg"],"color":ac["color"],
                                     "borderBottom":f"2px solid {ac['color']}",
                                     "borderRadius":"3px","padding":"1px 3px",
                                     "fontWeight":"600","cursor":"help"})
                )
        else:
            clarity_elements.append(html.Span(segment, style={"color":"#cbd5e1"}))

    clarity_annotation_section = html.Div(
        style={"background":"#1e293b","border":"1px solid #334155",
               "borderRadius":"12px","padding":"20px","marginBottom":"20px"},
        children=[
            html.P("TEXT ANNOTATION — ClarityConsensus Agent (with Jargon Layer Labels)",
                   style={"color":"#64748b","fontSize":"0.75rem","fontWeight":"700",
                          "letterSpacing":"0.1em","margin":"0 0 12px"}),
            html.P([
                "Hover for agent name. Jargon terms carry a superscript badge showing detection layer: ",
                html.Span("NER",   style={"background":JARGON_LAYER_COLOURS["NER"]["color"],
                                          "color":"#0f172a","borderRadius":"3px","padding":"0 5px",
                                          "fontSize":"0.7rem","fontWeight":"800","marginRight":"4px"}),
                html.Span("LEX",   style={"background":JARGON_LAYER_COLOURS["lexicon"]["color"],
                                          "color":"#0f172a","borderRadius":"3px","padding":"0 5px",
                                          "fontSize":"0.7rem","fontWeight":"800","marginRight":"4px"}),
                html.Span("MORPH", style={"background":JARGON_LAYER_COLOURS["morphology"]["color"],
                                          "color":"#0f172a","borderRadius":"3px","padding":"0 5px",
                                          "fontSize":"0.7rem","fontWeight":"800","marginRight":"4px"}),
                html.Span("SYL",   style={"background":JARGON_LAYER_COLOURS["syllable"]["color"],
                                          "color":"#0f172a","borderRadius":"3px","padding":"0 5px",
                                          "fontSize":"0.7rem","fontWeight":"800"}),
            ], style={"color":"#475569","fontSize":"0.78rem","margin":"0 0 12px"}),
            clarity_legend,
            html.Hr(style={"borderColor":"#334155","margin":"0 0 14px"}),
            html.Div(clarity_elements,
                     style={"fontSize":"0.95rem","lineHeight":"2.4",
                            "fontFamily":"'Courier New',monospace",
                            "whiteSpace":"pre-wrap","wordBreak":"break-word"}),
        ],
    )
    # ── Persona annotations ────────────────────────────────────────────────────
    bal_segments  = annotate_balanced(text)
    trans_segments = annotate_transitional(text)
    spec_segments  = annotate_specialized(text)

    fk_grade = textstat.flesch_kincaid_grade(text)
    reading_level_line = html.Div(
        style={"marginBottom": "10px"},
        children=[
            html.Span("Flesch-Kincaid grade level: ", style={"color": "#475569", "fontSize": "0.78rem"}),
            html.Span(
                f"{fk_grade:.1f}",
                style={
                    "color":      "#ef4444" if fk_grade > 7 else "#22c55e",
                    "fontFamily": "Courier New",
                    "fontWeight": "700",
                    "fontSize":   "0.85rem",
                },
            ),
            html.Span(
                "  (target: grade 5 – 7 for Transitional readers)",
                style={"color": "#475569", "fontSize": "0.75rem"},
            ),
        ],
    )

    balanced_annotation_section = _build_persona_annotation_section(
        persona_name  = "Balanced",
        accent_color  = PROFILE_COLOURS["Balanced"],
        description   = "Health-savvy readers who want evidence, tradeoffs, and shared decisions.",
        segments      = bal_segments,
        colour_map    = BALANCED_MARKER_COLOURS,
    )

    transitional_annotation_section = _build_persona_annotation_section(
        persona_name       = "Transitional",
        accent_color       = PROFILE_COLOURS["Transitional"],
        description        = "Low-HL readers who need plain language, analogies, and step-by-step safety guidance.",
        segments           = trans_segments,
        colour_map         = TRANSITIONAL_MARKER_COLOURS,
        reading_level_line = reading_level_line,
    )

    specialized_annotation_section = _build_persona_annotation_section(
        persona_name  = "Specialized",
        accent_color  = PROFILE_COLOURS["Specialized"],
        description   = "Research-strong (Digital) or narrative-strong (Functional) readers needing bridging.",
        segments      = spec_segments,
        colour_map    = SPECIALIZED_MARKER_COLOURS,
    )
    return html.Div([
        hero,
        clarity_section,
        annotation_section,
        clarity_annotation_section,
        balanced_annotation_section,
        transitional_annotation_section,
        specialized_annotation_section,
        charts,
        guidance,
        interp,
    ])


if __name__ == "__main__":
    app.run(debug=False, port=8050)