"""
Health Literacy Feature Extraction - Regex + POS Tagging Approach
==================================================================

Replaces keyword matching with flexible regex patterns and linguistic analysis.
Captures variations, handles inflections, and reduces false positives.
"""

import re
import spacy
from typing import Dict, List, Tuple, Set
import numpy as np

# Load spaCy models
try:
    nlp = spacy.load('en_core_sci_md')
except OSError:
    nlp = spacy.load('en_core_web_sm')

try:
    nlp_sm = spacy.load('en_core_web_sm')
except OSError:
    nlp_sm = nlp


# ══════════════════════════════════════════════════════════════════════════════
# REGEX PATTERN DEFINITIONS (Replace Keyword Lists)
# ══════════════════════════════════════════════════════════════════════════════

class HealthLiteracyPatterns:
    """
    Comprehensive regex patterns for health literacy feature extraction.
    Each pattern is compiled with variations and context awareness.
    """
    
    # ─── CERTAINTY & CONFIDENCE PATTERNS ──────────────────────────────────────
    
    CERTAINTY_PATTERNS = [
        # Absolute certainty
        r'\b(definitely|certainly|absolutely|clearly|obviously|undoubtedly)\b',
        r'\b(always|never|must|will|cannot|can\'?t)\b',
        
        # Causal certainty
        r'\b(causes?|lead(s|ing)?\s+to|results?\s+in)\b',
        r'\b(is\s+responsible\s+for|has\s+been\s+proven)\b',
        r'\b(is\s+proven|is\s+a\s+fact|evidence\s+shows?)\b',
        
        # Clinical certainty
        r'\b(clinically\s+proven|guarantees?|confirms?)\b',
        r'\b(there\s+is\s+no\s+doubt|it\s+is\s+certain)\b',
        r'\b(cures?|100%\s+effective|completely\s+eliminates?)\b',
        
        # Strength modifiers with certainty
        r'\b(always\s+(works?|effective|successful))\b',
        r'\b(never\s+(fails?|causes?))\b',
    ]
    
    HEDGING_PATTERNS = [
        # Modal hedges
        r'\b(may|might|could|would|maybe|possibly|probably)\b',
        r'\b(perhaps|potentially|presumably|conceivably)\b',
        
        # Approximation
        r'\b(approximately|roughly|around|about|nearly|almost)\b',
        r'\b(more\s+or\s+less|give\s+or\s+take)\b',
        
        # Frequency hedges
        r'\b(generally|often|sometimes|occasionally|typically)\b',
        r'\b(usually|frequently|rarely|seldom)\b',
        
        # Appearance/seeming
        r'\b(seems?|appears?|looks?\s+like|sounds?\s+like)\b',
        r'\b(suggests?|indicates?|implies?)\b',
        
        # Personal uncertainty
        r'\b(i\s+think|i\s+believe|i\s+guess|i\s+assume)\b',
        r'\b(i\'?m\s+not\s+sure|i\s+am\s+not\s+sure|not\s+certain)\b',
        r'\b(it\s+seems?|it\s+may\s+be|it\s+is\s+possible)\b',
        
        # Epistemic markers
        r'\b(unclear|unknown|uncertain|unsure)\b',
        r'\b(limited\s+evidence|inconclusive|debatable)\b',
        r'\b(in\s+some\s+cases|in\s+certain\s+situations)\b',
    ]
    
    # ─── COMMUNICATIVE HEALTH LITERACY PATTERNS ───────────────────────────────
    
    CONDITIONAL_PATTERNS = [
        # Basic conditionals
        r'\b(if|when|unless|whether|supposing)\b',
        
        # Complex conditionals
        r'\b(in\s+case(\s+of)?|provided(\s+that)?)\b',
        r'\b(as\s+long\s+as|given\s+that|on\s+condition\s+that)\b',
        r'\b(only\s+if|even\s+if|what\s+if)\b',
        
        # Dependency expressions
        r'\b(depending\s+on|contingent\s+on|subject\s+to)\b',
        r'\b(varies?\s+(with|according\s+to|based\s+on))\b',
    ]
    
    MODAL_VERBS_PATTERN = r'\b(can|could|may|might|shall|should|will|would|must)\b'
    
    CONTEXT_PATTERNS = [
        # Personal medical history
        r'\b(i\s+(have|had|was\s+diagnosed(\s+with)?|suffer(\s+from)?))\b',
        r'\b(my\s+(doctor|physician|specialist|condition))\b',
        r'\b(been\s+(taking|on|using)\s+.*\s+for)\b',
        
        # Temporal context
        r'\b(for\s+\d+\s+(days?|weeks?|months?|years?))\b',
        r'\b(since\s+(last|this|\d+))\b',
        r'\b(during\s+(the)?)\b',
        
        # Information seeking context
        r'\b(do\s+i\s+need|should\s+i\s+(mention|tell|ask))\b',
        r'\b(any\s+other\s+information|let\s+me\s+know\s+if)\b',
        r'\b(what\s+(else|other|additional))\b',
    ]
    
    # ─── CRITICAL HEALTH LITERACY PATTERNS ────────────────────────────────────
    
    CAUSAL_PATTERNS = [
        # Basic causation
        r'\b(because|since|therefore|thus|hence)\b',
        r'\b(as\s+a\s+result(\s+of)?|consequently)\b',
        
        # Direct causation
        r'\b(leads?\s+to|results?\s+in|causes?)\b',
        r'\b(due\s+to|owing\s+to|thanks\s+to)\b',
        r'\b(for\s+this\s+reason|that\'?s\s+why)\b',
        
        # Association
        r'\b(associated\s+with|linked\s+(to|with)|correlated\s+with)\b',
        r'\b(connected\s+(to|with)|related\s+to)\b',
        
        # Mechanism
        r'\b(responsible\s+for|triggers?|brings?\s+about)\b',
        r'\b(contributes?\s+to|plays\s+a\s+role\s+in)\b',
    ]
    
    CONTRASTIVE_PATTERNS = [
        # Basic contrast
        r'\b(however|although|though|yet|but|still)\b',
        r'\b(nevertheless|nonetheless|regardless)\b',
        
        # Strong contrast
        r'\b(on\s+the\s+(other\s+hand|contrary))\b',
        r'\b(in\s+contrast(\s+to)?|conversely)\b',
        
        # Concessive
        r'\b(while|whereas|even\s+(though|if))\b',
        r'\b(despite(\s+the\s+fact)?|notwithstanding)\b',
        r'\b(admittedly|granted)\b',
    ]
    
    EVIDENCE_PATTERNS = [
        # Research/studies
        r'\b(study|studies|research|trial|trials)\s+(shows?|indicate|suggest|found|demonstrate)\b',
        r'\b((clinical|randomized|controlled)\s+trial)\b',
        r'\b(systematic\s+review|meta-analysis|cohort\s+study)\b',
        
        # Evidence strength
        r'\b(evidence\s+(suggests?|shows?|indicates?|demonstrates?))\b',
        r'\b(data\s+(from|suggests?|shows?|indicates?))\b',
        r'\b(findings?\s+(suggest|show|indicate|demonstrate))\b',
        
        # Attribution
        r'\b(according\s+to(\s+the)?)\b',
        r'\b(published\s+in|reported\s+in|cited\s+in)\b',
        r'\b(as\s+(noted|reported|stated|mentioned)\s+in)\b',
        
        # Expert opinion
        r'\b((guidelines?|experts?|specialists?)\s+recommend)\b',
        r'\b(consensus\s+(is|suggests?))\b',
        r'\b(scientific\s+evidence(\s+shows?)?)\b',
    ]
    
    OPTIONS_PATTERNS = [
        # Alternatives
        r'\b(alternatively|another\s+(option|approach|method|way))\b',
        r'\b(instead(\s+of)?|rather\s+than|or)\b',
        
        # Comparison/evaluation
        r'\b(pros\s+and\s+cons|benefits?\s+and\s+risks?)\b',
        r'\b(trade-?offs?|weigh(\s+the)?|consider(ing)?)\b',
        r'\b(compare|comparison|versus|vs\.?)\b',
        
        # Choice/decision
        r'\b(choose\s+(between|from)|select|pick)\b',
        r'\b(options?\s+(are|include)|choices?)\b',
        r'\b(either\s+.*\s+or|whether\s+.*\s+or)\b',
    ]
    
    # ─── DIGITAL HEALTH LITERACY PATTERNS ─────────────────────────────────────
    
    URL_PATTERN = r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+'
    
    TRUSTED_SOURCES_PATTERNS = [
        # Health organizations
        r'\b(who|world\s+health\s+organization)\b',
        r'\b(cdc|centers?\s+for\s+disease\s+control)\b',
        r'\b(nih|national\s+institutes?\s+of\s+health)\b',
        r'\b(nice|nhs|fda|food\s+and\s+drug\s+administration)\b',
        
        # Medical institutions
        r'\b(mayo\s+clinic|cleveland\s+clinic)\b',
        r'\b(johns\s+hopkins|harvard\s+medical)\b',
        
        # Databases/journals
        r'\b(pubmed|ncbi|medline|cochrane)\b',
        r'\b(lancet|jama|bmj|nejm)\b',
        r'\b(new\s+england\s+journal(\s+of\s+medicine)?)\b',
        
        # Medical resources
        r'\b(uptodate|webmd|medscape|healthline)\b',
    ]
    
    CROSS_REF_PATTERNS = [
        r'\b(according\s+to(\s+the)?)\b',
        r'\b((as\s+)?(reported|noted|stated|mentioned)\s+(by|in))\b',
        r'\b(study\s+(shows?|found|suggests?))\b',
        r'\b(guidelines?\s+(say|state|recommend))\b',
        r'\b(based\s+on\s+(research|study|evidence))\b',
        r'\b(scientific\s+evidence(\s+shows?)?)\b',
        r'\b(peer-?reviewed|evidence-?based)\b',
    ]
    
    # ─── COMPILE ALL PATTERNS ─────────────────────────────────────────────────
    
    @classmethod
    def compile_patterns(cls) -> Dict[str, List[re.Pattern]]:
        """Compile all regex patterns for efficient matching."""
        compiled = {}
        
        for attr_name in dir(cls):
            if attr_name.endswith('_PATTERNS') or attr_name.endswith('_PATTERN'):
                patterns = getattr(cls, attr_name)
                
                # Handle single pattern vs list of patterns
                if isinstance(patterns, str):
                    compiled[attr_name] = re.compile(patterns, re.IGNORECASE)
                elif isinstance(patterns, list):
                    compiled[attr_name] = [
                        re.compile(p, re.IGNORECASE) for p in patterns
                    ]
        
        return compiled


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN MATCHING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

class PatternMatcher:
    """Efficient pattern matching with context awareness."""
    
    def __init__(self):
        self.compiled_patterns = HealthLiteracyPatterns.compile_patterns()
    
    def count_matches(self, text: str, pattern_key: str) -> int:
        """Count all matches for a pattern or list of patterns."""
        patterns = self.compiled_patterns.get(pattern_key)
        if not patterns:
            return 0
        
        text_lower = text.lower()
        
        if isinstance(patterns, re.Pattern):
            return len(patterns.findall(text_lower))
        elif isinstance(patterns, list):
            total = 0
            for pattern in patterns:
                total += len(pattern.findall(text_lower))
            return total
        
        return 0
    
    def find_all_matches(self, text: str, pattern_key: str) -> List[Tuple[str, int, int]]:
        """Find all matches with positions (useful for context analysis)."""
        patterns = self.compiled_patterns.get(pattern_key)
        if not patterns:
            return []
        
        text_lower = text.lower()
        matches = []
        
        if isinstance(patterns, re.Pattern):
            for match in patterns.finditer(text_lower):
                matches.append((match.group(), match.start(), match.end()))
        elif isinstance(patterns, list):
            for pattern in patterns:
                for match in pattern.finditer(text_lower):
                    matches.append((match.group(), match.start(), match.end()))
        
        return matches


# ══════════════════════════════════════════════════════════════════════════════
# POS-TAGGED FEATURE EXTRACTION (Context-Aware)
# ══════════════════════════════════════════════════════════════════════════════

class POSTaggedExtractor:
    """Extract features using POS tagging for context-aware analysis."""
    
    @staticmethod
    def extract_modal_verbs_pos(doc: spacy.tokens.Doc) -> Dict[str, any]:
        """
        Extract modal verbs using POS tags (more accurate than keyword matching).
        Captures: can, could, may, might, must, shall, should, will, would, etc.
        """
        modals = [token for token in doc if token.tag_ == 'MD']
        
        # Also check for phrasal modals
        phrasal_modals = []
        modal_phrases = ['ought to', 'have to', 'need to', 'has to', 'had to']
        text_lower = doc.text.lower()
        
        for phrase in modal_phrases:
            count = len(re.findall(r'\b' + phrase + r'\b', text_lower))
            phrasal_modals.extend([phrase] * count)
        
        total_modals = len(modals) + len(phrasal_modals)
        
        return {
            'modal_count': total_modals,
            'modal_density': total_modals / max(len(doc), 1),
            'modal_types': list(set([m.text.lower() for m in modals] + phrasal_modals)),
        }
    
    @staticmethod
    def extract_questions_pos(doc: spacy.tokens.Doc) -> Dict[str, any]:
        """
        Extract questions using sentence structure (not just '?' character).
        Handles: direct questions, wh-questions, yes/no questions.
        """
        sentences = list(doc.sents)
        n_sents = max(len(sentences), 1)
        
        questions = []
        question_types = []
        
        for sent in sentences:
            sent_text = sent.text.strip()
            
            # Direct questions (ending with ?)
            if sent_text.endswith('?'):
                questions.append(sent_text)
                
                # Classify question type
                first_word = sent[0].text.lower()
                if first_word in ['what', 'when', 'where', 'who', 'why', 'how', 'which']:
                    question_types.append('wh-question')
                elif first_word in ['is', 'are', 'do', 'does', 'did', 'can', 'could', 'should', 'would']:
                    question_types.append('yes-no-question')
                else:
                    question_types.append('other-question')
            
            # Implicit questions (starting with wh-words or modals, even without ?)
            elif sent[0].text.lower() in ['what', 'when', 'where', 'who', 'why', 'how']:
                questions.append(sent_text)
                question_types.append('wh-question-implicit')
        
        return {
            'question_count': len(questions),
            'question_ratio': len(questions) / n_sents,
            'question_types': question_types,
            'wh_question_count': sum(1 for qt in question_types if 'wh' in qt),
            'yesno_question_count': sum(1 for qt in question_types if 'yes-no' in qt),
        }
    
    @staticmethod
    def extract_verb_tense_pos(doc: spacy.tokens.Doc) -> Dict[str, any]:
        """
        Extract verb tenses using POS tags for EHL concreteness.
        Present tense indicates current/concrete situations.
        """
        verbs = [token for token in doc if token.pos_ == 'VERB']
        
        # Present tense verbs: VBP (non-3rd person), VBZ (3rd person)
        present_verbs = [v for v in verbs if v.tag_ in {'VBP', 'VBZ'}]
        
        # Past tense: VBD
        past_verbs = [v for v in verbs if v.tag_ == 'VBD']
        
        # Future markers (will, shall, going to)
        future_markers = [v for v in verbs if v.text.lower() in ['will', 'shall']]
        future_markers += re.findall(r'\bgoing to\b', doc.text.lower())
        
        n_verbs = max(len(verbs), 1)
        
        return {
            'present_verb_count': len(present_verbs),
            'present_verb_ratio': len(present_verbs) / max(len(doc), 1),
            'past_verb_count': len(past_verbs),
            'future_marker_count': len(future_markers),
            'present_dominance': len(present_verbs) / n_verbs,
        }
    
    @staticmethod
    def extract_pos_distributions(doc: spacy.tokens.Doc) -> Dict[str, any]:
        """
        Extract POS tag distributions for linguistic analysis.
        """
        n_tokens = max(len(doc), 1)
        
        # Determiners (the, a, an, this, that, etc.)
        determiners = [t for t in doc if t.tag_ == 'DT']
        
        # Adjectives (descriptive language)
        adjectives = [t for t in doc if t.tag_ in {'JJ', 'JJR', 'JJS'}]
        
        # Nouns (content words)
        nouns = [t for t in doc if t.pos_ == 'NOUN']
        
        # Adverbs (modification)
        adverbs = [t for t in doc if t.pos_ == 'ADV']
        
        return {
            'determiner_count': len(determiners),
            'determiner_ratio': len(determiners) / n_tokens,
            'adjective_count': len(adjectives),
            'adjective_ratio': len(adjectives) / n_tokens,
            'noun_count': len(nouns),
            'noun_ratio': len(nouns) / n_tokens,
            'adverb_count': len(adverbs),
            'adverb_ratio': len(adverbs) / n_tokens,
        }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FEATURE EXTRACTORS (Regex + POS Combined)
# ══════════════════════════════════════════════════════════════════════════════

# Initialize global matcher
MATCHER = PatternMatcher()
POS_EXTRACTOR = POSTaggedExtractor()


def extract_fhl_features(text: str) -> dict:
    """
    Functional Health Literacy - Regex + POS version.
    Uses regex patterns for hedging/certainty + POS for medical entities.
    """
    doc = nlp(text)
    
    # Medical entities using NER (more accurate than keyword lists)
    medical_entities = [
        ent for ent in doc.ents 
        if ent.label_ in {'DISEASE', 'CHEMICAL', 'ENTITY', 'GENE_OR_GENE_PRODUCT'}
    ]
    unique_medical = set([ent.text.lower() for ent in medical_entities])
    
    n_tokens = max(len(doc), 1)
    
    # Certainty and hedging using regex patterns
    certainty_count = MATCHER.count_matches(text, 'CERTAINTY_PATTERNS')
    hedging_count = MATCHER.count_matches(text, 'HEDGING_PATTERNS')
    
    # Calculate scores
    confidence_score = max(0, certainty_count - hedging_count)
    hedging_score = hedging_count / n_tokens if n_tokens > 0 else 0
    
    # Readability metrics (unchanged)
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    readability = flesch_reading_ease(text)
    
    # Sentence complexity
    sentences = list(doc.sents)
    avg_sent_len = np.mean([len(s) for s in sentences]) if sentences else 0
    
    # Clause counting (simplified - count by SBAR, subordinate clauses)
    clauses_per_sent = sum(1 for token in doc if token.dep_ in {'mark', 'advcl', 'acl'})
    clauses_per_sent = clauses_per_sent / max(len(sentences), 1)
    
    return {
        'readability_score': readability,
        'avg_sentence_length': avg_sent_len,
        'avg_clauses_per_sentence': clauses_per_sent,
        'medical_entity_count': len(medical_entities),
        'medical_entity_density': len(medical_entities) / n_tokens,
        'unique_medical_terms': len(unique_medical),
        'hedging_score': hedging_score,
        'confidence_score': confidence_score,
    }


def extract_chl_features(text: str) -> dict:
    """
    Communicative Health Literacy - POS + Regex version.
    """
    doc = nlp(text)
    
    # Questions using POS analysis
    question_features = POS_EXTRACTOR.extract_questions_pos(doc)
    
    # Modal verbs using POS tags
    modal_features = POS_EXTRACTOR.extract_modal_verbs_pos(doc)
    
    # Conditionals and context using regex
    cond_count = MATCHER.count_matches(text, 'CONDITIONAL_PATTERNS')
    ctx_count = MATCHER.count_matches(text, 'CONTEXT_PATTERNS')
    
    return {
        'question_count': question_features['question_count'],
        'question_ratio': question_features['question_ratio'],
        'wh_question_count': question_features['wh_question_count'],
        'yesno_question_count': question_features['yesno_question_count'],
        'conditional_expression_count': cond_count,
        'modal_verb_count': modal_features['modal_count'],
        'modal_verb_density': modal_features['modal_density'],
        'context_marker_count': ctx_count,
        'context_provided': int(ctx_count > 0),
    }


def extract_crhl_features(text: str) -> dict:
    """
    Critical Health Literacy - Regex patterns version.
    """
    # All features use regex patterns
    causal_count = MATCHER.count_matches(text, 'CAUSAL_PATTERNS')
    contrastive_count = MATCHER.count_matches(text, 'CONTRASTIVE_PATTERNS')
    evidence_count = MATCHER.count_matches(text, 'EVIDENCE_PATTERNS')
    options_count = MATCHER.count_matches(text, 'OPTIONS_PATTERNS')
    
    # Count URLs separately
    url_count = MATCHER.count_matches(text, 'URL_PATTERN')
    
    return {
        'causal_connective_count': causal_count,
        'contrastive_connective_count': contrastive_count,
        'evidence_reference_count': evidence_count + url_count,
        'multiple_options_count': options_count,
    }


def extract_dhl_features(text: str) -> dict:
    """
    Digital Health Literacy - Regex patterns version.
    """
    url_count = MATCHER.count_matches(text, 'URL_PATTERN')
    credible_count = MATCHER.count_matches(text, 'TRUSTED_SOURCES_PATTERNS')
    cross_ref_count = MATCHER.count_matches(text, 'CROSS_REF_PATTERNS')
    
    interp_score = url_count + credible_count + cross_ref_count
    
    return {
        'online_reference_count': url_count,
        'credible_source_count': credible_count,
        'cross_reference_count': cross_ref_count,
        'information_interpretation_score': max(interp_score, 0),
    }


def extract_ehl_features(text: str) -> dict:
    """
    Expressed Health Literacy - POS-based version.
    """
    doc = nlp_sm(text)
    
    # Verb tense analysis
    verb_features = POS_EXTRACTOR.extract_verb_tense_pos(doc)
    
    # POS distributions
    pos_features = POS_EXTRACTOR.extract_pos_distributions(doc)
    
    # Lexical diversity
    words = [t.text.lower() for t in doc if t.is_alpha]
    n_words = len(words)
    lexical_diversity = len(set(words)) / n_words if n_words > 0 else 0
    
    # Concreteness (simplified - based on POS)
    # More concrete texts have more nouns and present tense verbs
    concreteness = (pos_features['noun_ratio'] + verb_features['present_dominance']) / 2
    
    # Function words (common closed-class words)
    function_words = {'and', 'or', 'but', 'the', 'a', 'an', 'this', 'that', 
                     'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
    fw_count = sum(1 for w in words if w in function_words)
    
    return {
        'concreteness_score': concreteness,
        'lexical_diversity': lexical_diversity,
        'present_verb_count': verb_features['present_verb_count'],
        'present_verb_ratio': verb_features['present_verb_ratio'],
        'determiner_count': pos_features['determiner_count'],
        'determiner_ratio': pos_features['determiner_ratio'],
        'adjective_count': pos_features['adjective_count'],
        'adjective_ratio': pos_features['adjective_ratio'],
        'function_word_count': fw_count,
        'function_word_ratio': fw_count / n_words if n_words > 0 else 0,
    }


def extract_all_features(text: str) -> dict:
    """Run all 5 HL extractors and merge into one flat dict."""
    features = {}
    features.update(extract_fhl_features(text))
    features.update(extract_chl_features(text))
    features.update(extract_crhl_features(text))
    features.update(extract_dhl_features(text))
    features.update(extract_ehl_features(text))
    return features


# ══════════════════════════════════════════════════════════════════════════════
# TESTING & VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test examples
    test_texts = [
        # High certainty, medical expertise
        "The clinical trial definitely proves that metformin causes a 1.5% reduction in HbA1c levels. "
        "According to JAMA, this always leads to better outcomes.",
        
        # High hedging, uncertainty
        "I think maybe it could be related to my symptoms. I'm not sure if it might help, "
        "but perhaps I should ask my doctor?",
        
        # Critical thinking, multiple options
        "However, while statins reduce cholesterol, they may cause side effects. "
        "Alternatively, lifestyle changes could be considered. Studies show mixed results, "
        "so it's important to weigh the benefits and risks.",
        
        # Digital literacy
        "According to the CDC and research published in NEJM, evidence suggests that "
        "the vaccine is effective. See https://pubmed.ncbi.nlm.nih.gov/12345",
    ]
    
    print("=" * 80)
    print("REGEX + POS PATTERN MATCHING TEST")
    print("=" * 80)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'─' * 80}")
        print(f"TEST {i}: {text[:60]}...")
        print(f"{'─' * 80}")
        
        features = extract_all_features(text)
        
        # Print key features
        print(f"\n📊 FHL Features:")
        print(f"  • Confidence score: {features['confidence_score']}")
        print(f"  • Hedging score: {features['hedging_score']:.3f}")
        print(f"  • Medical entities: {features['medical_entity_count']}")
        
        print(f"\n💬 CHL Features:")
        print(f"  • Questions: {features['question_count']}")
        print(f"  • Modal verbs: {features['modal_verb_count']}")
        print(f"  • Conditionals: {features['conditional_expression_count']}")
        
        print(f"\n🔍 CRHL Features:")
        print(f"  • Causal connectives: {features['causal_connective_count']}")
        print(f"  • Contrastive: {features['contrastive_connective_count']}")
        print(f"  • Evidence references: {features['evidence_reference_count']}")
        
        print(f"\n🌐 DHL Features:")
        print(f"  • URLs: {features['online_reference_count']}")
        print(f"  • Credible sources: {features['credible_source_count']}")
        
        print(f"\n✍️ EHL Features:")
        print(f"  • Lexical diversity: {features['lexical_diversity']:.3f}")
        print(f"  • Present verb ratio: {features['present_verb_ratio']:.3f}")
