# Visual Pattern Matching Reference
## See Exactly What Each Pattern Captures

---

## 🎯 Pattern Anatomy

### Basic Pattern Structure:
```
\b(cause|causes|causing|caused)\b
│  │                           │
│  │                           └─ Word boundary (end)
│  └─ Alternation group (OR logic)
└─ Word boundary (start)
```

### Common Regex Components:

| Symbol | Meaning | Example | Matches | Doesn't Match |
|--------|---------|---------|---------|--------------|
| `\b` | Word boundary | `\bcause\b` | "the cause is" | "because" |
| `?` | 0 or 1 occurrence | `causes?` | "cause", "causes" | "causing" |
| `+` | 1 or more | `lead(s|ing)+` | "leads", "leading" | "lead" |
| `\s` | Whitespace | `according\s+to` | "according to" | "accordingto" |
| `\|` | OR | `(may\|might)` | "may", "might" | "maybe" |
| `()` | Capture group | `(study\|research)` | "study", "research" | - |
| `[]` | Character class | `[0-9]+` | "123", "5" | "abc" |

---

## 📋 Pattern Matching Examples

### Example 1: Certainty Markers

**Pattern:** `r'\b(definitely|certainly|absolutely)\b'`

#### ✅ Matches (with highlighting):
```
The treatment [definitely] works.
It is [certainly] effective.
This [absolutely] causes improvement.
```

#### ❌ Does NOT Match:
```
indefinitely     ← "definitely" is inside, but no word boundary
uncertain        ← "certain" is inside, but no word boundary  
absolution       ← "absolute" is inside, but no word boundary
```

#### 🔄 Comparison with Keyword:
```python
# KEYWORD (using substring search):
text = "This is indefinitely delayed"
'definitely' in text.lower()  # ✅ TRUE (WRONG!)

# REGEX (with word boundaries):
re.search(r'\bdefinitely\b', text.lower())  # ❌ None (CORRECT!)
```

---

### Example 2: Causal Connectives with Inflections

**Pattern:** `r'\b(causes?|lead(s|ing)?\s+to|results?\s+in)\b'`

#### ✅ Matches:
```
Smoking [cause] cancer.
Smoking [causes] cancer.
This can [lead to] problems.
This can [leads to] problems.
This is [leading to] issues.
The mutation [result in] disease.
The mutations [results in] disease.
```

#### ❌ Does NOT Match:
```
because          ← word boundary prevents partial match
causeway         ← different word entirely
leader           ← "lead" is inside but pattern requires "to"
```

#### 🔄 Pattern Breakdown:
```
causes?           → "cause" or "causes"
lead(s|ing)?      → "lead", "leads", or "leading"  
\s+               → one or more spaces
to                → literal "to"

Full pattern captures:
- "lead to", "leads to", "leading to"
```

---

### Example 3: Evidence Markers (Multi-word with Variations)

**Pattern:** `r'\b(study|studies|research|trial)\s+(shows?|indicates?|suggests?|found|demonstrates?)\b'`

#### ✅ Matches:
```
The [study shows] effectiveness.
Multiple [studies show] results.
Our [research indicates] benefits.
The [trial found] improvement.
Data [demonstrates] efficacy.
```

#### ❌ Does NOT Match:
```
study          ← needs verb after
show study     ← wrong order
studying       ← different form (would need separate pattern)
```

#### 🔄 All Captured Combinations:
```
Nouns (4):           Verbs (7):
- study              - show/shows
- studies            - indicate/indicates  
- research           - suggest/suggests
- trial              - found
                     - demonstrate/demonstrates

Total combinations matched: 4 × 7 = 28 variations
With keyword approach: would need 28 separate entries!
```

---

### Example 4: Modal Verbs (POS Tag vs Regex)

**Why POS tagging is better for modals:**

#### Text: "I can help you. Put it in the can."

**Method 1: Keyword/Regex (naive):**
```python
pattern = r'\bcan\b'
matches = re.findall(pattern, text.lower())
# Result: ['can', 'can']  ← BOTH matched! (one is wrong)
```

**Method 2: POS Tagging (correct):**
```python
doc = nlp(text)
for token in doc:
    print(f"{token.text}: {token.tag_}")

# Output:
# I: PRP
# can: MD      ← Modal verb (CORRECT!)
# help: VB
# you: PRP
# .
# Put: VB
# it: PRP
# in: IN
# the: DT
# can: NN      ← Noun (NOT a modal!)
# .: .

modals = [t for t in doc if t.tag_ == 'MD']
# Result: ['can']  ← Only the modal! (CORRECT)
```

---

### Example 5: Conditional Expressions

**Pattern:** `r'\b(if|when|unless|provided\s+that|as\s+long\s+as)\b'`

#### ✅ Matches:
```
[If] you take this medication...
Take it [when] you feel pain.
Don't stop [unless] directed.
Safe [provided that] you follow instructions.
Effective [as long as] you maintain dosage.
```

#### 🔄 Multi-word Pattern Details:
```
provided\s+that
│        ││    │
│        │└────┴─ literal "that"
│        └─ one or more whitespace chars
└─ literal "provided"

This matches:
- "provided that"    ← normal spacing
- "provided  that"   ← extra spaces (user typo)
- "provided\tthat"   ← tab character

But not:
- "providedthat"     ← no space (malformed)
```

---

### Example 6: Hedging Markers (Complex Patterns)

**Pattern:** `r'\b(i\s+(think|believe|guess)|i\'?m\s+not\s+sure)\b'`

#### ✅ Matches:
```
[I think] it might work.
[I believe] this is correct.
[I guess] that's possible.
[I'm not sure] about this.
[Im not sure] if it helps.  ← also matches without apostrophe
```

#### 🔄 Pattern Breakdown:
```
i\s+(think|believe|guess)
│ ││  └────────┬────────┘
│ ││           └─ verb options (OR)
│ │└─ one or more spaces
│ └─ literal "I"

i\'?m\s+not\s+sure
│ │││ ││  ││
│ │││ ││  │└─ literal "sure"
│ │││ ││  └─ spaces
│ │││ │└─ literal "not"
│ │││ └─ spaces
│ ││└─ optional apostrophe
│ │└─ literal "m"
│ └─ literal "I"
```

---

### Example 7: Trusted Sources with Variations

**Pattern:** `r'\b(who|world\s+health\s+organization)\b'`

#### ✅ Matches:
```
According to the [WHO]...
The [World Health Organization] recommends...
[WHO] guidelines state...
```

#### ❌ Does NOT Match (intentionally!):
```
Who should I see?        ← "Who" (question word, capitalized differently)
whoever                  ← "who" inside another word
```

#### 🔄 Why This Works:
```
Word boundaries \b ensure context:
- \bwho\b matches "WHO" (all caps - medical org)
- Doesn't match "who" in questions (different context)
- Case-insensitive flag handles "Who", "WHO", "who"

The pattern combines:
\b(who|world\s+health\s+organization)\b
   ├─── short form (WHO)
   └─── long form (World Health Organization)
```

---

### Example 8: Questions (POS-based Detection)

**Method: POS tagging + sentence analysis**

#### Text: "What is diabetes? I have been diagnosed."

**POS Analysis:**
```python
doc = nlp(text)
for sent in doc.sents:
    print(f"Sentence: {sent.text}")
    print(f"First token: {sent[0].text} | POS: {sent[0].pos_}")
    print(f"Ends with ?: {sent.text.strip().endswith('?')}")

# Output:
# Sentence: What is diabetes?
# First token: What | POS: PRON (wh-word!)
# Ends with ?: True
# → QUESTION DETECTED ✅

# Sentence: I have been diagnosed.
# First token: I | POS: PRON
# Ends with ?: False
# → NOT A QUESTION ✅
```

**Question Classification:**
```python
def classify_question(sent):
    first_word = sent[0].text.lower()
    
    if first_word in ['what', 'when', 'where', 'who', 'why', 'how', 'which']:
        return 'wh-question'
    elif first_word in ['is', 'are', 'do', 'does', 'did', 'can', 'could', 'should', 'would']:
        return 'yes-no-question'
    else:
        return 'other-question'

# "What is diabetes?" → wh-question
# "Should I see a doctor?" → yes-no-question
```

---

## 🎨 Visual Pattern Testing Tool

### Test Your Own Patterns:

```python
import re

def visualize_pattern(pattern, text):
    """Show what a pattern matches with highlighting."""
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    
    if not matches:
        print(f"❌ No matches found in: {text}")
        return
    
    print(f"✅ Found {len(matches)} match(es):")
    
    # Build highlighted string
    result = []
    last_end = 0
    
    for match in matches:
        # Add text before match
        result.append(text[last_end:match.start()])
        # Add highlighted match
        result.append(f"[{text[match.start():match.end()]}]")
        last_end = match.end()
    
    # Add remaining text
    result.append(text[last_end:])
    
    print("".join(result))
    print()

# Example usage:
pattern = r'\b(study|studies)\s+(shows?|found|indicates?)\b'

test_texts = [
    "The study shows positive results.",
    "Multiple studies found improvements.",
    "Research indicates effectiveness.",  # Won't match - different noun
    "The studying phase is complete.",   # Won't match - different verb form
]

for text in test_texts:
    visualize_pattern(pattern, text)

# Output:
# ✅ Found 1 match(es):
# The [study shows] positive results.
#
# ✅ Found 1 match(es):
# Multiple [studies found] improvements.
#
# ❌ No matches found in: Research indicates effectiveness.
#
# ❌ No matches found in: The studying phase is complete.
```

---

## 📊 Coverage Analysis

### Compare Pattern vs Keyword Coverage:

```python
def compare_coverage(pattern_str, keywords, test_texts):
    """Compare what pattern captures vs keyword list."""
    pattern = re.compile(pattern_str, re.IGNORECASE)
    
    print("=" * 60)
    print(f"Pattern: {pattern_str}")
    print(f"Keywords: {keywords}")
    print("=" * 60)
    
    for text in test_texts:
        # Pattern matching
        pattern_matches = len(pattern.findall(text))
        
        # Keyword matching
        keyword_matches = sum(text.lower().count(kw) for kw in keywords)
        
        # Compare
        status = "✅" if pattern_matches > 0 else "❌"
        kw_status = "✅" if keyword_matches > 0 else "❌"
        
        print(f"\nText: {text[:50]}...")
        print(f"  Pattern: {status} ({pattern_matches} matches)")
        print(f"  Keywords: {kw_status} ({keyword_matches} matches)")
        
        if pattern_matches != keyword_matches:
            print(f"  ⚠️  DIFFERENCE: Pattern found {pattern_matches}, Keywords found {keyword_matches}")

# Example:
pattern = r'\bcauses?\b'
keywords = ['cause', 'causes']

test_texts = [
    "Smoking causes cancer.",           # Both match
    "The root cause is unclear.",       # Both match
    "This is because of genetics.",     # Keywords: false positive!
    "What caused this problem?",        # Pattern: catches "caused"!
]

compare_coverage(pattern, keywords, test_texts)
```

---

## 🔧 Pattern Development Workflow

### Step 1: Identify Linguistic Phenomenon
```
Goal: Capture all forms of "evidence citation"

Base forms to consider:
- "study shows"
- "research indicates"  
- "data suggests"
```

### Step 2: Enumerate Variations
```
Nouns:        study, studies, research, data, trial(s), evidence
Verbs:        show(s), indicate(s), suggest(s), demonstrate(s), found
Inflections:  show/shows, indicate/indicates, etc.
```

### Step 3: Build Pattern
```python
# Start simple:
r'\bstudy shows\b'

# Add inflections:
r'\bstud(y|ies) shows?\b'

# Add verb variations:
r'\bstud(y|ies) (shows?|indicates?|suggests?)\b'

# Add noun variations:
r'\b(study|studies|research|data) (shows?|indicates?|suggests?|found)\b'

# Add spacing flexibility:
r'\b(study|studies|research|data)\s+(shows?|indicates?|suggests?|found)\b'
```

### Step 4: Test on Real Data
```python
pattern = re.compile(
    r'\b(study|studies|research|data)\s+(shows?|indicates?|suggests?|found)\b',
    re.IGNORECASE
)

test_corpus = [
    "The study shows...",
    "Research indicates...",
    "Multiple studies found...",
    "Data suggests...",
]

for text in test_corpus:
    match = pattern.search(text)
    print(f"{'✅' if match else '❌'} {text}")
```

### Step 5: Refine Based on False Positives/Negatives
```python
# Found issue: missing "evidence" as noun
# Add to pattern:
r'\b(study|studies|research|data|evidence)\s+(shows?|indicates?|suggests?|found)\b'

# Found issue: missing "demonstrate"
# Add to pattern:
r'\b(study|studies|research|data|evidence)\s+(shows?|indicates?|suggests?|found|demonstrates?)\b'
```

---

## 🎓 Advanced Pattern Techniques

### Lookahead/Lookbehind (Context Checking)
```python
# Match "study" only when followed by a verb
pattern = r'\bstudy(?=\s+(shows?|found|indicates?))\b'

# Matches:
"study shows"     ✅
"study found"     ✅
"study design"    ❌

# Match numbers only when preceded by "about" or "approximately"
pattern = r'(?<=about\s)\d+|(?<=approximately\s)\d+'

# Matches:
"about 50"              ✅
"approximately 100"     ✅
"exactly 50"            ❌
```

### Named Groups (For Extraction)
```python
pattern = r'\b(?P<noun>study|studies)\s+(?P<verb>shows?|found)\b'

match = re.search(pattern, "The study shows results")
if match:
    print(match.group('noun'))   # "study"
    print(match.group('verb'))   # "shows"
```

### Non-capturing Groups (Performance)
```python
# Capturing (slower, uses memory):
r'\b(study|studies)\s+(shows?|found)\b'

# Non-capturing (faster):
r'\b(?:study|studies)\s+(?:shows?|found)\b'

# Use non-capturing (?:...) when you don't need to extract the match
```

---

## 📚 Pattern Library Quick Reference

### High-Frequency Patterns:

```python
# Certainty
CERTAINTY = r'\b(definitely|certainly|absolutely|clearly|obviously)\b'

# Hedging  
HEDGING = r'\b(may|might|could|possibly|perhaps|seems?)\b'

# Modals (use POS instead!)
# MODALS = r'\b(can|could|may|might|must|should|will|would)\b'

# Questions (use sentence analysis instead!)
# QUESTIONS = r'\?$'

# Evidence
EVIDENCE = r'\b(study|studies|research)\s+(shows?|found|indicates?)\b'

# Causation
CAUSATION = r'\b(causes?|lead(s|ing)?\s+to|results?\s+in)\b'

# URLs
URLS = r'https?://[^\s]+'

# Organizations
ORGS = r'\b(who|cdc|nih|mayo clinic)\b'
```

---

## ✨ Summary

**Pattern matching advantages:**
1. **Flexible:** Captures morphological variations (cause/causes/causing)
2. **Precise:** Word boundaries prevent partial matches
3. **Comprehensive:** Single pattern = multiple keyword entries
4. **Maintainable:** Easy to extend and audit
5. **Linguistic:** Can combine with POS for context

**When to use POS tagging instead:**
- Ambiguous words ("can" - noun vs modal)
- Syntactic structure (questions, clauses)
- Grammatical categories (all verbs, all nouns)

**Best practice:** Combine both!
- Regex patterns for lexical features
- POS tagging for grammatical features
