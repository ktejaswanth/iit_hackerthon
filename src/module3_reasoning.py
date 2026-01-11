# Module 3: Reasoning Algorithms: Hybrid, SpeContext, JERR Graph, GCA Triples, SAT Encoding
import json
import os
from dotenv import load_dotenv
from typing import List, Dict, Tuple

from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def verify_claims_against_novel(claims: List[str], retrieved_contexts: List[Dict], algorithm: str | None = None) -> Tuple[int, List[Dict]]:
    """
    Router for verification algorithms.

    algorithm options:
    - "hybrid" (default): Debate-style Hybrid RAG
    - "specontext": Fast processing across large token windows
    - "jerr": Graph-based entity/relationship reasoning
    - "gca": Triple (SVO) extraction & clash detection
    - "sat": Constraint/SAT-style impossibility proof

    Uses env var ALGO when algorithm is None.
    """
    algo = (algorithm or os.getenv("ALGO") or "hybrid").lower()

    if algo == "specontext":
        return verify_specontext(claims, retrieved_contexts)
    if algo == "jerr":
        return verify_jerr_graph(claims, retrieved_contexts)
    if algo == "gca":
        return verify_gca_triples(claims, retrieved_contexts)
    if algo == "sat":
        return verify_sat_encoding(claims, retrieved_contexts)
    # default
    return verify_hybrid_debate(claims, retrieved_contexts)

def verify_hybrid_debate(claims: List[str], retrieved_contexts: List[Dict]) -> Tuple[int, List[Dict]]:
    """
    Hybrid RAG + Debate Logic with Gemini 1.5 Pro
    Returns: (overall_label, verification_details)
    """
    overall_label = 1
    verification_details: List[Dict] = []

    # RAG: limit to top relevant chunks (if any ordering exists)
    context_chunks = retrieved_contexts[:5] if retrieved_contexts else []
    full_context = "\n\n".join([chunk.get('text', '') for chunk in context_chunks])

    if not full_context.strip():
        for claim in claims:
            verification_details.append({
                "verdict": "CONSISTENT",
                "for": "No novel context available for verification",
                "against": "None",
                "reasoning": "Assuming consistency due to missing context"
            })
        return overall_label, verification_details

    for claim in claims:
        prompt = f"""You are a literary consistency validator. Analyze if this claim is CONSISTENT or CONTRADICTS the novel.

CLAIM TO VERIFY:
"{claim}"

NOVEL CONTEXT (excerpts):
{full_context[:3000]}

TASK: Debate-style reasoning
1. List ARGUMENTS supporting consistency (claim aligns with or is supported by context)
2. List ARGUMENTS against consistency (claim contradicts or conflicts with context)
3. Final VERDICT: CONSISTENT or CONTRADICT

DECISION RULE:
- Mark CONTRADICT only if there is CLEAR LOGICAL IMPOSSIBILITY or DIRECT CONTRADICTION
- Absence of evidence is NOT contradiction (mark CONSISTENT if no conflict found)
- Be strict but fair: require explicit contradiction

Return VALID JSON:
{{
  "verdict": "CONSISTENT" or "CONTRADICT",
  "for": "[arguments supporting consistency]",
  "against": "[arguments against]",
  "reasoning": "[brief explanation of verdict]"
}}"""

        try:
            response = client.models.generate_content(
                model="gemini-1.5-pro",
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            result = json.loads(response.text)

            verdict = result.get("verdict", "CONSISTENT").strip().upper()
            if verdict not in ["CONSISTENT", "CONTRADICT"]:
                verdict = "CONSISTENT"
            if verdict == "CONTRADICT":
                overall_label = 0

            verification_details.append({
                "verdict": verdict,
                "for": result.get("for", "[No arguments provided]"),
                "against": result.get("against", "[No arguments provided]"),
                "reasoning": result.get("reasoning", "[No reasoning provided]")
            })
        except Exception as e:
            print(f"Verification error: {e}")
            verdict = semantic_verify_fallback(claim, full_context)
            if verdict == "CONTRADICT":
                overall_label = 0
            verification_details.append({
                "verdict": verdict,
                "for": "Claim elements detected in context",
                "against": "Potential negation or conflict detected" if verdict == "CONTRADICT" else "None",
                "reasoning": "Fallback semantic matching with negation detection"
            })

    return overall_label, verification_details

def semantic_verify_fallback(claim, context):
    """
    Fallback semantic verification using lexical analysis.
    Detects explicit negations, factual conflicts, and contradictory patterns.
    """
    import re
    claim_lower = claim.lower()
    context_lower = context.lower()
    
    # Extract key terms from claim
    key_terms = re.findall(r'\b\w{4,}\b', claim_lower)  # Words 4+ chars
    
    # 1. Check for explicit negation patterns in claim (indicating false statement)
    negations = ["not ", "never ", "refused ", "unable ", "no longer", "wasn't", "didn't"]
    for neg in negations:
        if neg in claim_lower:
            # Extract action after negation
            action_after_neg = claim_lower.split(neg)[-1].strip()[:50]
            # If positive action appears in context, it's a contradiction
            if action_after_neg and action_after_neg in context_lower:
                return "CONTRADICT"
    
    # 2. Detect contradictory attributes (born in X vs Y, married vs single)
    birth_places = ["france", "london", "paris", "scotland", "ireland", "germany", "italy"]
    jobs = ["actress", "maid", "sailor", "soldier", "merchant", "physician"]
    
    for place in birth_places:
        if f"born in {place}" in claim_lower or f"born {place}" in claim_lower:
            # Check if different birthplace in context
            for other_place in birth_places:
                if other_place != place and f"born in {other_place}" in context_lower:
                    return "CONTRADICT"
    
    for job in jobs:
        if job in claim_lower:
            # Check for explicit opposite jobs in context
            if job == "married" and ("never married" in context_lower or "unmarried" in context_lower):
                return "CONTRADICT"
            elif job == "actress" and ("physician" in context_lower or "doctor" in context_lower):
                return "CONTRADICT"
            elif job == "maid" and ("educated" in context_lower or "learned" in context_lower):
                return "CONTRADICT"
    
    # 3. Check for family contradictions
    if "married" in claim_lower and ("never married" in context_lower or "single" in context_lower):
        return "CONTRADICT"
    if "children" in claim_lower and ("no children" in context_lower or "never had" in context_lower):
        return "CONTRADICT"
    
    # 4. Check if key terms appear in context (positive indicators of consistency)
    terms_found = sum(1 for term in key_terms if term in context_lower)
    coverage = terms_found / len(key_terms) if key_terms else 0
    
    # If most key terms missing, could be inconsistent
    if coverage < 0.3 and len(key_terms) > 3:
        # Check if context explicitly contradicts
        if "not " in claim_lower or "never " in claim_lower or "refused" in claim_lower:
            return "CONTRADICT"
    
    # Default to consistent if no clear contradiction found
    return "CONSISTENT"

# --- SpeContext: Fast processing across large token windows ---
def verify_specontext(claims: List[str], contexts: List[Dict]) -> Tuple[int, List[Dict]]:
    """SpeContext: Stream over all chunks; score by term-overlap and negation, no external index.
    Designed to handle very large corpora by single-pass scanning and keeping only top-k evidence per claim.
    """
    import re
    k = 5  # keep top-5 evidence snippets per claim
    overall_label = 1
    details: List[Dict] = []

    # Pre-normalize chunks to lower-case to reduce repeated work
    norm_chunks = [(c.get('index'), c.get('source'), c.get('text', ''), c.get('text', '').lower()) for c in (contexts or [])]

    for claim in claims:
        claim_l = claim.lower()
        key_terms = re.findall(r"\b\w{4,}\b", claim_l)
        scored = []
        for idx, src, text_raw, text_l in norm_chunks:
            if not text_l:
                continue
            # Fast term overlap
            overlap = sum(1 for t in key_terms if t in text_l)
            # Negation hint if claim asserts negation but context asserts positive
            neg_penalty = 0
            if any(neg in claim_l for neg in [" not ", " never ", " no ", " wasn't", " didn't"]):
                # reward contexts that mention the positive token (potential contradiction)
                neg_penalty = 1 if overlap > 0 else 0
            score = overlap + neg_penalty
            if score > 0:
                snippet = text_raw[:200].strip()
                scored.append((score, src, snippet))
        # Keep top-k evidence
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]

        # Decide verdict heuristically
        verdict = "CONSISTENT"
        if any(" never " in claim_l or " not " in claim_l for _ in [0]) and len(top) > 0:
            # If negated claim but strong overlap exists, flag possible contradiction
            verdict = "CONTRADICT"
            overall_label = 0

        details.append({
            "verdict": verdict,
            "for": "; ".join([f"{src}: {snip}" for _, src, snip in top]) or "[No evidence]",
            "against": "[Heuristic] Negation vs context overlap" if verdict == "CONTRADICT" else "None",
            "reasoning": "SpeContext fast pass based on overlap/negation heuristics"
        })

    return overall_label, details

# --- JERR (Graph): Entity/Relation Graph reasoning ---
def verify_jerr_graph(claims: List[str], contexts: List[Dict]) -> Tuple[int, List[Dict]]:
    """JERR: Build a simple entity-relation graph from claims, search contexts for conflicting edges.
    Heuristic NER and relation extraction to avoid heavy dependencies.
    """
    import re
    overall_label = 1
    details: List[Dict] = []

    def extract_entities(text: str) -> List[str]:
        # naive entities: capitalized words or multi-word names
        tokens = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
        return list(set(tokens))

    def extract_relation(text: str) -> str:
        # coarse relation: first verb-like word
        m = re.search(r"\b(is|was|were|became|married|born|lives|worked|met|killed|rescued)\b", text.lower())
        return m.group(1) if m else "related"

    # Build graph from claims
    claim_edges = []  # (subj, rel, obj)
    for c in claims:
        ents = extract_entities(c)
        rel = extract_relation(c)
        # object approximation: last capitalized phrase or key noun
        obj = ents[1] if len(ents) > 1 else (re.findall(r"\b\w{4,}\b", c.lower())[-1] if re.findall(r"\b\w{4,}\b", c.lower()) else "unknown")
        subj = ents[0] if ents else "unknown"
        claim_edges.append((subj, rel, obj))

    # Scan contexts for contradicting edges (same subj+rel, different obj with explicit contradiction cues)
    ctx_text = "\n".join([c.get('text', '') for c in (contexts or [])]).lower()
    for subj, rel, obj in claim_edges:
        contradict = False
        # look for patterns like "subj was not obj" or "never obj" etc.
        if subj != "unknown":
            if f"{subj.lower()} was not {obj.lower()}" in ctx_text or f"{subj.lower()} never {rel}" in ctx_text:
                contradict = True
        # or strong evidence mentioning subj with a different object token
        if not contradict and subj != "unknown":
            alt_objs = [w for w in ["soldier", "merchant", "physician", "maid", "sailor", "actor"] if w != obj.lower()]
            if any(f"{subj.lower()} {rel} {w}" in ctx_text for w in alt_objs):
                contradict = True

        verdict = "CONTRADICT" if contradict else "CONSISTENT"
        if verdict == "CONTRADICT":
            overall_label = 0
        details.append({
            "verdict": verdict,
            "for": f"Graph edge: ({subj}, {rel}, {obj})",
            "against": "Conflicting edge detected in context" if verdict == "CONTRADICT" else "None",
            "reasoning": "JERR graph-based heuristic for entity/relationship contradictions"
        })

    return overall_label, details

# --- GCA (Triples): Subject-Verb-Object triple extraction ---
def verify_gca_triples(claims: List[str], contexts: List[Dict]) -> Tuple[int, List[Dict]]:
    """GCA: Extract SVO triples from claims, scan contexts for subtle clashes (same S,V, different O or negation)."""
    import re
    overall_label = 1
    details: List[Dict] = []
    ctx = "\n".join([c.get('text', '') for c in (contexts or [])]).lower()

    def extract_svo(s: str) -> Tuple[str, str, str]:
        s_l = s.lower()
        # naive: first noun-ish token, first verb, next noun-ish token
        nouns = re.findall(r"\b[a-z]{3,}\b", s_l)
        verbs = re.findall(r"\b(is|was|were|became|met|killed|rescued|found|lost|visited|married|born|lives|works)\b", s_l)
        subj = nouns[0] if nouns else "unknown"
        verb = verbs[0] if verbs else "related"
        obj = nouns[1] if len(nouns) > 1 else "unknown"
        return subj, verb, obj

    for c in claims:
        subj, verb, obj = extract_svo(c)
        contradict = False
        # Clash 1: explicit negation of predicate
        if f"{subj} {verb} {obj}" in ctx and any(neg in ctx for neg in ["not ", "never "]):
            contradict = True
        # Clash 2: same subj+verb but different object mentioned frequently
        if not contradict and subj != "unknown":
            alt_hits = re.findall(fr"{subj}\s+{verb}\s+([a-z]{3,})", ctx)
            alt_objs = [o for o in alt_hits if o != obj]
            if len(set(alt_objs)) >= 1:
                contradict = True

        verdict = "CONTRADICT" if contradict else "CONSISTENT"
        if verdict == "CONTRADICT":
            overall_label = 0
        details.append({
            "verdict": verdict,
            "for": f"Triple: ({subj}, {verb}, {obj})",
            "against": "Different object or explicit negation in context" if verdict == "CONTRADICT" else "None",
            "reasoning": "GCA triples-based clash detection"
        })

    return overall_label, details

# --- SAT Encoding: Constraint satisfaction / impossibility proofs ---
def verify_sat_encoding(claims: List[str], contexts: List[Dict]) -> Tuple[int, List[Dict]]:
    """SAT/SMT-style: encode mutually exclusive attributes; if claim implies A and context implies not-A, mark contradiction.
    Attempts to use z3-solver when available; falls back to rule-based checks when not.
    """
    import re
    overall_label = 1
    details: List[Dict] = []
    ctx = "\n".join([c.get('text', '') for c in (contexts or [])]).lower()

    exclusives = {
        "married": ["never married", "unmarried", "single"],
        "born in paris": ["born in london", "born in scotland", "born in ireland", "born in germany", "born in italy"],
        "alive": ["died", "dead", "killed"],
    }

    # try z3
    z3_available = False
    try:
        from z3 import Bool, Solver, And, Not, sat
        z3_available = True
    except Exception:
        z3_available = False

    for c in claims:
        c_l = c.lower()
        contradiction = False
        reason = ""

        if z3_available:
            s = Solver()
            # map attributes to booleans
            attrs = {key: Bool(key.replace(" ", "_")) for key in exclusives.keys()}
            # assert claim attributes
            for key in exclusives:
                if key in c_l:
                    s.add(attrs[key])
            # assert context negatives
            for key, negs in exclusives.items():
                for neg in negs:
                    if neg in ctx:
                        s.add(Not(attrs.get(key, Bool(key.replace(" ", "_")))))

            if s.check() != sat:
                contradiction = True
                reason = "SAT solver found unsatisfiable constraints between claim and context."
        else:
            # rule-based: if claim asserts key and context asserts any negation, contradiction
            for key, negs in exclusives.items():
                if key in c_l and any(neg in ctx for neg in negs):
                    contradiction = True
                    reason = f"Claim asserts '{key}' but context states a mutually exclusive condition."

        verdict = "CONTRADICT" if contradiction else "CONSISTENT"
        if verdict == "CONTRADICT":
            overall_label = 0
        details.append({
            "verdict": verdict,
            "for": "Claim checked against exclusivity constraints",
            "against": reason if verdict == "CONTRADICT" else "None",
            "reasoning": "SAT/constraint-based contradiction detection"
        })

    return overall_label, details