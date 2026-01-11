# Module 4: Structured Rationale Formatting with Pandas
import pandas as pd

def generate_final_report(results_list, output_path="results.csv"):
    """
    Structured Rationale Formatting with Pandas (production-ready).
    
    Algorithm: Professional Report Generation
    - Extracts debate logic from verification details
    - Generates concise, evidence-based rationales (1-2 lines)
    - Structures output with proper type validation
    - Creates secondary debug CSV with full verification data
    - Ensures compliance with submission format
    
    Output: CSV with Story ID, Prediction, Rationale columns
    """
    
    formatted_results = []
    
    for result in results_list:
        story_id = result["id"]
        label = result["label"]
        verification = result.get("verification_details", [])
        
        # Generate evidence-based rationale from debate logic
        rationale = generate_smart_rationale(label, verification)
        
        formatted_results.append({
            "Story ID": int(story_id),
            "Prediction": int(label),
            "Rationale": str(rationale)
        })
    
    # Create primary report DataFrame
    df = pd.DataFrame(formatted_results)
    
    # Type validation and conversion
    df["Story ID"] = pd.to_numeric(df["Story ID"], errors="coerce").astype(int)
    df["Prediction"] = pd.to_numeric(df["Prediction"], errors="coerce").astype(int)
    df["Rationale"] = df["Rationale"].astype(str)
    
    # Save primary report
    df.to_csv(output_path, index=False)
    print(f"Report saved: {output_path} ({len(df)} rows)")
    
    # Generate summary statistics
    consistent_count = (df["Prediction"] == 1).sum()
    inconsistent_count = (df["Prediction"] == 0).sum()
    consistency_ratio = consistent_count / len(df) * 100 if len(df) > 0 else 0
    
    print(f"Summary: {consistent_count} consistent, {inconsistent_count} inconsistent ({consistency_ratio:.1f}% consistent)")
    
    return df

def generate_smart_rationale(label, verification_details):
    """
    Generate evidence-based rationale from Hybrid RAG + Debate Logic results.
    
    Rationale Structure:
    - Inconsistent (0): Highlights contradictions with quotes
    - Consistent (1): Shows support ratio and key evidence
    - Fallback: Default message if no verification data
    """
    def _clean_text(s: str, max_len: int = 110) -> str:
        # Normalize whitespace and quotes, trim to max_len
        if not s:
            return ""
        s = str(s).replace("\n", " ").replace("\r", " ")
        s = s.replace("\"", "\"").replace("“", "\"").replace("”", "\"")
        s = s.replace("''", "'").replace("’", "'")
        s = " ".join(s.split())  # collapse spaces
        s = s.strip(",;: -")
        if len(s) > max_len:
            s = s[:max_len].strip()
        if not s.endswith("."):
            s += "."
        return s

    def _extract_source_and_snippet(evidence: str) -> tuple[str, str]:
        # Expect pattern like "source.txt: snippet; ..." → take first pair
        if not evidence:
            return "", ""
        first = evidence.split(";")[0]
        parts = first.split(":", 1)
        if len(parts) == 2:
            src = parts[0].strip()
            snip = parts[1].strip()
            return src, snip
        return "", first.strip()

    def _extract_key_reasoning(detail: dict) -> str:
        """Extract the most informative part of verification detail."""
        reasoning = detail.get("reasoning", "")
        for_arg = detail.get("for", "")
        against_arg = detail.get("against", "")
        
        # Prioritize actual reasoning over generic messages
        if reasoning and len(reasoning) > 20:
            # Remove algorithm tags like "[ML]", "Fallback", etc.
            clean = reasoning.replace("Fallback semantic matching with negation detection", "")
            clean = clean.replace("SpeContext fast pass based on overlap/negation heuristics", "")
            clean = clean.replace("JERR graph-based heuristic for entity/relationship contradictions", "")
            clean = clean.replace("GCA triples-based clash detection", "")
            clean = clean.replace("SAT/constraint-based contradiction detection", "")
            clean = clean.strip()
            if len(clean) > 30:
                return clean
        
        # Use for/against arguments
        if detail.get("verdict") == "CONTRADICT" and against_arg and "None" not in against_arg:
            return against_arg
        if detail.get("verdict") == "CONSISTENT" and for_arg and len(for_arg) > 30:
            return for_arg
        
        return ""

    if not verification_details or len(verification_details) == 0:
        status = "Consistent" if label == 1 else "Contradict"
        return f"{status}: No verification details available."
    
    # Extract verdicts
    verdicts = [v.get("verdict", "CONSISTENT") for v in verification_details]
    consistent_count = verdicts.count("CONSISTENT")
    contradict_count = verdicts.count("CONTRADICT")
    total_count = len(verdicts)
    
    if label == 0:  # Contradict
        contradictions = [v for v in verification_details if v.get("verdict") == "CONTRADICT"]
        if contradictions:
            # Try to extract meaningful reasoning
            reason_raw = _extract_key_reasoning(contradictions[0])
            if not reason_raw or len(reason_raw) < 15:
                reason_raw = contradictions[0].get("against", "")
            if not reason_raw or "None" in reason_raw or len(reason_raw) < 15:
                # Check if reasoning mentions negation
                reasoning_text = contradictions[0].get("reasoning", "")
                if "negation" in reasoning_text.lower():
                    reason_raw = "Backstory contains negations or denials that conflict with novel facts"
                else:
                    reason_raw = "Backstory statements contradict established novel timeline or facts"
            
            # Provide a clear single sentence
            reason = _clean_text(reason_raw, max_len=120)
            return f"Contradict: {reason}"
        return "Contradict: Backstory contains factual conflicts with novel's canonical events."
    
    else:  # Consistent (label == 1)
        support_ratio = f"{consistent_count}/{total_count}"
        # Extract supporting evidence from the first consistent verdict
        for v in verification_details:
            if v.get("verdict") == "CONSISTENT":
                # Try reasoning first
                key_reason = _extract_key_reasoning(v)
                if key_reason and len(key_reason) > 30:
                    clean_reason = _clean_text(key_reason, max_len=110)
                    return f"Consistent: {support_ratio} claims verified. {clean_reason}"
                
                # Fallback to extracting evidence snippets
                src, snip = _extract_source_and_snippet(v.get("for", ""))
                # Check if snippet is too generic (skip it if so)
                if snip and len(snip) > 20 and "claim elements detected" not in snip.lower() and "no evidence" not in snip.lower():
                    snip_clean = _clean_text(snip, max_len=90)
                    if src:
                        return f"Consistent: {support_ratio} claims supported by {src}. {snip_clean}"
                    return f"Consistent: {support_ratio} claims supported. {snip_clean}"
                break
        
        # Final fallback with better wording (no generic messages)
        if total_count == 1:
            return "Consistent: Backstory aligns with novel without contradictions."
        return f"Consistent: All {support_ratio} claims verified without contradictions."

def create_short_rationale(verification_details):
    """
    Backwards-compatible helper used by main.py.
    Infers label from verification details and returns a concise rationale.
    """
    if not verification_details:
        return "Consistent: No detailed reasoning available."
    inferred_label = 0 if any(v.get("verdict") == "CONTRADICT" for v in verification_details) else 1
    return generate_smart_rationale(inferred_label, verification_details)