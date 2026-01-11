import os
import pandas as pd
from src.module1_ingestion import create_ingestion_pipeline
from src.module2_extraction import extract_claims
from src.module3_reasoning import verify_claims_against_novel
from src.ml_model import load_text_classifier, predict_label
from src.module4_report import generate_final_report, create_short_rationale

# Try to import hybrid classifier
try:
    from src.hybrid_classifier import predict_hybrid
    HYBRID_AVAILABLE = True
except:
    HYBRID_AVAILABLE = False
    print("Warning: Hybrid classifier not available, using standard ML")

def main(csv_path=None):
    algo = (os.getenv("ALGO") or "ml").lower()  # ML classifier is default
    use_ml = (algo == "ml")
    use_hybrid_ml = (algo == "hybrid-ml" and HYBRID_AVAILABLE)

    # 1. Prepare context or ML model depending on algorithm
    if not use_ml and not use_hybrid_ml:
        print("Step 1: Indexing novels with sliding window chunking...")
        novels_path = "data/Books/"
        novel_index = create_ingestion_pipeline(novels_path)
        if not novel_index:
            print(f"Warning: No novel chunks found under {novels_path}.")
    elif use_hybrid_ml:
        print("Step 1: Loading Hybrid ML classifier (ML + Rules)...")
        # Hybrid classifier will load on first use
    else:
        print("Step 1: Loading ML text classifier...")
        model = load_text_classifier()

    # 2. LOAD INPUTS: Read the test cases provided in the hackathon
    test_csv = csv_path or "data/test.csv"
    if not os.path.exists(test_csv):
        print(f"Error: Missing test CSV at {test_csv}. Please add your dataset.")
        return
    test_df = pd.read_csv(test_csv)
    final_results = []

    # 3. SYSTEMS REASONING / ML LOOP
    for _, row in test_df.iterrows():
        story_id = row['id'] # Match your test.csv column name
        backstory = row['content'] # The content column contains the backstory

        if use_hybrid_ml:
            # Use hybrid classifier (ML + rules)
            pred, rationale = predict_hybrid(backstory)
            label = pred
            details = [{
                "verdict": "CONSISTENT" if pred == 1 else "CONTRADICT",
                "for": rationale if pred == 1 else "",
                "against": "" if pred == 1 else rationale,
                "reasoning": rationale
            }]
            short_rationale = rationale
        elif use_ml:
            pred, rationale = predict_label(backstory, model=model)
            label = pred
            details = [{
                "verdict": "CONSISTENT" if pred == 1 else "CONTRADICT",
                "for": "[ML] TF-IDF logistic regression",
                "against": "None" if pred == 1 else "[ML] Feature-based contradiction",
                "reasoning": rationale
            }]
            short_rationale = rationale
        else:
            # Extract facts (Gemini 2.0 Flash)
            claims = extract_claims(backstory)

            # Verify against novel (Hybrid RAG + Debate Logic)
            label, details = verify_claims_against_novel(claims, novel_index)

            # Create the concise rationale (1-2 lines)
            short_rationale = create_short_rationale(details)

        final_results.append({
            "id": story_id,
            "label": label,
            "verification_details": details,
            "rationale": short_rationale
        })

    # 4. OUTPUT: Generate results.csv
    generate_final_report(final_results, "output/results.csv")

if __name__ == "__main__":
    main()