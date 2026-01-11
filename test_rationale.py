from src.module4_report import generate_smart_rationale

# Test fallback-style verification details (what you're currently getting)
fallback_details = [{
    "verdict": "CONSISTENT",
    "for": "Claim elements detected in context",
    "against": "None",
    "reasoning": "Fallback semantic matching with negation detection"
}]

# Test with label 1 (consistent)
rationale = generate_smart_rationale(1, fallback_details)
print("Single claim (consistent):", rationale)

# Test with multiple claims
multi_fallback = fallback_details * 3
rationale_multi = generate_smart_rationale(1, multi_fallback)
print("Multiple claims (consistent):", rationale_multi)

# Test contradiction
contradict_details = [{
    "verdict": "CONTRADICT",
    "for": "Claim elements detected in context",
    "against": "Potential negation or conflict detected",
    "reasoning": "Fallback semantic matching with negation detection"
}]
rationale_contra = generate_smart_rationale(0, contradict_details)
print("Contradiction:", rationale_contra)
