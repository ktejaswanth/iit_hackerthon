"""
Hybrid classifier that combines ML with rule-based contradiction detection.
"""
import re
from typing import Tuple
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from ml_improved import load_improved_classifier, MODEL_PATH as IMPROVED_MODEL_PATH
    USE_IMPROVED = os.path.exists(IMPROVED_MODEL_PATH)
except:
    USE_IMPROVED = False

if not USE_IMPROVED:
    from ml_model import load_text_classifier, MODEL_PATH as MODEL_PATH


class HybridClassifier:
    """
    Combines ML predictions with rule-based contradiction detection.
    """
    
    def __init__(self):
        # Load ML model
        if USE_IMPROVED:
            print("Loading improved ML model...")
            self.model = load_improved_classifier()
        else:
            print("Loading standard ML model...")
            self.model = load_text_classifier()
        
        # Strong contradiction indicators
        self.strong_contradiction_patterns = [
            r'\b(not|never|no)\s+(was|is|were|are|been)\b',
            r'\bforged\b.*\b(logbook|document|record)\b',
            r'\b(botched|failed)\b',
            r'\bconflict(s|ing|ed)?\b.*\b(with|against)\b',
            r'\b(contradict|opposite|reverse|contrary)\b',
            r'\bdisagree(s|d)?\b',
            r'\brefute(s|d)?\b',
            r'\bdeny|denied|denies\b',
            r'\b(incorrect|inaccurate|wrong|false)\b',
            r'\b(impossible|absurd)\b',
            r'\b(never\s+\w+|not\s+\w+)\b.*\b(actually|really|truly)\b'
        ]
        
        # Consistency indicators
        self.strong_consistency_patterns = [
            r'\bconfirm(s|ed|ing)?\b',
            r'\bverif(y|ied|ies)\b',
            r'\bsupport(s|ed|ing)?\b',
            r'\bagree(s|d)?\b.*\bwith\b',
            r'\bconsistent\s+with\b',
            r'\baccurate(ly)?\b',
            r'\bcorrect(ly)?\b',
            r'\btrue\s+(to|that)\b'
        ]
    
    def count_contradiction_signals(self, text: str) -> int:
        """Count strong contradiction signals in text."""
        text_lower = text.lower()
        count = 0
        for pattern in self.strong_contradiction_patterns:
            matches = re.findall(pattern, text_lower)
            count += len(matches)
        return count
    
    def count_consistency_signals(self, text: str) -> int:
        """Count strong consistency signals in text."""
        text_lower = text.lower()
        count = 0
        for pattern in self.strong_consistency_patterns:
            matches = re.findall(pattern, text_lower)
            count += len(matches)
        return count
    
    def predict(self, text: str) -> Tuple[int, str, float]:
        """
        Predict with hybrid approach.
        Returns: (label, rationale, confidence)
          label: 1=consistent, 0=contradict
        """
        # Get ML prediction
        ml_pred = int(self.model.predict([text])[0])
        
        # Try to get probability if available
        confidence = 0.5
        if hasattr(self.model, 'predict_proba'):
            try:
                proba = self.model.predict_proba([text])[0]
                confidence = max(proba)
            except:
                pass
        elif hasattr(self.model, 'decision_function'):
            try:
                decision = self.model.decision_function([text])[0]
                # Convert decision function to approximate probability
                from scipy.special import expit
                confidence = expit(abs(decision))
            except:
                pass
        
        # Count contradiction and consistency signals
        contra_signals = self.count_contradiction_signals(text)
        consis_signals = self.count_consistency_signals(text)
        
        # Build rationale parts
        rationale_parts = []
        
        # Rule-based override for strong signals
        final_pred = ml_pred
        override_reason = None
        
        if contra_signals >= 2:  # Strong contradiction signals
            final_pred = 0
            override_reason = f"Strong contradiction indicators detected ({contra_signals} signals)"
        elif contra_signals == 1 and ml_pred == 0:
            # ML agrees with contradiction signal
            override_reason = "Contradiction signal confirmed by ML"
        elif consis_signals >= 2 and contra_signals == 0:
            # Strong consistency signals
            final_pred = 1
            override_reason = f"Strong consistency indicators ({consis_signals} signals)"
        elif ml_pred == 0 and confidence < 0.6:
            # Low confidence contradiction - check signals
            if consis_signals > contra_signals:
                final_pred = 1
                override_reason = "Low ML confidence, consistency signals stronger"
        
        # Build rationale
        if final_pred == 0:  # Contradict
            if override_reason:
                rationale = f"Contradict: {override_reason}."
            else:
                rationale = f"Contradict: ML classifier indicates conflict (confidence: {confidence:.2f})."
            
            if contra_signals > 0:
                rationale += f" {contra_signals} contradiction signal(s) found."
        else:  # Consistent
            if override_reason:
                rationale = f"Consistent: {override_reason}."
            else:
                rationale = f"Consistent: ML classifier shows alignment (confidence: {confidence:.2f})."
            
            if consis_signals > 0:
                rationale += f" {consis_signals} consistency signal(s) found."
        
        return final_pred, rationale, confidence


# Global classifier instance
_classifier = None

def get_hybrid_classifier():
    """Get or create the global hybrid classifier."""
    global _classifier
    if _classifier is None:
        _classifier = HybridClassifier()
    return _classifier


def predict_hybrid(text: str) -> Tuple[int, str]:
    """
    Predict using hybrid classifier.
    Returns: (label, rationale)
    """
    classifier = get_hybrid_classifier()
    label, rationale, confidence = classifier.predict(text)
    return label, rationale
