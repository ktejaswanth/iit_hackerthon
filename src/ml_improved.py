"""
Improved ML classifier with better feature engineering and ensemble methods.
"""
import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, List
import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "text_clf_improved.joblib")

LABEL_MAP = {"consistent": 1, "contradict": 0}
INV_LABEL_MAP = {1: "consistent", 0: "contradict"}


class ContradictionFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features that indicate contradictions."""
    
    def __init__(self):
        # Words that often indicate contradictions
        self.contradiction_words = [
            'not', 'never', 'no', 'none', 'nothing', 'neither', 'nor',
            'contradict', 'against', 'opposite', 'however', 'but', 'although',
            'despite', 'though', 'except', 'rather', 'instead', 'unlike',
            'different', 'wrong', 'false', 'deny', 'denied', 'refute',
            'conflict', 'disagree', 'reverse', 'contrary', 'impossible',
            'failed', 'failure', 'botched', 'forged', 'secretly'
        ]
        
        # Words that indicate consistency
        self.consistency_words = [
            'confirm', 'verified', 'support', 'agree', 'consistent',
            'same', 'similar', 'align', 'match', 'accurate', 'correct',
            'true', 'exactly', 'indeed', 'certainly', 'obviously'
        ]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Extract contradiction/consistency features from text."""
        features = []
        for text in X:
            text_lower = text.lower()
            
            # Count contradiction indicators
            contra_count = sum(1 for word in self.contradiction_words if word in text_lower)
            
            # Count consistency indicators
            consis_count = sum(1 for word in self.consistency_words if word in text_lower)
            
            # Negation patterns (not X, never X, etc.)
            negation_count = len(re.findall(r'\b(not|never|no)\s+\w+', text_lower))
            
            # Question marks (uncertainty)
            question_count = text.count('?')
            
            # Exclamation (strong assertions)
            exclaim_count = text.count('!')
            
            # Length of text (longer might be more detailed/consistent)
            text_length = len(text)
            
            # Word count
            word_count = len(text.split())
            
            features.append([
                contra_count,
                consis_count,
                negation_count,
                question_count,
                exclaim_count,
                text_length,
                word_count,
                contra_count - consis_count,  # net contradiction score
            ])
        
        return np.array(features)


def _compose_text(row: pd.Series) -> str:
    """Compose a single text field from available columns."""
    parts = []
    
    # Add book name
    book = str(row.get("book_name", "")).strip()
    if book and book != "nan":
        parts.append(f"BOOK:{book}")
    
    # Add character name
    char = str(row.get("char", "")).strip()
    if char and char != "nan":
        parts.append(f"CHAR:{char}")
    
    # Add caption
    caption = str(row.get("caption", "")).strip()
    if caption and caption != "nan":
        parts.append(f"CAPTION:{caption}")
    
    # Add main content
    content = str(row.get("content", "")).strip()
    if content and content != "nan":
        content = re.sub(r'\s+', ' ', content)
        parts.append(content)
    
    return " ".join(parts)


def train_improved_classifier(train_csv_path: str, model_path: str = MODEL_PATH) -> Tuple[float, str]:
    """
    Train an improved ensemble classifier with better feature engineering.
    """
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Train CSV not found at {train_csv_path}")

    df = pd.read_csv(train_csv_path)
    if "label" not in df.columns:
        raise ValueError("Train CSV must contain a 'label' column.")

    # Build text and labels
    X_text = df.apply(_compose_text, axis=1).values
    y = df["label"].str.strip().str.lower().map(LABEL_MAP).values

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build feature pipeline combining TF-IDF and custom features
    print("Building ensemble classifier with enhanced features...")
    
    # Model 1: TF-IDF with LinearSVC
    model1 = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 3), 
            min_df=1, 
            max_df=0.95,
            max_features=8000,
            sublinear_tf=True,
            analyzer='word'
        )),
        ('clf', LinearSVC(C=0.5, class_weight='balanced', max_iter=3000, random_state=42))
    ])
    
    # Model 2: Character n-grams with LogisticRegression
    model2 = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(2, 5),
            analyzer='char_wb',
            min_df=1,
            max_df=0.95,
            max_features=5000
        )),
        ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=42))
    ])
    
    # Model 3: Combined features with RandomForest
    combined_features = FeatureUnion([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=3000
        )),
        ('custom', ContradictionFeatureExtractor())
    ])
    
    model3 = Pipeline([
        ('features', combined_features),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=3,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Train individual models and evaluate
    models = [
        ('linsvc', model1),
        ('logreg', model2),
        ('rf', model3)
    ]
    
    best_name = None
    best_model = None
    best_f1 = -1.0
    best_acc = 0.0
    best_pred = None
    
    for name, model in models:
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        acc = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')
        
        print(f"  Accuracy: {acc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        # Also show per-class metrics
        p_per_class, r_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_val, y_pred, labels=[0, 1], average=None
        )
        print(f"  Contradict - P: {p_per_class[0]:.3f}, R: {r_per_class[0]:.3f}, F1: {f1_per_class[0]:.3f}")
        print(f"  Consistent - P: {p_per_class[1]:.3f}, R: {r_per_class[1]:.3f}, F1: {f1_per_class[1]:.3f}")
        
        if f1 > best_f1:
            best_name = name
            best_model = model
            best_f1 = f1
            best_acc = acc
            best_pred = y_pred
    
    # Save the best model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    
    # Print detailed report
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_name}")
    print(f"Validation Accuracy: {best_acc:.3f}")
    print(f"Validation F1 Score: {best_f1:.3f}")
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_val, best_pred, target_names=["contradict", "consistent"]))
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_val, best_pred)
    print(f"                  Predicted")
    print(f"                 Cont  Cons")
    print(f"Actual Cont    [{cm[0][0]:4d} {cm[0][1]:4d}]")
    print(f"       Cons    [{cm[1][0]:4d} {cm[1][1]:4d}]")
    print(f"{'='*60}\n")
    
    return best_acc, model_path


def load_improved_classifier(model_path: str = MODEL_PATH):
    """Load the trained improved classifier."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
    return joblib.load(model_path)


def predict_with_improved(model, text: str) -> Tuple[int, str]:
    """
    Predict using the improved model.
    Returns (label, rationale).
    """
    pred = model.predict([text])[0]
    label_name = INV_LABEL_MAP[pred]
    
    # Generate rationale based on prediction
    if pred == 1:  # consistent
        rationale = f"Consistent: ML classifier (improved). The backstory aligns with novel context."
    else:  # contradict
        rationale = f"Contradict: ML classifier (improved). The backstory conflicts with established facts."
    
    return pred, rationale
