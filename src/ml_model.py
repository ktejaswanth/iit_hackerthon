import os
import joblib
import pandas as pd
from typing import Tuple
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
import numpy as np
import json

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "text_clf.joblib")

LABEL_MAP = {"consistent": 1, "contradict": 0}
INV_LABEL_MAP = {1: "consistent", 0: "contradict"}


def _compose_text(row: pd.Series) -> str:
    """Compose a single text field from available columns with enhanced preprocessing."""
    parts = []
    
    # Add book name as context
    book = str(row.get("book_name", "")).strip()
    if book and book != "nan":
        parts.append(f"BOOK:{book}")
    
    # Add character name
    char = str(row.get("char", "")).strip()
    if char and char != "nan":
        parts.append(f"CHAR:{char}")
    
    # Add caption if available
    caption = str(row.get("caption", "")).strip()
    if caption and caption != "nan":
        parts.append(f"CAPTION:{caption}")
    
    # Add main content - this is the most important
    content = str(row.get("content", "")).strip()
    if content and content != "nan":
        # Clean up content
        content = re.sub(r'\s+', ' ', content)  # normalize whitespace
        parts.append(content)
    
    return " ".join(parts)


def train_text_classifier(train_csv_path: str, model_path: str = MODEL_PATH) -> Tuple[float, str]:
    """
    Train a TF-IDF + Logistic Regression text classifier on train.csv.
    Saves the model pipeline with vectorizer to `model_path`.
    Returns (validation_accuracy, saved_model_path).
    """
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Train CSV not found at {train_csv_path}")

    df = pd.read_csv(train_csv_path)
    if "label" not in df.columns:
        raise ValueError("Train CSV must contain a 'label' column with values 'consistent' or 'contradict'.")

    # Build text and labels
    X = df.apply(_compose_text, axis=1)
    y = df["label"].str.strip().str.lower().map(LABEL_MAP)
    if y.isnull().any():
        raise ValueError("Label column contains unknown values. Expected 'consistent' or 'contradict'.")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define multiple candidate models with better parameters
    candidates = [
        ("logreg", Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_df=0.9, max_features=10000, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=42))
        ])),
        ("linsvc", Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_df=0.9, max_features=10000, sublinear_tf=True)),
            ("clf", LinearSVC(max_iter=2000, C=0.5, class_weight="balanced", random_state=42))
        ])),
        ("rf", Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9, max_features=5000)),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, 
                                          class_weight="balanced", random_state=42, n_jobs=-1))
        ])),
        ("gboost", Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9, max_features=5000)),
            ("clf", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, 
                                               random_state=42))
        ])),
    ]

    best_name = None
    best_model = None
    best_score = -1.0  # Use F1 score for imbalanced data
    best_pred = None

    for name, pipe in candidates:
        print(f"\nTraining {name}...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
        
        # Calculate metrics
        acc = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')
        
        # Use F1 score for model selection (better for imbalanced data)
        score = f1
        
        print(f"  Accuracy: {acc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        if score > best_score:
            best_name = name
            best_model = pipe
            best_score = score
            best_pred = y_pred
            best_acc = acc

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)

    # Print detailed report for best model
    try:
        print(f"\n{'='*50}")
        print(f"BEST MODEL: {best_name}")
        print(f"Validation Accuracy: {best_acc:.3f}")
        print(f"Validation F1 Score: {best_score:.3f}")
        print(f"\nClassification Report:")
        print(classification_report(y_val, best_pred, target_names=["contradict", "consistent"]))
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_val, best_pred)
        print(f"                  Predicted")
        print(f"                 Cont  Cons")
        print(f"Actual Cont    [{cm[0][0]:4d} {cm[0][1]:4d}]")
        print(f"       Cons    [{cm[1][0]:4d} {cm[1][1]:4d}]")
        print(f"{'='*50}\n")
    except Exception as e:
        print(f"Warning: Could not print detailed report: {e}")

    return best_acc, model_path


def load_text_classifier(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
    return joblib.load(model_path)

def explain_prediction(model, text: str, top_n: int = 5) -> Tuple[list, list]:
    """Return top positive and negative contributing tokens for a prediction."""
    try:
        vec = model.named_steps.get("tfidf")
        clf = model.named_steps.get("clf")
        if vec is None or clf is None:
            return [], []
        X = vec.transform([text]).tocsr()
        feature_names = vec.get_feature_names_out()
        indices = X.indices
        values = X.data
        # Coefficients
        if hasattr(clf, "coef_"):
            coef = clf.coef_
            coef_vec = coef[0] if coef.shape[0] == 1 else coef[1]
        else:
            return [], []
        contribs = [(feature_names[i], values[k] * coef_vec[i]) for k, i in enumerate(indices)]
        contribs.sort(key=lambda x: x[1], reverse=True)
        pos_tokens = [t for t, c in contribs[:top_n]]
        neg_tokens = [t for t, c in sorted(contribs, key=lambda x: x[1])[:top_n]]
        return pos_tokens, neg_tokens
    except Exception:
        return [], []


def predict_label(text: str, model=None) -> Tuple[int, str]:
    """Predict numeric label (1=consistent, 0=contradict) with brief rationale including top tokens."""
    if model is None:
        model = load_text_classifier()
    pred = int(model.predict([text])[0])
    pos_toks, neg_toks = explain_prediction(model, text)
    pos_str = ", ".join(pos_toks) if pos_toks else "-"
    neg_str = ", ".join(neg_toks) if neg_toks else "-"
    rationale = f"ML classifier (TF-IDF). Top tokens +[{pos_str}] -[{neg_str}]"
    return pred, rationale


def crossval_evaluate(train_csv_path: str, cv: int = 5, save_dir: str | None = None) -> dict:
    """Perform stratified K-fold cross-validation and return metrics; optionally save report and confusion plot."""
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Train CSV not found at {train_csv_path}")

    df = pd.read_csv(train_csv_path)
    if "label" not in df.columns:
        raise ValueError("Train CSV must contain a 'label' column.")
    X = df.apply(_compose_text, axis=1)
    y = df["label"].str.strip().str.lower().map(LABEL_MAP)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    pipelines = [
        ("logreg", Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000))
            , ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ])),
        ("linsvc", Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=80000))
            , ("clf", LinearSVC(class_weight="balanced"))
        ])),
    ]

    best_name = None
    best_score = -1.0
    best_pipe = None
    fold_metrics = []
    total_cm = np.zeros((2, 2), dtype=int)

    for name, pipe in pipelines:
        scores = []
        cms = np.zeros((2, 2), dtype=int)
        prfs = []
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            scores.append(acc)
            cms += confusion_matrix(y_val, y_pred, labels=[0,1])
            prfs.append(precision_recall_fscore_support(y_val, y_pred, average="binary", pos_label=1))
        avg_acc = float(np.mean(scores))
        print(f"CV {name} avg accuracy: {avg_acc:.3f}")
        if avg_acc > best_score:
            best_score = avg_acc
            best_name = name
            best_pipe = pipe
            total_cm = cms
            fold_metrics = scores

    report = {
        "best_model": best_name,
        "cv_accuracy_scores": [float(x) for x in fold_metrics],
        "cv_accuracy_mean": float(np.mean(fold_metrics)) if fold_metrics else 0.0,
        "confusion_matrix": total_cm.tolist(),
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join(save_dir, "ml_eval.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        # Save confusion plot
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(4,4))
            im = ax.imshow(total_cm, cmap="Blues")
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["contradict","consistent"]) ; ax.set_yticklabels(["contradict","consistent"]) 
            for (i,j), val in np.ndenumerate(total_cm):
                ax.text(j, i, int(val), ha="center", va="center", color="black")
            ax.set_title("Confusion Matrix (CV)")
            plt.tight_layout()
            fig_path = os.path.join(save_dir, "ml_confusion.png")
            fig.savefig(fig_path)
            plt.close(fig)
            report["confusion_plot"] = fig_path
        except Exception:
            pass

    return report
