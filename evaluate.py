import os
from src.ml_model import crossval_evaluate

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    train_csv = os.path.join(base_dir, "data", "train.csv")
    out_dir = os.path.join(base_dir, "output")
    report = crossval_evaluate(train_csv, cv=5, save_dir=out_dir)
    print("Best model:", report.get("best_model"))
    print("CV accuracy mean:", f"{report.get('cv_accuracy_mean', 0):.3f}")
    print("Confusion matrix:", report.get("confusion_matrix"))
    print("Saved JSON:", os.path.join(out_dir, "ml_eval.json"))
    if report.get("confusion_plot"):
        print("Saved plot:", report["confusion_plot"])