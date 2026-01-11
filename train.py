import os
from src.ml_model import train_text_classifier

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    train_csv = os.path.join(base_dir, "data", "train.csv")
    acc, path = train_text_classifier(train_csv)
    print(f"Model trained. Validation accuracy: {acc:.3f}. Saved to: {path}")
