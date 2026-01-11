"""
Train the improved ML classifier with better feature engineering.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ml_improved import train_improved_classifier

if __name__ == "__main__":
    train_csv = "data/train.csv"
    
    print("Training improved classifier with enhanced features...")
    print(f"Training data: {train_csv}")
    print("="*60)
    
    acc, model_path = train_improved_classifier(train_csv)
    
    print(f"\n✓ Model trained successfully!")
    print(f"✓ Final validation accuracy: {acc:.1%}")
    print(f"✓ Saved to: {model_path}")
