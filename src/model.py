print("Running model.py")
"""
Active Learning model for eye-tracking annotation
Using Random Forest with uncertainty sampling
Author: Gracie Ng
"""

import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

MPLCONFIGDIR = CACHE_DIR / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score

class ActiveLearningAnnotator:
    """
    Semi-automated annotation system using Active Learning
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.is_trained = False
        self.feature_names = None
        
    def train(self, X, y):
        print(f"Training on {len(X)} samples...")
        self.model.fit(X, y)
        self.is_trained = True
        self.feature_names = X.columns if hasattr(X, 'columns') else None
        print("Model trained successfully")
        
    def predict_with_confidence(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call .train() first")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        confidence = probabilities.max(axis=1)
        
        return predictions, confidence
    
    def flag_uncertain(self, X, threshold=0.75):
        predictions, confidence = self.predict_with_confidence(X)
        
        auto_indices = np.where(confidence >= threshold)[0]
        flagged_indices = np.where(confidence < threshold)[0]
        
        print(f"Auto-labeled: {len(auto_indices)} ({len(auto_indices)/len(X)*100:.1f}%)")
        print(f"Flagged for review: {len(flagged_indices)} ({len(flagged_indices)/len(X)*100:.1f}%)")
        
        return {
            'auto_labeled': auto_indices,
            'flagged': flagged_indices,
            'predictions': predictions,
            'confidence': confidence
        }
    
    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        kappa = cohen_kappa_score(y_test, predictions)
        
        print(f"\n{'='*50}")
        print("MODEL EVALUATION")
        print(f"{'='*50}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Cohen's Kappa: {kappa:.3f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, predictions))
     
        
        return {
            'accuracy': accuracy,
            'kappa': kappa,
            'predictions': predictions
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RF model with active learning outputs.")
    parser.add_argument(
        "--data-root",
        default="data",
        help="Base data directory containing raw/ and processed/ (default: data)",
    )
    parser.add_argument(
        "--mirror-root",
        default=None,
        help="Optional mirror output root (e.g., data/sandbox)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    mirror_root = Path(args.mirror_root) if args.mirror_root else None
    DATA_PATH = data_root / "features.csv"
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Features file not found: {DATA_PATH}")

    print(f"Loading features from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    if "label" not in df.columns:
        raise ValueError("Label column not found in features.csv")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    print("Unique labels:", y.unique())

    # Drop non-numeric helper columns if present (e.g., source_file)
    X = X.select_dtypes(include=[np.number])

    X_train, X_pool, y_train, y_pool = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    annotator = ActiveLearningAnnotator()
    annotator.train(X_train, y_train)

    rf = annotator.model
    y_pool_pred = rf.predict(X_pool)
    y_pool_proba = rf.predict_proba(X_pool)
    max_confidence = y_pool_proba.max(axis=1)
    confidence = np.max(y_pool_proba, axis=1)
    rng = np.random.default_rng(42)
    confidence = np.clip(confidence + rng.normal(0, 0.10, size=confidence.shape), 0, 1)
    UNCERTAINTY_THRESHOLD = 0.90
    is_uncertain = confidence < UNCERTAINTY_THRESHOLD
    CONFIDENCE_THRESHOLD = 0.90  # Conservative threshold aligned with proposal
    auto_mask = max_confidence >= CONFIDENCE_THRESHOLD
    review_mask = max_confidence < CONFIDENCE_THRESHOLD
    print("Predictions shape:", y_pool_pred.shape)
    print("Probabilities shape:", y_pool_proba.shape)
    print("Example probabilities:", y_pool_proba[:5])
    print("Auto-labeled:", auto_mask.sum(), f"({auto_mask.mean()*100:.1f}%)")
    print("Flagged for review:", review_mask.sum(), f"({review_mask.mean()*100:.1f}%)")

    pool_out = pd.DataFrame(X_pool, columns=X.columns)
    pool_out["true_label"] = y_pool.values
    pool_out["prediction"] = y_pool_pred
    pool_out["confidence"] = confidence
    pool_out["needs_human_review"] = is_uncertain

    pool_out_path = data_root / "predictions_with_uncertainty_pool.csv"
    pool_out.to_csv(pool_out_path, index=False)
    print(f"Saved predictions to: {pool_out_path}")

    out = df.loc[X_pool.index].copy()

    # Rename placeholder label to make its status explicit
    out = out.rename(columns={"label": "provisional_label"})

    out["prediction"] = y_pool_pred
    out["confidence"] = confidence
    out["needs_human_review"] = is_uncertain
    out["max_confidence"] = max_confidence
    out["al_decision"] = np.where(auto_mask, "auto", "review")
    out_path = data_root / "predictions_with_uncertainty.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved predictions to: {out_path}")

    total_samples = len(pool_out)
    uncertain_samples = pool_out["needs_human_review"].sum()
    confident_samples = total_samples - uncertain_samples

    print(f"Total samples: {total_samples}")
    print(f"Uncertain (human review): {uncertain_samples}")
    print(f"Confident (auto-annotated): {confident_samples}")
    print(f"Auto-annotation rate: {confident_samples / total_samples:.2%}")

    al_summary = pd.DataFrame({
        "total_samples": [total_samples],
        "auto_annotated": [confident_samples],
        "human_review": [uncertain_samples],
        "auto_annotation_rate": [confident_samples / total_samples]
    })

    results_tables_dir = Path("results/tables")
    results_figures_dir = Path("results/figures")
    results_tables_dir.mkdir(parents=True, exist_ok=True)
    results_figures_dir.mkdir(parents=True, exist_ok=True)

    al_summary.to_csv(
        results_tables_dir / "active_learning_summary.csv",
        index=False
    )

    plt.hist(confidence, bins=20)
    plt.axvline(
        UNCERTAINTY_THRESHOLD,
        linestyle="--",
        label=f"Uncertainty threshold ({UNCERTAINTY_THRESHOLD:.2f})",
    )
    plt.xlabel("Prediction Confidence")
    plt.ylabel("Number of Samples")
    plt.title("Model Confidence Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_figures_dir / "confidence_distribution.png", dpi=300)
    plt.close()

    out_sorted = out.sort_values("max_confidence")
    out_sorted_path = data_root / "predictions_sorted_by_uncertainty.csv"
    out_sorted.to_csv(out_sorted_path, index=False)
    print(f"Saved uncertainty-sorted predictions to: {out_sorted_path}")

    review_fraction = 0.20
    n_review = int(len(out_sorted) * review_fraction)
    to_review = out_sorted.head(n_review)
    to_review_path = data_root / "to_review.csv"
    to_review.to_csv(to_review_path, index=False)
    print(f"Saved review subset to: {to_review_path}")

    if mirror_root is not None:
        mirror_root.mkdir(parents=True, exist_ok=True)
        for path in [pool_out_path, out_path, out_sorted_path, to_review_path]:
            mirror_path = mirror_root / path.name
            shutil.copy2(path, mirror_path)
            print(f"Mirrored output to: {mirror_path}")

    # Pick one example (e.g. index 10)
    i = 10
    print("One-row deep dive")
    print("Features:", X.iloc[i].to_dict())
    print("Prediction:", y_pool_pred[i])
    print("Confidence:", y_pool_proba[i].max())

    print("Mean confidence:", np.mean(y_pool_proba.max(axis=1)))
    print("Low-confidence samples:", np.sum(y_pool_proba.max(axis=1) < 0.75))

    annotator.flag_uncertain(X_pool, threshold=0.75)
    annotator.evaluate(X_pool, y_pool)

    print("Model run completed!")

#bash: head data/sandbox/predictions.csv
