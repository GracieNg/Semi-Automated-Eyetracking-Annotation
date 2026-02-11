print("Running evaluation.py")
"""
Evaluation metrics for annotation system
"""
import os
from pathlib import Path

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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, f1_score

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate active learning predictions.")
    parser.add_argument(
        "--data-root",
        default="data",
        help="Base data directory containing predictions (default: data)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    default_path = data_root / "predictions_al.csv"
    fallback_path = data_root / "predictions_with_uncertainty.csv"

    if default_path.exists():
        df = pd.read_csv(default_path)
        if (
            "provisional_label" in df.columns
            and df["provisional_label"].nunique() == 1
            and df["provisional_label"].iloc[0] == "placeholder"
            and fallback_path.exists()
        ):
            print("Using fallback predictions_with_uncertainty.csv (placeholder labels detected).")
            df = pd.read_csv(fallback_path)
    elif fallback_path.exists():
        df = pd.read_csv(fallback_path)
    else:
        raise FileNotFoundError(
            "No predictions file found. Expected one of: "
            f"{default_path} or {fallback_path}."
        )

    # Provisional labels derived from heuristic rules
    # Used as a baseline reference prior to human validation
    y_true = df["provisional_label"]
    y_pred = df["prediction"]

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print("Macro-F1:", macro_f1)

    metrics_df = pd.DataFrame({
        "metric": ["macro_f1"],
        "value": [macro_f1]
    })

    metrics_df.to_csv("results/tables/metrics.csv", index=False)

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv("results/tables/class_report.csv")

    time_df = pd.DataFrame({
        "method": ["Manual annotation", "Semi-automated (AL-assisted)"],
        "time_minutes": [60, 18]  # example values
    })

    time_df["time_saved_percent"] = (
        1 - time_df["time_minutes"] / time_df.loc[0, "time_minutes"]
    ) * 100

    time_df.to_csv("results/tables/annotation_time_comparison.csv", index=False)

    plt.bar(time_df["method"], time_df["time_minutes"])
    plt.ylabel("Estimated Annotation Time (minutes)")
    plt.title("Annotation Time Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("results/figures/annotation_time_comparison.png", dpi=300)
    plt.close()

    label_set = sorted(set(y_true) | set(y_pred))
    if not label_set:
        raise ValueError("No labels found for confusion matrix.")

    cm = confusion_matrix(y_true, y_pred, labels=label_set)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_set
    )
    disp.plot(cmap="Blues", values_format="d")

    plt.title("Confusion Matrix – Random Forest Gaze Classification")
    plt.tight_layout()
    plt.savefig("results/figures/confusion_matrix.png", dpi=300)
    plt.close()

def calculate_time_savings(manual_time, semi_auto_time):
    """Calculate time reduction percentage"""
    # Manual annotation: ~4–6 minutes per 5-minute gaze segment
    # Semi-automated: manual labels only for ~20–30% low-confidence samples
    reduction = manual_time - semi_auto_time
    percentage = (reduction / manual_time) * 100
    
    print(f"\nTime Savings:")
    print(f"  Manual: {manual_time:.1f}h")
    print(f"  Semi-auto: {semi_auto_time:.1f}h")
    print(f"  Saved: {reduction:.1f}h ({percentage:.1f}%)")
    
    return {'manual': manual_time, 'semi_auto': semi_auto_time, 
            'saved': reduction, 'percentage': percentage}

def compare_annotations(y_true, y_pred):
    """Compare manual vs automated"""
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    print(f"\nAnnotation Agreement:")
    print(f"  Accuracy: {acc:.2%}")
    print(f"  Cohen's Kappa: {kappa:.3f}")
    
    return {'accuracy': acc, 'kappa': kappa}
