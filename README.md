# Eye-Tracking Active Learning Annotation System

CP5 Project: Semi-automated eye-tracking annotation using Random Forest and Active Learning.

## Overview
This project implements an end-to-end pipeline for processing Tobii TSV exports, engineering gaze features, running a Random Forest classifier, and applying an uncertainty-based Active Learning loop to reduce manual annotation effort. A Streamlit dashboard provides a working product for running the pipeline and reviewing outputs.

## Run Order
From project root:

```bash
python src/preprocessing.py --save
python src/features.py
python src/model.py
python src/evaluation.py
```

To run the same pipeline against the dashboard sandbox data:

```bash
python src/preprocessing.py --save --data-root data/sandbox
python src/features.py --data-root data/sandbox
python src/model.py --data-root data/sandbox
python src/evaluation.py --data-root data/sandbox
```

To run on `data/` but mirror outputs into `data/sandbox/` (for the dashboard):

```bash
python src/preprocessing.py --save --data-root data --mirror-root data/sandbox
python src/features.py --data-root data --mirror-root data/sandbox
python src/model.py --data-root data --mirror-root data/sandbox
python src/evaluation.py --data-root data
```

## Dashboard
Run the dashboard locally:

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

The dashboard supports file upload, pipeline execution, Active Learning outputs, evaluation visuals, and downloads.

## Data Layout
- `data/raw/` contains the master TSV exports (source of truth).
- `data/processed/` contains preprocessing outputs derived from `data/raw/` (can be regenerated).
- `data/ui_upload/` stores files uploaded through the dashboard UI (optional to keep).
- `data/sandbox/raw/` contains working copies for dashboard runs/experiments (often created from uploads).
- `data/sandbox/processed/` contains outputs derived from `data/sandbox/raw/`.
- `data/Pilot test/` contains pilot test TSVs used during early validation (optional to keep).

If you are cleaning up, it is generally safe to delete `data/processed/` and all of `data/sandbox/` because they can be regenerated.

## Key Outputs
Generated artifacts (after running the pipeline):

```
data/
  processed/
    *_processed.csv
  features.csv
  predictions_with_uncertainty_pool.csv
  predictions_with_uncertainty.csv
  predictions_sorted_by_uncertainty.csv
  to_review.csv
  human_reviewed.csv
results/
  figures/
    confidence_distribution.png
    confusion_matrix.png
    annotation_time_comparison.png
  tables/
    active_learning_summary.csv
    metrics.csv
    class_report.csv
    annotation_time_comparison.csv
```

---

If you run into issues, ensure your input data are in TSV format. For dashboard runs, place inputs under `data/sandbox/raw/` (the dashboard auto converts CSV uploads to TSV).
