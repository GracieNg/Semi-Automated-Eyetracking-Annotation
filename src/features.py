print("Running features.py")
"""
Feature extraction for eye-tracking ML classification
Author: Gracie Ng
Date: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil

def extract_features(df):
    """
    Extract ML features from eye-tracking fixations
    
    Args:
        df (pd.DataFrame): Fixation data from Tobii
        
    Returns:
        pd.DataFrame: Feature matrix for ML model
    """
    features = pd.DataFrame()
    
    # Basic fixation metrics (use available columns; fallback to NaN)
    if 'Eye movement event duration' in df.columns:
        features['fixation_duration'] = df['Eye movement event duration'].values
    elif 'FixationDuration' in df.columns:
        features['fixation_duration'] = df['FixationDuration'].values
    else:
        features['fixation_duration'] = np.nan

    if 'Gaze point X' in df.columns:
        features['X_coord'] = df['Gaze point X'].values
    elif 'GazePointX' in df.columns:
        features['X_coord'] = df['GazePointX'].values
    else:
        features['X_coord'] = np.nan

    if 'Gaze point Y' in df.columns:
        features['Y_coord'] = df['Gaze point Y'].values
    elif 'GazePointY' in df.columns:
        features['Y_coord'] = df['GazePointY'].values
    else:
        features['Y_coord'] = np.nan
    
    # Pupil size (if available)
    if 'PupilSize' in df.columns:
        features['pupil_size'] = df['PupilSize'].values
    elif 'pupil_mean' in df.columns:
        features['pupil_size'] = df['pupil_mean'].values
    else:
        features['pupil_size'] = 3.0  # Default
    
    # Saccade velocity (if available)
    if 'SaccadeVelocity' in df.columns:
        features['saccade_velocity'] = df['SaccadeVelocity'].fillna(0).values
    elif 'gaze_velocity' in df.columns:
        features['saccade_velocity'] = pd.Series(df['gaze_velocity']).fillna(0).values
    else:
        features['saccade_velocity'] = 0.0

    if 'label' in df.columns:
        features['label'] = df['label'].values
    
    print(f"✓ Extracted {len(features)} feature vectors with {features.shape[1]} features")
    return features


def add_temporal_features(df):
    """
    Add time-based features (fixation sequences)
    
    Args:
        df (pd.DataFrame): Fixation data with timestamps
        
    Returns:
        pd.DataFrame: Data with temporal features added
    """
    df = df.copy()
    
    # Calculate time between fixations
    df['time_since_last'] = df['RecordingTimestamp'].diff().fillna(0)
    
    # Fixation index (sequence number)
    df['fixation_index'] = range(len(df))
    
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract features from processed TSV data.")
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
    input_dir = data_root / "processed"
    feature_out = data_root / "features.csv"
    mirror_root = Path(args.mirror_root) if args.mirror_root else None

    input_paths = sorted(input_dir.glob("*.csv"))
    if not input_paths:
        print(f"No processed CSV files found in {input_dir}")
        raise SystemExit(1)

    all_features = []
    for input_path in input_paths:
        print(f"Loading processed file: {input_path}")
        df = pd.read_csv(input_path)
        feats = extract_features(df)
        feats["source_file"] = input_path.name
        all_features.append(feats)

    features = pd.concat(all_features, ignore_index=True)
    print("Feature label distribution:")
    print(features["label"].value_counts())

    # Ensure label is the last column (model expects this)
    label_col = features["label"]
    features = features.drop(columns=["label"])
    features["label"] = label_col
    assert features.columns[-1] == "label"

    feature_out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(feature_out, index=False)
    print(f"✓ Saved features to: {feature_out}")

    if mirror_root is not None:
        mirror_root.mkdir(parents=True, exist_ok=True)
        mirror_path = mirror_root / "features.csv"
        shutil.copy2(feature_out, mirror_path)
        print(f"✓ Mirrored features to: {mirror_path}")
