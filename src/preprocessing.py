print("Running preprocessing.py")
import pandas as pd
import numpy as np
from pathlib import Path
import shutil


def proxy_gaze_label(mean_velocity: float) -> str:
    if mean_velocity < 30:
        return "reading"
    elif mean_velocity < 100:
        return "scanning"
    else:
        return "navigation"

NUMERIC_COLS = [
    "Recording timestamp",
    "Computer timestamp",
    "Eyetracker timestamp",
    "Gaze point X", "Gaze point Y",
    "Gaze point left X", "Gaze point left Y",
    "Gaze point right X", "Gaze point right Y",
    "Pupil diameter left", "Pupil diameter right", "Pupil diameter filtered",
    "Eye openness left", "Eye openness right", "Eye openness filtered",
    "Fixation point X", "Fixation point Y",
    "Fixation point X (MCSnorm)", "Fixation point Y (MCSnorm)",
    "Eye movement event duration",
    "Viewport position X", "Viewport position Y",
    "Viewport width", "Viewport height",
]


def load_tobii_tsv(filepath: str | Path) -> pd.DataFrame:
    """Load Tobii Pro Lab TSV export."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, sep="\t", low_memory=False)
    print(f"✓ Loaded {len(df)} rows")
    print(f"✓ Columns available: {len(df.columns)}")
    return df


def get_column_info(df: pd.DataFrame) -> list[str]:
    """Print available columns in TSV."""
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns, start=1):
        print(f"  {i}. {col}")
    return df.columns.tolist()


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert selected columns to numeric (coerce errors to NaN)."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic features that are typically useful for ML:
    - time_sec from Eyetracker timestamp (or Recording timestamp fallback)
    - valid_gaze boolean from validity codes (0 usually best; we accept <=1 by default)
    - gaze_velocity (px/sec) based on consecutive gaze points
    - pupil_mean, pupil_diff
    """
    # Prefer Eyetracker timestamp (usually in microseconds), else Recording timestamp
    if "Eyetracker timestamp" in df.columns and df["Eyetracker timestamp"].notna().any():
        t = df["Eyetracker timestamp"].astype(float)
        # Tobii often uses microseconds -> convert to seconds
        df["time_sec"] = (t - t.iloc[0]) / 1_000_000.0
    else:
        t = df["Recording timestamp"].astype(float)
        # Recording timestamp often already ms -> convert to seconds
        df["time_sec"] = (t - t.iloc[0]) / 1000.0

    # Valid gaze mask (supports numeric or string validity codes)
    if "Validity left" in df.columns and "Validity right" in df.columns:
        left_num = pd.to_numeric(df["Validity left"], errors="coerce")
        right_num = pd.to_numeric(df["Validity right"], errors="coerce")
        if left_num.notna().any() or right_num.notna().any():
            df["valid_gaze"] = (left_num <= 1) & (right_num <= 1)
        else:
            left_str = df["Validity left"].astype(str).str.strip().str.lower()
            right_str = df["Validity right"].astype(str).str.strip().str.lower()
            valid_tokens = {"valid", "1", "true"}
            df["valid_gaze"] = left_str.isin(valid_tokens) & right_str.isin(valid_tokens)
    else:
        df["valid_gaze"] = True

    # Gaze velocity (based on combined gaze point X/Y)
    if "Gaze point X" in df.columns and "Gaze point Y" in df.columns:
        dx = df["Gaze point X"].diff()
        dy = df["Gaze point Y"].diff()
        dt = df["time_sec"].diff()
        dist = np.sqrt(dx**2 + dy**2)
        df["gaze_velocity"] = dist / dt
        df.loc[(dt <= 0) | (~np.isfinite(df["gaze_velocity"])), "gaze_velocity"] = np.nan

        df["label"] = df["gaze_velocity"].fillna(0).apply(proxy_gaze_label)

    # Pupil features
    if "Pupil diameter left" in df.columns and "Pupil diameter right" in df.columns:
        df["pupil_mean"] = df[["Pupil diameter left", "Pupil diameter right"]].mean(axis=1)
        df["pupil_diff"] = (df["Pupil diameter left"] - df["Pupil diameter right"]).abs()

    return df


def save_processed(
    df: pd.DataFrame,
    input_path: Path,
    out_dir: Path,
    mirror_root: Path | None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{input_path.stem}_processed.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved processed file to: {out_path}")

    if mirror_root is not None:
        mirror_dir = mirror_root / "processed"
        mirror_dir.mkdir(parents=True, exist_ok=True)
        mirror_path = mirror_dir / out_path.name
        shutil.copy2(out_path, mirror_path)
        print(f"✓ Mirrored processed file to: {mirror_path}")
    return out_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect + preprocess Tobii TSV export.")
    parser.add_argument("filepath", nargs="?", help="(Ignored) Path to Tobii .tsv export file")
    parser.add_argument("--head", type=int, default=3, help="Rows to preview (default: 3)")
    parser.add_argument("--save", action="store_true", help="Save processed CSV to data/processed/")
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
    raw_dir = data_root / "raw"
    out_dir = data_root / "processed"
    mirror_root = Path(args.mirror_root) if args.mirror_root else None

    input_paths = sorted(raw_dir.glob("*.tsv"))
    if not input_paths:
        print(f"No TSV files found in {raw_dir}")
        return

    for input_path in input_paths:
        df = load_tobii_tsv(input_path)
        get_column_info(df)

        df = coerce_numeric(df, NUMERIC_COLS)
        df = add_basic_features(df)

        print(f"\nFirst {args.head} rows:")
        print(df.head(args.head))

        # quick sanity stats
        if "valid_gaze" in df.columns:
            print(f"\nValid gaze rows: {df['valid_gaze'].mean()*100:.2f}%")

        if args.save:
            print("Label distribution:")
            print(df["label"].value_counts())
            save_processed(df, input_path, out_dir, mirror_root)


if __name__ == "__main__":
    main()
