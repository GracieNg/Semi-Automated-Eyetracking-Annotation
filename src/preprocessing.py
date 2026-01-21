"""
import pandas as pd
import numpy as np

def load_tobii_tsv(filepath):
    """
    Load Tobii Pro Lab TSV export
    
    Args:
        filepath (str): Path to .tsv file
        
    Returns:
        pd.DataFrame: Eye-tracking data
    """
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, sep='\t', low_memory=False)
    
    print(f"✓ Loaded {len(df)} rows")
    print(f"✓ Columns available: {len(df.columns)}")
    
    return df

def get_column_info(df):
    """Print available columns in TSV"""
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    return list(df.columns)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        df = load_tobii_tsv(sys.argv[1])
        get_column_info(df)
        print(f"\nFirst 3 rows:")
        print(df.head(3))
