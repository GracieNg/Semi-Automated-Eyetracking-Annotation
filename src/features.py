"""
Feature extraction for eye-tracking ML model
"""
import pandas as pd
import numpy as np

def extract_fixation_features(df):
    """Extract features from fixation data"""
    features = pd.DataFrame()
    
    # Basic fixation metrics
    features['fixation_duration'] = df['FixationDuration']
    features['X'] = df['GazePointX']
    features['Y'] = df['GazePointY']
    features['pupil_size'] = df.get('PupilSize', 3.0)  # default if missing
    
    return features

if __name__ == "__main__":
    print("✓ Features module loaded")
