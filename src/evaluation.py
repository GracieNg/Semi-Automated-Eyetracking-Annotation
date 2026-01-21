"""
Evaluation metrics for annotation system
"""
from sklearn.metrics import accuracy_score, cohen_kappa_score
import numpy as np

def calculate_time_savings(manual_time, semi_auto_time):
    """Calculate time reduction percentage"""
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
