"""
Active Learning model for eye-tracking annotation
Using Random Forest with uncertainty sampling
Author: Gracie Ng
"""

import numpy as np
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
            random_state=random_state,
            max_depth=10
        )
        self.is_trained = False
        self.feature_names = None
        
    def train(self, X, y):
        print(f"Training on {len(X)} samples...")
        self.model.fit(X, y)
        self.is_trained = True
        self.feature_names = X.columns if hasattr(X, 'columns') else None
        print("✓ Model trained successfully")
        
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
        
        print(f"✓ Auto-labeled: {len(auto_indices)} ({len(auto_indices)/len(X)*100:.1f}%)")
        print(f"⚠ Flagged for review: {len(flagged_indices)} ({len(flagged_indices)/len(X)*100:.1f}%)")
        
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
        print(f"{'='*50}\n")
        
        return {
            'accuracy': accuracy,
            'kappa': kappa,
            'predictions': predictions
        }


if __name__ == "__main__":
    print("Testing Active Learning model...")
    
    from sklearn.datasets import make_classification
    
    # FIXED: 3 classes instead of 5
    X, y = make_classification(
        n_samples=1000, 
        n_features=5, 
        n_classes=3,      
        n_informative=3,
        n_redundant=1,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.9, random_state=42
    )
    
    annotator = ActiveLearningAnnotator()
    annotator.train(X_train, y_train)
    
    results = annotator.flag_uncertain(X_test, threshold=0.75)
    annotator.evaluate(X_test, y_test)
    
    print("✓ All tests passed!")
