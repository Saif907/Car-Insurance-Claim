import os
import joblib
import pandas as pd
from typing import Dict, Any

class InsuranceClaimPredictor:
    """
    Wrapper class for the trained models and preprocessing pipeline.
    Handles inference for new data.
    """
    def __init__(self, models_dir: str = 'models'):
        preprocessor_path = os.path.join(models_dir, 'preprocessor.joblib')
        classifier_path = os.path.join(models_dir, 'classifier.joblib')
        regressor_path = os.path.join(models_dir, 'regressor.joblib')
        
        if not (os.path.exists(preprocessor_path) and os.path.exists(classifier_path) and os.path.exists(regressor_path)):
            raise FileNotFoundError("Model artifacts not found. Please run train.py first.")
            
        self.preprocessor = joblib.load(preprocessor_path)
        self.classifier = joblib.load(classifier_path)
        self.regressor = joblib.load(regressor_path)
        
    def predict(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Takes raw input DataFrame, applies preprocessing, predicts claim probability,
        and estimates claim value if a claim is predicted.
        """
        # Transform the data using the saved preprocessor
        X_prep = self.preprocessor.transform(input_data)
        
        # Predict claim probability and binary decision
        claim_prob = self.classifier.predict_proba(X_prep)[0][1]
        claim_decision = self.classifier.predict(X_prep)[0]
        
        # Predict claim amount (only applicable if a claim occurs, but we can compute it conditionally)
        if claim_decision == 1:
            estimated_value = self.regressor.predict(X_prep)[0]
            estimated_value = max(0.0, float(estimated_value)) # Cap at 0
        else:
            estimated_value = 0.0
            
        return {
            "claim_probability": float(claim_prob),
            "claim_decision": int(claim_decision),
            "estimated_claim_value": estimated_value
        }
