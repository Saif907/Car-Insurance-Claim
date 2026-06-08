from fastapi import FastAPI, HTTPException
import pandas as pd
import os

from src.predict import InsuranceClaimPredictor
from app.schemas import InsuranceClaimInput, InsuranceClaimOutput

app = FastAPI(title="Car Insurance Claim Prediction API")

# Initialize the predictor (it loads models at startup)
# Since the API runs from the root normally, models_dir is 'models'
models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

predictor = None
try:
    predictor = InsuranceClaimPredictor(models_dir=models_dir)
except Exception as e:
    print(f"Warning: Could not load predictor: {e}")

@app.get("/health")
def health_check():
    """Health check endpoint to ensure service is running."""
    if predictor is None:
        return {"status": "unhealthy", "reason": "Models not loaded"}
    return {"status": "ok"}

@app.post("/predict", response_model=InsuranceClaimOutput)
def predict_claim(input_data: InsuranceClaimInput):
    """
    Predict claim probability and value for a given insurance applicant.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable")
        
    try:
        # Convert input Pydantic model to a pandas DataFrame (with correct aliases)
        input_dict = input_data.model_dump(by_alias=True)
        input_df = pd.DataFrame([input_dict])
        
        # Make prediction
        prediction_result = predictor.predict(input_df)
        
        return InsuranceClaimOutput(**prediction_result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
