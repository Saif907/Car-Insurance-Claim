import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error

from src.preprocessing import build_preprocessing_pipeline, clean_raw_data
from src.config import TARGET_CLASS, TARGET_REG

def main():
    data_path = 'data/car_insurance_claim.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Please ensure you are running from the project root.")
        
    print("Loading data...")
    raw_df = pd.read_csv(data_path)
    
    print("Cleaning raw data...")
    df = clean_raw_data(raw_df)
    
    # Handle NaN in target variables (if any)
    df = df.dropna(subset=[TARGET_CLASS])
    
    # Separation into Classification Data
    # For classification, target is claim_flag.
    X = df.drop(columns=['id', 'birth', TARGET_CLASS, TARGET_REG], errors='ignore')
    y_class = df[TARGET_CLASS]
    
    print("Building preprocessing pipeline...")
    preprocessor = build_preprocessing_pipeline()
    
    # Train-test split for Classification (stratified)
    print("Splitting data...")
    X_train, X_test, y_train_class, y_test_class = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # Fit the preprocessor strictly on the training set
    print("Fitting preprocessing pipeline on training data...")
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    
    print("Training XGBClassifier...")
    classifier = XGBClassifier(random_state=42, n_estimators=100, max_depth=5)
    classifier.fit(X_train_prep, y_train_class)
    
    y_pred_class = classifier.predict(X_test_prep)
    print("\nClassification Report:")
    print(classification_report(y_test_class, y_pred_class))
    
    # Regression Model
    # For regression, we only train on samples where a claim actually occurred (claim_flag == 1)
    # We also drop records where target clm_amt is missing or 0 (if claim flag was 1 but amt 0, although usually consistent)
    print("Preparing data for Regression...")
    df_claims = df[(df[TARGET_CLASS] == 1) & (df[TARGET_REG].notna())].copy()
    X_reg = df_claims.drop(columns=['id', 'birth', TARGET_CLASS, TARGET_REG], errors='ignore')
    y_reg = df_claims[TARGET_REG]
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Transform using the already fitted preprocessor
    X_train_reg_prep = preprocessor.transform(X_train_reg)
    X_test_reg_prep = preprocessor.transform(X_test_reg)
    
    print("Training SGDRegressor...")
    regressor = SGDRegressor(random_state=42)
    regressor.fit(X_train_reg_prep, y_train_reg)
    
    y_pred_reg = regressor.predict(X_test_reg_prep)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    print("\nRegression Report:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    # Save models
    os.makedirs('models', exist_ok=True)
    print("Saving models to models/ directory...")
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    joblib.dump(classifier, 'models/classifier.joblib')
    joblib.dump(regressor, 'models/regressor.joblib')
    print("Training complete and models saved successfully!")

if __name__ == "__main__":
    main()
