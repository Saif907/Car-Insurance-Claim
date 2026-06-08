import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import sklearn

from src.custom_transformers import ColumnDropper
from src.config import (
    NUMERICAL_COLS,
    CAT_COLS_ORD,
    CAT_COLS_BIN,
    CAT_COLS_ONE_HOT,
    EDUCATION_RANK,
    COLS_TO_DROP
)

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw dataframe by formatting column names and cleaning string representations of currencies/categories.
    """
    df = df.copy()
    
    # Lowercase column names
    df.columns = df.columns.str.lower()
    
    # Currency columns to clean
    currency_cols = ['income', 'home_val', 'bluebook', 'oldclaim', 'clm_amt']
    for col in currency_cols:
        if col in df.columns:
            # Remove '$' and ',' and convert to float
            df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
            
    # Clean z_ prefix in some categorical columns
    z_cols = ['mstatus', 'gender', 'education', 'occupation', 'car_type']
    for col in z_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('z_', '')
            
    # Rename some columns to match notebook analysis
    rename_mapping = {
        'kidsdriv': 'num_young_drivers',
        'homekids': 'num_of_children',
        'yoj': 'years_job_held_for',
        'travtime': 'commute_dist',
        'car_use': 'type_of_use',
        'bluebook': 'vehicle_value',
        'tif': 'policy_tenure',
        'oldclaim': '5_year_total_claims_value',
        'clm_freq': '5_year_num_of_claims',
        'mvr_pts': 'license_points',
        'car_age': 'vehicle_age',
        'parent1': 'single_parent',
        'mstatus': 'married',
        'education': 'highest_education',
        'car_type': 'vehicle_type',
        'revoked': 'licence_revoked',
        'urbanicity': 'address_type'
    }
    df.rename(columns=rename_mapping, inplace=True)
    
    # Replace 'nan' string with actual np.nan
    df.replace('nan', np.nan, inplace=True)
    
    return df

def build_preprocessing_pipeline():
    """
    Builds and returns the full scikit-learn preprocessing pipeline.
    """
    # Configure scikit-learn to output pandas DataFrames for all transformers
    sklearn.set_config(transform_output="pandas")
    
    # 1. Numerical pipeline
    # The notebook used SimpleImputer for numerical values. We use 'median' here for robustness.
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 2. Categorical ordinal pipeline
    cat_ord_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=EDUCATION_RANK))
    ])
    
    # 3. Categorical binary pipeline
    cat_bin_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder())
    ])
    
    # 4. Categorical one-hot pipeline
    cat_one_hot_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine everything using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_COLS),
            ('cat_ord', cat_ord_pipeline, CAT_COLS_ORD),
            ('cat_bin', cat_bin_pipeline, CAT_COLS_BIN),
            ('cat_one_hot', cat_one_hot_pipeline, CAT_COLS_ONE_HOT)
        ],
        remainder='drop'
    )
    
    # Full pipeline with ColumnDropper to avoid perfect multicollinearity
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('dropper', ColumnDropper(columns_to_drop=COLS_TO_DROP))
    ])
    
    return full_pipeline
