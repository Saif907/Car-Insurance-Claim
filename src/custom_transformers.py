import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class SqrtTransformer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to apply square root transformation to numerical features.
    Clips values at 0 to avoid negative inputs to sqrt.
    """
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        """Fit method (does nothing, returns self)."""
        return self
        
    def transform(self, X):
        """Apply square root transformation."""
        if isinstance(X, pd.DataFrame):
            X_copy = X.copy()
            # Apply clip and sqrt
            return np.sqrt(np.clip(X_copy, a_min=0, a_max=None))
        elif isinstance(X, np.ndarray):
            return np.sqrt(np.clip(X, a_min=0, a_max=None))
        else:
            raise TypeError("Input must be a pandas DataFrame or numpy ndarray")

    def get_feature_names_out(self, input_features=None):
        """Return the same feature names."""
        return input_features


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to drop specific columns from a pandas DataFrame.
    Useful for removing baseline dummy variables to avoid perfect multicollinearity (dummy variable trap).
    """
    def __init__(self, columns_to_drop=None):
        """
        Args:
            columns_to_drop (list): List of column names to drop.
        """
        if columns_to_drop is None:
            columns_to_drop = []
        self.columns_to_drop = columns_to_drop
        
    def fit(self, X, y=None):
        """Fit method (does nothing, returns self)."""
        return self
        
    def transform(self, X):
        """Drop specified columns."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame. ColumnDropper expects DataFrames to match column names.")
            
        X_copy = X.copy()
        cols_to_drop = [col for col in self.columns_to_drop if col in X_copy.columns]
        return X_copy.drop(columns=cols_to_drop)

    def get_feature_names_out(self, input_features=None):
        """Return the remaining feature names."""
        if input_features is not None:
            return np.array([col for col in input_features if col not in self.columns_to_drop])
        return None
