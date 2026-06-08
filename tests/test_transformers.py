import pytest
import pandas as pd
import numpy as np
from src.custom_transformers import SqrtTransformer, ColumnDropper

def test_sqrt_transformer():
    """
    Test that the SqrtTransformer correctly applies sqrt and clips negative values.
    """
    transformer = SqrtTransformer()
    df = pd.DataFrame({'A': [1, 4, 9, -4]})
    result = transformer.transform(df)
    
    assert result['A'][0] == 1.0
    assert result['A'][1] == 2.0
    assert result['A'][2] == 3.0
    assert result['A'][3] == 0.0  # clipped to 0 before sqrt

def test_column_dropper():
    """
    Test that the ColumnDropper correctly drops specified columns.
    """
    transformer = ColumnDropper(columns_to_drop=['B', 'C'])
    df = pd.DataFrame({'A': [1], 'B': [2], 'C': [3], 'D': [4]})
    result = transformer.transform(df)
    
    assert 'A' in result.columns
    assert 'B' not in result.columns
    assert 'C' not in result.columns
    assert 'D' in result.columns
    assert list(result.columns) == ['A', 'D']
    
def test_column_dropper_missing_col():
    """
    Test that the ColumnDropper handles cases where the column is not in the DataFrame.
    """
    transformer = ColumnDropper(columns_to_drop=['B', 'X'])
    df = pd.DataFrame({'A': [1], 'B': [2]})
    result = transformer.transform(df)
    
    assert 'A' in result.columns
    assert 'B' not in result.columns
