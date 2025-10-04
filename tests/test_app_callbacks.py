# Direct imports using the conftest.py setup
import LoadA3model
import importlib

# Reload LoadA3model to get the a3_model with proper setup
importlib.reload(LoadA3model)
a3_model = LoadA3model.a3_model

from dash._callback_context import context_value
from dash._utils import AttributeDict

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

# Try to import a2_preprocessor, but handle if it fails
try:
    import app
    a2_preprocessor = getattr(app, 'a2_preprocessor', None)
except ImportError:
    a2_preprocessor = None

# Check the expected input shape
def test_a3model_input_shape():
    synthetic_data = {
        'year': [2014, 2015],
        'km_driven': [50000, 30000],
        'fuel': ['Diesel', 'Petrol'],
        'transmission': ['Manual', 'Automatic'],
        'owner': [1, 1],
        'engine': [1500, 1200],
        'max_power': [100, 80],
        'brand': ['Maruti', 'Hyundai'],
        'mileage': [18.0, 20.0]
        }
    df = pd.DataFrame(synthetic_data)
    df = a2_preprocessor.transform(df)
    input_shape = df.shape
    assert input_shape == (2, 38), f"Expected input shape (2, 38), but got {input_shape}"

# Check the output shape
def test_a3model_output_shape():
    # Skip test if a3_model is not available
    if a3_model is None:
        pytest.skip("A3 model not available for testing")
    
    # Create synthetic data
    synthetic_data = {
        'year': [2014, 2015],
        'km_driven': [50000, 30000],
        'fuel': ['Diesel', 'Petrol'],
        'transmission': ['Manual', 'Automatic'],
        'owner': [1, 1],
        'engine': [1500, 1200],
        'max_power': [100, 80],
        'brand': ['Maruti', 'Hyundai'],
        'mileage': [18.0, 20.0]
        }
    df = pd.DataFrame(synthetic_data)
    

    df_transformed = a2_preprocessor.transform(df)
    output = a3_model.predict(df_transformed)
    assert output.shape == (2, ), f"Expected output shape (2,), but got {output.shape}"