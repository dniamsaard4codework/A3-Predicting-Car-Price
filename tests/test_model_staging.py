# Direct imports using the conftest.py setup
import LoadA3model
import importlib
import pytest
import pandas as pd
import numpy as np

# Reload LoadA3model to get the a3_model with proper setup
importlib.reload(LoadA3model)

# Try to import a2_preprocessor, but handle if it fails
try:
    import app
    a2_preprocessor = getattr(app, 'a2_preprocessor', None)
except ImportError:
    a2_preprocessor = None

def test_load_model():
    a3_model = LoadA3model.a3_model
    assert a3_model is not None

@pytest.mark.depends(on=["test_load_model"])
def test_model_input():
    a3_model = LoadA3model.a3_model
    # Create synthetic data matching your car price structure
    synthetic_data = {
        'year': [2014],
        'km_driven': [50000],
        'fuel': ['Diesel'],
        'transmission': ['Manual'],
        'owner': [1],
        'engine': [1500],
        'max_power': [100],
        'brand': ['Maruti'],
        'mileage': [18.0]
    }
    df = pd.DataFrame(synthetic_data)
    df_transformed = a2_preprocessor.transform(df)
    pred = a3_model.predict(df_transformed)
    assert pred is not None

@pytest.mark.depends(on=["test_model_input"])
def test_model_output():
    a3_model = LoadA3model.a3_model
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
    pred = a3_model.predict(df_transformed)
    assert pred.shape == (2,), f"Expected shape (2,), but got {pred.shape}"

@pytest.mark.depends(on=["test_load_model"])
def test_model_properties():
    a3_model = LoadA3model.a3_model
    # Test that model has expected attributes (adjust based on your model type)
    assert hasattr(a3_model, 'predict'), "Model should have predict method"
    # Add more model-specific property tests if needed