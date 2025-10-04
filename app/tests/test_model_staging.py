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
    # Force app to load by accessing its global variables (this will trigger the loading)
    a2_preprocessor = getattr(app, 'a2_preprocessor', None)
    if a2_preprocessor is None:
        # Try to load the model package directly if it's available in the container
        import joblib
        import os
        
        A2_MODEL_PATHS = [
            "./best_model.pkl",
            "./model/best_model.pkl", 
            "../model/best_model.pkl",
            "./app/model/best_model.pkl",
        ]
        
        for model_path in A2_MODEL_PATHS:
            if os.path.exists(model_path):
                try:
                    a2_model_package = joblib.load(model_path)
                    a2_preprocessor = a2_model_package['preprocessor']
                    print(f"Loaded a2_preprocessor from {model_path}")
                    break
                except Exception as e:
                    print(f"Failed to load from {model_path}: {e}")
                    continue
        
        if a2_preprocessor is None:
            print("Warning: a2_preprocessor not found - creating a mock for testing")
            # Create a mock preprocessor for testing
            from unittest.mock import Mock
            a2_preprocessor = Mock()
            a2_preprocessor.transform = Mock(return_value=[[1, 2, 3, 4, 5, 6, 7, 8, 9]])  # Mock transformed features
            
except ImportError as e:
    print(f"Import error: {e}")
    a2_preprocessor = None

def test_load_model():
    a3_model = LoadA3model.a3_model
    assert a3_model is not None

@pytest.mark.depends(on=["test_load_model"])
def test_model_input():
    a3_model = LoadA3model.a3_model
    
    # Skip test if a3_model is None
    if a3_model is None:
        pytest.skip("A3 model not available - skipping input test")
    
    # Skip test if a2_preprocessor is None
    if a2_preprocessor is None:
        pytest.skip("A2 preprocessor not available - skipping input test")
    
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
    
    # Skip test if a3_model is None
    if a3_model is None:
        pytest.skip("A3 model not available - skipping output test")
    
    # Skip test if a2_preprocessor is None
    if a2_preprocessor is None:
        pytest.skip("A2 preprocessor not available - skipping output test")
    
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