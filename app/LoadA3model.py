"""
A3 Model Loading Module
This module handles loading the A3 model from MLflow with compatibility patches.
"""

import os
import ssl
import urllib3
import mlflow
import mlflow.artifacts
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import traceback


def load_a3_model():
    """
    Load A3 model from MLflow with compatibility patches.
    
    Returns:
        model: The loaded MLflow model or None if loading fails
    """
    a3_model = None
    
    try:
        # Set MLflow tracking URI and credentials for A3 model
        # Disable SSL warnings for self-signed certificates
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Set environment variables for MLflow
        os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
        os.environ['MLFLOW_TRACKING_PASSWORD'] = 'password'
        os.environ['MLFLOW_TRACKING_URI'] = 'https://mlflow.ml.brain.cs.ait.ac.th/'
        
        print("Attempting to load A3 model from MLflow...")
        print(f"MLflow tracking URI: https://mlflow.ml.brain.cs.ait.ac.th/")
        
        # CRITICAL FIX: Apply monkey patch BEFORE any MLflow calls
        print("Applying MLflow compatibility patch...")
        
        # Get the original function
        original_download_fn = mlflow.artifacts._download_artifact_from_uri
        
        # Create a fixed version that handles the signature mismatch
        def fixed_download_artifact_from_uri(artifact_uri, output_path=None, *args, **kwargs):
            """Fixed version that removes problematic tracking_uri parameter"""
            # Remove the tracking_uri parameter if it exists
            cleaned_kwargs = {k: v for k, v in kwargs.items() if k != 'tracking_uri'}
            
            # Call the original function with cleaned parameters
            try:
                return original_download_fn(artifact_uri, output_path, *args, **cleaned_kwargs)
            except TypeError as e:
                if "tracking_uri" in str(e):
                    # If it still fails, try with minimal parameters
                    return original_download_fn(artifact_uri, output_path)
                else:
                    raise e
        
        # Replace the function globally
        mlflow.artifacts._download_artifact_from_uri = fixed_download_artifact_from_uri
        
        # Also patch it in the download_artifacts function's scope
        mlflow.artifacts._download_artifact_from_uri = fixed_download_artifact_from_uri
        
        print("MLflow compatibility patch applied successfully!")
        
        # Set MLflow tracking URI AFTER patching
        mlflow.set_tracking_uri("https://mlflow.ml.brain.cs.ait.ac.th/")
        
        # Now try to load the model with the patch applied
        try:
            print("Trying direct model loading with patch...")
            a3_model = mlflow.pyfunc.load_model("models:/st126235-a3-model/staging")
            print("A3 Model loaded successfully from MLflow with compatibility patch!")
            
        except Exception as e1:
            print(f"Direct loading still failed: {e1}")
            try:
                print("Trying version-specific loading...")
                a3_model = mlflow.pyfunc.load_model("models:/st126235-a3-model/4")
                print("A3 Model loaded successfully using version number!")
                
            except Exception as e2:
                print(f"Version loading failed: {e2}")
                try:
                    print("Trying run-based loading...")
                    client = MlflowClient()
                    
                    model_versions = client.get_latest_versions("st126235-a3-model", stages=["staging"])
                    if model_versions:
                        run_id = model_versions[0].run_id
                        model_uri = f"runs:/{run_id}/model"
                        print(f"Using run URI: {model_uri}")
                        
                        a3_model = mlflow.pyfunc.load_model(model_uri)
                        print("A3 Model loaded successfully using run URI!")
                    else:
                        raise Exception("No model versions found in staging")
                        
                except Exception as e3:
                    print(f"All loading methods failed: {e3}")
                    raise e1
                
    except ImportError as e:
        print(f"Import error when loading A3 model: {e}")
        a3_model = None
    except mlflow.exceptions.MlflowException as e:
        print(f"MLflow exception when loading A3 model: {e}")
        a3_model = None
    except Exception as e:
        print(f"Failed to load A3 model from MLflow. Error: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()
        a3_model = None
    
    # Add final status message
    if a3_model is not None:
        print("A3 model is ready for predictions!")
    else:
        print("A3 model not available - application will work without A3 predictions")
    
    return a3_model


# Load the model when this module is imported
a3_model = load_a3_model()
