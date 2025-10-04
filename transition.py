#!/usr/bin/env python3
"""
Model Transition Script
This script handles transitioning models from staging to production in MLflow.
"""

import os
import sys
import mlflow
from mlflow.tracking import MlflowClient
import urllib3

def transition_to_production():
    """
    Transition the A3 model from staging to production.
    """
    try:
        # Disable SSL warnings for self-signed certificates
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Set MLflow configuration
        os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
        os.environ['MLFLOW_TRACKING_PASSWORD'] = 'password'
        mlflow.set_tracking_uri("https://mlflow.ml.brain.cs.ait.ac.th/")
        
        client = MlflowClient()
        model_name = "st126235-a3-model"
        
        print(f"Attempting to transition {model_name} from staging to production...")
        
        # Get all model versions and find staging ones
        all_versions = client.search_model_versions(f"name='{model_name}'")
        staging_versions = [v for v in all_versions if hasattr(v, 'current_stage') and v.current_stage == "Staging"]
        
        # Show current model status
        print(f"Found {len(all_versions)} total model versions for {model_name}")
        for version in all_versions:
            stage = getattr(version, 'current_stage', 'None')
            print(f"  Version {version.version}: {stage}")
        
        if not staging_versions:
            print("No model found in staging. Nothing to transition.")
            print("This is normal - either no models are staged or they've already been promoted.")
            return True  # This is a successful scenario, not an error
        
        latest_staging = staging_versions[0]
        print(f"Found staging model version: {latest_staging.version}")
        
        # Archive any existing production models
        try:
            production_versions = [v for v in all_versions if hasattr(v, 'current_stage') and v.current_stage == "Production"]
            for prod_version in production_versions:
                print(f"Archiving existing production version: {prod_version.version}")
                client.transition_model_version_stage(
                    name=model_name,
                    version=prod_version.version,
                    stage="Archived"
                )
        except Exception as e:
            print(f"Note: No existing production models to archive or archiving failed: {e}")
        
        # Transition staging to production
        client.transition_model_version_stage(
            name=model_name,
            version=latest_staging.version,
            stage="Production"
        )
        
        print(f"Successfully transitioned model version {latest_staging.version} to production!")
        return True
        
    except Exception as e:
        print(f"Error transitioning model to production: {e}")
        return False

if __name__ == "__main__":
    success = transition_to_production()
    sys.exit(0 if success else 1)