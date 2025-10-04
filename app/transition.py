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
        
        # Get the latest version in staging
        staging_versions = client.get_latest_versions(model_name, stages=["staging"])
        
        if not staging_versions:
            print("No model found in staging. Nothing to transition.")
            return False
        
        latest_staging = staging_versions[0]
        print(f"Found staging model version: {latest_staging.version}")
        
        # Archive any existing production models
        try:
            production_versions = client.get_latest_versions(model_name, stages=["production"])
            for prod_version in production_versions:
                print(f"Archiving existing production version: {prod_version.version}")
                client.transition_model_version_stage(
                    name=model_name,
                    version=prod_version.version,
                    stage="archived"
                )
        except Exception as e:
            print(f"Note: No existing production models to archive: {e}")
        
        # Transition staging to production
        client.transition_model_version_stage(
            name=model_name,
            version=latest_staging.version,
            stage="production"
        )
        
        print(f"Successfully transitioned model version {latest_staging.version} to production!")
        return True
        
    except Exception as e:
        print(f"Error transitioning model to production: {e}")
        return False

if __name__ == "__main__":
    success = transition_to_production()
    sys.exit(0 if success else 1)