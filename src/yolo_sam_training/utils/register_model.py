"""Utility script to register an existing trained model with MLflow Model Registry."""

import argparse
import logging
import os
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_mlflow_tracking(tracking_uri=None):
    """Setup MLflow tracking to use remote MLflow server."""
    # Use the provided tracking URI or default to localhost
    if not tracking_uri:
        tracking_uri = "http://localhost:5000"
    
    logger.info(f"Using MLflow tracking server at {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set environment variables for S3 artifact store access if needed
    if "localhost" in tracking_uri:
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
        os.environ["AWS_ACCESS_KEY_ID"] = "mibadmin"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "cuhkminio"

def register_existing_model(
    run_id,
    model_name="yolo_object_detection",
    model_path="model",
    description=None,
    tags=None,
    tracking_uri=None
):
    """Register an existing model from a completed MLflow run.
    
    Args:
        run_id (str): The MLflow run ID containing the model artifacts
        model_name (str): Name to register the model under in Model Registry
        model_path (str): Path within the run artifacts where the model is stored
        description (str, optional): Description for the model version
        tags (dict, optional): Tags to add to the model version
        tracking_uri (str, optional): MLflow tracking server URI
    
    Returns:
        ModelVersion: The registered model version object
    """
    # Setup MLflow tracking
    setup_mlflow_tracking(tracking_uri)
    
    # Create model URI pointing to the model artifacts in the run
    model_uri = f"runs:/{run_id}/{model_path}"
    logger.info(f"Registering model from URI: {model_uri}")
    
    # Register the model
    try:
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        logger.info(f"Model registered successfully with name: {model_name}, version: {registered_model.version}")
        
        # Add description and tags if provided
        client = MlflowClient()
        
        if description:
            client.update_model_version(
                name=model_name,
                version=registered_model.version,
                description=description
            )
            logger.info(f"Added description to model version {registered_model.version}")
        
        if tags:
            for key, value in tags.items():
                client.set_model_version_tag(
                    name=model_name,
                    version=registered_model.version,
                    key=key,
                    value=str(value)
                )
            logger.info(f"Added {len(tags)} tags to model version {registered_model.version}")
        
        return registered_model
    
    except Exception as e:
        logger.error(f"Error registering model: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Register an existing model with MLflow Model Registry")
    parser.add_argument("--run-id", required=True, help="MLflow run ID containing the model artifacts")
    parser.add_argument("--model-name", default="yolo_object_detection", help="Name to register the model under")
    parser.add_argument("--model-path", default="model", help="Path within run artifacts where model is stored")
    parser.add_argument("--description", help="Description for the model version")
    parser.add_argument("--tracking-uri", help="MLflow tracking server URI")
    parser.add_argument("--tag", action="append", nargs=2, metavar=("KEY", "VALUE"), 
                        help="Tags to add to the model version (can be used multiple times)")
    
    args = parser.parse_args()
    
    # Convert tags list to dictionary
    tags = {}
    if args.tag:
        for key, value in args.tag:
            tags[key] = value
    
    # Register the model
    register_existing_model(
        run_id=args.run_id,
        model_name=args.model_name,
        model_path=args.model_path,
        description=args.description,
        tags=tags,
        tracking_uri=args.tracking_uri
    )

if __name__ == "__main__":
    main() 