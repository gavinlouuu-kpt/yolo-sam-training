"""Utility script to manage model versions in the MLflow Model Registry."""

import argparse
import logging
import os
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import model_version_status

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

def list_model_versions(model_name, tracking_uri=None):
    """List all versions of a registered model.
    
    Args:
        model_name (str): Name of the registered model
        tracking_uri (str, optional): MLflow tracking server URI
    """
    setup_mlflow_tracking(tracking_uri)
    client = MlflowClient()
    
    try:
        # Get all versions of the model
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            logger.info(f"No versions found for model '{model_name}'")
            return
        
        # Print header
        logger.info(f"\nVersions for model '{model_name}':")
        logger.info(f"{'Version':<8} {'Status':<12} {'Stage':<12} {'Created':<20} {'Description'}")
        logger.info("-" * 80)
        
        # Print each version
        for version in versions:
            logger.info(f"{version.version:<8} {version.status:<12} {version.current_stage:<12} {version.creation_timestamp:<20} {version.description or 'N/A'}")
        
        logger.info("\n")
    
    except Exception as e:
        logger.error(f"Error listing model versions: {str(e)}")

def transition_model_stage(model_name, version, stage, tracking_uri=None):
    """Transition a model version to a different stage.
    
    Args:
        model_name (str): Name of the registered model
        version (str): Version of the model to transition
        stage (str): Target stage (Staging, Production, Archived, None)
        tracking_uri (str, optional): MLflow tracking server URI
    """
    setup_mlflow_tracking(tracking_uri)
    client = MlflowClient()
    
    try:
        # Check if the model version exists
        model_version = client.get_model_version(model_name, version)
        
        # Transition the model version to the specified stage
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=(stage == "Production")  # Archive existing versions in Production if transitioning to Production
        )
        
        logger.info(f"Model '{model_name}' version {version} transitioned to '{stage}' stage")
        
        # If transitioning to Production, log a message about archiving other versions
        if stage == "Production":
            logger.info(f"All existing versions in Production stage have been archived")
    
    except Exception as e:
        logger.error(f"Error transitioning model version: {str(e)}")

def delete_model_version(model_name, version, tracking_uri=None):
    """Delete a specific version of a registered model.
    
    Args:
        model_name (str): Name of the registered model
        version (str): Version of the model to delete
        tracking_uri (str, optional): MLflow tracking server URI
    """
    setup_mlflow_tracking(tracking_uri)
    client = MlflowClient()
    
    try:
        # Delete the model version
        client.delete_model_version(
            name=model_name,
            version=version
        )
        
        logger.info(f"Model '{model_name}' version {version} deleted successfully")
    
    except Exception as e:
        logger.error(f"Error deleting model version: {str(e)}")

def compare_model_versions(model_name, versions, tracking_uri=None):
    """Compare metrics between different versions of a model.
    
    Args:
        model_name (str): Name of the registered model
        versions (list): List of versions to compare
        tracking_uri (str, optional): MLflow tracking server URI
    """
    setup_mlflow_tracking(tracking_uri)
    client = MlflowClient()
    
    try:
        # Get all specified versions
        model_versions = []
        for version in versions:
            try:
                model_version = client.get_model_version(model_name, version)
                model_versions.append(model_version)
            except Exception:
                logger.warning(f"Version {version} not found for model '{model_name}'")
        
        if not model_versions:
            logger.info(f"No valid versions found for model '{model_name}'")
            return
        
        # Collect tags for each version
        version_tags = {}
        for model_version in model_versions:
            tags = client.get_model_version_tags(model_name, model_version.version)
            version_tags[model_version.version] = tags
        
        # Find common metrics to compare
        common_metrics = set()
        for version, tags in version_tags.items():
            metrics = [tag for tag in tags.keys() if tag in ['mAP50', 'mAP50-95', 'precision', 'recall', 'val_mAP50', 'val_mAP50-95']]
            common_metrics.update(metrics)
        
        # Print comparison table
        logger.info(f"\nComparison of model '{model_name}' versions:")
        
        # Print header
        header = "Metric".ljust(15)
        for version in sorted([v.version for v in model_versions]):
            header += f"Version {version}".ljust(15)
        logger.info(header)
        logger.info("-" * (15 + 15 * len(model_versions)))
        
        # Print metrics
        for metric in sorted(common_metrics):
            row = metric.ljust(15)
            for version in sorted([v.version for v in model_versions]):
                value = version_tags.get(version, {}).get(metric, "N/A")
                row += value.ljust(15)
            logger.info(row)
        
        logger.info("\n")
    
    except Exception as e:
        logger.error(f"Error comparing model versions: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Manage model versions in MLflow Model Registry")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List model versions
    list_parser = subparsers.add_parser("list", help="List all versions of a registered model")
    list_parser.add_argument("--model-name", required=True, help="Name of the registered model")
    list_parser.add_argument("--tracking-uri", help="MLflow tracking server URI")
    
    # Transition model stage
    transition_parser = subparsers.add_parser("transition", help="Transition a model version to a different stage")
    transition_parser.add_argument("--model-name", required=True, help="Name of the registered model")
    transition_parser.add_argument("--version", required=True, help="Version of the model to transition")
    transition_parser.add_argument("--stage", required=True, choices=["Staging", "Production", "Archived", "None"], 
                                  help="Target stage")
    transition_parser.add_argument("--tracking-uri", help="MLflow tracking server URI")
    
    # Delete model version
    delete_parser = subparsers.add_parser("delete", help="Delete a specific version of a registered model")
    delete_parser.add_argument("--model-name", required=True, help="Name of the registered model")
    delete_parser.add_argument("--version", required=True, help="Version of the model to delete")
    delete_parser.add_argument("--tracking-uri", help="MLflow tracking server URI")
    
    # Compare model versions
    compare_parser = subparsers.add_parser("compare", help="Compare metrics between different versions of a model")
    compare_parser.add_argument("--model-name", required=True, help="Name of the registered model")
    compare_parser.add_argument("--versions", required=True, nargs="+", help="List of versions to compare")
    compare_parser.add_argument("--tracking-uri", help="MLflow tracking server URI")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_model_versions(args.model_name, args.tracking_uri)
    elif args.command == "transition":
        transition_model_stage(args.model_name, args.version, args.stage, args.tracking_uri)
    elif args.command == "delete":
        delete_model_version(args.model_name, args.version, args.tracking_uri)
    elif args.command == "compare":
        compare_model_versions(args.model_name, args.versions, args.tracking_uri)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 