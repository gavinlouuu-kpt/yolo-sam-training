"""Script to register the most recently completed YOLO model run."""

import logging
import os
import mlflow
from mlflow.tracking import MlflowClient
from yolo_sam_training.utils.register_model import register_existing_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_mlflow_tracking():
    """Setup MLflow tracking to use remote MLflow server."""
    # Use the remote MLflow server running in Docker
    tracking_uri = "http://localhost:5000"
    logger.info(f"Using remote MLflow tracking server at {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set environment variables for S3 artifact store access
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "mibadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "cuhkminio"

def get_latest_run_id(experiment_name="yolo_training"):
    """Get the run ID of the most recently completed run in the specified experiment.
    
    Args:
        experiment_name (str): Name of the MLflow experiment
        
    Returns:
        str: Run ID of the most recent run
    """
    client = MlflowClient()
    
    # Get experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    # Get all runs in the experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")
    
    latest_run = runs[0]
    logger.info(f"Found latest run: {latest_run.info.run_id} (started at {latest_run.info.start_time})")
    
    return latest_run.info.run_id

def main():
    # Setup MLflow tracking
    setup_mlflow_tracking()
    
    try:
        # Get the latest run ID
        run_id = get_latest_run_id()
        
        # Get metrics from the run to use in description
        client = MlflowClient()
        run = client.get_run(run_id)
        metrics = run.data.metrics
        
        # Helper function to format metric values
        def format_metric(metric_name):
            value = metrics.get(metric_name)
            if value is None:
                return "N/A"
            if isinstance(value, (int, float)):
                return f"{value:.4f}"
            return str(value)
        
        # Create description with metrics
        description = (
            f"YOLO model trained on custom dataset. "
            f"Training metrics - mAP50: {format_metric('train_mAP50')}, "
            f"mAP50-95: {format_metric('train_mAP50-95')}. "
            f"Validation metrics - mAP50: {format_metric('val_mAP50')}, "
            f"mAP50-95: {format_metric('val_mAP50-95')}"
        )
        
        # Extract tags from run parameters
        params = run.data.params
        
        tags = {
            "mAP50": format_metric('train_mAP50'),
            "mAP50-95": format_metric('train_mAP50-95'),
            "val_mAP50": format_metric('val_mAP50'),
            "val_mAP50-95": format_metric('val_mAP50-95'),
            "precision": format_metric('train_precision'),
            "recall": format_metric('train_recall'),
            "image_size": params.get('image_size', 'N/A'),
            "batch_size": params.get('batch_size', 'N/A'),
            "epochs": params.get('num_epochs', 'N/A'),
            "learning_rate": params.get('learning_rate', 'N/A'),
            "device": params.get('device', 'N/A')
        }
        
        # Register the model
        register_existing_model(
            run_id=run_id,
            model_name="yolo_object_detection",
            model_path="model",
            description=description,
            tags=tags
        )
        
        logger.info("Model registered successfully!")
        
    except Exception as e:
        logger.error(f"Error registering model: {str(e)}")

if __name__ == "__main__":
    main() 