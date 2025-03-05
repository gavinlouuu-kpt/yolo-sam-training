"""Script to register the most recently completed SAM model run."""

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

def get_latest_run_id(experiment_name="sam_training"):
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
        
        # Get metrics and parameters from the run
        client = MlflowClient()
        run = client.get_run(run_id)
        metrics = run.data.metrics
        params = run.data.params
        
        # Helper function to format metric values
        def format_metric(metric_name):
            value = metrics.get(metric_name)
            if value is None:
                return "N/A"
            if isinstance(value, (int, float)):
                return f"{value:.4f}"
            return str(value)
        
        # Create description with training details
        description = (
            f"SAM model fine-tuned on custom dataset. "
            f"Training parameters: {params.get('num_epochs', 'N/A')} epochs, "
            f"LR: {params.get('learning_rate', 'N/A')}, "
            f"Batch size: {params.get('batch_size', 'N/A')}, "
            f"Dataset size: {params.get('dataset_size', 'N/A')} images "
            f"({params.get('train_size', 'N/A')} train, {params.get('val_size', 'N/A')} validation)"
        )
        
        # Extract best metrics from run
        best_metrics = {k.replace('best_', ''): v for k, v in metrics.items() if k.startswith('best_')}
        final_metrics = {k.replace('epoch_', ''): v for k, v in metrics.items() if k.startswith('epoch_')}
        
        # Create tags dictionary
        tags = {
            "num_epochs": params.get('num_epochs', 'N/A'),
            "learning_rate": params.get('learning_rate', 'N/A'),
            "batch_size": params.get('batch_size', 'N/A'),
            "device": params.get('device', 'N/A'),
            "dataset_size": params.get('dataset_size', 'N/A'),
            "train_size": params.get('train_size', 'N/A'),
            "val_size": params.get('val_size', 'N/A')
        }
        
        # Add best metrics to tags
        for key, value in best_metrics.items():
            tags[f"best_{key}"] = format_metric(f"best_{key}")
        
        # Add final metrics to tags
        for key, value in final_metrics.items():
            if key in best_metrics:  # Only add metrics that also have best values
                tags[f"final_{key}"] = format_metric(f"epoch_{key}")
        
        # Register the model
        register_existing_model(
            run_id=run_id,
            model_name="sam_segmentation",
            model_path="model",
            description=description,
            tags=tags
        )
        
        logger.info("Model registered successfully!")
        
    except Exception as e:
        logger.error(f"Error registering model: {str(e)}")

if __name__ == "__main__":
    main() 