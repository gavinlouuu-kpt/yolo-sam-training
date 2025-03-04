"""Example script demonstrating how to train/fine-tune YOLO model."""

import logging
from pathlib import Path
import torch
import os
import mlflow
from mlflow.tracking import MlflowClient

from yolo_sam_training.yolo_training import (
    prepare_yolo_dataset,
    train_yolo_model,
    validate_yolo_model
)

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
    
    # Create experiment if it doesn't exist
    client = MlflowClient()
    experiment_name = "yolo_training"
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(experiment_name)
            logger.info(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")
        else:
            logger.info(f"Using existing experiment '{experiment_name}' with ID: {experiment.experiment_id}")
            mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.warning(f"Error setting up experiment: {str(e)}")
        # Continue with default experiment if there's an issue

def main():
    # Check for CUDA
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Training will be slow on CPU!")
    else:
        logger.info(f"Found CUDA device: {torch.cuda.get_device_name(0)}")
        
    # Set default device
    device = "0" if torch.cuda.is_available() else "cpu"  # YOLO uses "0" for first GPU
    
    # Setup MLflow with fallback
    setup_mlflow_tracking()
    
    # Get data directory from environment variable or use default
    data_dir = os.getenv('TRAINING_DATA_DIR', '/home/mib-p5-a5000/code/ai-cyto/data')
    source_dir = Path(data_dir) / 'example_yolo_dataset'
    prepared_data_dir = Path(data_dir) / 'prepared_yolo_dataset'
    
    # Prepare dataset
    try:
        yaml_path = prepare_yolo_dataset(
            source_dir=source_dir,
            output_dir=prepared_data_dir,
            split_ratio=0.2
        )
    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        return
    
    # Set paths for model saving
    model_save_path = Path('models/yolo_fine_tuned')
    
    # Train model
    try:
        # Start MLflow run
        experiment_name = "yolo_training"
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="yolo_training") as run:
            # Log training parameters
            training_params = {
                'pretrained_model': 'yolov8n.pt',
                'num_epochs': 200,
                'image_size': 640,
                'batch_size': 200,
                'device': device,
                'learning_rate': 0.001,
                'yaml_path': str(yaml_path),
                'model_save_path': str(model_save_path)
            }
            mlflow.log_params(training_params)
            
            # Train the model
            metrics = train_yolo_model(
                yaml_path=yaml_path,
                model_save_path=model_save_path,
                pretrained_model='yolov8n.pt',
                num_epochs=training_params['num_epochs'],
                image_size=training_params['image_size'],
                batch_size=training_params['batch_size'],
                device=training_params['device'],
                learning_rate=training_params['learning_rate']
            )
            
            # Log training metrics
            mlflow.log_metrics({f"train_{k}": v for k, v in metrics.items()})
            
            logger.info("Training metrics:")
            for metric_name, value in metrics.items():
                logger.info(f"{metric_name}: {value:.4f}")
            
            # Validate the trained model
            validation_metrics = validate_yolo_model(
                model_path=model_save_path / 'weights/best.pt',
                data_yaml=yaml_path,
                image_size=640,
                device=device
            )
            
            # Log validation metrics
            mlflow.log_metrics({f"val_{k}": v for k, v in validation_metrics.items()})
            
            logger.info("\nValidation metrics:")
            for metric_name, value in validation_metrics.items():
                logger.info(f"{metric_name}: {value:.4f}")
            
            # Log model artifacts
            mlflow.log_artifacts(str(model_save_path), "model")
            
    except Exception as e:
        logger.error(f"Error during training/validation: {str(e)}")
        return
    
    logger.info("Training and validation complete!")
    logger.info(f"Model saved to: {model_save_path}")

if __name__ == '__main__':
    main() 