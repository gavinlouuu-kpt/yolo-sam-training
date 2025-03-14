"""Example script demonstrating how to train/fine-tune SAM model."""

import logging
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import os
import mlflow
from mlflow.tracking import MlflowClient
from tqdm.auto import tqdm
import time
from datetime import datetime
from dotenv import load_dotenv
import yaml
import tempfile

from yolo_sam_training.data import (
    load_dataset_from_summary,
    process_dataset_with_sam,
    split_dataset,
    create_dataloaders
)
from yolo_sam_training.sam_training import train_sam_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingProgressCallback:
    def __init__(self, num_epochs, mlflow_run=None):
        self.num_epochs = num_epochs
        self.mlflow_run = mlflow_run
        self.epoch_pbar = tqdm(total=num_epochs, desc='Epochs', position=0)
        self.batch_pbar = None
        self.start_time = time.time()
        self.metrics = {}
        self.best_metrics = {}
        self.current_epoch = 0
        
    def on_train_start(self, num_batches):
        """Called when training starts."""
        self.batch_pbar = tqdm(total=num_batches, desc='Batches', position=1, leave=False)
        logger.info(f"\nStarting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    def on_epoch_start(self, epoch):
        """Called at the start of each epoch."""
        self.current_epoch = epoch
        self.epoch_pbar.set_description(f'Epoch {epoch}/{self.num_epochs}')
        if self.batch_pbar:
            self.batch_pbar.reset()
            
    def on_batch_end(self, batch_metrics):
        """Called at the end of each batch."""
        if self.batch_pbar:
            self.batch_pbar.update(1)
            # Update batch progress bar with current metrics
            desc = ' '.join(f"{k}: {v:.4f}" for k, v in batch_metrics.items())
            self.batch_pbar.set_description(desc)
            
            # Log batch metrics to MLflow
            if self.mlflow_run:
                step = (self.current_epoch - 1) * self.batch_pbar.total + self.batch_pbar.n
                for key, value in batch_metrics.items():
                    # Convert tensor to float if needed
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    mlflow.log_metric(f"batch_{key}", value, step=step)
            
    def on_epoch_end(self, epoch_metrics):
        """Called at the end of each epoch."""
        self.epoch_pbar.update(1)
        
        # Convert any tensor metrics to Python native types
        converted_metrics = {}
        for key, value in epoch_metrics.items():
            if isinstance(value, torch.Tensor):
                converted_metrics[key] = value.item()
            else:
                converted_metrics[key] = value
                
        self.metrics = converted_metrics
        
        # Update best metrics
        for key, value in converted_metrics.items():
            if key not in self.best_metrics or value < self.best_metrics[key]:
                self.best_metrics[key] = value
        
        # Calculate ETA
        elapsed_time = time.time() - self.start_time
        epochs_remaining = self.num_epochs - self.epoch_pbar.n
        eta = elapsed_time / self.epoch_pbar.n * epochs_remaining if self.epoch_pbar.n > 0 else 0
        
        # Format metrics for display
        metrics_str = ' '.join(f"{k}: {v:.4f}" for k, v in converted_metrics.items())
        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
        logger.info(f"Epoch {self.epoch_pbar.n}/{self.num_epochs} - {metrics_str} - ETA: {eta_str}")
        
        # Log metrics to MLflow if available
        if self.mlflow_run:
            # Log epoch metrics
            for key, value in converted_metrics.items():
                mlflow.log_metric(f"epoch_{key}", value, step=self.epoch_pbar.n)
            
            # Log best metrics so far
            for key, value in self.best_metrics.items():
                mlflow.log_metric(f"best_{key}", value, step=self.epoch_pbar.n)
            
            # Log speed metrics
            speed = elapsed_time / self.epoch_pbar.n
            mlflow.log_metric("epoch_time", speed, step=self.epoch_pbar.n)
            mlflow.log_metric("eta_seconds", eta, step=self.epoch_pbar.n)
    
    def on_train_end(self):
        """Called when training ends."""
        if self.batch_pbar:
            self.batch_pbar.close()
        self.epoch_pbar.close()
        
        total_time = time.time() - self.start_time
        time_str = time.strftime('%H:%M:%S', time.gmtime(total_time))
        logger.info(f"\nTraining completed in {time_str}")
        logger.info("Final metrics:")
        for key, value in self.metrics.items():
            logger.info(f"{key}: {value:.4f}")
        
        # Log final metrics to MLflow
        if self.mlflow_run:
            mlflow.log_metric("total_training_time", total_time)
            
            # Log best metrics as parameters for easy comparison
            for key, value in self.best_metrics.items():
                mlflow.log_param(f"best_{key}", f"{value:.4f}")

def load_env_config():
    """Load environment configuration from .env file."""
    # Try to find .env file in parent directories
    env_path = Path(__file__).resolve()
    for _ in range(5):  # Look up to 5 directories up
        env_file = env_path.parent / '.env'
        if env_file.exists():
            load_dotenv(env_file)
            logger.info(f"Loaded environment from: {env_file}")
            return
        env_path = env_path.parent
    
    # If no .env file found, try loading from current directory
    load_dotenv()
    logger.warning("No .env file found in parent directories, using default values")

def setup_mlflow_tracking():
    """Setup MLflow tracking for remote use."""
    # Use MLflow URI from .env
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if not tracking_uri:
        logger.error("MLFLOW_TRACKING_URI not set in environment or .env file")
        logger.error("Please create a .env file with MLFLOW_TRACKING_URI=<your_mlflow_server_uri>")
        raise ValueError("MLFLOW_TRACKING_URI environment variable is required")
    
    logger.info(f"Using MLflow tracking at {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)

def main():
    # Load environment configuration
    load_env_config()
    
    # Check for CUDA
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Training will be slow on CPU!")
    else:
        logger.info(f"Found CUDA device: {torch.cuda.get_device_name(0)}")
        
    # Set default device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup MLflow tracking
    setup_mlflow_tracking()
    mlflow.set_experiment("sam_training")
    
    # Get data directory from environment
    data_dir = os.getenv('TRAINING_DATA_DIR')
    if not data_dir:
        logger.error("TRAINING_DATA_DIR not set in environment or .env file")
        logger.error("Please create a .env file with TRAINING_DATA_DIR=path/to/data")
        return
        
    summary_path = Path(data_dir) / 'example_training_data' / 'summary.json'
    
    # Check if the file exists
    if not summary_path.exists():
        logger.error(f"Summary file not found at: {summary_path}")
        logger.error(f"Please check if {data_dir} contains the correct data")
        return
    
    logger.info(f"Loading dataset from: {summary_path}")
    dataset = load_dataset_from_summary(summary_path)
    
    processed_dataset = process_dataset_with_sam(
        dataset=dataset,
        save_visualizations=False
    )
    
    # Split dataset
    train_dataset, test_dataset = split_dataset(
        dataset=processed_dataset,
        test_size=0.2,
        stratify_by_box_count=True
    )
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=4,
        num_workers=4 if device == "cuda" else 0  # Disable workers on CPU
    )
    
    # Train model with MLflow tracking
    try:
        with mlflow.start_run(run_name="sam_training") as run:
            # Prepare training parameters
            training_params = {
                'num_epochs': 100,  # Reduced from 200
                'learning_rate': 1e-5,
                'batch_size': 4,
                'device': device,
                'test_size': 0.2,
                'num_workers': 4 if device == "cuda" else 0,
                'dataset_size': len(dataset),
                'train_size': len(train_dataset),
                'val_size': len(test_dataset),
                'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                'torch_version': torch.__version__,
                'early_stopping_patience': 15,
                'early_stopping_min_delta': 0.0001
            }
            
            # Log parameters
            mlflow.log_params(training_params)
            
            # Save parameters to YAML and log directly to MLflow
            with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as temp_file:
                temp_path = temp_file.name
                with open(temp_path, 'w') as f:
                    yaml.dump(training_params, f)
                mlflow.log_artifact(temp_path, "config")
                os.unlink(temp_path)
            
            # Create progress callback
            progress_callback = TrainingProgressCallback(
                num_epochs=training_params['num_epochs'],
                mlflow_run=run
            )
            
            # Train the model
            best_model_state, loss_plot = train_sam_model(
                train_loader=train_loader,
                val_loader=test_loader,
                model_save_path=None,  # No local model saving
                visualization_dir=None,  # No local visualization directory
                num_epochs=training_params['num_epochs'],
                learning_rate=training_params['learning_rate'],
                device=device,
                progress_callback=progress_callback,
                mlflow_run=run,  # Pass MLflow run to train_sam_model
                early_stopping_patience=training_params['early_stopping_patience'],
                early_stopping_min_delta=training_params['early_stopping_min_delta']
            )
            
            # Register the model in MLflow Model Registry
            model_uri = f"runs:/{run.info.run_id}/pytorch_model"
            registered_model_name = "sam_segmentation"
            
            # Register model with appropriate tags based on metrics
            try:
                registered_model = mlflow.register_model(
                    model_uri=model_uri,
                    name=registered_model_name
                )
                
                # Add tags and description to the registered model version
                client = MlflowClient()
                
                # Create description with training details
                description = (
                    f"SAM model fine-tuned on custom dataset. "
                    f"Training parameters: {training_params['num_epochs']} epochs, "
                    f"LR: {training_params['learning_rate']}, "
                    f"Batch size: {training_params['batch_size']}, "
                    f"Dataset size: {training_params['dataset_size']} images "
                    f"({training_params['train_size']} train, {training_params['val_size']} validation)"
                )
                
                client.update_model_version(
                    name=registered_model_name,
                    version=registered_model.version,
                    description=description
                )
                
                # Add tags with key metrics and training parameters
                # Extract best metrics from progress callback
                best_metrics = progress_callback.best_metrics
                
                tags = {
                    "num_epochs": str(training_params['num_epochs']),
                    "learning_rate": str(training_params['learning_rate']),
                    "batch_size": str(training_params['batch_size']),
                    "device": training_params['device'],
                    "dataset_size": str(training_params['dataset_size']),
                    "train_size": str(training_params['train_size']),
                    "val_size": str(training_params['val_size'])
                }
                
                # Add best metrics to tags if available
                for key, value in best_metrics.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    tags[f"best_{key}"] = f"{value:.4f}"
                
                # Add final metrics to tags
                for key, value in progress_callback.metrics.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    tags[f"final_{key}"] = f"{value:.4f}"
                
                for key, value in tags.items():
                    # Ensure value is a string
                    value = str(value)
                    client.set_model_version_tag(
                        name=registered_model_name,
                        version=registered_model.version,
                        key=key,
                        value=value
                    )
                
                logger.info(f"Model registered in MLflow Model Registry with name: {registered_model_name}, version: {registered_model.version}")
            
            except Exception as e:
                logger.warning(f"Error registering model: {str(e)}")
                logger.warning("Training completed successfully, but model registration failed")
            
            logger.info(f"MLflow run ID: {run.info.run_id}")
            
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return

if __name__ == '__main__':
    main() 