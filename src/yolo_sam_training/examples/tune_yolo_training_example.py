"""Example script demonstrating how to train/fine-tune YOLO model with hyperparameter optimization."""

import logging
from pathlib import Path
import torch
import os
import optuna
import mlflow
from mlflow.tracking import MlflowClient
import tempfile
import shutil
from ultralytics import YOLO

from yolo_sam_training.yolo_training import (
    prepare_yolo_dataset,
    train_yolo_model,
    validate_yolo_model
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOOptimization:
    def __init__(self, yaml_path: Path, base_save_path: Path, device: str):
        self.yaml_path = yaml_path
        self.base_save_path = base_save_path
        self.device = device
        
        # Create temporary directory for trial models
        self.temp_dir = Path(tempfile.mkdtemp())
        logger.info(f"Created temporary directory for trials: {self.temp_dir}")
        
        # Initialize base YOLO model once
        logger.info("Initializing base YOLO model...")
        self.base_model = YOLO('yolov8n.pt')
    
    def __del__(self):
        # Cleanup temporary directory
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function for hyperparameter optimization."""
        # Suggest hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_int('batch_size', 16, 128, log=True)
        image_size = trial.suggest_categorical('image_size', [416, 512, 640])
        
        # Create trial-specific save directory
        trial_save_path = self.temp_dir / f"trial_{trial.number}"
        
        with mlflow.start_run(nested=True) as run:
            # Log parameters
            mlflow.log_params({
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'image_size': image_size,
                'trial_number': trial.number
            })
            
            try:
                # Train model with suggested parameters
                metrics = train_yolo_model(
                    yaml_path=self.yaml_path,
                    model_save_path=trial_save_path,
                    pretrained_model=self.base_model,  # Use pre-initialized model
                    num_epochs=50,  # Reduced epochs for trials
                    image_size=image_size,
                    batch_size=batch_size,
                    device=self.device,
                    learning_rate=learning_rate
                )
                
                # Log training metrics
                mlflow.log_metrics({f"train_{k}": v for k, v in metrics.items()})
                
                # Get validation metrics
                validation_metrics = validate_yolo_model(
                    model_path=trial_save_path / 'weights/best.pt',
                    data_yaml=self.yaml_path,
                    image_size=image_size,
                    device=self.device
                )
                
                # Log validation metrics
                mlflow.log_metrics({f"val_{k}": v for k, v in validation_metrics.items()})
                
                # Log model artifacts
                mlflow.log_artifacts(str(trial_save_path), "model")
                
                # Return the metric to optimize
                objective_value = validation_metrics.get('mAP50-95', 0.0)
                
                # Report intermediate value for pruning
                trial.report(objective_value, step=1)
                
                return objective_value
                
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {str(e)}")
                mlflow.log_param("error", str(e))
                raise optuna.TrialPruned()

def main():
    # Check for CUDA
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Training will be slow on CPU!")
    else:
        logger.info(f"Found CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Set default device
    device = "0" if torch.cuda.is_available() else "cpu"
    
    # Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("yolo_optimization")
    
    # Get data directory from environment variable or use default
    data_dir = os.getenv('TRAINING_DATA_DIR', 'data')
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
    
    # Create optimization instance
    optimization = YOLOOptimization(yaml_path, model_save_path, device)
    
    # Start the main MLflow run
    with mlflow.start_run(run_name="yolo_optimization") as parent_run:
        # Create study
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=5,
                interval_steps=1
            ),
            study_name="yolo_hyperparameter_optimization"
        )
        
        # Run optimization
        study.optimize(
            optimization.objective,
            n_trials=20,
            timeout=3600 * 8  # 8 hours timeout
        )
        
        # Log the best parameters and results
        mlflow.log_params({
            "best_trial": study.best_trial.number,
            **study.best_params
        })
        mlflow.log_metric("best_value", study.best_value)
        
        # Save optimization plots
        try:
            # Optimization history plot
            history_fig = optuna.visualization.plot_optimization_history(study)
            mlflow.log_figure(history_fig, "optimization_history.png")
            
            # Parameter importance plot
            param_imp_fig = optuna.visualization.plot_param_importances(study)
            mlflow.log_figure(param_imp_fig, "parameter_importances.png")
            
            # Parallel coordinate plot
            parallel_fig = optuna.visualization.plot_parallel_coordinate(study)
            mlflow.log_figure(parallel_fig, "parallel_coordinate.png")
        except Exception as e:
            logger.warning(f"Could not create some visualization plots: {str(e)}")
        
        # Train final model with best parameters
        logger.info("Training final model with best parameters...")
        try:
            final_metrics = train_yolo_model(
                yaml_path=yaml_path,
                model_save_path=model_save_path / "final_model",
                pretrained_model=self.base_model,  # Use pre-initialized model
                num_epochs=200,  # Full training for final model
                device=device,
                **study.best_params
            )
            
            mlflow.log_metrics({f"final_{k}": v for k, v in final_metrics.items()})
            
            # Log final model artifacts
            mlflow.log_artifacts(str(model_save_path / "final_model"), "final_model")
            
        except Exception as e:
            logger.error(f"Error training final model: {str(e)}")
            return
        
        logger.info(f"Best trial number: {study.best_trial.number}")
        logger.info("Best parameters:")
        for param_name, param_value in study.best_params.items():
            logger.info(f"  {param_name}: {param_value}")
        logger.info(f"Best validation mAP50-95: {study.best_value:.4f}")

if __name__ == '__main__':
    main() 