"""Example script demonstrating how to train/fine-tune YOLO model."""

import logging
from pathlib import Path
import torch
import os

from yolo_sam_training.yolo_training import (
    prepare_yolo_dataset,
    train_yolo_model,
    validate_yolo_model
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Check for CUDA
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Training will be slow on CPU!")
    else:
        logger.info(f"Found CUDA device: {torch.cuda.get_device_name(0)}")
        
    # Set default device
    device = "0" if torch.cuda.is_available() else "cpu"  # YOLO uses "0" for first GPU
    
    # Get data directory from environment variable or use default
    data_dir = os.getenv('TRAINING_DATA_DIR', '/Users/kpt/Code/data')
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
        metrics = train_yolo_model(
            yaml_path=yaml_path,
            model_save_path=model_save_path,
            pretrained_model='yolov8n.pt',  # Use nano model
            num_epochs=100,
            image_size=640,
            batch_size=16,
            device=device,
            learning_rate=0.01
        )
        
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
        
        logger.info("\nValidation metrics:")
        for metric_name, value in validation_metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
            
    except Exception as e:
        logger.error(f"Error during training/validation: {str(e)}")
        return
    
    logger.info("Training and validation complete!")
    logger.info(f"Model saved to: {model_save_path}")

if __name__ == '__main__':
    main() 