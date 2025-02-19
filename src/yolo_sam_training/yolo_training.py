"""Module containing YOLO model training functionality using Ultralytics."""

import logging
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, Optional, Tuple
import yaml
import shutil
import os
import random

logger = logging.getLogger(__name__)

def prepare_yolo_dataset(
    source_dir: Path,
    output_dir: Path,
    split_ratio: float = 0.2
) -> Path:
    """
    Prepare a YOLO dataset by creating train/val splits and generating yaml config.
    
    Args:
        source_dir: Directory containing 'images' and 'labels' folders
        output_dir: Directory to save the prepared dataset
        split_ratio: Validation split ratio (0-1)
        
    Returns:
        Path to the generated dataset.yaml file
    """
    logger.info(f"Preparing YOLO dataset from: {source_dir}")
    
    # Check source directory structure
    images_dir = source_dir / 'images'
    labels_dir = source_dir / 'labels'
    
    if not images_dir.exists() or not labels_dir.exists():
        raise ValueError(f"Expected 'images' and 'labels' directories in {source_dir}")
    
    # Create output directory structure
    train_img_dir = output_dir / 'images' / 'train'
    val_img_dir = output_dir / 'images' / 'val'
    train_label_dir = output_dir / 'labels' / 'train'
    val_label_dir = output_dir / 'labels' / 'val'
    
    for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    
    if not image_files:
        raise ValueError(f"No images found in {images_dir}")
    
    # Calculate split
    num_val = max(1, int(len(image_files) * split_ratio))  # Ensure at least 1 validation sample
    
    # Randomly shuffle files
    random.shuffle(image_files)
    
    val_files = image_files[:num_val]
    train_files = image_files[num_val:]
    
    logger.info(f"Found {len(image_files)} images, splitting into {len(train_files)} train and {len(val_files)} validation")
    
    # Copy files to train/val directories
    for img_path in train_files:
        # Copy image
        shutil.copy2(img_path, train_img_dir / img_path.name)
        # Copy corresponding label if it exists
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, train_label_dir / label_path.name)
        else:
            logger.warning(f"Label not found for training image: {img_path.name}")
    
    for img_path in val_files:
        # Copy image
        shutil.copy2(img_path, val_img_dir / img_path.name)
        # Copy corresponding label if it exists
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, val_label_dir / label_path.name)
        else:
            logger.warning(f"Label not found for validation image: {img_path.name}")
    
    # Create dataset.yaml
    yaml_path = output_dir / 'dataset.yaml'
    dataset_config = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'bead'}  # Assuming single class 'bead'
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, sort_keys=False)
    
    logger.info(f"Dataset prepared successfully:")
    logger.info(f"- Training images: {len(train_files)}")
    logger.info(f"- Validation images: {len(val_files)}")
    logger.info(f"- Config saved to: {yaml_path}")
    
    return yaml_path

def train_yolo_model(
    yaml_path: Path,
    model_save_path: Path,
    pretrained_model: str | YOLO = 'yolov8n.pt',
    num_epochs: int = 100,
    image_size: int = 640,
    batch_size: int = 16,
    device: Optional[str] = None,
    learning_rate: float = 0.01,
) -> Dict[str, float]:
    """
    Train a YOLO model using Ultralytics on the provided dataset.
    
    Args:
        yaml_path: Path to dataset YAML file
        model_save_path: Path to save the trained model
        pretrained_model: Path/name of pretrained model or pre-initialized YOLO model
        num_epochs: Number of training epochs
        image_size: Input image size
        batch_size: Training batch size
        device: Device to train on (None for auto-selection)
        learning_rate: Initial learning rate
        
    Returns:
        Dictionary containing final training metrics
    """
    # Validate YAML file
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found at: {yaml_path}")
    
    try:
        with open(yaml_path) as f:
            dataset_config = yaml.safe_load(f)
            required_keys = ['train', 'val', 'names']
            if not all(key in dataset_config for key in required_keys):
                raise ValueError(f"Dataset YAML must contain: {required_keys}")
            logger.info(f"Loaded dataset config with {len(dataset_config['names'])} classes")
    except Exception as e:
        logger.error(f"Error loading dataset YAML: {str(e)}")
        raise
    
    # Create save directory
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing YOLO model training...")
    
    # Initialize model
    if isinstance(pretrained_model, str):
        logger.info(f"Using pretrained model: {pretrained_model}")
        model = YOLO(pretrained_model)
    else:
        logger.info("Using pre-initialized YOLO model")
        model = pretrained_model
    
    # Configure training parameters
    training_args = {
        'data': str(yaml_path),
        'epochs': num_epochs,
        'imgsz': image_size,
        'batch': batch_size,
        'lr0': learning_rate,
        'device': device,
        'project': str(model_save_path.parent),
        'name': model_save_path.name,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'Adam',
        'verbose': True,
        'patience': 50,  # Early stopping patience
        'save': True,
        'save_period': -1,  # Save only best and last
        'amp': True,  # Use mixed precision training
        'plots': True,  # Generate training plots
    }
    
    try:
        # Train the model
        results = model.train(**training_args)
        
        # Extract final metrics - using more robust access method
        metrics = {}
        results_dict = results.results_dict
        
        # Map common metric names and provide fallbacks
        metric_mappings = {
            'mAP50': ['metrics/mAP50(B)', 'metrics/mAP50'],
            'mAP50-95': ['metrics/mAP50-95(B)', 'metrics/mAP50-95'],
            'precision': ['metrics/precision(B)', 'metrics/precision'],
            'recall': ['metrics/recall(B)', 'metrics/recall']
        }
        
        for metric_name, possible_keys in metric_mappings.items():
            for key in possible_keys:
                if key in results_dict:
                    metrics[metric_name] = results_dict[key]
                    break
            
        # Add any available loss metrics
        for key in results_dict:
            if 'loss' in key.lower():
                metrics[key] = results_dict[key]
        
        logger.info("Training completed successfully!")
        if 'mAP50' in metrics:
            logger.info(f"Final mAP50: {metrics['mAP50']:.4f}")
        if 'mAP50-95' in metrics:
            logger.info(f"Final mAP50-95: {metrics['mAP50-95']:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def validate_yolo_model(
    model_path: Path,
    data_yaml: Path,
    image_size: int = 640,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Validate a trained YOLO model.
    
    Args:
        model_path: Path to the trained model
        data_yaml: Path to dataset YAML file
        image_size: Input image size
        device: Device to run validation on
        
    Returns:
        Dictionary containing validation metrics
    """
    logger.info(f"Starting validation of model: {model_path}")
    
    model = YOLO(str(model_path))
    
    try:
        # Run validation
        results = model.val(
            data=str(data_yaml),
            imgsz=image_size,
            device=device
        )
        
        metrics = {
            'mAP50': results.results_dict['metrics/mAP50(B)'],
            'mAP50-95': results.results_dict['metrics/mAP50-95(B)'],
            'precision': results.results_dict['metrics/precision(B)'],
            'recall': results.results_dict['metrics/recall(B)']
        }
        
        logger.info("Validation completed successfully")
        logger.info(f"Validation mAP50: {metrics['mAP50']:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        raise 