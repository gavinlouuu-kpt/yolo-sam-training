"""Example script demonstrating how to train/fine-tune SAM model."""

import logging
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from yolo_sam_training.data import (
    load_dataset_from_summary,
    process_dataset_with_sam,
    split_dataset,
    create_dataloaders
)
from yolo_sam_training.training import train_sam_model

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load and preprocess dataset
    summary_path = Path('D:/code/ai_cytometry/data/example_training_data/summary.json')
    logger.info("Loading dataset...")
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
    
    # Train model
    model_save_path = Path('models/sam_fine_tuned')
    visualization_dir = Path('visualization_output/training')
    
    best_model_state, loss_plot = train_sam_model(
        train_loader=train_loader,
        val_loader=test_loader,
        model_save_path=model_save_path,
        visualization_dir=visualization_dir,
        num_epochs=10,
        learning_rate=1e-5,
        device=device
    )
    
    # Save loss plot
    loss_plot.savefig(visualization_dir / 'loss_plot.png')
    plt.close(loss_plot)
    
    logger.info("Training complete!")
    logger.info(f"Model saved to: {model_save_path}")
    logger.info(f"Visualizations saved to: {visualization_dir}")

if __name__ == '__main__':
    main() 