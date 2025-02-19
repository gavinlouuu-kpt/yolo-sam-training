"""Example script demonstrating how to preprocess data for SAM model."""

import logging
from pathlib import Path
import torch
import os

from yolo_sam_training.data import (
    load_dataset_from_summary,
    process_dataset_with_sam,
    split_dataset,
    create_dataloaders
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Get data directory from environment variable or use default
    data_dir = os.getenv('TRAINING_DATA_DIR', '/Users/kpt/Code/data')
    summary_path = Path(data_dir) / 'example_training_data' / 'summary.json'
    
    # Check if the file exists
    if not summary_path.exists():
        logger.error(f"Summary file not found at: {summary_path}")
        logger.error("Please set TRAINING_DATA_DIR environment variable to point to your data directory")
        return
    
    logger.info(f"Loading dataset from: {summary_path}")
    dataset = load_dataset_from_summary(summary_path)
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Display initial dataset statistics
    total_masks = sum(len(sample['masks']) for sample in dataset.values())
    avg_masks = total_masks / len(dataset) if dataset else 0
    logger.info(f"Total masks in dataset: {total_masks}")
    logger.info(f"Average masks per image: {avg_masks:.2f}")
    
    # Process all samples
    output_dir = Path('visualization_output')
    
    try:
        # Process with visualizations disabled by default for speed
        processed_dataset = process_dataset_with_sam(
            dataset=dataset,
            output_dir=output_dir,
            save_visualizations=False,  # Set to True if visualizations are needed
            batch_size=4  # Adjust based on available memory
        )
        
        # Split into train and test sets
        train_dataset, test_dataset = split_dataset(
            dataset=processed_dataset,
            test_size=0.2,  # 20% for testing
            random_seed=42,  # For reproducibility
            stratify_by_box_count=True  # Maintain box count distribution
        )
        
        # Create dataloaders
        train_loader, test_loader = create_dataloaders(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=4,
            num_workers=4,  # Adjust based on CPU cores
            pin_memory=torch.cuda.is_available()  # Use pin_memory if GPU available
        )
        
        # Demonstrate batch iteration
        logger.info("\nDemonstrating batch iteration:")
        
        # Get a sample batch from training loader
        sample_batch = next(iter(train_loader))
        logger.info("\nSample training batch contents:")
        for key, value in sample_batch.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: shape = {value.shape}, dtype = {value.dtype}")
            elif isinstance(value, (list, tuple)):
                logger.info(f"  {key}: length = {len(value)}")
                # Show details of first item if it's a list of tensors
                if len(value) > 0 and isinstance(value[0], torch.Tensor):
                    logger.info(f"    First item shape: {value[0].shape}")
            else:
                logger.info(f"  {key}: type = {type(value)}")
        
        # Example of iterating through batches (just first 2 for demonstration)
        logger.info("\nDemonstrating training loop:")
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 2:  # Only show first 2 batches
                break
                
            # Calculate masks per sample using the first tensor's batch dimension
            batch_size = len(batch['dataset_key'])
            total_boxes = len(batch['input_boxes'])
            masks_per_sample = total_boxes // batch_size
            
            logger.info(f"Batch {batch_idx}:")
            logger.info(f"  Samples in batch: {batch_size}")
            logger.info(f"  Total boxes/masks: {total_boxes}")
            logger.info(f"  Average masks per sample: {masks_per_sample}")
            
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main() 