"""Example script demonstrating how to load and visualize the dataset."""

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from yolo_sam_training.data import load_dataset_from_summary, yolo_to_pixel_coords

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_sample(image: np.ndarray, mask: np.ndarray, boxes: list, title: str = None):
    """Visualize a single sample with its mask and boxes."""
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original image with boxes
    ax1.imshow(image)
    height, width = image.shape[:2]
    
    # Convert and draw boxes
    for box in boxes:
        pixel_box = yolo_to_pixel_coords(box, width, height)
        x1, y1, x2, y2 = pixel_box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, edgecolor='red', linewidth=2)
        ax1.add_patch(rect)
    
    ax1.set_title('Image with Boxes')
    ax1.axis('off')
    
    # Plot mask
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Mask')
    ax2.axis('off')
    
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    
def main():
    # Path to the summary file
    summary_path = Path('D:/code/ai_cytometry/data/example_training_data/summary.json')
    
    # Load the dataset
    logger.info("Loading dataset...")
    dataset = load_dataset_from_summary(summary_path)
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Display some statistics
    logger.info("\nDataset statistics:")
    logger.info(f"Number of samples: {len(dataset)}")
    
    # Get a sample task_id
    sample_task_id = list(dataset.keys())[0]
    sample = dataset[sample_task_id]
    
    logger.info(f"\nSample task ID: {sample_task_id}")
    logger.info(f"Image shape: {sample['image'].shape}")
    logger.info(f"Mask shape: {sample['mask'].shape}")
    logger.info(f"Number of boxes: {len(sample['boxes'])}")
    
    # Visualize the sample
    visualize_sample(
        sample['image'], 
        sample['mask'], 
        sample['boxes'],
        f'Sample Task ID: {sample_task_id}'
    )
    plt.show()

if __name__ == '__main__':
    main() 