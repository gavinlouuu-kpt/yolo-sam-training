"""Example script demonstrating how to load and visualize the dataset."""

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import List

from yolo_sam_training.data import load_dataset_from_summary, yolo_to_pixel_coords

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_sample(image: np.ndarray, masks: List[np.ndarray], boxes: list, title: str = None):
    """Visualize a single sample with its masks and boxes."""
    # Create figure with subplots - one for image with boxes and one for each mask
    n_masks = len(masks)
    fig, axes = plt.subplots(1, n_masks + 1, figsize=(6 * (n_masks + 1), 6))
    
    # If only one subplot, wrap in list for consistent indexing
    if n_masks == 0:
        axes = [axes]
    
    # Plot original image with boxes
    axes[0].imshow(image)
    height, width = image.shape[:2]
    
    # Convert and draw boxes
    for box in boxes:
        pixel_box = yolo_to_pixel_coords(box, width, height)
        x1, y1, x2, y2 = pixel_box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, edgecolor='red', linewidth=2)
        axes[0].add_patch(rect)
    
    axes[0].set_title('Image with Boxes')
    axes[0].axis('off')
    
    # Plot each mask
    for i, mask in enumerate(masks, 1):
        axes[i].imshow(mask, cmap='gray')
        axes[i].set_title(f'Mask {i}')
        axes[i].axis('off')
    
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
    total_masks = sum(len(sample['masks']) for sample in dataset.values())
    avg_masks = total_masks / len(dataset)
    
    logger.info("\nDataset statistics:")
    logger.info(f"Number of samples: {len(dataset)}")
    logger.info(f"Total masks: {total_masks}")
    logger.info(f"Average masks per image: {avg_masks:.2f}")
    
    # Get a sample task_id
    sample_task_id = list(dataset.keys())[0]
    sample = dataset[sample_task_id]
    
    logger.info(f"\nSample task ID: {sample_task_id}")
    logger.info(f"Image shape: {sample['image'].shape}")
    logger.info(f"Number of masks: {len(sample['masks'])}")
    logger.info(f"Number of boxes: {len(sample['boxes'])}")
    
    # Visualize the sample
    visualize_sample(
        sample['image'], 
        sample['masks'], 
        sample['boxes'],
        f'Sample Task ID: {sample_task_id}'
    )
    plt.show()

if __name__ == '__main__':
    main() 