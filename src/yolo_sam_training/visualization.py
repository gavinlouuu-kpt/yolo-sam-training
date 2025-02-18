"""Visualization utilities for displaying samples and results."""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .data import Sample

def plot_sample(sample: Sample, show: bool = True, save_path: Optional[Path] = None) -> None:
    """Plot a sample with its image, mask, and bounding boxes.
    
    Args:
        sample: Sample to visualize
        show: Whether to display the plot
        save_path: Optional path to save the visualization
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    ax1.imshow(sample.image)
    ax1.set_title(f"Image {sample.id}")
    ax1.axis('off')
    
    # Plot mask
    ax2.imshow(sample.mask, cmap='gray')
    ax2.set_title("Mask")
    ax2.axis('off')
    
    # Plot image with bounding boxes
    ax3.imshow(sample.image)
    height, width = sample.image.shape[:2]
    for bbox in sample.bboxes:
        # Convert normalized coordinates to pixel coordinates
        # YOLO format is [x_center, y_center, width, height]
        x_center, y_center, w, h = bbox
        x_center = x_center * width
        y_center = y_center * height
        w = w * width
        h = h * height
        
        # Calculate top-left corner from center
        x1 = int(x_center - w/2)
        y1 = int(y_center - h/2)
        
        # Draw rectangle
        rect = plt.Rectangle((x1, y1), w, h, fill=False, color='red')
        ax3.add_patch(rect)
    ax3.set_title("Bounding Boxes")
    ax3.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    else:
        plt.close() 