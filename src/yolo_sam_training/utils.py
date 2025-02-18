"""Utility functions and data structures."""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image

@dataclass
class Sample:
    """Represents a single training sample."""
    image: np.ndarray
    mask: np.ndarray
    bboxes: List[List[float]]  # List of [x, y, width, height] in normalized coordinates

def visualize_sample(sample: Sample) -> None:
    """Visualize a sample with its mask and bounding boxes.
    
    Args:
        sample: A Sample object containing image, mask, and bounding boxes
    """
    # Convert numpy arrays to PIL Images for visualization
    image = Image.fromarray(sample.image)
    mask = Image.fromarray((sample.mask * 255).astype(np.uint8))
    
    # Create a figure with subplots for image, mask, and overlay
    # TODO: Implement visualization using matplotlib
    # For now, just save the images
    image.save("sample_image.png")
    mask.save("sample_mask.png") 