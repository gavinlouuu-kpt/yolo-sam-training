"""Data loading and processing functionality."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm

import numpy as np
from PIL import Image
import torch
from transformers import SamProcessor

import cv2

logger = logging.getLogger(__name__)

def load_image(image_path: Path) -> np.ndarray:
    """Load image from disk.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array in RGB format
    """
    if not image_path.exists():
        raise FileNotFoundError(f'Image file not found: {image_path}')
        
    # Read image using PIL to ensure proper RGB format
    image = Image.open(image_path)
    image = np.array(image)
    
    # Convert grayscale to RGB if needed
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # Convert RGBA to RGB
        image = image[..., :3]
    
    return image

def load_mask(mask_path: Path) -> np.ndarray:
    """Load mask from disk.
    
    Args:
        mask_path: Path to the mask file
        
    Returns:
        Binary mask as numpy array with float32 dtype
    """
    if not mask_path.exists():
        raise FileNotFoundError(f'Mask file not found: {mask_path}')
        
    # Read mask using PIL
    mask = Image.open(mask_path)
    mask = np.array(mask)
    
    # Convert to binary
    mask = (mask > 0).astype(np.float32)
    return mask

def load_boxes(box_path: Path) -> List[List[float]]:
    """Load bounding boxes in YOLO format from disk.
    
    Args:
        box_path: Path to the text file containing YOLO format boxes
        
    Returns:
        List of boxes, each box is [x_center, y_center, width, height]
    """
    boxes = []
    
    if not box_path.exists():
        logger.warning(f'Box file not found: {box_path}')
        return boxes
        
    with open(box_path, 'r') as f:
        for line in f:
            # YOLO format: class x_center y_center width height
            values = line.strip().split()
            if len(values) != 5:
                logger.warning(f'Invalid box format in {box_path}: {line}')
                continue
                
            try:
                # Extract box coordinates (skip class index)
                box = [float(x) for x in values[1:]]
                boxes.append(box)
            except ValueError as e:
                logger.warning(f'Failed to parse box in {box_path}: {line}')
                continue
                
    return boxes

def yolo_to_pixel_coords(box: List[float], image_width: int, image_height: int) -> List[float]:
    """Convert YOLO format box to pixel coordinates.
    
    Args:
        box: Box in YOLO format [x_center, y_center, width, height]
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        
    Returns:
        Box in pixel coordinates [x1, y1, x2, y2]
    """
    x_center, y_center, width, height = box
    
    # Convert normalized coordinates to pixel values
    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height
    
    # Convert center coordinates to top-left and bottom-right
    x1 = x_center - width/2
    y1 = y_center - height/2
    x2 = x_center + width/2
    y2 = y_center + height/2
    
    # Clip to image boundaries
    x1 = max(0, min(x1, image_width))
    y1 = max(0, min(y1, image_height))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))
    
    return [x1, y1, x2, y2]

def prepare_sam_data(image: np.ndarray, boxes: List[List[float]], mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Prepare data for SAM model input.
    
    Args:
        image: RGB image as numpy array
        boxes: List of boxes in YOLO format [x_center, y_center, width, height]
        mask: Optional ground truth mask
        
    Returns:
        Dictionary containing processed data ready for SAM:
            - image: RGB image as numpy array
            - input_boxes: List of boxes in pixel coordinates [x1, y1, x2, y2]
            - original_size: Original image dimensions
            - ground_truth_mask: Processed mask if provided
    """
    # Ensure image is RGB
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        image = image[..., :3]
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Convert boxes to pixel coordinates
    input_boxes = [yolo_to_pixel_coords(box, width, height) for box in boxes]
    
    # Prepare output dictionary
    output = {
        'image': image,
        'input_boxes': input_boxes,
        'original_size': (height, width)
    }
    
    # Add mask if provided
    if mask is not None:
        output['ground_truth_mask'] = mask.astype(np.float32)
    
    return output

def load_dataset_from_summary(summary_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load all corresponding files for each task ID from the summary.json.
    
    Args:
        summary_path: Path to the summary.json file
        
    Returns:
        Dictionary with task IDs as keys, each containing:
            - image: np.ndarray of the RGB image
            - mask: np.ndarray of the mask
            - boxes: List of boxes in YOLO format
            - sam_data: Dictionary of SAM-ready data
    """
    import json
    
    # Load summary file
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Get base directory (parent of summary.json)
    base_dir = summary_path.parent
    
    dataset = {}
    for task_id in tqdm(summary['task_ids'], desc='Loading dataset'):
        try:
            # Construct paths
            image_path = base_dir / 'images' / f'{task_id}.png'
            mask_path = base_dir / 'masks' / f'{task_id}.png'
            box_path = base_dir / 'boxes' / f'{task_id}.txt'
            
            # Load files
            image = load_image(image_path)
            mask = load_mask(mask_path)
            boxes = load_boxes(box_path)
            
            # Prepare SAM data
            sam_data = prepare_sam_data(image, boxes, mask)
            
            # Store in dictionary
            dataset[task_id] = {
                'image': image,
                'mask': mask,
                'boxes': boxes,
                'sam_data': sam_data
            }
        except Exception as e:
            logger.warning(f'Failed to load task {task_id}: {str(e)}')
            continue
            
    return dataset
