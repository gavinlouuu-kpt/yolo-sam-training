"""Data loading and processing functionality for SAM model training.

This module provides functionality for:
1. Loading and preprocessing image data
2. Dataset splitting and management
3. DataLoader creation with custom collation
4. Batch processing utilities

Key Components:
- Data Loading: Functions to load images, masks, and boxes
- Preprocessing: SAM-specific data preparation
- Dataset Management: Dataset splitting and organization
- DataLoader: Custom batch handling for variable-sized inputs

Potential Issues and TODOs:
1. Multiple Masks:
   - Current implementation assumes one mask per image
   - Need to handle multiple instance masks within same image
   - Consider mask indexing and instance separation

2. Data Augmentation:
   - Should be implemented in SAMDataset.__getitem__
   - Consider: rotation, flipping, color jittering
   - Need to handle box coordinates in augmentations
   - Ensure mask consistency with augmentations

3. Memory Management:
   - Large datasets might need lazy loading
   - Consider implementing data streaming
   - Add memory usage monitoring

4. Error Handling:
   - Add validation for mask-box correspondence
   - Implement data integrity checks
   - Add input validation for all functions
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm

import numpy as np
from PIL import Image
import torch
from transformers import SamProcessor
import matplotlib.pyplot as plt

import cv2

logger = logging.getLogger(__name__)

def load_image(image_path: Path) -> np.ndarray:
    """Load image from disk.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array in RGB format
        
    TODO:
        - Add support for different image formats
        - Implement lazy loading for large images
        - Add image validation and error checking
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

def load_mask(base_path: Path, task_id: str) -> List[np.ndarray]:
    """Load all masks for a given task ID.
    
    Args:
        base_path: Base path to the masks directory
        task_id: Task ID to load masks for
        
    Returns:
        List of binary masks as numpy arrays with float32 dtype
        
    Raises:
        FileNotFoundError: If no masks found for the task ID
    """
    masks = []
    mask_dir = base_path / 'masks'
    
    # Find all mask files for this task ID
    mask_files = sorted(mask_dir.glob(f"{task_id}_*.png"))
    if not mask_files:
        raise FileNotFoundError(f'No mask files found for task {task_id}')
    
    # Load each mask
    for mask_path in mask_files:
        # Read mask using PIL
        mask = Image.open(mask_path)
        mask = np.array(mask)
        
        # Convert to binary
        mask = (mask > 0).astype(np.float32)
        masks.append(mask)
    
    return masks

def load_boxes(box_path: Path) -> List[List[float]]:
    """Load bounding boxes in YOLO format from disk.
    
    Args:
        box_path: Path to the text file containing YOLO format boxes
        
    Returns:
        List of boxes, each box is [x_center, y_center, width, height]
        
    TODO:
        - Validate box-mask correspondence
        - Add support for different box formats
        - Implement box validation
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

def prepare_sam_data(image: np.ndarray, boxes: List[List[float]], masks: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
    """Prepare data for SAM model input.
    
    Args:
        image: RGB image as numpy array
        boxes: List of boxes in YOLO format [x_center, y_center, width, height]
        masks: Optional list of ground truth masks
        
    Returns:
        Dictionary containing processed data ready for SAM:
            - image: Original RGB image
            - input_boxes: List of boxes in pixel coordinates
            - original_size: Tuple of (height, width)
            - ground_truth_masks: List of masks if provided
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
    
    # Add masks if provided
    if masks is not None:
        output['ground_truth_masks'] = [mask.astype(np.float32) for mask in masks]
    
    return output

def load_dataset_from_summary(summary_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load all corresponding files for each task ID from the summary.json.
    
    Args:
        summary_path: Path to the summary.json file
        
    Returns:
        Dictionary with task IDs as keys, each containing:
            - image: np.ndarray of the RGB image
            - masks: List of np.ndarray masks
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
    total_masks = 0
    
    for task_id in tqdm(summary['task_ids'], desc='Loading dataset'):
        try:
            # Construct paths
            image_path = base_dir / 'images' / f'{task_id}.png'
            box_path = base_dir / 'boxes' / f'{task_id}.txt'
            
            # Load files
            image = load_image(image_path)
            masks = load_mask(base_dir, task_id)  # Now returns list of masks
            boxes = load_boxes(box_path)
            
            # Verify we have the same number of boxes as masks
            if len(boxes) != len(masks):
                logger.warning(f'Mismatch in number of boxes ({len(boxes)}) and masks ({len(masks)}) for task {task_id}')
                continue
            
            # Prepare SAM data
            sam_data = prepare_sam_data(image, boxes, masks)
            
            # Store in dictionary
            dataset[task_id] = {
                'image': image,
                'masks': masks,
                'boxes': boxes,
                'sam_data': sam_data
            }
            
            total_masks += len(masks)
            
        except Exception as e:
            logger.warning(f'Failed to load task {task_id}: {str(e)}')
            continue
    
    logger.info(f'Successfully loaded {len(dataset)} images with {total_masks} total masks')
    return dataset

def preprocess_mask_for_sam(mask: np.ndarray) -> Image.Image:
    """Convert numpy mask to PIL Image format required by SAM.
    
    Args:
        mask: Binary mask as numpy array
        
    Returns:
        Mask as PIL Image in correct format for SAM
    """
    # Ensure mask is binary and in correct dtype
    mask_processed = mask.astype(np.float32)
    if mask_processed.max() > 1:
        mask_processed = mask_processed / 255.0
        
    # Convert mask to PIL Image
    mask_2d = mask_processed.squeeze()  # Remove channel dimension if present
    mask_uint8 = (mask_2d * 255).astype(np.uint8)
    return Image.fromarray(mask_uint8)

def preprocess_for_sam(
    image: np.ndarray, 
    boxes: List[List[float]], 
    masks: Optional[List[np.ndarray]] = None,
    processor: Optional[SamProcessor] = None,
    **processor_kwargs
) -> Dict[str, torch.Tensor]:
    """Preprocess data for SAM model input.
    
    Args:
        image: RGB image as numpy array
        boxes: List of boxes in YOLO format [x_center, y_center, width, height]
        masks: Optional list of ground truth masks
        processor: Optional SamProcessor instance. If None, will create new one.
        **processor_kwargs: Additional arguments to pass to the processor
        
    Returns:
        Dictionary containing preprocessed tensors for SAM model
    """
    # Initialize processor if not provided
    if processor is None:
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        
    # Get image dimensions and convert boxes
    height, width = image.shape[:2]
    pixel_boxes = [yolo_to_pixel_coords(box, width, height) for box in boxes]
    
    # Process masks if provided
    mask_pils = None
    if masks is not None:
        mask_pils = [preprocess_mask_for_sam(mask) for mask in masks]
    
    # Set default processor arguments if not provided
    default_kwargs = {
        'return_tensors': "pt",
        'do_resize': True,
        'size': {"longest_edge": 1024},
        'do_normalize': True,
        'do_pad': True
    }
    processor_kwargs = {**default_kwargs, **processor_kwargs}
    
    # Prepare inputs for SAM
    try:
        # Process each box and mask pair
        all_inputs = []
        for i, box in enumerate(pixel_boxes):
            inputs = processor(
                images=image,
                input_boxes=[[box]],  # Process single box
                segmentation_maps=[mask_pils[i]] if mask_pils else None,
                **processor_kwargs
            )
            all_inputs.append(inputs)
        
        # Combine all inputs
        combined_inputs = {
            'pixel_values': torch.cat([inp['pixel_values'] for inp in all_inputs]),
            'input_boxes': torch.cat([inp['input_boxes'] for inp in all_inputs]),
            'original_sizes': torch.cat([inp['original_sizes'] for inp in all_inputs]),
            'reshaped_input_sizes': torch.cat([inp['reshaped_input_sizes'] for inp in all_inputs])
        }
        
        # Add labels if masks were provided
        if mask_pils:
            combined_inputs['labels'] = torch.cat([inp['labels'] for inp in all_inputs])
        
        return combined_inputs
        
    except Exception as e:
        logger.error(f"Error during SAM preprocessing: {str(e)}")
        logger.error("Input types:")
        logger.error(f"Image type: {type(image)}, shape: {image.shape}, dtype: {image.dtype}")
        logger.error(f"Boxes type: {type(pixel_boxes)}, count: {len(pixel_boxes)}")
        if masks is not None:
            logger.error(f"Masks type: {type(masks)}, count: {len(masks)}")
        raise

def visualize_sam_preprocessing(
    image: np.ndarray,
    boxes: List[List[float]],
    preprocessed_image: torch.Tensor,
    preprocessed_boxes: torch.Tensor,
    masks: Optional[List[np.ndarray]] = None,
    preprocessed_masks: Optional[torch.Tensor] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Visualize original and preprocessed data for SAM.
    
    Args:
        image: Original image as numpy array
        boxes: Original boxes in YOLO format
        preprocessed_image: Preprocessed image tensor from SAM processor
        preprocessed_boxes: Preprocessed boxes tensor from SAM processor
        masks: Optional list of original masks
        preprocessed_masks: Optional preprocessed masks tensor
        save_path: Optional path to save visualization
        
    Returns:
        Matplotlib figure object
    """
    n_masks = len(masks) if masks is not None else 0
    n_cols = 2 + (2 if n_masks > 0 else 0)  # 2 for images, 2 for masks if present
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
    
    # Original image with boxes
    axes[0].imshow(image)
    height, width = image.shape[:2]
    for box in boxes:
        pixel_box = yolo_to_pixel_coords(box, width, height)
        x1, y1, x2, y2 = pixel_box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, edgecolor='red', linewidth=2)
        axes[0].add_patch(rect)
    axes[0].set_title('Original Image with Boxes')
    axes[0].axis('off')
    
    # Preprocessed image with boxes
    # Convert preprocessed image from tensor to numpy and rescale to [0, 1] range
    proc_img = preprocessed_image.squeeze().permute(1, 2, 0).numpy()
    proc_img = (proc_img - proc_img.min()) / (proc_img.max() - proc_img.min())
    axes[1].imshow(proc_img)
    
    # Handle preprocessed boxes - they come in shape [batch, num_boxes, 4]
    boxes_array = preprocessed_boxes.squeeze().numpy()
    if len(boxes_array.shape) == 1:  # Single box
        boxes_array = boxes_array.reshape(1, 4)
    for box in boxes_array:
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                           fill=False, edgecolor='red', linewidth=2)
        axes[1].add_patch(rect)
    axes[1].set_title('Preprocessed Image with Boxes')
    axes[1].axis('off')
    
    # Original masks
    if masks is not None and n_masks > 0:
        # Create a combined mask visualization
        combined_mask = np.zeros_like(masks[0])
        for i, mask in enumerate(masks):
            # Add mask with different intensity for visualization
            combined_mask += mask * ((i + 1) / len(masks))
        
        axes[2].imshow(combined_mask, cmap='viridis')
        axes[2].set_title('Original Masks')
        axes[2].axis('off')
        
        # Preprocessed masks
        if preprocessed_masks is not None:
            proc_masks = preprocessed_masks.squeeze().numpy()
            if len(proc_masks.shape) == 2:
                proc_masks = proc_masks[None, ...]
            
            # Create combined preprocessed mask visualization
            combined_proc_mask = np.zeros_like(proc_masks[0])
            for i, mask in enumerate(proc_masks):
                combined_proc_mask += mask * ((i + 1) / len(proc_masks))
            
            axes[3].imshow(combined_proc_mask, cmap='viridis')
            axes[3].set_title('Preprocessed Masks')
            axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(exist_ok=True)
        fig.savefig(save_path)
        
    return fig

def process_dataset_with_sam(
    dataset: Dict[str, Dict[str, Any]], 
    output_dir: Optional[Path] = None,
    save_visualizations: bool = False,
    batch_size: int = 4,
    processor: Optional[SamProcessor] = None
) -> Dict[str, Dict[str, Any]]:
    """Process all samples in the dataset with SAM preprocessing.
    Handles multiple masks per image.
    
    Args:
        dataset: Dictionary of samples with images, masks, and boxes
        output_dir: Optional path to save visualizations
        save_visualizations: Whether to save visualizations
        batch_size: Batch size for processing
        processor: Optional SAM processor instance
        
    Returns:
        Dictionary of processed samples with SAM-ready data
    """
    # Create output directory if needed
    if save_visualizations and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize or reuse processor
    if processor is None:
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    
    # Process each sample
    processed_samples = 0
    total_masks = 0
    errors = []
    processed_dataset = {}
    
    # Convert dataset items to list for easier batching
    items = list(dataset.items())
    
    for batch_start in tqdm(range(0, len(items), batch_size), desc="Processing batches"):
        batch_items = items[batch_start:batch_start + batch_size]
        
        for task_id, sample in batch_items:
            try:
                # Get image, boxes and masks
                image = sample['image']
                boxes = sample['boxes']
                masks = sample['masks']
                
                # Skip if no boxes or masks found
                if not boxes or not masks:
                    logger.warning(f"No boxes or masks found for sample {task_id}, skipping")
                    continue
                
                # Verify matching counts
                if len(boxes) != len(masks):
                    logger.warning(f"Mismatch in number of boxes ({len(boxes)}) and masks ({len(masks)}) for sample {task_id}, skipping")
                    continue
                
                # Preprocess data for SAM
                inputs = preprocess_for_sam(
                    image=image,
                    boxes=boxes,
                    masks=masks,
                    processor=processor
                )
                
                # Store processed data
                processed_dataset[task_id] = {
                    'original_data': sample,
                    'preprocessed_data': inputs
                }
                
                if save_visualizations and output_dir is not None:
                    # Create visualization
                    fig = visualize_sam_preprocessing(
                        image=image,
                        boxes=boxes,
                        preprocessed_image=inputs['pixel_values'],
                        preprocessed_boxes=inputs['input_boxes'],
                        masks=masks,
                        preprocessed_masks=inputs.get('labels')
                    )
                    
                    # Save visualization
                    output_path = output_dir / f'preprocessed_sample_{task_id}.png'
                    fig.savefig(output_path)
                    plt.close(fig)  # Close figure to free memory
                
                processed_samples += 1
                total_masks += len(masks)
                    
            except Exception as e:
                error_msg = f"Error processing sample {task_id}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                continue
    
    # Final summary
    logger.info("\nProcessing complete!")
    logger.info(f"Successfully processed {processed_samples} samples with {total_masks} total masks")
    
    if len(processed_dataset) == 0:
        raise ValueError("No samples were successfully processed!")
        
    if errors:
        logger.info(f"Encountered {len(errors)} errors:")
        for error in errors:
            logger.info(f"  - {error}")
            
    return processed_dataset

def split_dataset(
    dataset: Dict[str, Dict[str, Any]],
    test_size: float = 0.2,
    random_seed: int = 42,
    stratify_by_box_count: bool = True
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Split dataset into training and test sets.
    
    Args:
        dataset: Dictionary of preprocessed samples
        test_size: Fraction of dataset to use for testing (default: 0.2)
        random_seed: Random seed for reproducibility (default: 42)
        stratify_by_box_count: Whether to maintain box count distribution in split (default: True)
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    # Get all task IDs
    task_ids = list(dataset.keys())
    
    if stratify_by_box_count:
        # Get number of boxes for each sample
        box_counts = [len(dataset[task_id]['original_data']['boxes']) for task_id in task_ids]
        # Convert to categorical for stratification
        box_count_categories = np.array([min(count, 5) for count in box_counts])  # Cap at 5 for reasonable stratification
        stratify = box_count_categories
    else:
        stratify = None
    
    # Split task IDs
    train_ids, test_ids = train_test_split(
        task_ids,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify
    )
    
    # Create train and test datasets
    train_dataset = {task_id: dataset[task_id] for task_id in train_ids}
    test_dataset = {task_id: dataset[task_id] for task_id in test_ids}
    
    # Log split information
    logger.info(f"\nDataset split complete:")
    logger.info(f"Total samples: {len(dataset)}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    if stratify_by_box_count:
        # Log box count distribution
        logger.info("\nBox count distribution:")
        
        def get_box_count_stats(ids):
            counts = [len(dataset[task_id]['original_data']['boxes']) for task_id in ids]
            return {
                'mean': np.mean(counts),
                'median': np.median(counts),
                'min': np.min(counts),
                'max': np.max(counts)
            }
        
        train_stats = get_box_count_stats(train_ids)
        test_stats = get_box_count_stats(test_ids)
        
        logger.info("\nTraining set:")
        logger.info(f"  Mean boxes per image: {train_stats['mean']:.2f}")
        logger.info(f"  Median boxes per image: {train_stats['median']:.2f}")
        logger.info(f"  Range: {train_stats['min']} to {train_stats['max']} boxes")
        
        logger.info("\nTest set:")
        logger.info(f"  Mean boxes per image: {test_stats['mean']:.2f}")
        logger.info(f"  Median boxes per image: {test_stats['median']:.2f}")
        logger.info(f"  Range: {test_stats['min']} to {test_stats['max']} boxes")
    
    return train_dataset, test_dataset

class SAMDataset(torch.utils.data.Dataset):
    """Dataset class for SAM model training/evaluation.
    
    TODO:
        - Implement data augmentation in __getitem__
        - Add support for multiple instance masks
        - Consider adding caching mechanism
        - Implement validation methods
    """
    
    def __init__(self, dataset: Dict[str, Dict[str, Any]], prefix: str = '', 
                 transform: Optional[Any] = None):
        """Initialize SAM dataset.
        
        Args:
            dataset: Dictionary of preprocessed samples
            prefix: Optional prefix for dataset keys
            transform: Optional transform for data augmentation
        """
        self.dataset = dataset
        self.prefix = prefix
        self.transform = transform
        
        # Store task IDs in a list for indexing
        self.task_ids = list(dataset.keys())
        self.task_ids.sort()
        
        logger.info(f"Initialized {prefix} dataset with {len(self)} samples")
    
    def __len__(self) -> int:
        return len(self.task_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset.
        
        TODO:
            - Implement data augmentation here
            - Add input validation
            - Consider adding caching
        """
        if not isinstance(idx, int):
            raise TypeError(f"Index must be an integer, got {type(idx)}")
        
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self)} items")
        
        # Get task ID and corresponding data
        task_id = self.task_ids[idx]
        sample = self.dataset[task_id]
        
        # Get preprocessed data
        inputs = sample['preprocessed_data']
        
        # Add metadata
        inputs['dataset_key'] = f"{self.prefix}_{task_id}"
        inputs['original_data'] = sample['original_data']
        
        return inputs

def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function to handle varying image sizes and other data types.
    
    TODO:
        - Add support for multiple instance masks
        - Implement more efficient padding
        - Consider adding batch statistics
    """
    # Initialize output dictionary
    output = {}
    
    # Get all keys from first sample
    keys = batch[0].keys()
    
    for key in keys:
        values = [sample[key] for sample in batch]
        
        # Handle different types of data
        if isinstance(values[0], torch.Tensor):
            if key == 'pixel_values':
                # Log shape information for debugging
                logger.debug(f"First image shape: {values[0].shape}")
                
                # For images, pad to max size in batch
                # Handle both [C, H, W] and [H, W, C] formats
                if len(values[0].shape) == 3:
                    if values[0].shape[-1] == 3:  # [H, W, C] format
                        max_h = max(x.shape[0] for x in values)
                        max_w = max(x.shape[1] for x in values)
                        
                        # Pad each image to max size
                        padded_values = []
                        for img in values:
                            h, w = img.shape[:2]
                            pad_h = max_h - h
                            pad_w = max_w - w
                            if pad_h > 0 or pad_w > 0:
                                padding = (0, 0, 0, pad_w, 0, pad_h)  # pad last two dims
                                img = torch.nn.functional.pad(img, padding)
                            padded_values.append(img)
                        
                        # Stack and permute to [B, C, H, W]
                        output[key] = torch.stack(padded_values).permute(0, 3, 1, 2)
                    else:  # [C, H, W] format
                        max_h = max(x.shape[1] for x in values)
                        max_w = max(x.shape[2] for x in values)
                        
                        # Pad each image to max size
                        padded_values = []
                        for img in values:
                            h, w = img.shape[1:]
                            pad_h = max_h - h
                            pad_w = max_w - w
                            if pad_h > 0 or pad_w > 0:
                                padding = (0, pad_w, 0, pad_h)  # left, right, top, bottom
                                img = torch.nn.functional.pad(img, padding)
                            padded_values.append(img)
                        
                        output[key] = torch.stack(padded_values)
                else:
                    # If not 3D tensor, just try stacking
                    try:
                        output[key] = torch.stack(values)
                    except:
                        output[key] = values
            elif key == 'labels' or key == 'mask':
                # For masks, ensure proper padding
                if len(values[0].shape) == 2:  # [H, W] format
                    max_h = max(x.shape[0] for x in values)
                    max_w = max(x.shape[1] for x in values)
                    
                    padded_values = []
                    for mask in values:
                        h, w = mask.shape
                        pad_h = max_h - h
                        pad_w = max_w - w
                        if pad_h > 0 or pad_w > 0:
                            padding = (0, pad_w, 0, pad_h)
                            mask = torch.nn.functional.pad(mask, padding)
                        padded_values.append(mask)
                    
                    output[key] = torch.stack(padded_values)
                else:
                    try:
                        output[key] = torch.stack(values)
                    except:
                        output[key] = values
            else:
                # For other tensors, try regular stacking
                try:
                    output[key] = torch.stack(values)
                except:
                    output[key] = values
        elif isinstance(values[0], (list, tuple)):
            # Keep lists/tuples as is
            output[key] = values
        elif isinstance(values[0], dict):
            # For nested dictionaries (like original_data)
            output[key] = values
        else:
            # For other types (strings, etc)
            output[key] = values
    
    return output

def create_dataloaders(
    train_dataset: Dict[str, Dict[str, Any]],
    test_dataset: Dict[str, Dict[str, Any]],
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create PyTorch DataLoaders for training and testing.
    
    TODO:
        - Add validation dataloader
        - Implement memory usage monitoring
        - Consider adding sampler support
    """
    # Create dataset objects
    train_set = SAMDataset(train_dataset, prefix='train')
    test_set = SAMDataset(test_dataset, prefix='test')
    
    # Create dataloaders with custom collate function
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch during training
        collate_fn=custom_collate_fn
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for test set
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # Keep all samples in test set
        collate_fn=custom_collate_fn
    )
    
    # Log dataloader information
    logger.info("\nDataLoader creation complete:")
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Log sample batch information
    try:
        sample_batch = next(iter(train_loader))
        logger.info("\nSample batch shapes:")
        for key, value in sample_batch.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: {value.shape}")
            elif isinstance(value, (list, tuple)):
                logger.info(f"  {key}: list of length {len(value)}")
    except Exception as e:
        logger.warning(f"Could not log sample batch info: {str(e)}")
    
    return train_loader, test_loader
