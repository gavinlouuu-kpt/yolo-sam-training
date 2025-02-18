"""Example script demonstrating how to preprocess data for SAM model."""

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from transformers import SamProcessor
import torch
import PIL.Image

from yolo_sam_training.data import load_dataset_from_summary, yolo_to_pixel_coords

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_preprocessed_sample(image, boxes, preprocessed_image, preprocessed_boxes, mask=None, preprocessed_mask=None):
    """Visualize original and preprocessed data."""
    n_plots = 4 if mask is not None else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    
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
    proc_img = (proc_img - proc_img.min()) / (proc_img.max() - proc_img.min())  # Normalize to [0, 1]
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
    
    # Original mask
    if mask is not None:
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title('Original Mask')
        axes[2].axis('off')
        
        # Preprocessed mask
        if preprocessed_mask is not None:
            proc_mask = preprocessed_mask.squeeze().numpy()
            axes[3].imshow(proc_mask, cmap='gray')
            axes[3].set_title('Preprocessed Mask')
            axes[3].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    # Load the dataset using the existing functionality
    summary_path = Path('D:/code/ai_cytometry/data/example_training_data/summary.json')
    logger.info("Loading dataset...")
    dataset = load_dataset_from_summary(summary_path)
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Initialize SAM processor
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    
    # Process a sample
    sample_task_id = list(dataset.keys())[0]
    sample = dataset[sample_task_id]
    
    # Get image, boxes and mask
    image = sample['image']
    boxes = sample['boxes']
    mask = sample['mask']
    
    # Debug mask information
    logger.info("\nMask debug information:")
    logger.info(f"Mask shape: {mask.shape}")
    logger.info(f"Mask dtype: {mask.dtype}")
    logger.info(f"Mask unique values: {np.unique(mask)}")
    logger.info(f"Mask min/max: {mask.min()}, {mask.max()}")
    
    # Convert YOLO format boxes to pixel coordinates
    height, width = image.shape[:2]
    pixel_boxes = [yolo_to_pixel_coords(box, width, height) for box in boxes]
    
    # Debug box information
    logger.info("\nBox debug information:")
    logger.info(f"Number of boxes: {len(pixel_boxes)}")
    logger.info(f"First box coordinates: {pixel_boxes[0]}")
    
    try:
        # Ensure mask is binary and in correct dtype
        mask_processed = mask.astype(np.float32)
        if mask_processed.max() > 1:
            mask_processed = mask_processed / 255.0
            
        # Convert mask to PIL Image
        mask_2d = mask_processed.squeeze()  # Remove channel dimension if present
        mask_uint8 = (mask_2d * 255).astype(np.uint8)
        mask_pil = PIL.Image.fromarray(mask_uint8)
        
        # Debug mask conversion
        logger.info("\nMask conversion debug:")
        logger.info(f"PIL Image size: {mask_pil.size}")
        logger.info(f"PIL Image mode: {mask_pil.mode}")
        
        # Prepare inputs for SAM
        inputs = processor(
            images=image,
            input_boxes=[[pixel_boxes[0]]],  # Process first box as example
            segmentation_maps=[mask_pil],  # Pass as list of PIL Images
            return_tensors="pt",
            do_resize=True,
            size={"longest_edge": 1024},
            do_normalize=True,
            do_pad=True,
        )
    
        logger.info("\nPreprocessed data shapes:")
        logger.info(f"Image tensor shape: {inputs['pixel_values'].shape}")
        logger.info(f"Input boxes shape: {inputs['input_boxes'].shape}")
        logger.info(f"Available keys in inputs: {inputs.keys()}")
        for key in inputs.keys():
            if isinstance(inputs[key], torch.Tensor):
                logger.info(f"{key} shape: {inputs[key].shape}")
            elif isinstance(inputs[key], list):
                logger.info(f"{key} length: {len(inputs[key])}")
        
        # Visualize original and preprocessed data
        fig = visualize_preprocessed_sample(
            image=image,
            boxes=boxes,
            preprocessed_image=inputs['pixel_values'],
            preprocessed_boxes=inputs['input_boxes'],
            mask=mask,
            preprocessed_mask=inputs['labels'] if 'labels' in inputs else None
        )
        
        # Save visualization
        output_dir = Path('visualization_output')
        output_dir.mkdir(exist_ok=True)
        fig.savefig(output_dir / f'preprocessed_sample_{sample_task_id}.png')
        logger.info(f"\nVisualization saved to: {output_dir}/preprocessed_sample_{sample_task_id}.png")
        plt.show()

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        logger.error("Processor input types:")
        logger.error(f"Image type: {type(image)}, shape: {image.shape}, dtype: {image.dtype}")
        logger.error(f"Boxes type: {type(pixel_boxes)}, First box type: {type(pixel_boxes[0])}")
        logger.error(f"Mask type: {type(mask_processed)}, shape: {mask_processed.shape}, dtype: {mask_processed.dtype}")
        raise

if __name__ == '__main__':
    main() 