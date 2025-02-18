# YOLO-SAM Training

A Python package for training and fine-tuning Segment Anything Model (SAM) using YOLO-format annotations.

## Features

- Load and preprocess images, masks, and YOLO-format annotations
- Support for multiple instance masks per image
- Prepare data for SAM model training
- Split datasets into train/test sets with stratification
- Custom DataLoader with support for variable-sized images
- Visualization tools for preprocessing steps

## Installation

```bash
pip install -e .
```

## Usage

### Basic Usage

```python
from yolo_sam_training.data import (
    load_dataset_from_summary,
    process_dataset_with_sam,
    split_dataset,
    create_dataloaders
)

# Load dataset
dataset = load_dataset_from_summary('path/to/summary.json')

# Preprocess for SAM
processed_dataset = process_dataset_with_sam(
    dataset=dataset,
    save_visualizations=True  # Optional
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
    batch_size=4
)
```

## Data Format

### Expected Directory Structure
```
data/
├── images/
│   ├── image1.png
│   └── image2.png
├── masks/
│   ├── image1_0.png  # First mask for image1
│   ├── image1_1.png  # Second mask for image1
│   ├── image2_0.png  # First mask for image2
│   └── image2_1.png  # Second mask for image2
├── boxes/
│   ├── image1.txt    # Contains multiple boxes, one per mask
│   └── image2.txt    # Contains multiple boxes, one per mask
└── summary.json
```

### YOLO Box Format
Each line in the box file corresponds to a mask:
```
class x_center y_center width height  # First mask
class x_center y_center width height  # Second mask
```

### Multiple Instance Support

The package now fully supports multiple instance masks per image:

1. **Loading Multiple Masks**
   - Masks are named with index suffixes (e.g., `image1_0.png`, `image1_1.png`)
   - Each mask corresponds to one box in the YOLO format file
   - Automatic validation ensures matching mask-box pairs

2. **Data Structure**
   ```python
   dataset = {
       'task_id': {
           'image': np.ndarray,  # Single image
           'masks': List[np.ndarray],  # List of masks
           'boxes': List[List[float]],  # List of YOLO boxes
           'sam_data': Dict  # SAM-ready data
       }
   }
   ```

3. **Batch Processing**
   - Each sample can have different numbers of masks
   - Batches maintain mask-box correspondence
   - Automatic handling of variable instance counts

4. **Visualization**
   - Support for visualizing multiple masks per image
   - Different intensities for distinguishing instances
   - Combined mask visualization options

## Current Limitations and TODOs

### Data Augmentation
- To be implemented in `SAMDataset.__getitem__`
- Planned augmentations:
  - Rotation
  - Flipping
  - Color jittering
  - Random cropping
- Will ensure consistency across instances

### Memory Management
- Large datasets might need lazy loading
- Planning to implement:
  - Data streaming
  - Memory usage monitoring
  - Caching mechanisms

### Error Handling
- Need to add:
  - More robust validation for mask-box correspondence
  - Data integrity checks
  - Input validation
  - Better error messages

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

This package takes output of label-studio-processor as input for fine tuning yolo and sam.

The input would consist of:
1. image
2. mask
3. bounding box

In Yolo we will be using the ultralytics framework.

In SAM we will be getting the model from huggingface using the bounding box as prompt. We will be using pytorch as the ML framework.

## SAM Preprocessing Guide

### Input Format Requirements

When preprocessing data for the Segment Anything Model (SAM), the following format requirements must be met:

1. **Images**
   - Format: NumPy array
   - Shape: `(H, W, 3)` for RGB images
   - Dtype: `uint8` or `float32`

2. **Bounding Boxes**
   - Format: List of boxes or torch tensor
   - Shape: `(batch_size, num_instances, 4)` where 4 represents `[x1, y1, x2, y2]` in pixel coordinates
   - Supports multiple boxes per image

3. **Masks**
   - Format: List of PIL Images
   - Mode: 'L' (grayscale)
   - Values: Binary (0 or 255)
   - Supports multiple masks per image

### Example Usage with Multiple Instances

```python
# Process multiple masks and boxes
processed = processor(
    images=image,
    input_boxes=boxes,  # List of boxes for multiple instances
    segmentation_maps=mask_pils,  # List of PIL Images
    return_tensors="pt"
)

# Access processed data
pixel_values = processed['pixel_values']  # Processed image
input_boxes = processed['input_boxes']    # Processed boxes
labels = processed['labels']              # Processed masks
```

### Common Issues and Solutions

1. **Multiple Mask Processing**
   - Issue: Mismatched number of masks and boxes
   - Solution: Ensure equal number of masks and boxes, use validation checks

2. **Batch Processing**
   - Issue: Variable number of instances per image
   - Solution: Use proper batching with padding or dynamic batch sizes

3. **Memory Usage**
   - Issue: Large batches with multiple instances
   - Solution: Adjust batch size based on instance count and available memory