# YOLO-SAM Training

A Python package for training and fine-tuning Segment Anything Model (SAM) using YOLO-format annotations.

## Features

- Load and preprocess images, masks, and YOLO-format annotations
- Support for multiple instance masks per image
- Prepare data for SAM model training
- Split datasets into train/test sets with stratification
- Custom DataLoader with support for variable-sized images
- Robust training pipeline with validation and visualization
- Modular design for easy extension and customization

## Integration with Label Studio

This package works seamlessly with data exported from Label Studio using our companion package `label-studio-interface`. The workflow is:

1. **Export Data from Label Studio**:
   ```python
   from label_studio_processor.examples.prepare_training_data import main as prepare_data
   
   # Export and prepare data from Label Studio
   prepare_data()  # This creates the expected directory structure
   ```

2. **Train SAM Model**:
   ```python
   from yolo_sam_training.examples.sam_training_example import main as train_sam
   
   # Train model using the exported data
   train_sam()
   ```

The `label-studio-interface` package handles:
- Downloading images from Label Studio
- Converting brush annotations to masks
- Generating YOLO format boxes
- Creating the directory structure expected by this package

You can use this training package with any data in the correct format, not just from Label Studio.

## Installation

```bash
pip install -e .
```

## Project Structure

```
yolo-sam-training/
├── src/
│   └── yolo_sam_training/
│       ├── data.py           # Data loading and preprocessing
│       ├── training.py       # Training utilities and functions
│       └── examples/
│           ├── dataset_loading_example.py
│           ├── sam_preprocessing_example.py
│           └── sam_training_example.py
├── tests/
└── README.md
```

## Usage

### 1. Data Loading and Preprocessing

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

### 2. Training SAM Model

```python
from yolo_sam_training.training import train_sam_model

# Train model
best_model_state, loss_plot = train_sam_model(
    train_loader=train_loader,
    val_loader=test_loader,
    model_save_path='models/sam_fine_tuned',
    visualization_dir='visualization_output/training',
    num_epochs=10,
    learning_rate=1e-5
)
```

## Training Module Features

The `training.py` module provides a comprehensive set of utilities for training SAM:

1. **Shape Validation**
   - `validate_batch_shapes`: Ensures tensor shapes match SAM requirements
   - `validate_pred_masks`: Normalizes and validates predicted mask shapes

2. **Loss Computation**
   - `compute_batch_loss`: Handles variable numbers of masks per sample
   - Supports any PyTorch loss function
   - Properly handles batch-level loss aggregation

3. **Training Loop**
   - `train_one_epoch`: Manages single epoch training
   - `validate_one_epoch`: Handles validation
   - Progress tracking with tqdm
   - Comprehensive error handling

4. **Visualization**
   - `save_prediction_visualization`: Creates detailed mask visualizations
   - Training progress plots
   - Support for multiple masks per image

5. **Model Management**
   - Automatic device handling (CPU/CUDA)
   - Model checkpoint saving
   - Best model tracking
   - Training state logging

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

## Training Configuration

The training process can be customized through various parameters:

```python
train_sam_model(
    train_loader,
    val_loader,
    model_save_path,
    num_epochs=10,          # Number of training epochs
    learning_rate=1e-5,     # Learning rate for optimizer
    weight_decay=0,         # Weight decay for regularization
    visualization_dir=None, # Directory for saving visualizations
    device=None            # Device to use (auto-detected if None)
)
```

### Key Features:

1. **Automatic Device Selection**
   - Automatically uses CUDA if available
   - Falls back to CPU if CUDA is not available
   - Configurable through device parameter

2. **Model Architecture**
   - Uses SAM base model from HuggingFace
   - Freezes encoder parameters by default
   - Only fine-tunes mask decoder

3. **Loss Function**
   - Uses DiceCELoss from MONAI
   - Handles variable numbers of masks
   - Properly weighted loss computation

4. **Visualization**
   - Per-epoch visualization of predictions
   - Training/validation loss plots
   - Sample-level mask comparisons

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