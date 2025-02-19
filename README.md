# YOLO-SAM Training

This repository contains training scripts and utilities for YOLO and SAM model training for cytometry applications.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## MLflow Experiment Tracking

The project uses MLflow for experiment tracking. You can either use a centralized MLflow server (recommended for team collaboration) or local tracking.

### Option 1: Centralized MLflow Server (Recommended)

1. Start the MLflow tracking server and database:
```bash
docker-compose up -d
```

2. Access the MLflow UI at http://localhost:5000

The server setup includes:
- PostgreSQL database for metadata storage
- Persistent volume for artifacts
- Automatic fallback to local tracking if server is unavailable

### Option 2: Local Tracking

If the MLflow server is not available, the training script automatically falls back to local tracking:
- Experiments are stored in `./mlruns` directory
- No additional setup required

### Environment Variables

- `MLFLOW_TRACKING_URI`: MLflow tracking server URL (default: http://localhost:5000)
- `TRAINING_DATA_DIR`: Directory containing training data (default varies by script)

## Training Scripts

### YOLO Training

Run the YOLO training script:
```bash
python -m yolo_sam_training.examples.yolo_training_example
```

The script will:
1. Prepare the dataset
2. Train/fine-tune YOLO model
3. Log metrics and artifacts to MLflow
4. Validate the trained model

### Metrics Tracked

Training metrics logged to MLflow include:
- Training metrics (prefixed with 'train_')
- Validation metrics (prefixed with 'val_')
- Model artifacts and parameters

## Development

### Docker Setup

The MLflow tracking server uses Docker Compose with:
- PostgreSQL 14 for metadata storage
- MLflow server with PostgreSQL driver
- Persistent volumes for database and artifacts

To rebuild the containers after changes:
```bash
docker-compose down
docker-compose up --build -d
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Features

- Load and preprocess images and annotations for both YOLO and SAM training
- Support for multiple instance masks and bounding boxes per image
- Automatic dataset splitting with train/validation sets
- Integration with Ultralytics YOLO for object detection
- SAM fine-tuning with bounding box prompts
- Comprehensive logging and visualization tools
- Modular design for easy extension and customization

## Integration with Label Studio

This package works seamlessly with data exported from Label Studio using our companion package `label-studio-interface`. The workflow is:

1. **Export Data from Label Studio**:
   ```python
   from label_studio_processor.examples.prepare_training_data import main as prepare_data
   
   # Export and prepare data from Label Studio
   prepare_data()  # Creates the expected directory structure
   ```

2. **Set Data Directory Environment Variable**:
   ```bash
   export TRAINING_DATA_DIR=/path/to/your/data
   ```

3. **Train Models**:
   ```python
   # Train YOLO model
   python -m yolo_sam_training.examples.yolo_training_example
   
   # Train SAM model
   python -m yolo_sam_training.examples.sam_training_example
   ```

## Data Directory Structure

Expected structure after Label Studio export:
```
data/
├── images/
│   ├── image1.png
│   └── image2.png
├── labels/           # YOLO format annotations
│   ├── image1.txt
│   └── image2.txt
└── masks/           # For SAM training
    ├── image1_0.png
    └── image2_0.png
```

## YOLO Training

### Dataset Preparation
```python
from yolo_sam_training.yolo_training import prepare_yolo_dataset

# Prepare dataset with train/val split
yaml_path = prepare_yolo_dataset(
    source_dir='/path/to/source',
    output_dir='/path/to/prepared_dataset',
    split_ratio=0.2
)
```

### Training
```python
from yolo_sam_training.yolo_training import train_yolo_model

# Train YOLO model
metrics = train_yolo_model(
    yaml_path=yaml_path,
    model_save_path='models/yolo_fine_tuned',
    pretrained_model='yolov8n.pt',
    num_epochs=100,
    image_size=640,
    batch_size=16,
    device='0',  # Use '0' for first GPU, 'cpu' for CPU
    learning_rate=0.01
)
```

### Validation
```python
from yolo_sam_training.yolo_training import validate_yolo_model

# Validate trained model
validation_metrics = validate_yolo_model(
    model_path='models/yolo_fine_tuned/weights/best.pt',
    data_yaml=yaml_path,
    image_size=640,
    device='0'
)
```

## SAM Training

### Data Processing
```python
from yolo_sam_training.data import (
    load_dataset_from_summary,
    process_dataset_with_sam,
    split_dataset,
    create_dataloaders
)

# Load and process dataset
dataset = load_dataset_from_summary('path/to/summary.json')
processed_dataset = process_dataset_with_sam(dataset)

# Create train/test split
train_dataset, test_dataset = split_dataset(
    dataset=processed_dataset,
    test_size=0.2
)

# Create dataloaders
train_loader, test_loader = create_dataloaders(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    batch_size=4
)
```

### Training
```python
from yolo_sam_training.sam_training import train_sam_model

# Train SAM model
best_model_state, loss_plot = train_sam_model(
    train_loader=train_loader,
    val_loader=test_loader,
    model_save_path='models/sam_fine_tuned',
    visualization_dir='visualization_output/training',
    num_epochs=10,
    learning_rate=1e-5
)
```

## Model Configuration

### YOLO Configuration
- Uses Ultralytics YOLOv8
- Supports various model sizes (nano to extra large)
- Configurable training parameters:
  - Image size
  - Batch size
  - Learning rate
  - Number of epochs
  - Early stopping patience

### SAM Configuration
- Uses HuggingFace SAM implementation
- Supports bounding box prompts
- Configurable training parameters:
  - Learning rate
  - Weight decay
  - Number of epochs
  - Visualization options

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
│       ├── yolo_training.py  # YOLO training utilities
│       ├── sam_training.py   # SAM training utilities
│       └── examples/
│           ├── yolo_training_example.py
│           └── sam_training_example.py
├── tests/
└── README.md
```

## Current Limitations and TODOs

### Data Augmentation
- Implement consistent augmentation across YOLO and SAM training
- Add support for:
  - Rotation
  - Flipping
  - Color jittering
  - Random cropping

### Memory Management
- Implement efficient data loading for large datasets
- Add support for:
  - Data streaming
  - Memory usage monitoring
  - Caching mechanisms

### Error Handling
- Add comprehensive validation for:
  - Dataset integrity
  - Model configurations
  - Training parameters
  - Input formats

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