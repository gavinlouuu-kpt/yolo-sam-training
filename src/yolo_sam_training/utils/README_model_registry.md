# MLflow Model Registry Tools

This directory contains utilities for working with the MLflow Model Registry, which helps you manage the full lifecycle of your YOLO and SAM models.

## Overview

The MLflow Model Registry provides:
- Model versioning
- Stage transitions (None → Staging → Production)
- Model lineage tracking
- Model metadata and metrics

## Available Tools

### 1. Register a Model

The `register_model.py` utility allows you to register a model from a completed MLflow run:

```bash
# Register a model from a specific run
python -m yolo_sam_training.utils.register_model \
  --run-id <RUN_ID> \
  --model-name "yolo_object_detection" \
  --model-path "model" \
  --description "YOLO model trained on cell dataset" \
  --tag "mAP50" "0.9950" \
  --tag "precision" "0.9992"
```

### 2. Register the Most Recent Model

The `register_recent_model.py` and `register_recent_sam_model.py` scripts automatically register the most recently completed training runs:

```bash
# Register the most recent YOLO model
python -m yolo_sam_training.examples.register_recent_model

# Register the most recent SAM model
python -m yolo_sam_training.examples.register_recent_sam_model
```

### 3. Manage Model Versions

The `manage_model_versions.py` utility provides commands for managing model versions:

```bash
# List all versions of a model
python -m yolo_sam_training.utils.manage_model_versions list \
  --model-name "yolo_object_detection"

# Transition a model version to a new stage
python -m yolo_sam_training.utils.manage_model_versions transition \
  --model-name "yolo_object_detection" \
  --version 1 \
  --stage "Production"

# Compare metrics between model versions
python -m yolo_sam_training.utils.manage_model_versions compare \
  --model-name "yolo_object_detection" \
  --versions 1 2 3

# Delete a model version
python -m yolo_sam_training.utils.manage_model_versions delete \
  --model-name "yolo_object_detection" \
  --version 1
```

## Model Stages

Models in the registry can be in one of the following stages:

- **None**: The initial stage for newly registered models
- **Staging**: Models that are being validated or tested
- **Production**: Models that are deployed in production
- **Archived**: Models that are no longer active but kept for reference

## Best Practices

1. **Consistent naming**: Use consistent model names for the same type of model
2. **Descriptive tags**: Add tags with key metrics and parameters for easy comparison
3. **Stage transitions**: Move models through stages as they are validated
4. **Version comparison**: Compare metrics between versions before promoting to production
5. **Archiving**: Archive old versions instead of deleting them to maintain history

## Accessing Models from the Registry

To load a model from the registry in your inference code:

```python
import mlflow

# Load a specific YOLO model version
yolo_model = mlflow.pyfunc.load_model(
    model_uri=f"models:/yolo_object_detection/1"
)

# Load the production version of a YOLO model
yolo_model = mlflow.pyfunc.load_model(
    model_uri=f"models:/yolo_object_detection/Production"
)

# Load a specific SAM model version
sam_model = mlflow.pyfunc.load_model(
    model_uri=f"models:/sam_segmentation/1"
)

# Load the production version of a SAM model
sam_model = mlflow.pyfunc.load_model(
    model_uri=f"models:/sam_segmentation/Production"
)
```

## Viewing Models in the UI

You can view registered models in the MLflow UI:

1. Open the MLflow UI at http://localhost:5000
2. Click on "Models" in the top navigation bar
3. Select a model to view its versions and details 