"""Training utilities for SAM model fine-tuning.

This module provides functions for training and fine-tuning the Segment Anything Model (SAM).
It includes utilities for:
- Shape validation
- Loss computation
- Training and validation loops
- Visualization
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
import torch
from torch.optim import Adam
import monai.losses
from tqdm import tqdm
import matplotlib.pyplot as plt
from statistics import mean
from transformers import SamModel
import numpy as np
import mlflow
import mlflow.pytorch
import tempfile
import os
import shutil
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def save_prediction_visualization(
    image: torch.Tensor,
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    batch_idx: int,
    epoch: int,
    output_dir: Path = None,
    mlflow_run = None
):
    """Save visualization of model predictions.
    
    Args:
        image: Input image tensor [3, H, W] or [1, 3, H, W]
        pred_masks: Predicted masks tensor [N, H, W] or [1, N, H, W]
        gt_masks: Ground truth masks tensor [N, H, W] or [1, N, H, W]
        batch_idx: Current batch index
        epoch: Current epoch
        output_dir: Directory to save visualizations (optional)
        mlflow_run: MLflow run to log artifacts to (optional)
    """
    try:
        # Ensure all tensors are on CPU and convert to numpy
        def prepare_tensor(x: torch.Tensor) -> np.ndarray:
            x = x.detach().cpu()
            if x.ndim == 4:  # Remove batch dimension if present
                x = x.squeeze(0)
            return x.numpy()
        
        # Prepare image
        image = prepare_tensor(image)
        if image.shape[0] == 3:  # If channels first
            image = np.transpose(image, (1, 2, 0))
        
        # Normalize image to [0, 1] range for display
        image = (image - image.min()) / (image.max() - image.min())
        
        # Prepare masks
        pred_masks = prepare_tensor(pred_masks)
        gt_masks = prepare_tensor(gt_masks)
        
        # Get number of masks
        n_masks = pred_masks.shape[0]
        
        # Create figure with rows for each mask
        fig, axes = plt.subplots(n_masks, 3, figsize=(15, 5 * n_masks))
        if n_masks == 1:
            axes = axes[None, :]  # Add dimension for consistent indexing
        
        # Plot each mask
        for i in range(n_masks):
            # Plot original image
            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f'Original Image (Mask {i+1})')
            axes[i, 0].axis('off')
            
            # Plot predicted mask
            pred_mask = pred_masks[i]
            if pred_mask.ndim == 3:  # If shape is (1, H, W)
                pred_mask = pred_mask.squeeze(0)
            axes[i, 1].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title(f'Predicted Mask {i+1}')
            axes[i, 1].axis('off')
            
            # Plot ground truth mask
            gt_mask = gt_masks[i]
            if gt_mask.ndim == 3:  # If shape is (1, H, W)
                gt_mask = gt_mask.squeeze(0)
            axes[i, 2].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title(f'Ground Truth Mask {i+1}')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        # Create a temporary file to save the figure
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Save figure to temporary file
        plt.savefig(temp_path)
        plt.close(fig)  # Close the figure to free memory
        
        # If MLflow run is provided, log the visualization as an artifact
        if mlflow_run:
            artifact_path = f"visualizations/epoch_{epoch}"
            mlflow.log_artifact(temp_path, artifact_path)
            logger.debug(f"Saved visualization to MLflow as {artifact_path}/batch_{batch_idx}.png")
            # Remove the temporary file after logging
            os.unlink(temp_path)
        # If output_dir is provided, save to local directory
        elif output_dir:
            # Create output directory
            vis_dir = output_dir / f'epoch_{epoch}'
            vis_dir.mkdir(parents=True, exist_ok=True)
            
            # Save figure
            output_file = vis_dir / f'batch_{batch_idx}.png'
            logger.debug(f"Saving visualization to {output_file}")
            # Move the temporary file to the output directory
            shutil.move(temp_path, output_file)
        else:
            # If neither MLflow nor output_dir is provided, just remove the temporary file
            os.unlink(temp_path)
            
    except Exception as e:
        logger.warning(f"Error saving visualization for batch {batch_idx} in epoch {epoch}: {str(e)}")
        # Continue without failing the training process

def validate_batch_shapes(batch: Dict[str, torch.Tensor], batch_idx: int = None) -> None:
    """Validate shapes of tensors in a batch.
    
    Args:
        batch: Dictionary containing batch tensors
        batch_idx: Optional batch index for logging
    
    Raises:
        ValueError: If tensor shapes are invalid
    """
    # Expected shapes
    expected_shapes = {
        'pixel_values': (None, 3, 1024, 1024),  # [B, 3, 1024, 1024]
        'input_boxes': (None, None, 4),         # [B, max_masks, 4]
        'labels': (None, None, 256, 256),       # [B, max_masks, H, W]
        'num_masks_per_sample': (None,)         # [B]
    }
    
    # Validate each tensor
    for key, expected_shape in expected_shapes.items():
        if key not in batch:
            raise ValueError(f"Missing required key '{key}' in batch")
            
        tensor = batch[key]
        actual_shape = tensor.shape
        
        # Check number of dimensions
        if len(actual_shape) != len(expected_shape):
            raise ValueError(
                f"Wrong number of dimensions for '{key}'. "
                f"Expected {len(expected_shape)}, got {len(actual_shape)}"
            )
        
        # Check each dimension that's not None in expected shape
        for i, (expected_dim, actual_dim) in enumerate(zip(expected_shape, actual_shape)):
            if expected_dim is not None and expected_dim != actual_dim:
                raise ValueError(
                    f"Wrong shape for '{key}' at dimension {i}. "
                    f"Expected {expected_dim}, got {actual_dim}"
                )
    
    # Additional validation
    batch_size = batch['pixel_values'].size(0)
    max_masks = batch['input_boxes'].size(1)
    
    # Validate batch size consistency
    for key in ['input_boxes', 'labels']:
        if batch[key].size(0) != batch_size:
            raise ValueError(
                f"Inconsistent batch size for '{key}'. "
                f"Expected {batch_size}, got {batch[key].size(0)}"
            )
    
    # Validate max_masks consistency
    if batch['labels'].size(1) != max_masks:
        raise ValueError(
            f"Inconsistent number of masks between boxes and labels. "
            f"Got {batch['input_boxes'].size(1)} and {batch['labels'].size(1)}"
        )
    
    # Log shapes if batch_idx provided
    if batch_idx is not None and batch_idx == 0:
        logger.info("Batch shapes:")
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                logger.info(f"  {key}: {tensor.shape}")

def smooth_masks(masks: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Apply Gaussian smoothing to predicted masks.
    
    Args:
        masks: Predicted masks tensor [B, N, H, W]
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Smoothed masks tensor [B, N, H, W]
    """
    # Ensure masks are in the right shape
    original_shape = masks.shape
    if len(original_shape) == 3:
        masks = masks.unsqueeze(1)  # Add channel dimension if needed
    
    # Apply Gaussian blur
    padding = int(sigma * 4)
    kernel_size = 2 * padding + 1
    
    # Process each mask in the batch
    smoothed_masks = []
    for i in range(masks.shape[0]):
        batch_masks = []
        for j in range(masks.shape[1]):
            # Apply 2D Gaussian blur
            mask = masks[i, j].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            smoothed = F.gaussian_blur(mask, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
            batch_masks.append(smoothed.squeeze(0).squeeze(0))
        smoothed_masks.append(torch.stack(batch_masks))
    
    # Stack back to original shape
    result = torch.stack(smoothed_masks)
    
    # Threshold to get binary masks
    result = (result > 0.5).float()
    
    # Return in the original shape format
    if len(original_shape) == 3:
        result = result.squeeze(1)
        
    return result

def validate_pred_masks(pred_masks: torch.Tensor, expected_shape: tuple, apply_smoothing: bool = True) -> torch.Tensor:
    """Validate and normalize predicted masks.
    
    Args:
        pred_masks: Predicted masks from model output
        expected_shape: Expected shape of the output masks (H, W)
        apply_smoothing: Whether to apply smoothing to the masks
        
    Returns:
        Normalized and validated masks
    """
    # Original validation code
    if pred_masks is None:
        raise ValueError("Predicted masks cannot be None")
    
    # Reshape if needed
    if len(pred_masks.shape) == 4 and pred_masks.shape[1] == 1:
        pred_masks = pred_masks.squeeze(1)
    
    # Ensure masks match expected shape
    if pred_masks.shape[-2:] != expected_shape:
        pred_masks = F.interpolate(
            pred_masks.unsqueeze(1),
            size=expected_shape,
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
    
    # Apply smoothing if requested
    if apply_smoothing:
        pred_masks = smooth_masks(pred_masks)
    
    return pred_masks

def compute_batch_loss(
    pred_masks: torch.Tensor,
    labels: torch.Tensor,
    num_masks: torch.Tensor,
    loss_fn: torch.nn.Module,
    batch_idx: int = None
) -> torch.Tensor:
    """Compute loss for a batch, handling variable numbers of masks per sample.
    
    Args:
        pred_masks: Predicted masks tensor [B, max_masks, H, W]
        labels: Ground truth masks tensor [B, max_masks, H, W]
        num_masks: Number of valid masks per sample [B]
        loss_fn: Loss function module
        batch_idx: Optional batch index for logging
    
    Returns:
        Average loss value
    """
    batch_size = pred_masks.size(0)
    total_loss = 0
    valid_mask_count = 0
    
    for i in range(batch_size):
        valid_masks = num_masks[i]
        
        if valid_masks > 0:
            # Get predictions and targets for valid masks only
            sample_pred = pred_masks[i:i+1, :valid_masks]    # [1, valid_masks, H, W]
            sample_target = labels[i:i+1, :valid_masks]      # [1, valid_masks, H, W]
            
            # Log shapes for first batch
            if batch_idx == 0 and i == 0:
                logger.info(f"Sample shapes:")
                logger.info(f"  pred: {sample_pred.shape}")
                logger.info(f"  target: {sample_target.shape}")
            
            # Compute loss for this sample
            sample_loss = loss_fn(sample_pred, sample_target)
            total_loss += sample_loss
            valid_mask_count += 1
    
    # Return average loss
    if valid_mask_count > 0:
        return total_loss / valid_mask_count
    else:
        return torch.tensor(0.0, device=pred_masks.device)

def train_one_epoch(
    model: SamModel,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: str,
    epoch: int,
    visualization_dir: Optional[Path] = None
) -> List[float]:
    """Train model for one epoch.
    
    Args:
        model: SAM model
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to use
        epoch: Current epoch number
        visualization_dir: Optional directory for saving visualizations
    
    Returns:
        List of loss values for each batch
    """
    model.train()
    epoch_losses = []
    
    for batch_idx, batch in tqdm(enumerate(train_loader), desc=f"Training epoch {epoch}"):
        try:
            # Move data to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Validate batch shapes
            validate_batch_shapes(batch, batch_idx)
            
            # Forward pass
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_boxes=batch["input_boxes"],
                multimask_output=False
            )
            
            # Validate and normalize predicted masks
            pred_masks = validate_pred_masks(
                outputs.pred_masks,
                expected_shape=batch["labels"].shape[-2:]
            )
            
            # Compute loss
            loss = compute_batch_loss(
                pred_masks=pred_masks,
                labels=batch["labels"],
                num_masks=batch["num_masks_per_sample"],
                loss_fn=loss_fn,
                batch_idx=batch_idx
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Save visualizations
            if visualization_dir and batch_idx % 10 == 0:
                save_prediction_visualization(
                    image=batch["pixel_values"][0],
                    pred_masks=pred_masks[0, :batch["num_masks_per_sample"][0]],
                    gt_masks=batch["labels"][0, :batch["num_masks_per_sample"][0]],
                    batch_idx=batch_idx,
                    epoch=epoch,
                    output_dir=visualization_dir
                )
                
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            raise
            
    return epoch_losses

def validate_one_epoch(
    model: SamModel,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: str,
    epoch: int
) -> List[float]:
    """Validate model for one epoch.
    
    Args:
        model: SAM model
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to use
        epoch: Current epoch number
    
    Returns:
        List of loss values for each batch
    """
    model.eval()
    epoch_losses = []
    
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(val_loader), desc=f"Validation epoch {epoch}"):
            try:
                # Move data to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Validate batch shapes
                validate_batch_shapes(batch)
                
                # Forward pass
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    input_boxes=batch["input_boxes"],
                    multimask_output=False
                )
                
                # Validate and normalize predicted masks
                pred_masks = validate_pred_masks(
                    outputs.pred_masks,
                    expected_shape=batch["labels"].shape[-2:]
                )
                
                # Compute loss
                loss = compute_batch_loss(
                    pred_masks=pred_masks,
                    labels=batch["labels"],
                    num_masks=batch["num_masks_per_sample"],
                    loss_fn=loss_fn
                )
                
                epoch_losses.append(loss.item())
                
            except Exception as e:
                logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                raise
                
    return epoch_losses

def train_sam_model(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    model_save_path: Optional[Path] = None,
    num_epochs: int = 10,
    learning_rate: float = 1e-5,
    weight_decay: float = 0,
    visualization_dir: Optional[Path] = None,
    device: Optional[str] = None,
    progress_callback: Optional[Any] = None,
    mlflow_run = None,
    early_stopping_patience: int = 20,
    early_stopping_min_delta: float = 0.0001
) -> Tuple[Dict[str, torch.Tensor], plt.Figure]:
    """Train/fine-tune SAM model.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        model_save_path: Optional path to save trained model locally
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        visualization_dir: Optional directory for saving visualizations
        device: Device to use for training
        progress_callback: Optional callback for tracking progress
        mlflow_run: Optional MLflow run to log artifacts to
        early_stopping_patience: Number of epochs to wait for improvement before stopping
        early_stopping_min_delta: Minimum change to qualify as improvement
        
    Returns:
        Tuple containing:
        - Best model state dictionary
        - Figure with training/validation loss plots
    """
    # Initialize device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    
    # Freeze encoder parameters
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    
    model.to(device)
    
    # Initialize optimizer and loss
    optimizer = Adam(
        model.mask_decoder.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    loss_fn = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    
    # Training tracking
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    
    # Early stopping variables
    patience_counter = 0
    
    # Notify callback of training start if provided
    if progress_callback:
        progress_callback.on_train_start(len(train_loader))
    
    # Training loop
    for epoch in range(num_epochs):
        if progress_callback:
            progress_callback.on_epoch_start(epoch + 1)
        
        # Training phase
        model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Move data to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Validate batch shapes
                validate_batch_shapes(batch, batch_idx)
                
                # Forward pass
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    input_boxes=batch["input_boxes"],
                    multimask_output=False
                )
                
                # Validate and normalize predicted masks
                pred_masks = validate_pred_masks(
                    outputs.pred_masks,
                    expected_shape=batch["labels"].shape[-2:],
                    apply_smoothing=(epoch > 0)  # Only apply smoothing after epoch 0
                )
                
                # Compute loss
                loss = compute_batch_loss(
                    pred_masks=pred_masks,
                    labels=batch["labels"],
                    num_masks=batch["num_masks_per_sample"],
                    loss_fn=loss_fn,
                    batch_idx=batch_idx
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
                
                # Update progress callback
                if progress_callback:
                    progress_callback.on_batch_end({'loss': loss.item()})
                
                # Save visualizations
                if (visualization_dir or mlflow_run) and batch_idx % 10 == 0:
                    save_prediction_visualization(
                        image=batch["pixel_values"][0],
                        pred_masks=pred_masks[0, :batch["num_masks_per_sample"][0]],
                        gt_masks=batch["labels"][0, :batch["num_masks_per_sample"][0]],
                        batch_idx=batch_idx,
                        epoch=epoch,
                        output_dir=visualization_dir,
                        mlflow_run=mlflow_run
                    )
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                raise
        
        train_loss = mean(epoch_losses)
        train_losses.append(train_loss)
        
        # Validation phase
        val_epoch_losses = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch
        )
        val_loss = mean(val_epoch_losses)
        val_losses.append(val_loss)
        
        # Track best model
        if val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            logger.info(f'New best validation loss: {val_loss:.4f}')
            patience_counter = 0
            
            # Log best model to MLflow directly
            if mlflow_run:
                logger.info("Logging best model to MLflow")
                # Save model to a temporary directory first
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_model_path = Path(tmp_dir) / "best_model"
                    model.save_pretrained(str(tmp_model_path))
                    # Log the model directory as an artifact
                    mlflow.log_artifacts(str(tmp_model_path), "best_model")
        else:
            patience_counter += 1
            logger.info(f'No improvement in validation loss. Patience: {patience_counter}/{early_stopping_patience}')
            
            # Check if early stopping criteria is met
            if patience_counter >= early_stopping_patience:
                logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                break
        
        # Update progress callback with epoch metrics
        if progress_callback:
            progress_callback.on_epoch_end({
                'train_loss': train_loss,
                'val_loss': val_loss
            })
    
    # Notify callback of training completion
    if progress_callback:
        progress_callback.on_train_end()
    
    # Save final model
    if mlflow_run:
        # Log final model to MLflow
        logger.info("Logging final model to MLflow")
        # Save model to a temporary directory first
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_model_path = Path(tmp_dir) / "model"
            model.save_pretrained(str(tmp_model_path))
            # Log the model directory as an artifact
            mlflow.log_artifacts(str(tmp_model_path), "model")
            
            # Also log as a PyTorch model for inference
            # Create a sample input for the model using an actual sample from the validation set
            logger.info("Creating input example from validation data for model signature")
            try:
                # Get a sample batch from the validation loader
                sample_batch = next(iter(val_loader))
                
                # Move tensors to CPU for MLflow
                sample_input = {
                    "pixel_values": sample_batch["pixel_values"][0:1].detach().cpu(),
                    "input_points": sample_batch["input_points"][0:1].detach().cpu() if "input_points" in sample_batch else torch.zeros((1, 1, 2)),
                    "input_labels": sample_batch["input_labels"][0:1].detach().cpu() if "input_labels" in sample_batch else torch.ones((1, 1)),
                    "input_boxes": None,
                    "input_masks": None
                }
                logger.info(f"Created input example with image shape: {sample_input['pixel_values'].shape}")
                
                # Log the model with MLflow
                mlflow.pytorch.log_model(
                    model,
                    "pytorch_model",
                    registered_model_name="sam_segmentation",
                    input_example=sample_input
                )
            except Exception as e:
                logger.warning(f"Failed to create input example from validation data: {str(e)}")
                # Fallback to zeros if there's an issue
                logger.info("Using fallback zeros tensor for input example")
                sample_input = {
                    "pixel_values": torch.zeros((1, 3, 1024, 1024)),
                    "input_points": torch.zeros((1, 1, 2)),
                    "input_labels": torch.ones((1, 1)),
                    "input_boxes": None,
                    "input_masks": None
                }
                
                # Log the model with MLflow
                mlflow.pytorch.log_model(
                    model,
                    "pytorch_model",
                    registered_model_name="sam_segmentation",
                    input_example=sample_input
                )
    elif model_save_path:
        # Save model locally if MLflow is not available and path is provided
        logger.info(f"Saving model locally to {model_save_path}")
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(model_save_path))
    else:
        logger.warning("No MLflow run or model_save_path provided, model will not be saved")
    
    # Create loss plot
    fig = plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    
    # If MLflow run is provided, log the loss plot directly
    if mlflow_run:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
        plt.savefig(temp_path)
        mlflow.log_artifact(temp_path, "plots")
        os.unlink(temp_path)
    
    return best_model_state, fig 