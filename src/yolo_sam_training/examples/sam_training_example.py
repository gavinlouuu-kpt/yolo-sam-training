"""Example script demonstrating how to train/fine-tune SAM model."""

import logging
from pathlib import Path
import torch
from torch.optim import Adam
import monai.losses
from tqdm import tqdm
import matplotlib.pyplot as plt
from statistics import mean
from transformers import SamModel, SamProcessor

from yolo_sam_training.data import (
    load_dataset_from_summary,
    process_dataset_with_sam,
    split_dataset,
    create_dataloaders
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_prediction_visualization(
    image,
    original_size,
    reshaped_input_size,
    pred_mask,
    gt_mask,
    batch_idx,
    epoch,
    output_dir: Path,
    processor: SamProcessor
):
    """Save visualization of model predictions.
    
    TODO:
        - Add more visualization options
        - Implement better formatting
        - Add metrics visualization
    """
    # Create output directory
    vis_dir = output_dir / f'epoch_{epoch}'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot predicted mask
    axes[1].imshow(pred_mask.cpu().numpy(), cmap='gray')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')
    
    # Plot ground truth mask
    axes[2].imshow(gt_mask.cpu().numpy(), cmap='gray')
    axes[2].set_title('Ground Truth Mask')
    axes[2].axis('off')
    
    # Save figure
    plt.savefig(vis_dir / f'batch_{batch_idx}.png')
    plt.close(fig)

def train_sam_model(
    train_loader,
    val_loader,
    model_save_path: Path,
    num_epochs: int = 10,
    learning_rate: float = 1e-5,
    weight_decay: float = 0,
    visualization_dir: Optional[Path] = None,
    device: str = None
) -> Tuple[dict, plt.Figure]:
    """Train/fine-tune SAM model.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        model_save_path: Path to save trained model
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        visualization_dir: Optional directory for saving visualizations
        device: Device to use for training ('cuda' or 'cpu')
        
    Returns:
        Tuple of (best model state dict, loss plot figure)
    """
    # Initialize model and move to device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    
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
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_losses = []
        
        for batch_idx, batch in tqdm(enumerate(train_loader), desc=f"Training epoch {epoch}"):
            # Move data to device
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            
            # Forward pass
            outputs = model(
                pixel_values=pixel_values,
                input_boxes=input_boxes,
                multimask_output=False
            )
            
            # Compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            loss = loss_fn(predicted_masks, ground_truth_masks.unsqueeze(1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Save visualizations
            if visualization_dir and batch_idx % 10 == 0:
                save_prediction_visualization(
                    image=batch["original_data"][0]['image'],
                    original_size=batch["original_data"][0]['sam_data']['original_size'],
                    reshaped_input_size=pixel_values.shape[-2:],
                    pred_mask=predicted_masks[0],
                    gt_mask=ground_truth_masks[0],
                    batch_idx=batch_idx,
                    epoch=epoch,
                    output_dir=visualization_dir,
                    processor=processor
                )
        
        train_loss = mean(epoch_losses)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_epoch_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation epoch {epoch}"):
                # Move data to device
                pixel_values = batch["pixel_values"].to(device)
                input_boxes = batch["input_boxes"].to(device)
                ground_truth_masks = batch["ground_truth_mask"].float().to(device)
                
                # Forward pass
                outputs = model(
                    pixel_values=pixel_values,
                    input_boxes=input_boxes,
                    multimask_output=False
                )
                
                # Compute loss
                predicted_masks = outputs.pred_masks.squeeze(1)
                val_loss = loss_fn(predicted_masks, ground_truth_masks.unsqueeze(1))
                val_epoch_losses.append(val_loss.item())
        
        val_loss = mean(val_epoch_losses)
        val_losses.append(val_loss)
        
        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            logger.info(f'New best validation loss: {val_loss:.4f}')
        
        # Log progress
        logger.info(f'Epoch {epoch}:')
        logger.info(f'  Training loss: {train_loss:.4f}')
        logger.info(f'  Validation loss: {val_loss:.4f}')
    
    # Save model
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_save_path))
    processor.save_pretrained(str(model_save_path) + '_processor')
    
    # Create loss plot
    fig = plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    
    return best_model_state, fig

def main():
    # Load and preprocess dataset
    summary_path = Path('D:/code/ai_cytometry/data/example_training_data/summary.json')
    logger.info("Loading dataset...")
    dataset = load_dataset_from_summary(summary_path)
    
    processed_dataset = process_dataset_with_sam(
        dataset=dataset,
        save_visualizations=False
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
        batch_size=4,
        num_workers=4
    )
    
    # Train model
    model_save_path = Path('models/sam_fine_tuned')
    visualization_dir = Path('visualization_output/training')
    
    best_model_state, loss_plot = train_sam_model(
        train_loader=train_loader,
        val_loader=test_loader,
        model_save_path=model_save_path,
        visualization_dir=visualization_dir,
        num_epochs=10,
        learning_rate=1e-5
    )
    
    # Save loss plot
    loss_plot.savefig(visualization_dir / 'loss_plot.png')
    plt.close(loss_plot)
    
    logger.info("Training complete!")
    logger.info(f"Model saved to: {model_save_path}")
    logger.info(f"Visualizations saved to: {visualization_dir}")

if __name__ == '__main__':
    main() 