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
   - Shape: `(batch_size, num_boxes, 4)` where 4 represents `[x1, y1, x2, y2]` in pixel coordinates
   - Example: For a single box, use shape `(1, 1, 4)`

3. **Masks**
   - Format: PIL Image (important!)
   - Mode: 'L' (grayscale)
   - Values: Binary (0 or 255)
   - Note: Must be converted from numpy array to PIL Image before processing

### Example Usage

```python
# Convert mask from numpy to PIL
mask_2d = mask.squeeze()  # Remove channel dimension if present
mask_uint8 = (mask_2d * 255).astype(np.uint8)  # Convert to uint8
mask_pil = PIL.Image.fromarray(mask_uint8)

# Process with SAM
processed = processor(
    images=image,
    input_boxes=[[box]],  # Note the double brackets for shape (1, 1, 4)
    segmentation_maps=[mask_pil],  # Pass as list of PIL Images
    return_tensors="pt"
)

# Access processed mask
if 'labels' in processed:
    processed_mask = processed['labels']
```

### Common Issues and Solutions

1. **Mask Processing**
   - Issue: `segmentation_maps` parameter doesn't work as expected
   - Solution: Convert masks to PIL Images and access them using the 'labels' key in the output

2. **Box Format**
   - Issue: Box dimensions must match SAM's expectations
   - Solution: Ensure boxes are in format `(batch_size, num_boxes, 4)` using proper nesting or tensor reshaping

3. **Output Format**
   - Processed images: Available in `pixel_values` key
   - Processed boxes: Available in `input_boxes` key
   - Processed masks: Available in `labels` key (not `segmentation_maps`)