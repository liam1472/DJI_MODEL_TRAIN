# DJI YOLOv8 Training Guide

A comprehensive guide for training YOLOv8 models compatible with DJI AI Inside program, including optimizations for different dataset sizes and use cases.

## Table of Contents

- [Overview](#overview)
- [DJI Compatibility Requirements](#dji-compatibility-requirements)
- [Dataset Size Guidelines](#dataset-size-guidelines)
- [Configuration Files](#configuration-files)
- [Class Modification Guide](#class-modification-guide)
- [Training Commands](#training-commands)
- [Use Cases and Examples](#use-cases-and-examples)
- [Model Conversion Limitations](#model-conversion-limitations)
- [Fine-tuning Strategies](#fine-tuning-strategies)
- [Troubleshooting](#troubleshooting)

## Overview

This repository contains DJI-compatible YOLOv8 configurations for object detection training. The models trained with these configurations can be deployed to DJI drones through the DJI AI Inside program.

### Key Features

- ✅ **DJI Compatible**: Maintains all DJI architectural requirements
- 🎯 **Optimized for Small Datasets**: Special configurations for limited data scenarios
- 🚀 **False Positive Reduction**: Tuned parameters to minimize false detections
- 📊 **Multiple Use Cases**: Configurations for different dataset sizes and scenarios
- 🔧 **Easy Customization**: Simple class modification and parameter tuning

## DJI Compatibility Requirements

All configurations maintain these DJI-specific requirements:

```python
# Required activation function (NOT SiLU)
act_cfg=dict(type='ReLU', inplace=True)

# Required preprocessing normalization
mean=[128., 128., 128.]
std=[128., 128., 128.]

# Required SPPF kernel size
kernel_sizes=3  # NOT 5

# Required DFL configuration
skip_dfl=False
```

⚠️ **Important**: These requirements make direct conversion from Ultralytics `.pt` files impossible.

## Dataset Size Guidelines

### Small Dataset (< 500 images)
- **Recommended**: Use `yolov8_s_syncbn_fast_8xb16-500e_coco_small_dataset_optimized.py`
- **Characteristics**: Aggressive data augmentation, reduced epochs, higher learning rate
- **Training time**: 2-4 hours on single GPU
- **Expected performance**: Good with proper augmentation

### Medium Dataset (500-2000 images)
- **Recommended**: Use standard `yolov8_s_syncbn_fast_8xb16-500e_coco.py` with modifications
- **Characteristics**: Moderate augmentation, standard training schedule
- **Training time**: 4-8 hours on single GPU
- **Expected performance**: Very good

### Large Dataset (> 2000 images)
- **Recommended**: Use standard configuration with minimal augmentation
- **Characteristics**: Focus on model capacity, longer training
- **Training time**: 8-16 hours on single GPU
- **Expected performance**: Excellent

## Configuration Files

### 1. Small Dataset Optimized Configuration

**File**: `configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco_small_dataset_optimized.py`

**Use Case**: Datasets with < 500 images total

**Key Features**:
- Enhanced data augmentation pipeline
- Reduced training epochs (200 instead of 500)
- Higher score threshold (0.25) to reduce false positives
- AdamW optimizer for better small dataset performance
- More frequent validation (every 5 epochs)

### 2. Standard DJI Configuration

**File**: `configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py`

**Use Case**: Standard datasets with 500+ images

**Key Features**:
- Balanced augmentation
- Standard training schedule (500 epochs)
- SGD optimizer
- Regular validation intervals

## Class Modification Guide

### Step 1: Update Class Information

```python
# Modify these parameters in your config file
num_classes = 3  # Change to your number of classes

class_name = ['pothole', 'crack', 'manhole']  # Your class names
metainfo = dict(
    classes=class_name,
    palette=[(20, 220, 60), (0, 0, 255), (255, 0, 0)]  # RGB colors for visualization
)
```

### Step 2: Update Dataset Paths

```python
# Update these paths to point to your dataset
data_root = 'path/to/your/dataset/'
train_ann_file = 'path/to/your/train_annotations.json'
train_data_prefix = 'path/to/your/train/images/'
val_ann_file = 'path/to/your/val_annotations.json'
val_data_prefix = 'path/to/your/val/images/'
```

### Step 3: Adjust Model Configuration

```python
# The model will automatically use your num_classes
model = dict(
    bbox_head=dict(
        head_module=dict(
            num_classes=num_classes,  # Automatically updated
            # ... other parameters
        )
    ),
    train_cfg=dict(
        assigner=dict(
            num_classes=num_classes,  # Automatically updated
            # ... other parameters
        )
    )
)
```

### Example: 5-Class Road Defect Detection

```python
num_classes = 5
class_name = ['pothole', 'crack', 'manhole', 'debris', 'marking_fade']
metainfo = dict(
    classes=class_name,
    palette=[
        (20, 220, 60),   # Green for pothole
        (0, 0, 255),     # Red for crack
        (255, 0, 0),     # Blue for manhole
        (255, 255, 0),   # Yellow for debris
        (255, 0, 255)    # Magenta for marking_fade
    ]
)
```

## Training Commands

### Small Dataset Training (< 500 images)

```bash
# Navigate to mmyolo directory
cd /path/to/DJI_MODEL_TRAIN/mmyolo

# Make training script executable
chmod +x train_small_dataset.sh

# Run optimized training for small datasets
./train_small_dataset.sh
```

**Manual command**:
```bash
python tools/train.py \
    configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco_small_dataset_optimized.py \
    --work-dir ./work_dirs/your_project_name \
    --amp \
    --auto-scale-lr
```

### Standard Dataset Training (500+ images)

```bash
python tools/train.py \
    configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py \
    --work-dir ./work_dirs/your_project_name \
    --amp
```

### Multi-GPU Training

```bash
# For 2 GPUs
bash tools/dist_train.sh \
    configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py \
    2 \
    --work-dir ./work_dirs/your_project_name
```

### Resume Training

```bash
python tools/train.py \
    configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py \
    --work-dir ./work_dirs/your_project_name \
    --resume
```

## Use Cases and Examples

### Use Case 1: Road Infrastructure Inspection

**Dataset**: 300 images of road defects
**Classes**: 3 (pothole, crack, manhole)
**Configuration**: Small dataset optimized

```python
# Configuration snippet
num_classes = 3
class_name = ['pothole', 'crack', 'manhole']
max_epochs = 200
base_lr = 0.02
score_thr = 0.25  # High threshold to reduce false positives
```

**Training Command**:
```bash
./train_small_dataset.sh
```

**Expected Results**:
- Training time: ~3 hours
- mAP@0.5: 0.6-0.8 (depending on data quality)
- False positive rate: Low due to optimized thresholds

### Use Case 2: Vehicle Detection

**Dataset**: 1500 images of vehicles
**Classes**: 3 (car, truck, bus)
**Configuration**: Standard with modifications

```python
# Configuration snippet
num_classes = 3
class_name = ['car', 'truck', 'bus']
max_epochs = 500
base_lr = 0.01
score_thr = 0.001  # Lower threshold for vehicle detection
```

**Training Command**:
```bash
python tools/train.py \
    configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py \
    --work-dir ./work_dirs/vehicle_detection
```

**Expected Results**:
- Training time: ~6 hours
- mAP@0.5: 0.8-0.9
- Good generalization to different vehicle types

### Use Case 3: Construction Site Safety

**Dataset**: 800 images of safety equipment
**Classes**: 4 (helmet, vest, person, vehicle)
**Configuration**: Standard with safety-specific tuning

```python
# Configuration snippet
num_classes = 4
class_name = ['helmet', 'vest', 'person', 'vehicle']
max_epochs = 400
loss_cls_weight = 1.5  # Emphasize classification for safety
```

## Model Conversion Limitations

### Why Direct .pt to .pth Conversion Fails

❌ **Architecture Differences**:
- Ultralytics YOLOv8: Uses SiLU activation
- DJI YOLOv8: Requires ReLU activation

❌ **Preprocessing Incompatibility**:
- Ultralytics: `mean=[0,0,0], std=[255,255,255]`
- DJI: `mean=[128,128,128], std=[128,128,128]`

❌ **Layer Structure Changes**:
- DJI patch modifies SPPF kernel sizes
- Custom deconvolution upsampling layers
- Different CSP layer implementations

### Alternative Approaches

❌ **No DJI-Compatible Pre-trained Models Available**:
```python
# NO load_from parameter - must train from scratch
# OpenMMLab weights are incompatible with DJI architecture modifications
```

✅ **Knowledge Distillation**:
- Train DJI-compatible student model
- Use Ultralytics model as teacher
- Transfer knowledge, not weights

✅ **Feature-based Transfer Learning**:
- Extract features from Ultralytics models
- Train DJI-compatible classifier on features
- Combine approaches for best results

## Fine-tuning Strategies

### Strategy 1: Backbone Freezing

```python
# Freeze backbone layers for faster fine-tuning
model = dict(
    backbone=dict(
        frozen_stages=4,  # Freeze all backbone stages
        # ... other parameters
    )
)
```

### Strategy 2: Progressive Unfreezing

```python
# Custom hook for progressive unfreezing
custom_hooks = [
    dict(
        type='UnfreezeBackboneHook',
        unfreeze_epoch=50,  # Start unfreezing after 50 epochs
        # ... other parameters
    )
]
```

### Strategy 3: Layer-wise Learning Rates

```python
# Different learning rates for different layers
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),  # Lower LR for backbone
            'neck': dict(lr_mult=0.5),      # Medium LR for neck
            'bbox_head': dict(lr_mult=1.0), # Full LR for head
        }
    )
)
```

## Troubleshooting

### High False Positive Rate

**Symptoms**: Model detects objects that aren't there

**Solutions**:
1. Increase `score_thr` in test configuration:
   ```python
   score_thr = 0.3  # Increase from 0.25
   ```

2. Add more background images to training data

3. Reduce `max_per_img` limit:
   ```python
   max_per_img = 50  # Reduce from 100
   ```

### Low Recall (Missing Objects)

**Symptoms**: Model misses actual objects

**Solutions**:
1. Decrease `score_thr`:
   ```python
   score_thr = 0.15  # Decrease from 0.25
   ```

2. Increase `loss_cls_weight`:
   ```python
   loss_cls_weight = 1.5  # Increase from 1.0
   ```

3. Add more positive examples to dataset

### Training Instability

**Symptoms**: Loss curves are erratic, training doesn't converge

**Solutions**:
1. Add warmup epochs:
   ```python
   warmup_epochs = 10  # Increase from 5
   ```

2. Reduce learning rate:
   ```python
   base_lr = 0.01  # Reduce from 0.02
   ```

3. Increase gradient clipping:
   ```python
   clip_grad=dict(max_norm=5.0)  # Reduce from 10.0
   ```

### Out of Memory Errors

**Solutions**:
1. Reduce batch size:
   ```python
   train_batch_size_per_gpu = 2  # Reduce from 4
   ```

2. Use gradient accumulation:
   ```python
   accumulate_grad_batches = 2
   ```

3. Enable mixed precision training:
   ```bash
   python tools/train.py config.py --amp
   ```

### Poor Performance on Small Dataset

**Solutions**:
1. Use the small dataset optimized configuration
2. Increase data augmentation probability:
   ```python
   dict(type='RandomBrightnessContrast', p=0.5)  # Increase from 0.2
   ```

3. Consider synthetic data generation
4. Implement cross-validation for robust evaluation

## Performance Optimization Tips

### For Small Datasets (< 500 images)

1. **Aggressive Augmentation**: Use high probability augmentations
2. **Frequent Validation**: Monitor overfitting closely
3. **Early Stopping**: Implement to prevent overfitting
4. **Cross-validation**: Use k-fold for robust evaluation

### For Large Datasets (> 2000 images)

1. **Reduce Augmentation**: Focus on model capacity
2. **Longer Training**: Use full 500 epochs
3. **Learning Rate Scheduling**: Implement cosine annealing
4. **Multi-scale Training**: Enable for better generalization

### General Tips

1. **Monitor GPU Utilization**: Ensure efficient resource usage
2. **Use Mixed Precision**: Enable AMP for faster training
3. **Batch Size Tuning**: Find optimal batch size for your GPU
4. **Regular Checkpointing**: Save models frequently

## Contributing

When contributing to this repository:

1. Maintain DJI compatibility requirements
2. Test configurations thoroughly
3. Update documentation for new features
4. Follow the existing code style
5. Add appropriate use case examples

## License

This project follows the same license as the base MMYOLO repository.

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review the DJI compatibility requirements
3. Consult the MMYOLO documentation
4. Open an issue with detailed information about your use case

---

**Note**: This guide is specifically for DJI AI Inside compatible YOLOv8 training. For general YOLOv8 training, refer to the standard MMYOLO documentation.
