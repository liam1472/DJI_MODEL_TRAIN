#!/bin/bash


echo "Starting DJI YOLOv8 training for large dataset..."

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/ubuntu/repos/DJI_MODEL_TRAIN/mmyolo:$PYTHONPATH

cd /home/ubuntu/repos/DJI_MODEL_TRAIN/mmyolo

mkdir -p ./work_dirs/standard_training

python tools/train.py \
    configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py \
    --work-dir ./work_dirs/standard_training \
    --amp

echo "Training completed. Check results in ./work_dirs/standard_training/"
echo ""
echo "Standard training configuration:"
echo "- Full epochs (500) for comprehensive training"
echo "- Low score threshold (0.001) for high recall"
echo "- Minimal augmentation to focus on model capacity"
echo "- SGD optimizer with standard schedule"
echo "- Regular validation (every 10 epochs)"
echo ""
echo "Dataset size: 2000+ images"
echo "Expected training time: 8-16 hours on single GPU"
echo "Expected performance: Excellent (mAP@0.5: 0.8-0.95)"
