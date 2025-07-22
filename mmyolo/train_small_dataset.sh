#!/bin/bash


echo "Starting DJI YOLOv8 training optimized for small dataset..."

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/ubuntu/repos/DJI_MODEL_TRAIN/mmyolo:$PYTHONPATH

cd /home/ubuntu/repos/DJI_MODEL_TRAIN/mmyolo

mkdir -p ./work_dirs/pothole_detection_optimized

python tools/train.py \
    configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco_small_dataset_optimized.py \
    --work-dir ./work_dirs/pothole_detection_optimized \
    --amp \
    --auto-scale-lr

echo "Training completed. Check results in ./work_dirs/pothole_detection_optimized/"
echo ""
echo "Key optimizations applied:"
echo "- Reduced epochs from 500 to 200 for small dataset"
echo "- Increased score threshold to 0.25 to reduce false positives"
echo "- Enhanced data augmentation pipeline"
echo "- AdamW optimizer for better small dataset performance"
echo "- More frequent validation (every 5 epochs)"
echo "- Disabled multi-label prediction to reduce FPs"
echo ""
echo "Dataset recommendations:"
echo "- Current: 161 images (very small)"
echo "- Recommended: 500-1000 images per class minimum"
echo "- Consider data collection or synthetic data generation"
