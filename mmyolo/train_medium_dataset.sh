#!/bin/bash


echo "Starting DJI YOLOv8 training optimized for medium dataset..."

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/ubuntu/repos/DJI_MODEL_TRAIN/mmyolo:$PYTHONPATH

cd /home/ubuntu/repos/DJI_MODEL_TRAIN/mmyolo

mkdir -p ./work_dirs/medium_dataset_training

python tools/train.py \
    configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco_medium_dataset.py \
    --work-dir ./work_dirs/medium_dataset_training \
    --amp

echo "Training completed. Check results in ./work_dirs/medium_dataset_training/"
echo ""
echo "Medium dataset optimizations applied:"
echo "- Moderate epochs (300) for balanced training"
echo "- Balanced score threshold (0.15) for good precision/recall"
echo "- Moderate data augmentation pipeline"
echo "- SGD optimizer for stable convergence"
echo "- Standard validation frequency (every 10 epochs)"
echo ""
echo "Dataset size: 500-2000 images"
echo "Expected training time: 4-8 hours on single GPU"
echo "Expected performance: Very good (mAP@0.5: 0.7-0.9)"
