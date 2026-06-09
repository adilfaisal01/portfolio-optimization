#!/bin/bash
set -e

export TRAIN_LR=3e-4
export TRAIN_NUM_EPOCHS=50
export TRAIN_SAVE_INTERVAL=5
export TRAIN_MODEL_PATH="/workspace/outputs/jepa_model"
export JEPA_VIX_FAIRWEATHER=20
export JEPA_NUM_PATCHES=20
export JEPA_DIM_IN_ENCODER=49
export JEPA_KERNEL_SIZE=49
export JEPA_MASK_RATIO=0.2
export JEPA_ENCODER_EMBED_DIM=256
export JEPA_NHEAD_ENCODER=8
export JEPA_PREDICTOR_EMBED_DIM=512
export JEPA_N_HEAD_PREDICTOR=4
export JEPA_NUM_LAYERS_ENCODER=4
export JEPA_NUM_LAYERS_PREDICTOR=2
export TRAIN_BATCH_SIZE=16
export TRAIN_EMA_MOMENTUM=0.998

mkdir -p /workspace/outputs

echo "=== Starting single training run ==="
python3 -u /portfolio-opt/jepa-training.py
echo "=== Training complete ==="

sleep infinity
