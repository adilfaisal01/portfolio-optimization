#!/bin/bash
set -e
service ssh start

echo "JEPA Training Pipeline"
echo "Device: $(python3 -c 'import torch; print("CUDA" if torch.cuda.is_available() else "CPU")')"
echo "GPUs visible: $(python3 -c 'import torch; print(torch.cuda.device_count())')"

case "$1" in
    sweep)
        echo "Starting JEPA hyperparameter sweep..."
        exec bash /portfolio-opt/sweep.sh runpod
        ;;
    *)
        echo "Starting single JEPA training run..."
        python3 -u /portfolio-opt/jepa-training.py
        ;;
esac
