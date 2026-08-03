#!/bin/bash
# Download MobileNet-SSD Caffe model files for person detection
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../models"

echo "=== Downloading MobileNet-SSD model ==="
echo "Target directory: ${MODEL_DIR}"

if [ ! -f "${MODEL_DIR}/MobileNetSSD_deploy.prototxt" ]; then
    echo "Downloading prototxt..."
    wget -q -O "${MODEL_DIR}/MobileNetSSD_deploy.prototxt" \
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
    echo "  prototxt downloaded."
else
    echo "  prototxt already exists, skipping."
fi

if [ ! -f "${MODEL_DIR}/MobileNetSSD_deploy.caffemodel" ]; then
    echo "Downloading caffemodel (this may take a moment)..."
    wget -q --show-progress -O "${MODEL_DIR}/MobileNetSSD_deploy.caffemodel" \
        "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"
    echo "  caffemodel downloaded."
else
    echo "  caffemodel already exists, skipping."
fi

echo "=== Done. Model files in ${MODEL_DIR} ==="
ls -lh "${MODEL_DIR}/"
