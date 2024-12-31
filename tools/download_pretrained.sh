mkdir pretrained &&
cd pretrained &&
wget https://bevfusion.mit.edu/files/pretrained_updated/bevfusion-det.pth &&
wget https://bevfusion.mit.edu/files/pretrained_updated/bevfusion-seg.pth &&
wget https://bevfusion.mit.edu/files/pretrained/lidar-only-det.pth &&
wget https://bevfusion.mit.edu/files/pretrained/lidar-only-seg.pth &&
wget https://bevfusion.mit.edu/files/pretrained_updated/camera-only-det.pth &&
wget https://bevfusion.mit.edu/files/pretrained_updated/camera-only-seg.pth &&
wget https://bevfusion.mit.edu/files/pretrained_updated/swint-nuimages-pretrained.pth

#!/usr/bin/env bash

MODEL_URL="https://github.com/ldtho/DifFUSER/releases/download/1.0.0/DifFUSER-seg.pth"

echo "Downloading the model from: $MODEL_URL"
wget "$MODEL_URL" -O DifFUSER-seg.pth

echo "Model downloaded to $(pwd)/DifFUSER-seg.pth"
