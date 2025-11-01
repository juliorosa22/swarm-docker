#!/bin/bash

docker run -it --rm \
  --gpus all \
  --env="NVIDIA_VISIBLE_DEVICES=all" \
  --env="NVIDIA_DRIVER_CAPABILITIES=all" \
  --env="SDL_VIDEODRIVER=offscreen" \
  --env="SDL_HINT_CUDA_DEVICE=0" \
  --env="DISPLAY=${DISPLAY}" \
  --env="XDG_RUNTIME_DIR=/tmp/runtime-ue4" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /tmp/runtime-ue4:/tmp/runtime-ue4:rw \
  -v ./settings.json:/home/ue4/Documents/AirSim/settings.json \
  --name ue4-gpu-headless \
  ue-airsim:4.27