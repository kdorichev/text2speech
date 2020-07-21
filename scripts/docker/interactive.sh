#!/usr/bin/env bash

docker run --gpus=all -it --rm -e CUDA_VISIBLE_DEVICES -e "TZ=Europe/Moscow" --ipc=host -v $PWD:/workspace/fastpitch/ -v /etc/timezone:/etc/timezone:ro fastpitch:latest bash 
