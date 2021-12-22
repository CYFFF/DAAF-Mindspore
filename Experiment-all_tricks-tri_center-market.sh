#!/usr/bin/env bash
# Experiment all tricks with center loss : 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005
# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# with center loss
nohup python3 tools/train.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('Market-1501')" DATASETS.ROOT_DIR "('/home/chenyf')" OUTPUT_DIR "('/home/chenyf/experiments/reid_baseline_review/exp_106')" > exp_106.file 2>&1 &

