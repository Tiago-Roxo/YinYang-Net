#!/bin/bash

DATASET="PETA"
DATASET_MODEL="PETA"
MODEL_CKPTS="exp_result/PETA/peta_best.pth"
CUDA_VISIBLE_DEVICES=0 python3 infer.py $DATASET $DATASET_MODEL --model_ckpts $MODEL_CKPTS