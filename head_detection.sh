#!/bin/bash

INPUT_JSON_PATH="data/jsons/peta_kp.json"
IMG_DIR="data/PETA/images"
DATASET="PETA"

python3 head_detection/main.py -i $INPUT_JSON_PATH --img-dir $IMG_DIR --method $DATASET #> cropping_details.txt   
