"""
@author: Tiago Roxo, UBI
@date: 2021
"""
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.argparser import *
from draw_keypoints import *

from os import listdir
from os.path import isfile, join

from tqdm import tqdm

import os
from utils.head_detect_config import OUTPUT_DIR

def parse_arguments():
    """
    Parser arguments given. Enables the display of "help" funcionality

    @Returns: arguments parsed and stores the parameters given to the python file in appropriate variables
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(INPUT, INPUT_EXTENDED,default=INPUT_DEFAULT, help=INPUT_HELP)
    parser.add_argument(METHOD, METHOD_EXTENDED,default=METHOD_DEFAULT, help=METHOD_HELP)
    parser.add_argument(DRAWING_DIR_EXTENDED)
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse_arguments()

    list_imgs = [f for f in listdir(args.img_dir) if isfile(join(args.img_dir, f))]
    list_imgs.sort()

    os.makedirs(OUTPUT_DIR, exist_ok = True)
    output_dir   = None
    json_kp_path = None
    method_name  = None 

    output_dir      = os.path.join(OUTPUT_DIR, args.method+"/draw")
    output_dir_crop = os.path.join(OUTPUT_DIR, args.method+"/crop")
    os.makedirs(output_dir, exist_ok = True)
    os.makedirs(output_dir_crop, exist_ok = True)

    json_kp_path = args.input
    method_name = args.method

    for img_id in tqdm(list_imgs):
        img_id = img_id
        draw_keypoints_img(json_kp_path, method_name, img_id, output_dir, output_dir_crop, args.img_dir)
        
