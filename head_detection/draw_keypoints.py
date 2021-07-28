"""
@author: Tiago Roxo, UBI
@date: 2021
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from utils.file_use import *
from utils.read_image import *
from utils.visualizer import Visualizer
from utils.head_detect_config import NUMBER_KEYPOINTS

def convert_keypoints_to_list_triplets(list_keypoints):
    """
    Transform list of keypoints into tripplets, which is the expected input for drawing methods.

    @Input: list of keypoints for a singe image. In the case of COCO it is a list with 51 Keypoints (17 x 3)

    @Return: List of lists, with each sublist having 3 elements: [1, 2, 3, 4, 5, 6] => [[1,2,3], [4,5,6]]
    """
    list_list_keypoints = np.reshape(list_keypoints, (NUMBER_KEYPOINTS, 3))

    return list_list_keypoints


def draw_head(img_file, list_list_keypoints):
    """
    Reads image and draws keypoints based on list of keypoints given (predictions). Heavily based on Detectron2 code
    """
    image = read_image(img_file)
    # image = image[:, :, ::-1]
    visualizer = Visualizer(image)
    visualized_output, bbox = visualizer.draw_head(predictions=list_list_keypoints)
    return visualized_output, bbox

def get_bb_image(list_bb_image, image_id):
    list_bb = []
    for e in list_bb_image:
        if e[IMAGE_ID] == image_id:
            dict_img = e
            list_bb.append(dict_img[BOUNDING_BOX])
    return list_bb

def draw_keypoints_img(json_kp_path, method_name, img_id, output_dir, output_dir_crop, input_dir):

    list_keypoints_detect_file = []
    list_keypoints_detect_file = read_json_into_list(json_kp_path)

    list_bb        = get_bb_image(list_keypoints_detect_file, img_id)
    list_keypoints = get_list_keypoints_img(list_keypoints_detect_file, img_id)

    # Conver to triplets
    list_keypoints_ = []
    for e in list_keypoints:
        list_keypoints_.append(convert_keypoints_to_list_triplets(e))

    image_filename_path = os.path.join(input_dir, img_id)
    image_drawn_kp, bbox = draw_head(image_filename_path, list_keypoints_)

    if not image_drawn_kp:
        print("Image {} does not have KP prediction".format(img_id))
        return

    image_drawn_kp.save(os.path.join(output_dir, img_id))

    # Crop
    try:
        image = crop_image(image_filename_path, bbox)
        image.save(os.path.join(output_dir_crop, img_id)) 
    except:
        print("Problems with cropping image {}".format(image_filename_path))
   
    return
