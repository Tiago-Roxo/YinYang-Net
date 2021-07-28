"""
@author: Tiago Roxo, UBI
@date: 2021
"""

import json 
from utils.head_detect_config import SCORE, IMAGE_ID, BOUNDING_BOX, KEYPOINTS, SCORE_THRESHOLD, COCO_FORMAT


def read_json_file(file_path):
    """
    Read json file given its name and using input directory information, from config.py

    @Return: If the json file read is from methods it will be a list of dict. If it is from ground truth it will be a dictionary
    """

    with open(file_path) as json_file: 
        keypoints_file_content = json.load(json_file)
    
    return keypoints_file_content


def get_list_keypoints_img(list_keypoints_dict, compared_image):
    """
    Given a list of keypoints dict, each with a key from different images, stores the keypoints (list) of dict whose keys 
    match the id of compared image, given as argument

    @Input: 
        - list_keypoints_dict: list of keypoints dictionaries, each dict following the COCO submission format.
        - compared_image     : image id (int)

    @Return: list of keypoints (list of list). Each keypoint is a list with 51 elements (17x3)
    """
    list_keypoints_img = []
    for e in list_keypoints_dict:
        for k, v in e.items():
            if k == IMAGE_ID and v == compared_image:
                list_keypoints_img.append(e[KEYPOINTS])
    return list_keypoints_img


def read_json_into_list(filename):
    """
    Receives JSON file input of methods, according to COCO submission format

    @Return: list of keypoints detection (COCO format)
    """
    keypoints_file_content = read_json_file(filename)

    # Get list with COCO format for keypoints
    list_keypoints = []
    
    for e in keypoints_file_content:
        dict_keypoints = {}
        for k, _ in e.items():
            if k in COCO_FORMAT:
                dict_keypoints[k] = e[k]
        if dict_keypoints[SCORE] > SCORE_THRESHOLD: # Make sure proposal score is higher than threshold
            list_keypoints.append(dict_keypoints)
        
    return list_keypoints

