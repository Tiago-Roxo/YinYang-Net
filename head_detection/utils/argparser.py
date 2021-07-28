"""
@author: Tiago Roxo, UBI
@date: 2021
"""

# Optional Arguments
INPUT              = '-i'
INPUT_EXTENDED     = '--input'
INPUT_DEFAULT      = 'data/jsons/peta.json'
INPUT_HELP         = 'Location of JSON file with method prediction for keypoints in COCO 2017. By default the file is {}'.format(INPUT_DEFAULT)

METHOD             = '-m'
METHOD_EXTENDED    = '--method'
METHOD_DEFAULT     = 'tested_method'
METHOD_HELP        = 'Name of method to create approriate name directories when outputing results'

DRAWING_DIR_EXTENDED   = '--img-dir'  
DRAWING_HELP           = 'Directory of images to detect and crop head region'