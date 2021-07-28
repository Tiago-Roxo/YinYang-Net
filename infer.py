import os
import pprint
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import argument_parser
from dataset.AttrDataset import AttrDataset, get_transform

# from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50

from tools.function import get_model_log_path, get_pedestrian_metrics, get_reload_weight, match_pedict_gt_scale, get_gender_accuracy
from tools.utils import time_str


# ----------- Variables -----------
parser = argument_parser()
args = parser.parse_args()


backbone = resnet50()

dataset_name       = args.dataset.lower().split("_")[0] # remove variations of peta
dataset_model_name = args.dataset_model_name.lower().split("_")[0]

num_att = 1

model_path = args.model_ckpts
dict_attrib_index_dataset = {"peta" : -1, "pa100k": -4, "rap": -14} # gender attribute index

# -------------------------------

# Frontal model
if "frontal" in args.dataset.lower():
    from models.base_block_meta_frontal import MetaModel
# Any pose model
else:
    from models.base_block_meta import MetaModel


# -------------------------------

def main(args):
    visenv_name = args.dataset
    exp_dir = os.path.join('exp_result', args.dataset)
    model_dir, log_dir = get_model_log_path(exp_dir, visenv_name)
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')

    pprint.pprint(OrderedDict(args.__dict__))

    print('-' * 60)
    print(f'use GPU{args.device} for training')
    print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')

    train_tsfm, valid_tsfm = get_transform(args)
    print(train_tsfm)

    train_set = AttrDataset(args=args, split=args.train_split, transform=train_tsfm)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    valid_set = AttrDataset(args=args, split=args.valid_split, transform=valid_tsfm)

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f'{args.train_split} set: {len(train_loader.dataset)}, '
          f'{args.valid_split} set: {len(valid_loader.dataset)}, '
          f'attr_num : {train_set.attr_num}')

    # ---------------------------------
    model = MetaModel(nattr=num_att)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    print("reloading pretrained models")
    model = get_reload_weight(model_path, model)

    model.eval()

    preds_probs = []
    gt_list = []

    with torch.no_grad():
        for step, (imgs_all, imgs_face, gt_label, imgname, img_info) in enumerate(tqdm(valid_loader)):
                        
            imgs, imgs_face = imgs_all.cuda(), imgs_face.cuda()
            gt_label = gt_label.cuda()
            gt_list.append(gt_label.cpu().numpy())

            valid_logits = model(imgs_all, imgs_face, img_info.float())

            valid_probs = torch.sigmoid(valid_logits)
            preds_probs.append(valid_probs.cpu().numpy())

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    gender_dataset_index = dict_attrib_index_dataset[dataset_name]
    gender_model_index = -1

    gt_label = gt_label[:,gender_dataset_index] 
    preds_probs = preds_probs[:,gender_model_index] 

    # Match labels in cross domain settings (train and evaluate in different datasets)
    gt_label = match_pedict_gt_scale(gt_label, dataset_name, dataset_model_name)

    valid_result = get_pedestrian_metrics(gt_label, preds_probs)

    print(f'Gender evaluation: ',
    'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f}'.format(
        valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)))

    tot_right, tot = get_gender_accuracy(gt_label, preds_probs)

    print("Gender Accuracy = {:.2f}".format((tot_right/tot)*100))



if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()

    main(args)


