import os

import numpy as np
import torch
from easydict import EasyDict

from tools.utils import may_mkdirs


def ratio2weight(targets, ratio):
    ratio = torch.from_numpy(ratio).type_as(targets)
    pos_weights = targets * (1 - ratio)
    neg_weights = (1 - targets) * ratio
    weights = torch.exp(neg_weights + pos_weights)

    # for RAP dataloader, targets element may be 2, with or without smooth, some element must great than 1
    weights[targets > 1] = 0.0

    return weights

def get_reload_weight(model_path, model): 
    checkpoint = torch.load(model_path) 
    model.load_state_dict(checkpoint['state_dicts']) 
    return model

def get_model_log_path(root_path, visenv):
    multi_attr_model_dir = os.path.join(root_path, f'{visenv}', 'img_model')
    may_mkdirs(multi_attr_model_dir)

    multi_attr_log_dir = os.path.join(root_path, f'{visenv}', 'log')
    may_mkdirs(multi_attr_log_dir)

    return multi_attr_model_dir, multi_attr_log_dir


def get_pkl_rootpath(dataset):
    root = os.path.join("./data", f"{dataset}")
    data_path = os.path.join(root, 'dataset.pkl')

    return data_path


def get_pedestrian_metrics(gt_label, preds_probs, threshold=0.5):
    
    pred_label = preds_probs > threshold

    eps = 1e-20
    result = EasyDict()

    ###############################
    # label metrics
    # TP + FN
    gt_pos = np.sum((gt_label == 1), axis=0).astype(float)
    # TN + FP
    gt_neg = np.sum((gt_label == 0), axis=0).astype(float)
    # TP
    true_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=0).astype(float)
    # TN
    true_neg = np.sum((gt_label == 0) * (pred_label == 0), axis=0).astype(float)
    # FP
    false_pos = np.sum(((gt_label == 0) * (pred_label == 1)), axis=0).astype(float)
    # FN
    false_neg = np.sum(((gt_label == 1) * (pred_label == 0)), axis=0).astype(float)

    label_pos_recall = 1.0 * true_pos / (gt_pos + eps)  # true positive
    label_neg_recall = 1.0 * true_neg / (gt_neg + eps)  # true negative
    # mean accuracy
    label_ma = (label_pos_recall + label_neg_recall) / 2

    result.label_pos_recall = label_pos_recall
    result.label_neg_recall = label_neg_recall
    result.label_prec = true_pos / (true_pos + false_pos + eps)
    result.label_acc = true_pos / (true_pos + false_pos + false_neg + eps)
    result.label_f1 = 2 * result.label_prec * result.label_pos_recall / (
            result.label_prec + result.label_pos_recall + eps)

    result.label_ma = label_ma
    result.ma = np.mean(label_ma)

    return result


def match_pedict_gt_scale(target, dataset_name, dataset_model_name):
    if (dataset_name == "rap" or dataset_name == "pa100k") and dataset_model_name == "peta":
        target = np.where(target == 1, 0, 1) # Invert

    if dataset_name == "peta" and (dataset_model_name == "rap" or dataset_model_name == "pa100k"):
        target = np.where(target == 1, 0, 1) # Invert

    return target


def get_gender_accuracy(target, output):

    #target = target.cpu().numpy()
    output = np.where(output > 0.5, 1, 0)
    
    return np.sum(target == output), target.size
