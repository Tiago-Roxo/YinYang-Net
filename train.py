import os
import pprint
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import AttrDataset, get_transform
from torch.nn import BCEWithLogitsLoss as BCLogit

from models.resnet import resnet50
from tools.function import get_model_log_path, get_pedestrian_metrics, match_pedict_gt_scale, get_gender_accuracy
from tools.utils import time_str, save_ckpt


# ----------- Variables -----------
parser = argument_parser()
args = parser.parse_args()

backbone = resnet50()
criterion = BCLogit() 

dataset_name       = args.dataset.lower().split("_")[0] # remove variations of peta
dataset_model_name = args.dataset_model_name.lower()

dict_attrib_index_dataset = {"peta" : -1, "pa100k": -4, "rap": -14} # gender attribute index

att_dataset = dict_attrib_index_dataset[dataset_name]
att_model   = -1

dict_attrib_dataset = {"peta" : 35, "pa100k": 26, "rap": 51}
num_att = dict_attrib_dataset[dataset_model_name]
num_att = 1


# Frontal model
if "frontal" in args.dataset_model_name.lower():
    from models.base_block_meta_frontal import MetaModel
# Any pose model
else:
    from models.base_block_meta import MetaModel

# -------------------------------


def main(args):

    visenv_name = args.dataset
    exp_dir = os.path.join('exp_result')
    model_dir, log_dir = get_model_log_path(exp_dir, visenv_name)
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')
    save_model_path = os.path.join(model_dir, 'ckpt_max.pth')

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


    model = MetaModel(nattr=num_att)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    model_params = model.module.fresh_params()
    param_groups = [{'params': model_params, 'lr': args.lr}]

    optimizer = torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)

    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=4)

    best_metric, epoch = trainer(epoch=args.train_epoch,
                                 model=model,
                                 train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 lr_scheduler=lr_scheduler,
                                 path=save_model_path)

    print(f'{visenv_name},  best_metrc : {best_metric} in epoch{epoch}')


def trainer(epoch, model, train_loader, valid_loader, criterion, optimizer, lr_scheduler,
            path):
    maximum = float(-np.inf)
    best_epoch = 0
    
    result_list = defaultdict()

    for i in range(epoch):

        train_loss, train_gt, train_probs = batch_trainer(
            epoch=i,
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            att_dataset=att_dataset,
            att_model=att_model
        )

        valid_loss, valid_gt, valid_probs = valid_trainer(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
            att_dataset=att_dataset,
            att_model=att_model
        )

        lr_scheduler.step(metrics=valid_loss, epoch=i)

        print(f'{time_str()}')
        print('-' * 60)


        valid_gt = match_pedict_gt_scale(valid_gt, dataset_name, dataset_model_name)

        valid_result = get_pedestrian_metrics(valid_gt, valid_probs)

        print(f'Gender evaluation: ',
        'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f}'.format(
            valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)))
    
        tot_right, tot = get_gender_accuracy(valid_gt, valid_probs)

        print("Gender Accuracy = {:.2f}".format((tot_right/tot)*100))

        cur_metric = valid_result.ma

        if cur_metric > maximum:
            print(path)
            print("Saving model in epoch {}. Acc = {} (previous {})".format(i, cur_metric, maximum))
            maximum = cur_metric
            best_epoch = i
            save_ckpt(model, path, i, maximum)

       
    torch.save(result_list, os.path.join(os.path.dirname(path), 'metric_log.pkl'))

    return maximum, best_epoch


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)

