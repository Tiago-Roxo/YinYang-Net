import os
import pickle

import numpy as np
import torch.utils.data as data
from PIL import Image

from tools.function import get_pkl_rootpath
import torchvision.transforms as T
from pathlib import Path

class AttrDataset(data.Dataset):

    def __init__(self, split, args, transform=None, target_transform=None):

        data_path = get_pkl_rootpath(args.dataset)

        dataset_info = pickle.load(open(data_path, 'rb+'))

        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.dataset = args.dataset
        self.transform = transform
        self.target_transform = target_transform

        self.root_path = dataset_info.root

        self.attr_id = dataset_info.attr_name
        self.attr_num = len(self.attr_id)

        self.img_idx = dataset_info.partition[split]

        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]  # default partition 0
        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]
        self.label = attr_label[self.img_idx]

        self.image_info = dataset_info.partition[split+"_info"][0] # default partition 0
        

    def __getitem__(self, index):

        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]
        imgpath = os.path.join(self.root_path, imgname)

        img = Image.open(imgpath)

        if self.dataset == "PETA" or self.dataset == "peta_frontal":
            face_imgpath = imgpath.replace("images", "face_images")
        elif self.dataset == "RAP" or self.dataset == "rap_frontal":
            face_imgpath = imgpath.replace("RAP/RAP_dataset", "RAP/face_images")
        elif self.dataset == "PA100k" or self.dataset == "pa100k_frontal":
            face_imgpath = imgpath.replace("PA100k/data", "PA100k/face_images")
        # elif self.dataset == "peta_frontal":
        #     face_imgpath = imgpath.replace("images", "face_images_frontal")
        # elif self.dataset == "rap_frontal":
        #     face_imgpath = imgpath.replace("RAP/RAP_dataset", "RAP/face_images_frontal")
        # elif self.dataset == "pa100k_frontal":
        #     face_imgpath = imgpath.replace("PA100k/data", "PA100k/face_images_frontal")
        else:
            assert False

        if Path(face_imgpath).exists():
            face_img = Image.open(face_imgpath)
            if self.transform is not None:
                face_img = self.transform(face_img)
        else:
            assert False
            face_img = self.transform(img)
        
        # --------------------------------

        if self.transform is not None:
            img = self.transform(img)

        gt_label = gt_label.astype(np.float32)

        if self.target_transform is not None:
            gt_label = self.transform(gt_label)

        image_info = self.image_info[index]

        return img, face_img, gt_label, imgname, image_info
        

    def __len__(self):
        return len(self.img_id)


def get_transform(args):
    height = args.height
    width = args.width
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform


