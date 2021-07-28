
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import torch
from models.resnet import *

class MetaBaseClassifier(nn.Module):
    def __init__(self, nattr):
        super().__init__()
        self.logits = nn.Sequential(
            nn.Linear(2048, nattr), # 2048
            nn.BatchNorm1d(nattr)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1) 

        self.w2 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1),
        )
        # Squeeze and Excitation
        self.w1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(8,6)),
            nn.Flatten(),
            nn.Linear(2048, 128), # 2048/16
            nn.ReLU(),
            nn.Linear(128, 2048),
            nn.Sigmoid()
        )
        
        # Fusion Matrix
        self.A_front = nn.Parameter(torch.randn(2048, 8, 6).clone().detach(), requires_grad=True)

    def fresh_params(self):
        return self.parameters()

    def A_operation(self, x_body_feat, x_face_feat, pose):
        # Pose:
        # 0   = Backside
        # 0.5 = Sideways
        # 1   = Frontal

        feat_product = []
        batch_size = x_body_feat.size(0)

        for i in range(batch_size):
            batch_body          = x_body_feat[i] # [2048,8,6]
            batch_face          = x_face_feat[i] # [2048,8,6]

            pose_index = pose[i].item()
            # Frontal
            if pose_index == 1:
                comb_feat = batch_body * self.A_front * batch_face
                comb_feat = self.w1(comb_feat[None,:,:,:])[:,:,None,None] + batch_body
                comb_feat = torch.squeeze(comb_feat)
            # Any other pose
            else:
                comb_feat = batch_body
            
            feat_product.append(comb_feat)

        feat_product = torch.stack(feat_product)

        return feat_product


    def forward(self, x_body_feat, x_face_feat, pose):

        # [32, 2048, 8, 6] * [2048, 8, 6] -> [32, 2048, 8, 6]
        comb_ = self.A_operation(x_body_feat, x_face_feat, pose)

        comb_feat = self.avg_pool(comb_).view(comb_.size(0), -1)

        x = self.logits(comb_feat)

        return x


class MetaModel(nn.Module):
    def __init__(self, nattr):
        super().__init__()

        self.face_model = resnet50()
        self.body_model = resnet50()

        self.classifier = MetaBaseClassifier(nattr)

    def fresh_params(self):
        return self.parameters()

    def forward(self, x_body, x_face, img_info, label=None):

        # Orientation information
        img_orientation = img_info[:,6]

        x_face_feat = self.face_model(x_face)
        x_body_feat = self.body_model(x_body)

        x = self.classifier(x_body_feat, x_face_feat, img_orientation)

        return x
