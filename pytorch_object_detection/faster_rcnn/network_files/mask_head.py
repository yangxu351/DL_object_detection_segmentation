import torch
import torch.nn as nn
import torch.nn.functional as F  
from torch import Tensor
from typing import List, Optional, Dict, Tuple
import math

from torch.nn.modules.conv import Conv2d
from .det_utils import attention_loss

class InceptionModule(nn.Module):
    def __init__(self,):
        super(InceptionModule,self).__init__()
        self.branch_0_conv = Conv2d(512,out_channels=384,kernel_size=(1,1),stride=1)

        self.branch_1_conv1 = Conv2d(512,out_channels=192,kernel_size=(1,1),stride=1)
        self.branch_1_conv2 = Conv2d(192,out_channels=224,kernel_size= (1, 7),stride=1)
        self.branch_1_conv3 = Conv2d(224,out_channels=256,kernel_size= (7, 1),stride=1)

        self.branch_2_conv1 = Conv2d(512,out_channels=192,kernel_size=(1,1),stride=1)
        self.branch_2_conv2 = Conv2d(192,out_channels=192,kernel_size= (7, 1),stride=1)
        self.branch_2_conv3 = Conv2d(192,out_channels=224,kernel_size= (1, 7),stride=1)
        self.branch_2_conv4 = Conv2d(224,out_channels=224,kernel_size= (7, 1),stride=1)
        self.branch_2_conv5 = Conv2d(224,out_channels=256,kernel_size= (1, 7),stride=1)

        self.branch_3_avgpool = nn.MaxPool2d(kernel_size=3,stride=1)
        self.branch_3_conv = Conv2d(512,out_channels=128,kernel_size= (1, 1),stride=1)


    def forward(self,x):
        # branch_0
        branch_0 = self.branch_0_conv(x)
        branch_0 = nn.relu(branch_0)

        # branch_1
        branch_1 = self.branch_1_conv1(x)
        branch_1 = nn.relu(branch_1)
        branch_1 = self.branch_1_conv2(branch_1)
        branch_1 = nn.relu(branch_1)
        branch_1 = self.branch_1_conv3(branch_1)
        branch_1 = nn.relu(branch_1)

        # branch_2 
        branch_2 = self.branch_2_conv1(x)
        branch_2 = nn.relu(branch_2)
        branch_2 = self.branch_2_conv2(branch_2)
        branch_2 = nn.relu(branch_2)
        branch_2 = self.branch_2_conv3(branch_2)
        branch_2 = nn.relu(branch_2)
        branch_2 = self.branch_2_conv4(branch_2)
        branch_2 = nn.relu(branch_2)
        branch_2 = self.branch_2_conv5(branch_2)
        branch_2 = nn.relu(branch_2)

        # branch_3
        branch_3 = self.branch_3_avgpool(x)
        branch_3 = self.branch_3_conv(branch_3)
        branch_3 = nn.relu(branch_3)

        out =  torch.cat([branch_0, branch_1, branch_2, branch_3],dim=1)
        return out

class InceptionAttention(nn.Module):
    
    def __init__(self,):
        super(InceptionAttention,self).__init__()
        self.inception_module = InceptionModule()
        self.inception_attention_conv = nn.Conv2d(in_channels=1024, out_channels=2)

    def forward(self,x):
        x = self.inception_module(x)
        x = self.inception_attention_conv(x)
        return x 

class MaskHead(nn.Module):
    def __init__(self):# ,out_dim ,ratio
        super(MaskHead,self).__init__()
        # self.fusion_conv = nn.Conv2d(2*4, out_channels=1, kernel_size=(1, 1), stride=1)
        self.fusion_conv = nn.Conv2d(4, out_channels=1, kernel_size=(1, 1), stride=1)
    
    def fusion_all_layer(self, features, h, w):
        keys = features.keys()
        up_feats = []
        for k in keys:
            feat = features.get(k)
            up_feat = F.interpolate(feat, size=(h, w))
            up_feats.append(up_feat)
        cat_feats = torch.cat(up_feats, dim=1)
        reduce_dim_f = self.fusion_conv(cat_feats)
        return reduce_dim_f

    def forward(self, 
                masks=None,        # type: List[Tensor]
                mask_features=None      # type: Dict[str, Tensor]
        ):
        # type: ( List[Tensor],  Dict[str, Tensor]) -> Tensor
        h, w = masks.shape[-2:]
        pixel_masks = self.fusion_all_layer(mask_features, h, w)
        # ------------------ Attention losses -------------------#
        loss_mask = attention_loss(masks, pixel_masks)
        mask_losses = {"loss_mask": loss_mask}
        return mask_losses

