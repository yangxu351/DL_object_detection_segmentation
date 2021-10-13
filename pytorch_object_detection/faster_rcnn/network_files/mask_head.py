import torch
import torch.nn as nn
import torch.nn.functional as F  
from torch import Tensor
from typing import List, Optional, Dict, Tuple
import math

from torch.nn.modules.conv import Conv2d
from .det_utils import attention_loss
from .visualize_feature import visualize_feature_map, visualize_feature_mask
    


class MaskHead(nn.Module):
    def __init__(self, in_channels=4, channels=1):
        super(MaskHead, self).__init__()
        
        # self.fusion_conv = nn.Conv2d(2*4, out_channels=2, kernel_size=(1, 1), stride=1)
        self.fusion_conv = nn.Conv2d(in_channels, out_channels=1, kernel_size=(1, 1), stride=1)
    
    def fusion_all_layer(self, features, h, w):
        keys = features.keys()
        up_feats = []
        for k in keys:
            feat = features.get(k)
            up_feat = F.interpolate(feat, size=(h, w))
            up_feats.append(up_feat)
        cat_feats = torch.cat(up_feats, dim=1)
        return cat_feats

    def forward(self, 
                targets=None,        # type: List[Tensor]
                mask_features=None      # type: Dict[str, Tensor]
        ):
        # type: ( List[Tensor],  Dict[str, Tensor]) -> Tensor
        h, w = targets.shape[-2:]
        # fusion
        # pixel_masks = self.fusion_all_layer(mask_features, h, w) # 8,2,608,608
        # pixel_masks = self.block(pixel_masks)
        # loss_mask = attention_loss(pixel_masks, targets)
        
        # separatelly 
        loss_mask = {}
        for name, ms in mask_features.items():
            # feat = self.block(ms)
            # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
            up_feat = F.interpolate(ms, size=(h, w), mode='bilinear', align_corners=False)
            # visualize_feature_mask(up_feat, targets)
            loss_mask[name] = attention_loss(up_feat, targets, 'tversky')
            
        loss = 0.5*(loss_mask['0'] + loss_mask['1'])
        mask_losses = {"loss_mask": loss}
        return mask_losses

