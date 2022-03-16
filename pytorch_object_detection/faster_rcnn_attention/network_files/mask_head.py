import torch
import torch.nn as nn
import torch.nn.functional as F  
from torch import Tensor
from typing import List, Optional, Dict, Tuple
import math

from torch.nn.modules.conv import Conv2d
from .det_utils import attention_loss
from .visualize_feature import visualize_feature_map, visualize_feature_mask
from .focal_loss import FocalLoss_v2    


class MaskHead(nn.Module):
    def __init__(self, in_channels=4, channels=1):
        super(MaskHead, self).__init__()
        
        self.FL2 = FocalLoss_v2(num_class=1, alpha=[0.1])

    def forward(self, 
                targets=None,        # type: List[Tensor]
                mask_features=None      # type: Dict[str, Tensor]
        ):
        # type: ( List[Tensor],  Dict[str, Tensor]) -> Tensor

        h, w = targets.shape[-2:]

        # separatelly 
        loss_mask = {}
        for name, ms in mask_features.items():
            # feat = self.block(ms)
            # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
            # up_feat = F.interpolate(ms, size=(h, w), mode='bilinear', align_corners=False)
            # visualize_feature_mask(up_feat, targets)
            # loss_mask[name] = self.FL2(up_feat, targets)

            scale_tgt = F.interpolate(targets, size=ms.shape[-2:], mode='nearest')
            ###### focal loss
            loss_mask[name] = self.FL2(ms, scale_tgt)
            # visualize_feature_mask(ms, scale_tgt)
            ###### tversky loss
            # loss_mask[name] = attention_loss(up_feat, targets, 'tversky')
            
        loss = 0.5*(loss_mask['0'] + loss_mask['1'])
        mask_losses = {"loss_mask": loss}
        return mask_losses

