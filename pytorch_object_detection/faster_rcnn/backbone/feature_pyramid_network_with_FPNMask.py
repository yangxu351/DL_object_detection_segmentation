from collections import OrderedDict

import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

from torch.jit.annotations import Tuple, List, Dict
import copy

from network_files import visualize_feature
from backbone import resnet50_fpn_model


class Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attention,self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, out_channels, 1)
        )
    def forward(self,x, idx=0):
        x = self.block(x)
        # visualize_feature.visualize_feature_map(x, idx)
        return x 


class FeaturePyramidNetworkwithFPNMask(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps. This is based on
    `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.
    The feature maps are currently supposed to be in increasing depth
    order.
    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the FPN will be added.
    Arguments:
        in_channels_list (list[int]): number of channels for each feature map that
            is passed to the module
        out_channels (int): number of channels of the FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names
    """

    def __init__(self, in_channels_list, out_channels, extra_blocks=None, soft_val=1.):
        super(FeaturePyramidNetworkwithFPNMask, self).__init__()
        # 用来调整resnet特征矩阵(layer1,2,3,4)的channel（kernel_size=1）
        self.inner_blocks = nn.ModuleList()
        # 对调整后的特征矩阵使用3x3的卷积核来得到对应的预测特征矩阵
        self.layer_blocks = nn.ModuleList()
        # pixel attention branch
        self.pixel_blocks = nn.ModuleList()
        for in_channels in in_channels_list: # [256, 512, 1024, 2048]
            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

            pixel_block_module = Attention(out_channels, out_channels=1)
            self.pixel_blocks.append(pixel_block_module)
        
        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        self.extra_blocks = extra_blocks
        self.soft_val = soft_val

    def get_result_from_inner_blocks(self, x, idx):
        # type: (Tensor, int) -> Tensor
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x, idx):
        # type: (Tensor, int) -> Tensor
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_pixel_blocks(self, x, idx):
        # type: (Tensor, int) -> Tensor
        """
        This is equivalent to self.pixel_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.pixel_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.pixel_blocks:
            if i == idx:
                pa_mask = module(x, idx)
            i += 1
        
        pa_mask_sigm = torch.sigmoid(pa_mask)
        out = out*pa_mask_sigm
        return out, pa_mask
        
        

    def forward(self, x, masks):
        # type: (Dict[str, Tensor], Tensor, float) -> Dict[str, Tensor]
        """
        Computes the FPN for a set of feature maps.
        Arguments:
            x (OrderedDict[Tensor]): feature maps for each feature level.
        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys()) # ['0', '1', '2', '3']
        x = list(x.values())  # [256, 512, 1024, 2048]
        # print('name', names)
        # print('x[0]', x[0].shape)

        # 将resnet layer4的channel调整到指定的out_channels
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        # result中保存着每个预测特征层
        results = []
        
        # 将layer4调整channel后的特征矩阵，通过3x3卷积后得到对应的预测特征矩阵
        results.append(self.get_result_from_layer_blocks(last_inner, -1))
        
        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            if idx == 1: # layer2 100
                msk = F.interpolate(masks, size=feat_shape, mode="nearest")
                msk[msk==0] = self.soft_val 
                mask_layer = msk*last_inner
                results.insert(0, self.get_result_from_layer_blocks(mask_layer, idx))
            elif idx == 0 : # layer1 200
                msk = F.interpolate(masks, size=feat_shape, mode="nearest")
                msk[msk==0] = self.soft_val
                mask_layer = msk*last_inner
                results.insert(0, self.get_result_from_layer_blocks(mask_layer, idx))
            else:
                results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))
            
        # 在layer4对应的预测特征层基础上生成预测特征矩阵5
        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        return out

