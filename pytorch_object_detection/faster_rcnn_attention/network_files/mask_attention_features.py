import torch
import torch.nn as nn
from torch.nn import functional as F

class MaskAttention(nn.Module):
    def __init__(self, soft_val=1.0, layer_levels=['0', '1', '2', '3']):
        super(MaskAttention, self).__init__()
        self.soft_val = soft_val
        self.layer_levels = layer_levels
    
    def forward(self, features, masks=None):
        for k, feat in features.items():
            if masks is not None and k in self.layer_levels:
                feat_shape = feat.shape[-2:]
                msk = F.interpolate(masks, size=feat_shape, mode="nearest")
                if self.soft_val == -1:
                    soft_msk = torch.rand_like(msk)
                    ## fixme
                    soft_msk[msk==1] = 1
                    ### fixme softval-1_nonzero
                    #  soft_msk[msk!=0] = 1
                    msk=soft_msk
                elif self.soft_val == -0.5:
                    ### fixme softval-1_halfmax
                    soft_msk = torch.rand_like(msk)//2
                    ## fixme
                    soft_msk[msk==1] = 1
                    ### fixme softval-1_nonzero
                    #  soft_msk[msk!=0] = 1
                else:
                    # msk[msk==0] = soft_val
                    soft_msk = torch.ones_like(msk)*self.soft_val
                    ## fixme
                    soft_msk[msk==1] = 1
                    ### fixme softval-1_nonzero
                    #  soft_msk[msk!=0] = 1
                features[k] = feat*soft_msk
            
        return features