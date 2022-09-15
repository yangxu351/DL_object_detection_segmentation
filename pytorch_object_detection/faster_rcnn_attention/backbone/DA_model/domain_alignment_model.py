import torch
import torch.nn as nn
from torch.autograd import Variable, Function

from .model_global_local import netD, netD_pixel


class Domain_Alignment(nn.Module):
    def __init__(self, in_channels, lc=False, gl=False, context=False):
        """ local alignment """
        super(Domain_Alignment, self).__init__()
        self.lc = lc
        self.gl = gl
        self.context = context
        if self.lc:
            self.lc_blocks = netD_pixel(in_channels, context=self.context)
        if self.gl:
            self.gl_blocks = netD(in_channels, context=self.context)

    def forward(self, features, is_target=False, alpha=1.0):
        lc_domain_results = {}
        lc_features = {}
        gl_domain_results = {}
        gl_features = {}
        alpha = Variable(torch.tensor(alpha))
        for k, feat in features.items():
            if self.lc and (k == '3' or k == '2'):
                if self.context:
                    d_pixel, _ = self.lc_blocks(grad_reverse(feat, alpha=alpha)) # (B, 1, H, W) # 50, 25
                    if not is_target:
                        _, lc_feat = self.lc_blocks(feat.detach()) # (B, 128, 1, 1)
                        lc_features[k] = lc_feat
                else:
                    d_pixel = self.lc_blocks(grad_reverse(feat, alpha=alpha))
                lc_domain_results[k] = d_pixel

            if self.gl and (k == '1' or k == '0'):
                if self.context:
                    domain_p, _ = self.gl_blocks(grad_reverse(feat, alpha=alpha)) # (B, 2)
                    if is_target:
                        gl_domain_results[k] = domain_p
                        continue
                    _, gc_feat = self.gl_blocks(feat.detach()) # (B, 128)
                    gl_features[k] = gc_feat
                else:
                    domain_p = self.gl_blocks(grad_reverse(feat, alpha=alpha))
                gl_domain_results[k] = domain_p

        if is_target:
            return lc_domain_results, gl_domain_results
        else:
            return features, lc_domain_results, lc_features, gl_domain_results, gl_features 


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        # if ctx.needs_input_grad[0]:
        grad_input = grad_output.neg()*alpha
        return grad_input, None


def grad_reverse(x, alpha):
    return GradientReversal.apply(x, alpha)