import torch
import torch.nn as nn
import torch.nn.functional as F
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

class Local_Alignment(nn.Module):
    def __init__(self, in_channels, lc=False, context=False, soft_val=1.0):
        """ local alignment """
        super(Local_Alignment, self).__init__()
        self.soft_val = soft_val
        self.lc = lc
        self.context = context
        self.lc_blocks = netD_pixel(in_channels, context=self.context)

    def forward(self, features, masks=None, is_target=False, la_weight=1.0, eta=1.0):
        lc_domain_results = {}
        lc_features = {}
        la_loss = []
        la_weight = Variable(torch.tensor(la_weight))
        for k, feat in features.items():
            if self.lc and k != 'pool': #  and (k == '3' or k == '2'):
                if self.context:
                    d_pixel, _ = self.lc_blocks(grad_reverse(feat, alpha=la_weight)) # (B, 1, H, W) # 50, 25
                    if not is_target:
                        _, lc_feat = self.lc_blocks(feat.detach()) # (B, 128, 1, 1)
                        lc_features[k] = lc_feat
                else:
                    d_pixel = self.lc_blocks(grad_reverse(feat, alpha=la_weight))
                # tag: soft mask attention
                if masks is not None:
                    feat_shape = feat.shape[-2:]
                    msk = F.interpolate(masks, size=feat_shape, mode="nearest")
                    if self.soft_val == -1:
                        soft_msk = torch.rand_like(msk)
                        soft_msk[msk==1] = 1
                        msk=soft_msk
                    elif self.soft_val == -0.5:
                        soft_msk = torch.rand_like(msk)//2
                        soft_msk[msk==1] = 1
                        msk=soft_msk
                    else:
                        soft_msk = torch.ones_like(msk)*self.soft_val
                        soft_msk[msk==1] = 1
                        msk=soft_msk
                    d_pixel = d_pixel*msk
                lc_domain_results[k] = d_pixel
        
        if is_target:
            for k,v in lc_domain_results.items():
                dloss_lc_t = 0.5 * torch.mean((1 - v) ** 2) # 0.1191
                # target adv loss
                la_loss.append(dloss_lc_t)
            la_loss_dict = {'loss_lc_t':torch.mean(torch.stack(la_loss))*eta}
            return la_loss_dict
        else:
            for k,v in lc_domain_results.items():
                dloss_lc_s = 0.5 * torch.mean(v ** 2)
                # source adv loss
                la_loss.append(dloss_lc_s) # 0.048
            la_loss_dict = {'loss_lc_s':torch.mean(torch.stack(la_loss))*eta}
            return features, la_loss_dict, lc_features



class Global_Alignment(nn.Module):
    def __init__(self, in_channels, gl=False, context=False):
        """ global alignment """
        super(Global_Alignment, self).__init__()
        self.gl = gl
        self.context = context
        self.gl_blocks = netD(in_channels, context=self.context)

    def forward(self, features, is_target=False, alpha=1.0):
        gl_domain_results = {}
        gl_features = {}
        alpha = Variable(torch.tensor(alpha))
        for k, feat in features.items():
            #fixme: redesign this module??????????
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
            return gl_domain_results
        else:
            return features, gl_domain_results, gl_features 

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