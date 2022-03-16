import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class FocalLoss_v2(nn.Module):
    def __init__(self, num_class=1, gamma=2, alpha=None):
        '''
        alpha: tensor of shape (C)
        '''
        super(FocalLoss_v2, self).__init__()
        self.gamma = gamma
        self.num_class = num_class
        if alpha==None:
            self.alpha = torch.ones(num_class)
        if isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class)
            self.alpha = alpha / alpha.sum()

    def forward(self, logit, target):
        '''
        args: logits: tensor before the softmax of shape (N,C) where C = number of classes 
            or (N, C, H, W) in case of 2D Loss, 
            or (N,C,d1,d2,...,dK) where K≥1 in the case of K-dimensional loss.
        args: label: (N) where each value is in [0,C-1],
            or (N, d1, d2, ..., dK)
        Focal_Loss= -1*alpha*(1-pt)**gamma*log(pt)
        '''
        if self.alpha.device != logit.device:
            self.alpha = self.alpha.to(logit.device)
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)#(N,C,d=d1*d2*d3)
            logit = logit.permute(0,2,1)#(N,d,C)
            logit = logit.view(-1) #(N*d*C)
        target = target.view(-1) #(N*H*W)
        logpt = - F.binary_cross_entropy_with_logits(logit, target)
        pt    = torch.exp(logpt)
        focal_loss = -(self.alpha * (1 - pt) ** self.gamma) * logpt

        return focal_loss.mean()
        
def test_focal():
    num_class = 5

    nodes = 100
    N = 100
    # model1d = torch.nn.Linear(nodes, num_class).cuda()
    model2d = torch.nn.Conv2d(16, num_class, 3, padding=1).cuda()
    alpha = [0.1,0.1,0.1,0.2,0.5]
    FL2 = FocalLoss_v2(num_class=num_class, alpha=alpha)
    for i in range(10):
        input  = torch.rand(3, 16, 32, 32).cuda() #(B,C,H,W)
        target = torch.rand(3, 32, 32).random_(num_class).cuda() #(B,H,W)
        target = target.long().cuda()
        output = model2d(input) #(B,num_classes,H,W)
        loss2 = FL2(output, target)
        print(loss2.item())



class BinaryTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, ignore_index=None, reduction='mean'):
        """Dice loss of binary class
        https://github.com/CorcovadoMing/Researchlib/blob/cb7da2687a9f92f9ca5c0658be2ceb51647eda22/researchlib/loss/segmentation/tverskey.py#L96
        Args:
            alpha: controls the penalty for false positives.
            beta: penalty for false negative.
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        Shapes:
            output: A tensor of shape [N, 1,(d,) h, w] without sigmoid activation function applied
            target: A tensor of shape same with output
        Returns:
            Loss tensor according to arg reduction
        Raise:
            Exception if unexpected reduction
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.epsilon = 1e-6
        self.reduction = reduction
        s = self.beta + self.alpha
        if sum != 1:
            self.beta = self.beta / s
            self.alpha = self.alpha / s

    def forward(self, output, target):
        batch_size = output.size(0)

        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            output = output.float().mul(valid_mask)  # can not use inplace for bp
            target = target.float().mul(valid_mask)

        output = torch.sigmoid(output).view(batch_size, -1)
        target = target.view(batch_size, -1)

        P_G = torch.sum(output * target, 1)  # TP
        P_NG = torch.sum(output * (1 - target), 1)  # FP
        NP_G = torch.sum((1 - output) * target, 1)  # FN

        tversky_index = P_G / (P_G + self.alpha * P_NG + self.beta * NP_G + self.epsilon)

        loss = 1. - tversky_index
        # target_area = torch.sum(target_label, 1)
        # loss[target_area == 0] = 0
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            loss = torch.mean(loss)
        return loss

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    test_focal()