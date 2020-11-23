import torch
import torch.nn as nn


class CalcLossWithLabelSmooth(nn.Module):

    def __init__(self, gt_val=0.9):
        super(CalcLossWithLabelSmooth, self).__init__()
        self.gt_val = gt_val
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):



        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()