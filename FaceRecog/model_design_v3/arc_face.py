import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class ArcFace(nn.Module):

    def __init__(self,
                 embedding_size=512,
                 n_cls=10,
                 scale=30.0,
                 margin=0.50,
                 easy_margin=False):
        super(ArcFace, self).__init__()
        # self.embedding_size = embedding_size
        # self.n_cls = n_cls
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.FloatTensor(n_cls, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        self.th = math.cos(math.pi - margin)  # cos(pi - m)
        # self.mm = math.sin(math.pi - margin) * margin  # sin(pi - m) * m
        self.mm = math.sin(margin) * margin # sin(m) * m

    def forward(self, embedd_features, gt_labels):
        cos_theta = F.linear(F.normalize(embedd_features), F.normalize(self.weight))
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        sin_theta = sin_theta.clamp(0, 1)
        # cos(a+b) = cos(a)*cos(b) - sin(a)*sin(b)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        if self.easy_margin:
            # theta < pi/2, use cos(theta+m), else cos(theta)
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            # theta + m < pi, use cos(theta+m), else cos(theta) - sin(theta)*m
            cos_theta_m = torch.where(cos_theta > self.th, cos_theta_m, cos_theta - sin_theta * self.margin)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cos_theta.size(), device='cuda')
        one_hot = torch.zeros(cos_theta.size()).to(embedd_features.device)
        one_hot.scatter_(1, gt_labels.view(-1, 1).long(), 1)
        output = (one_hot * cos_theta_m) + ((1.0 - one_hot) * cos_theta)
        output *= self.scale
        return output

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

if __name__=='__main__':

    metric_fn = ArcFace(n_cls=10)

    embeddings = torch.randn(2, 512, dtype=torch.float32)
    gt_labels = torch.tensor([1, 2], dtype=torch.long)

    output = metric_fn(embeddings, gt_labels)
    print(output.size())
    print(output)



