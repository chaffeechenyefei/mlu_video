import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
Method 	             m1 	m2 	  m3    	LFW 	CFP-FP 	AgeDB-30
W&F Norm Softmax 	  1 	0 	  0 	    99.28 	88.50 	95.13
SphereFace 	          1.5 	0 	  0 	    99.76 	94.17 	97.30
CosineFace 	          1 	0 	  0.35 	    99.80 	94.4 	97.91
ArcFace 	          1 	0.5     0 	    99.83 	94.04 	98.08
Combined Margin 	  1.2 	0.4 	0 	    99.80 	94.08 	98.05
Combined Margin 	  1.1 	0 	  0.35 	    99.81 	94.50 	98.08
Combined Margin 	  1 	0.3 	0.2 	99.83 	94.51 	98.13
Combined Margin 	  0.9 	0.4 	0.15 	99.83 	94.20 	98.16
'''

class CombinedMargin(nn.Module):

    def __init__(self,
                 embedding_size=512,
                 n_cls=10,
                 m1=1.0,
                 m2=0.3,
                 m3=0.2,
                 s=64.0,
                 easy_margin=False):
        super(CombinedMargin, self).__init__()
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.s = s
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.FloatTensor(n_cls, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        '''
        COM(θ) = cos(m_1*θ+m_2) - m_3
        '''
        self.cos_m2 = math.cos(self.m2)
        self.sin_m2 = math.sin(self.m2)
        self.threshold = math.cos(math.pi - self.m2)


    def forward(self, embedd_features, gt_labels):
        embedd_features = F.normalize(embedd_features, dim=1)
        weights = F.normalize(self.weight, dim=1)
        cos_t = F.linear(embedd_features, weights)
        sin_t = torch.sqrt(1.0 - torch.pow(cos_t, 2))
        cos_tm = cos_t * self.cos_m2 - sin_t * self.sin_m2 - self.m3
        if self.easy_margin:
            # theta < pi/2, use cos(theta+m), else cos(theta)
            cos_tm = torch.where(cos_t > 0, cos_tm, cos_t)
        else:
            # theta + m < pi, use cos(theta+m), else cos(theta) - sin(theta)*m
            cos_tm = torch.where(cos_t > self.threshold, cos_tm, cos_t - self.m3 - sin_t * self.m2)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cos_theta.size(), device='cuda')
        one_hot = torch.zeros(cos_t.size()).to(embedd_features.device)
        one_hot.scatter_(1, gt_labels.view(-1, 1).long(), 1)
        output = (one_hot * cos_tm) + ((1.0 - one_hot) * cos_t)
        output *= self.s
        return output





if __name__=='__main__':

    metric_fn = CombinedMargin()

    embeddings = torch.randn(2, 512, dtype=torch.float32)
    label = torch.tensor([1, 2], dtype=torch.long)

    output = metric_fn(embeddings, label)
    print(output.size())
