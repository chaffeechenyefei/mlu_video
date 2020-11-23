import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, mutual_flag=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        dist.addmm_(beta=1.0, alpha=-2.0, mat1=inputs, mat2=inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss


class TripletLoss2(nn.Module):

    def __init__(self, margin=0.5):
        super(TripletLoss2, self).__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, embeddings):
        n_sample = embeddings.size()[0]
        n_each = int(n_sample / 3)
        anchors = embeddings[0: n_each]
        positives = embeddings[n_each: 2*n_each]
        negtives = embeddings[2*n_each:3*n_each]
        loss = self.triplet_loss(anchors, positives, negtives)
        return loss




if __name__=='__main__':

    metric_fn = TripletLoss()

    embeddings = torch.randn(10, 512, dtype=torch.float32)
    gt_labels = torch.tensor([1, 2, 3, 1, 1, 2, 3, 1, 3, 2], dtype=torch.long)

    output = metric_fn(embeddings, gt_labels)
    print(output.size())
    print(output)

    metric_fn2 = TripletLoss2()
    output2 = metric_fn2(embeddings, gt_labels)



    loss_func = torch.nn.TripletMarginLoss()
    a = torch.randn(3, 128, dtype=torch.float32)
    b = torch.randn(3, 128, dtype=torch.float32)
    c = torch.randn(3, 128, dtype=torch.float32)
    loss = loss_func(anchor=a, positive=b, negative=c)
    print(loss)


