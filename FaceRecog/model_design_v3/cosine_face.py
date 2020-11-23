import torch
import torch.nn as nn
import torch.nn.functional as F



class CosineFace(nn.Module):

    def __init__(self, in_features, out_features, s=64.0, m=0.40):
        super(CosineFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):

        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size()).to(input.device)
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        # print(output)
        return output


if __name__=='__main__':

    metric_fn = CosineFace(in_features=512, out_features=10).cuda()

    embeddings = torch.randn(2, 512, dtype=torch.float32).cuda()
    label = torch.tensor([1, 2], dtype=torch.long).cuda()

    output = metric_fn(embeddings, label)
    print(output)





