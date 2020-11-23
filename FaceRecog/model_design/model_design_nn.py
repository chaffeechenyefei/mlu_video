import torch
import torch.nn as nn
import torch.nn.functional as F

from model_design import backbone_module
# from model_design.backbone_res50_v1 import ResNet50 as Backbone
from model_design.backnone_mobilenet import MobileFaceNet as Backbone

from model_design import head_module


class InsightFace(nn.Module):

    def __init__(self, n_person=123):
        super(InsightFace, self).__init__()
        self.backbone = Backbone(dropout_ratio=0.1)
        self.head = head_module.Arcface(embedding_size=512, classnum=n_person)
        self.criteria = nn.CrossEntropyLoss()

    def forward_train(self, input, labels):
        embedding_features = self.backbone(input)
        thetas = self.head(embedding_features, labels)
        # print(thetas)
        loss = self.criteria(thetas, labels)
        return dict(loss=loss,
                    embedding_features=embedding_features)

    def forward_test(self, input):
        embedding_features = self.backbone(input)
        return dict(embedding_features=embedding_features)

    def forward(self, input, labels=None):
        if self.training:
            return self.forward_train(input, labels)
        else:
            return self.forward_test(input)


if __name__ == '__main__':
    model = InsightFace()
    model.train()

    input = torch.randn(2, 3, 112, 112, dtype=torch.float32)
    label = torch.tensor([1, 12], dtype=torch.long)
    loss = model(input=input, labels=label)
    print(loss)
