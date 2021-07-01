from torchvision import models
import torch.nn as nn 
import torch.nn.functional as F
import torch 

class CLS(nn.Module):
    """
    From: https://github.com/thuml/Universal-Domain-Adaptation
    a two-layer MLP for classification
    """
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
        super(CLS, self).__init__()
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        self.bn = nn.BatchNorm1d(bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, out_dim)
        self.main = nn.Sequential(self.bottleneck, self.fc, nn.Softmax(dim=-1))

    def forward(self, x):
        out = [x]
#        for module in self.main.children():
 #           x = module(x)
  #          out.append(x)
        x = self.bn(self.bottleneck(x))
        out.append(x)
        x =self.fc(x)
        out.append(x)
        out.append(F.softmax(x, dim=-1))
        return out


class Res50(nn.Module):
    def __init__(self, num_classes, bottleneck=True, pretrained=True, extra=False):
        super(Res50, self).__init__()

        self.bottleneck = bottleneck
        features = models.resnet50(pretrained=pretrained)
        self.features =  nn.Sequential(*list(features.children())[:-1])
        if bottleneck:
            self.classifer = CLS(2048, num_classes)
        else:
            ori_fc  = features.fc
            self.classifer = nn.Linear(2048, num_classes)


        self.num_classes = num_classes 
    def forward(self, x):
        if len(x.shape)>4:
            x = x.squeeze()
        assert len(x.shape)==4
        feat = self.features(x)
        feat = feat.squeeze()
        if self.bottleneck:
            _, bottleneck, prob, af_softmax = self.classifer(feat)
        else:
            prob = self.classifer(feat)        
            bottleneck = feat
        return feat, bottleneck, prob, F.softmax(prob, dim=-1)


    def optim_parameters(self, lr):
        d = [{'params': self.features.parameters(), 'lr': lr},
                {'params': self.classifer.parameters(), 'lr':  lr*10}]
        return d
