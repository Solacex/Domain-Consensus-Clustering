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
        self.fc = nn.Linear(bottle_neck_dim, out_dim)
        self.main = nn.Sequential(self.bottleneck, self.fc, nn.Softmax(dim=-1))

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out
class VGGBase(nn.Module):
    # Model VGG
    def __init__(self):
        super(VGGBase, self).__init__()
        model_ft = models.vgg19(pretrained=True)
        mod = list(model_ft.features.children())
        self.lower = nn.Sequential(*mod)
        mod = list(model_ft.classifier.children())
        mod.pop()
        self.upper = nn.Sequential(*mod)
        self.linear1 = nn.Linear(4096, 100)
        self.bn1 = nn.BatchNorm1d(100, affine=True)
        self.linear2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100, affine=True)

    def forward(self, x, target=False):
        x = self.lower(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        x = self.upper(x)
        x = F.dropout(F.leaky_relu(self.bn1(self.linear1(x))), training=False)
        x = F.dropout(F.leaky_relu(self.bn2(self.linear2(x))), training=False)
        return x

class Classifier(nn.Module):
    def __init__(self, num_classes=12):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(100, 100)
        self.bn1 = nn.BatchNorm1d(100, affine=True)
        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100, affine=True)
        self.fc3 = nn.Linear(100, num_classes)  # nn.Linear(100, num_classes)

    def forward(self, x, dropout=False, return_feat=False, reverse=False):
        feat = x
        x = self.fc3(x)
        return feat, x

class VGG19(nn.Module):
    def __init__(self, num_classes, bottleneck=True, pretrained=True, extra=False):
        super(VGG19, self).__init__()
        self.features = VGGBase()
        self.classifer = Classifier(num_classes=num_classes)

    def forward(self, x):
        if len(x.shape)>4:
            x = x.squeeze()
        assert len(x.shape)==4
        feat = self.features(x)
        feat = feat.view(feat.shape[0], -1)
        feat, prob = self.classifer(feat)
        return feat, feat, prob, F.softmax(prob, dim=-1)


    def optim_parameters(self, lr):
        d = [{'params': self.features.parameters(), 'lr': lr},
                {'params': self.classifer.parameters(), 'lr':  lr*10}]
        return d
