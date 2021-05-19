import torch
import torch.nn as nn
import torch.nn.functional as F

class Memory(nn.Module):
    def __init__(self, num_cls=10, feat_dim=256, momentum=0.9, T=2):
        super(Memory,self).__init__()
        self.num_cls = num_cls
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.memory = torch.zeros(self.num_cls, feat_dim, dtype=torch.float).cuda()
        self.T = T

    def init(self, center):
        center = F.normalize(center, p=2, dim=-1)
        self.memory = center

    def calc_memo_change(self, a, b):
        diff = (a * b).sum(dim=-1)
        return diff.mean()

    def update_center(self, batch_center, flags):
        self.memory = F.normalize(self.memory, p=2, dim=-1)
        flags = flags.expand_as(self.memory)
        update_wei = torch.ones_like(flags) - (1-self.momentum) * flags
        old_memo = self.memory
        self.memory =  update_wei * self.memory + (1 - update_wei) * batch_center
        
        self.memory = F.normalize(self.memory, p=2, dim=-1)
        self.memo_change = self.calc_memo_change(self.memory, old_memo)

    def update_center_by_simi(self, batch_center, flags):
        old_center = self.memory 
        update_wei = (old_center * batch_center).sum(dim=-1).squeeze()
        update_wei = update_wei.view(-1, 1).expand_as(old_center)
        flags = flags.expand_as(self.memory)

        update_wei = torch.ones_like(flags) - (1 - update_wei) * flags# update_wei

        self.memory = update_wei * self.memory + (1-update_wei) * batch_center
        self.memory = F.normalize(self.memory, p=2, dim=-1)
        self.memo_change =  self.calc_memo_change(self.memory, old_center)#torch.matmul(self.memory, old_center.transpose(0, 1)).mean()

    def update(self, feat, label):
        feat = feat.detach()
        batch_center = [] 
        empty = torch.zeros((1, self.feat_dim), dtype=torch.float).cuda()
        flags = []
        for i in range(self.num_cls):
            mask = label==i
            if mask.sum()==0:
                flags.append(torch.Tensor([.0]).cuda())
                batch_center.append(empty)
                continue
            index = mask.squeeze().nonzero()
            cur_feat = feat[index,:]
            count = cur_feat.shape[0]
            cur_feat = cur_feat.sum(dim=0)
            cur_feat = F.normalize(cur_feat, p=2, dim=-1)
            cur_feat = cur_feat.view(1, -1)

            flags.append(torch.Tensor([1.0]).cuda())
            batch_center.append(cur_feat)
        batch_center = torch.cat(batch_center, dim=0)
        flags = torch.stack(flags).cuda()
        self.update_center_by_simi(batch_center, flags)

    def l2_distance(self, feat):
        a, c = feat.shape
        b, c = self.memory.shape
        feat = feat.unsqueeze(1).repeat(1, b, 1)
        dis = F.mse_loss(feat, self.memo_change, reduction='none')
        
        return dis.sum(-1)
    def forward(self, feat, label, t=1.0, slabel=None):
        feat = F.normalize(feat, p=2, dim=-1)
        self.update(feat, label.unsqueeze(0))
        simis = torch.matmul(feat, self.memory.transpose(0, 1))

        simis = simis/t
        loss = F.cross_entropy(simis, label.squeeze())

        return loss.mean()
