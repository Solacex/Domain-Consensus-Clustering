import torch
import torch.nn as nn
import torch.nn.functional as F

class Memory(nn.Module):
    def __init__(self, num_cls=10, num_src=0, feat_dim=256, momentum=0.9):
        super(Memory,self).__init__()
        self.num_cls = num_cls
        self.num_src = num_src
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.memory = torch.zeros(self.num_cls, feat_dim, dtype=torch.float).cuda()
        self.source_memo = torch.zeros(self.num_src, feat_dim, dtype=torch.float).cuda()

    def init(self, center):
        center = F.normalize(center, p=2, dim=-1)
        self.memory = center
    def init_source(self, center):
        center = F.normalize(center, p=2, dim=-1)
        self.source_memo = center
        self.num_src = center.shape[0]
    def calc_memo_change(self, a, b):
        diff = (a * b).sum(dim=-1)
        return diff.mean()

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
            index = mask.squeeze().nonzero(as_tuple=False)
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

    def update_source_center(self, batch_center, flags):
        flags = flags.view(-1,1 )
        old_center = self.source_memo
        update_wei = (old_center * batch_center).sum(dim=-1).squeeze()
        update_wei = update_wei.view(-1, 1).expand_as(old_center)
        flags = flags.expand_as(self.source_memo)

        update_wei = torch.ones_like(flags) - (1 - update_wei) * flags# update_wei

        self.source_memo = update_wei * self.source_memo + (1-update_wei) * batch_center
        self.source_memo = F.normalize(self.source_memo, p=2, dim=-1)
        self.source_memo_change =  self.calc_memo_change(self.source_memo, old_center)

    def update_source(self, feat, label, mapping=None):
        if len(mapping)==0:
            return 0 
        feat = feat.detach()
        batch_center = {}
        empty = torch.zeros((1, self.feat_dim), dtype=torch.float).cuda()
        flags = {}
        for i in label.unique().tolist():
            mask = label==i
            if i not in mapping:
                continue
            memo_index = mapping[i]
            if mask.sum()==0:
                flags[memo_index] = torch.Tensor([.0]).cuda()
                continue
            index = mask.squeeze().nonzero(as_tuple=False)
            cur_feat = feat[index,:]
            count = cur_feat.shape[0]
            cur_feat = cur_feat.sum(dim=0)
            cur_feat = F.normalize(cur_feat, p=2, dim=-1)
            cur_feat = cur_feat.view(1, -1)

            flags[memo_index] = torch.Tensor([1.0]).cuda()
            batch_center[memo_index] = cur_feat
        cat_centers = []
        cat_flags = []

        for i in range(self.num_src):
            if i in batch_center:
                cat_centers.append(batch_center[i])
            else:
                cat_centers.append(empty)
            if i in flags:
                cat_flags.append(flags[i])
            else:
                cat_flags.append(torch.Tensor([.0]).cuda())
        cat_centers = torch.cat(cat_centers, dim=0 )
        cat_flags = torch.cat(cat_flags)
        self.update_source_center(cat_centers, cat_flags)

    def forward(self, feat, label, t=1.0, joint=True):
        feat = F.normalize(feat, p=2, dim=-1)
        self.update(feat, label.unsqueeze(0))
        if joint and self.source_memo.shape[0]>0:
            memo = torch.cat([self.memory,self.source_memo], dim=0)
        else:
            memo = self.memory
        simis = torch.matmul(feat, memo.transpose(0, 1))
        simis = simis/t
        loss = F.cross_entropy(simis, label.squeeze())
        return loss.mean()

