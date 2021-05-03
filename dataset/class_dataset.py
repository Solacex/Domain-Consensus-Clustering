import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from PIL import Image
import torchvision.transforms.functional as TF
import torch
import imageio
import random
import os.path as osp

class ClassAwareDataset(data.Dataset):
    def __init__(self, root, src_list_path, num_pclass, transform, src_class_set, tgt_class_set, tgt_plabel_dict, num_steps=None, uk_list=None):
        self.num_pclass = num_pclass # number of samples per class in each domain
        self.files = []
        self.transform = transform
        labels = []
        self.num_steps=num_steps
        assert src_class_set == tgt_class_set
        
        self.ind2label = {}
        for i in range(len(src_class_set)):
            self.ind2label[i] = src_class_set[i]

        self.src_files = {i:[] for i in src_class_set}
        self.tgt_files = {i:[] for i in tgt_class_set}
        if uk_list is not None:
            self.uk_pool = list(uk_list.keys()) 
        with open(src_list_path) as f:
            for item in f.readlines():
                feilds = item.strip()
                name, label = feilds.split(' ')
                if name[0]=='/':
                    name = name[1:]
                label = int(label)

                if label not in src_class_set:
                    continue
                labels.append(label)
                path = osp.join(root, name)
                self.src_files[int(label)].append(path)

        for k, v in tgt_plabel_dict.items():
            if v in tgt_class_set:
                self.tgt_files[int(v)].append(k)


    def __getitem__(self, index):
        if self.num_steps is not None:
            index = index % (len(self.src_files))
        label = self.ind2label[index]
        if label not in self.src_files:
            src_pool = self.uk_pool
            tgt_pool = self.uk_pool
        else:
            src_pool = self.src_files[label]
            tgt_pool = self.tgt_files[label]

        src_index = np.random.choice(len(src_pool), self.num_pclass)
        tgt_index = np.random.choice(len(tgt_pool), self.num_pclass)

        src_path = [src_pool[i] for i in src_index]
        tgt_path = [tgt_pool[i] for i in tgt_index]
        
        src_labels = [label for i in src_index] 
        tgt_labels = [label for i in tgt_index]
        src_labels = torch.Tensor(src_labels).long()
        tgt_labels = torch.Tensor(tgt_labels).long()

        src_imgs = []
        for p in src_path:
            cur_img = Image.open(p).convert('RGB')
            cur_img = self.transform(cur_img)
            src_imgs.append(cur_img)
        src_imgs = torch.stack(src_imgs, 0)

        tgt_imgs = []
        for p in tgt_path:
            cur_img = Image.open(p).convert('RGB')
            cur_img = self.transform(cur_img)
            tgt_imgs.append(cur_img)
        tgt_imgs = torch.stack(tgt_imgs, 0)
        return src_imgs, tgt_imgs, src_labels, src_path, tgt_path


    def __len__(self):
        assert len(self.src_files) == len(self.tgt_files)
        if self.num_steps is None:
            return len(self.src_files)
        else:
            return self.num_steps
    
