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

class BaseDataset(data.Dataset):
    def __init__(self, root, list_path, transform, dataset, class_set, label_list=None, num_steps=None, plabel_dict=None, binary_label=None):
        
        self.dataset = dataset
        self.files = []
        self.transform = transform
        labels = []
        self.plabel_dict = plabel_dict

        self.binary_label_dict = binary_label 
#        if self.plabel_dict is not None:
#            print(len(self.plabel_dict), 'wewewa')
        with open(list_path) as f:
            for item in f.readlines():
                #print(item)
                feilds = item.strip()
                name, label = feilds.split(' ')
                label = int(label)

#                print(name,
                if label not in class_set and self.plabel_dict is None:
                    continue
                labels.append(label)
                if label_list is not None:
                    label = label_list[name]
                path = osp.join(root, name)
                if plabel_dict is not None and path not in plabel_dict:
                    continue 
                    print(path)
                if not osp.exists(path):
                    print(path)
                    if name[0]=='/':
                        name = name[1:]
                    path = osp.join(root, dataset,  name)
                self.files.append([path, int(label)])
        print('Length of {}:{}'.format(dataset, len(self.files)))


        if num_steps is not None:
            self.files = self.files * int(np.ceil(num_steps)/len(self.files) + 1)
    def __getitem__(self, index):
        files = self.files[index]
        name = files[0]
        if self.plabel_dict is not None:
            if name in self.plabel_dict: 
                label = self.plabel_dict[name]
            else:
                label = 255
        else:
            label = files[1]
        if self.binary_label_dict is not None:
            bi_label = self.binary_label_dict[name]
        else:
            bi_label = 2.0
        try:
            img = Image.open(files[0]).convert('RGB')
            img = self.transform(img)
        except Exception as e:
            print(files)
        return img, label, files[0], bi_label

    def __len__(self):
        return len(self.files)
    
