import numpy as np
import os.path as osp
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils import data
import torchvision.transforms.functional as TF
from torchvision.transforms.transforms import *
from .base_dataset import BaseDataset
from .class_dataset import ClassAwareDataset
from .path_dataset import PathDataset
import torch
import random 
from .target_dataset import TargetClassAwareDataset

def get_transform(train=True):
    if train:
        transforms = Compose([
            Resize(256),
            RandomCrop(224),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
    else:
        transforms = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
    return transforms
def get_path_dataset(config, path_dict, length=None, test=False, batch_size=None, get_loader=False):
    transforms = get_transform(train=not test)

    if length is not None and not test:
        num_steps = length * config.batch_size
    elif length is not None and test:
        num_steps = length
    else:
        num_steps = None

    dataset = PathDataset(path_dict, transforms,num_steps=num_steps)
    if batch_size is None:
        batch_size = config.batch_size
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=config.num_workers, shuffle=not test, drop_last=not test)
    if get_loader:
        return loader
    else:
        return enumerate(loader)

def get_dataset(config, dataset, class_set, label_list=None, test=False, batch_size=None, plabel_dict=None, get_loader=True, length=None, binary_label=None, class_wise=False, validate=False):

    list_path = './dataset/list/{}/{}.txt'.format(config.task,dataset)
    if config.target =='caltech' and config.target==dataset:
        list_path = './dataset/list/{}/{}_tgt.txt'.format(config.task,dataset)
    if config.task=='domainnet' and validate:
        list_path = './dataset/list/{}/{}_test.txt'.format(config.task,dataset)
    if config.task =='visda':
        root_path = osp.join(config.root[config.task], dataset)
    else:
        root_path = config.root[config.task]
    transforms = get_transform(train=not test)
    if length is not None and not test:
        num_steps = length * config.batch_size
    elif length is not None and test:
        num_steps = length
    else:
        num_steps = None
    if class_wise:
        TargetClassAwareDataset(root_path, config.num_pclass, transform, class_set, plabel_dict, num_steps=length*config.num_sample)
    else:
        dataset = BaseDataset(root_path, list_path, transforms, dataset, class_set, num_steps=num_steps, plabel_dict=plabel_dict, binary_label=binary_label)
    if batch_size is None:
        batch_size = config.batch_size

    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=config.num_workers, shuffle=not test, drop_last=not test)
    if get_loader: 
        return loader
    else:
        return dataset

def init_pair_dataset(config, src_wei=None, label_list=None, plabel_dict=None, length=None, binary_label=None):

    src_loader  = get_dataset(config, config.source, config.source_classes, length=length)
    tgt_loader  = get_dataset(config, config.target, config.target_classes, label_list=label_list,  plabel_dict=plabel_dict, length=length, binary_label=binary_label)

    src_loader = enumerate(src_loader)
    tgt_loader = enumerate(tgt_loader)

    return src_loader, tgt_loader#, s_test_loader, t_test_loader

def init_class_dataset(config, plabel_dict, src_class_set, tgt_class_set, length=None, uk_list=None):
    transform = get_transform()
    src_list_path = './dataset/list/{}/{}.txt'.format(config.task, config.source)
    if config.task =='visda' or config.task=='imagenet-caltech':
        root_path = osp.join(config.root[config.task], config.source)
    else:
        root_path = config.root[config.task]
    if uk_list is not None:
        src_class_set.append(config.num_classes)
        tgt_class_set.append(config.num_classes)

    dataset = ClassAwareDataset(root_path, src_list_path, config.num_pclass, transform, src_class_set, tgt_class_set, plabel_dict, num_steps=length*config.num_sample, uk_list=uk_list)
    dataloader = DataLoader(dataset=dataset, batch_size=config.num_sample, num_workers=config.num_workers, shuffle=True, drop_last=True)
    dataloader = enumerate(dataloader)
    return dataloader

def init_target_dataset(config, plabel_dict, tgt_class_set, length=None, uk_list=None, binary_label=None):
    transform = get_transform()
    if config.task =='visda' or config.task=='imagenet-caltech':
        root_path = osp.join(config.root[config.task], config.source)
    else:
        root_path = config.root[config.task]
    if uk_list is not None:
        tgt_class_set.append(config.num_classes)
    dataset = TargetClassAwareDataset(root_path, config.num_pclass, transform, tgt_class_set, plabel_dict, num_steps=length*config.num_sample)
    dataloader = DataLoader(dataset=dataset, batch_size=config.num_sample, num_workers=config.num_workers, shuffle=True, drop_last=True)
    dataloader = enumerate(dataloader)
    return dataloader
