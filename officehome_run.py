import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from dataset import *  # init_dataset
from model import *
from init_config import *
from easydict import EasyDict as edict
import sys
import trainer
import time, datetime
import copy
import numpy as np
import random
import importlib
domain_list = {}
domain_list['officehome'] = ['Art','Product','Clipart', 'RealWorld']
domain_list['office'] = ['amazon', 'webcam', 'dslr']

def main():
    cudnn.enabled = True
    cudnn.benchmark = True

    config, writer = init_config("config/oh.yml", sys.argv)

    Param = importlib.import_module('trainer.{}{}_trainer'.format(config.trainer, config.version))
    if config.setting=='uda':
        config.cls_share = 10
        config.cls_src   = 5
        config.cls_total = 65
    elif config.setting=='osda':
        config.cls_share = 25
        config.cls_src   = 0
        config.cls_total = 65
    elif config.setting=='pda':
        config.cls_share = 25
        config.cls_src   = 40
        config.cls_total = 65


    config.num_classes = config.cls_share + config.cls_src
    config.uk_index=config.cls_share + config.cls_src
    config.transfer_all = 1 
    a,b,c =  config.cls_share, config.cls_src, config.cls_total
    c = c-a-b
    share_classes  = [i for i in range(a)]
    source_classes = [a+i for i in range(b)]
    target_classes = [a+b+i for i in range(c)]
    config.share_classes  = share_classes
    config.source_classes = share_classes + source_classes
    config.target_classes = share_classes + target_classes
    index_list = [(2,1), (1,2), (0,2), (2,0), (1,0), (3,2)]
    #index_list = index_list[2:]
    transfer_list  = []
    domains = domain_list[config.task]
    for src in domains: 
        for tgt in domains:
            if src != tgt:
                transfer_list.append((src, tgt))
    if not config.transfer_all:
        trainer = Param.Trainer(config, writer)
        trainer.train()
    else:
        print(transfer_list)
        for i, (src, tgt) in enumerate(transfer_list):
            print('{}-->{}'.format(src, tgt))
            config.source = src
            config.target = tgt
            trainer = Param.Trainer(config, writer)
            trainer.train()
if __name__ == "__main__":

    main()
