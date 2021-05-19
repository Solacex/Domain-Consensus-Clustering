import torch
import os.path as osp
import os
import pickle
#class Recorder(object):
def print_loss(loss_dic):
    result = []
    for key in loss_dict.keys():
        value = loss_dict[key]
        if isinstance(value, torch.Tensor):
            value = value.item()
        tmp = "{} : {:.4f}".format(key, value)
        result.append(tmp)
    result = '  '.join(result)
    return result
def mkdir(p):
    if not osp.exists(p):
        os.makedirs(p)
        print('DIR {} created'.format(p))
    return p

def pickle_save(dic, path):
    if path[-4:]!='.pkl':
        path = path+'.pkl'
    files = open(path, 'wb')
    pickle.dump(dic, files)
    files.close()
def pickle_load(path):
    with open(path ,'rb' ) as f:
        return pickle.load(f)

def get_list(path):
    result = []
    with open(path) as f:
        for item in f.readlines():
            image = item.strip().split(' ')[0]
            result.append(image)
    return result    
def save_list(path, ls):
    with open(path, 'w') as filehandle:
        if isinstance(ls ,list):
            for i in ls:
                filehandle.write(str(i)+ '\n')
        elif isinstance(ls, dict):
            for k, v in ls.items():
                filehandle.write(str(k)+ ' ' + str(v) +'\n')
