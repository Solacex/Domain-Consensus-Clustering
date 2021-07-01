import torch
import os.path as osp
import torch.nn as nn
import neptune.new as neptune 
from tqdm import tqdm
import operator 
import math
import torch.optim as optim
from utils.optimize import *
from easydict import EasyDict as edict
from utils import *
from utils.memory import * 
from utils.flatwhite import * 
from dataset import * 
from sklearn import metrics
import sklearn
from sklearn.cluster import KMeans
from torch.utils.tensorboard import SummaryWriter

class BaseTrainer(object):
    def __init__(self, config,  writer):

        self.config = config
        self.writer = writer
        if self.config.neptune:
            name = self.config['note'] + '_' + self.config.source + '_' + self.config.target
            self.run = neptune.init(project='solacex/UniDA-Extension', name=name,  source_files=[], capture_hardware_metrics=False)#, mode="offline")
            self.run['config'] = self.config
            self.run['name'] = self.config['note'] + '_' + self.config.source + '_' + self.config.target
        if self.config.tensorboard:
            self.writer  = SummaryWriter(osp.join(self.config.snapshot, 'log'))
        self.best = 0.0
        self.acc_best_h = 0.0
        self.h_best_acc = 0.0 
        self.k_best = 0.0
        self.h_best = 0.0
        self.label_mask = None
        self.k_converge=False
        self.score_vec = None 
        self.test_loader = get_dataset(self.config, self.config.target, self.config.target_classes, batch_size=100, test=True, validate=True)
        if self.config.task=='imagenet-caltech':
            self.src_loader = get_cls_sep_dataset(self.config, self.config.source, self.config.source_classes, batch_size=100, test=True)
        else:
            self.src_loader = get_dataset(self.config, self.config.source, self.config.source_classes, batch_size=100, test=True)
        self.tgt_loader = get_dataset(self.config, self.config.target, self.config.target_classes, batch_size=100, test=True)
        self.best_prec = 0.0
        self.best_recall = 0.0 

    def forward(self):
        pass
    def backward(self):
        pass

    def iter(self):
        pass
    def train(self):
        for i_iter in range(self.config.num_steps):
            losses = self.iter(i_iter)
            if i_iter % self.config.print_freq ==0:
                self.print_loss(i_iter)
            if i_iter % self.config.save_freq ==0 and i_iter != 0:
                self.save_model(i_iter)
            if self.config.val and i_iter % self.config.val_freq ==0 and i_iter!=0:
                self.validate()

    def save_model(self, iter):
        tmp_name = '_'.join((key, str(iter))) + '.pth'
        torch.save(self.model.state_dict(), osp.join(self.config['snapshot'], tmp_name))
    def save_txt(self):
        with open(osp.join(self.config.snapshot, 'result.txt'), 'a') as f:
            f.write(self.config.source[:2] + '->' + self.config.target[:2] +'[best]: ' + str(self.best) + ' '+ str(self.k_best) + ' [H-Score]: '+ str(self.h_best) + ' ' +  str(self.acc_best_h) + ' '+ str(self.h_best_acc) + ' ' + str(self.best_prec) + ' ' + str(self.best_recall) + '\n')
            f.write(self.config.source[:2] + '->' + self.config.target[:2] +'[last]: ' + str(self.last) + ' '+ str(self.k_last) + ' [H-Score]: '+ str(self.h_last) + ' ' + str(self.last_prec) + ' ' + str(self.last_recall) + '\n')
        f.close()
    def neptune_metric(self, name, value, display=True):
        if self.config.neptune:
            self.run[name].log(value)
        if self.config.tensorboard:
            self.writer.add_scalar(name, value) 
        if display:
            print('{} is {:.2f}'.format(name, value))
    def print_loss(self, iter):
        iter_infor = ('iter = {:6d}/{:6d}, exp = {}'.format(iter, self.config.num_steps, self.config.note))
        to_print = ['{}:{:.4f}'.format(key, self.losses[key].item()) for key in self.losses.keys()]
        loss_infor = '  '.join(to_print)
        if self.config.screen:
            print(iter_infor +'  '+ loss_infor)
        if self.config.neptune:
            for key in self.losses.keys():
                self.neptune_metric('train/'+key, self.losses[key].item(), False)
        if self.config.tensorboard and self.writer is not None:
            for key in self.losses.keys():
                self.writer.add_scalar('train/'+key, self.losses[key], iter)
    def print_acc(self, acc_dict):
        str_dict = [str(k) + ': {:.2f}'.format(v) for k, v in acc_dict.items() ]
        output = ' '.join(str_dict)
        print(output)
    def cos_simi(self, x1, x2):
        simi = torch.matmul(x1, x2.transpose(0, 1))
        return simi    

    def gather_feats(self):
        # self.model.set_bn_domain(1)
        data_feat, data_gt, data_paths, data_probs  = [], [], [], []
        gts = []
        gt = {}
        preds = []
        names = []
        for _, batch in tqdm(enumerate(self.tgt_loader)):
            img, label, name, _ = batch
            names.extend(name)
            with torch.no_grad():
                _, output, _, prob = self.model(img.cuda())
            feature = output.squeeze()#view(1,-1)
            N, C = feature.shape
            data_feat.extend(torch.chunk(feature, N, dim=0))
            gts.extend(torch.chunk(label, N, dim=0))

        for k,v in zip(names, gts):
            gt[k]=v.cuda()
        feats =  torch.cat(data_feat, dim=0)
        feats = F.normalize(feats, p=2, dim=-1)

        return feats, gt, preds

    def validate(self, i_iter, class_set):
        print(self.config.source, self.config.target, self.config.note)
        print(self.global_label_set)
        if not self.config.prior:
            if self.config.num_centers == len(self.cluster_mapping):
                result = self.close_validate(i_iter)
            else:
                result = self.open_validate(i_iter)
        elif self.config.setting in ['uda', 'osda']:
            result = self.open_validate(i_iter)
        else:
            result = self.close_validate(i_iter)
        over_all, k, h_score, recall, precision = result 
        
        if over_all > self.best:
            self.best = over_all
            self.k_best = k
            self.acc_best_h = h_score
        if h_score > self.h_best:
            self.h_best = h_score
            self.h_best_acc = over_all
            self.best_recall = recall
            self.best_prec = precision
        if i_iter+1 == self.config.stop_steps:
            self.last = over_all
            self.k_last = k
            self.h_last = h_score
            self.last_recall = recall
            self.last_prec = precision

        return result 

    def close_validate(self, i_iter):
        self.model.train(False)
        knows = 0.0
        unknows = 0.0
        k_co = 0.0
        uk_co = 0.0
        accs = GroupAverageMeter()
        test_loader = get_dataset(self.config, self.config.target, self.config.target_classes, batch_size=100, test=True)
        common_index = torch.Tensor(self.global_label_set).cuda().long()

        for _, batch in tqdm(enumerate(test_loader)):
            acc_dict = {}
            img, label, name, _ = batch
            label = label.cuda()
            with torch.no_grad():
                _, neck, pred, pred2 = self.model(img.cuda())
 #           pred2 = pred2[:, common_index]
            pred_label =  pred2.argmax(dim=-1)
#            pred_label = common_index[pred_label]

            label = torch.where(label>=self.config.num_classes, torch.Tensor([self.config.num_classes]).cuda(),label.float())
            for i in label.unique().tolist():
                mask = label==i
                count = mask.sum().float()
                correct = (pred_label==label) * mask
                correct = correct.sum().float()

                acc_dict[i] = ((correct/count).item(), count.item())
            accs.update(acc_dict)
        acc = np.mean(list(accs.avg.values()))
        self.print_acc(accs.avg)
        if acc > self.best:
            self.best = acc
        self.model.train(True)
        self.neptune_metric('val/Test Accuracy', acc)
        return acc, 0.0, 0.0, 0.0, 0.0

    def open_validate(self, i_iter):
        # self.model.set_bn_domain(1)
        self.model.train(False)
        knows = 0.0
        unknows = 0.0
        accs = GroupAverageMeter()
        t_centers = self.memory.memory

        length = len(self.test_loader.sampler)
        cls_pred_all = torch.zeros(length).cuda()
        memo_pred_all = torch.zeros(length).cuda()
        gt_all = torch.zeros(length).cuda()
        uk_index = self.config.num_classes

        cnt = 0
        for _, batch in tqdm(enumerate(self.test_loader)):
            acc_dict = {}
            img, label, name, _ = batch
            label = label.cuda()
            img = img.cuda()
            with torch.no_grad():
                _, neck, pred, pred2 = self.model(img)
            N = neck.shape[0]
            simi2cluster = self.cos_simi(F.normalize(neck, p=2, dim=-1), t_centers)
            clus_index = simi2cluster.argmax(dim=-1)
            cls_pred = pred2.argmax(-1)
            cls_pred_all[cnt:cnt+N] = cls_pred.squeeze()
            memo_pred_all[cnt:cnt+N] = clus_index.squeeze()
            gt_all[cnt:cnt+N] = label.squeeze()
            cnt+=N

        clus_mapping = self.cluster_mapping # mapping between source label and target cluster index 

        uk_null = torch.ones_like(memo_pred_all).float().cuda() * uk_index
        map_mask =  torch.zeros_like(memo_pred_all).float().cuda() 

        for k,v in self.cluster_mapping.items():
            if v in self.global_label_set:
                map_mask += torch.where(memo_pred_all==k, torch.Tensor([1.0]).cuda().float(), map_mask.float()) 


        pred_label = torch.where(map_mask>0, cls_pred_all, uk_null)

        gt_all = torch.where(gt_all>=self.config.num_classes, torch.Tensor([uk_index]).cuda(), gt_all.float())
        mask = pred_label!=uk_index
        pred_binary = (pred_label==uk_index).squeeze().tolist()
        gt_binary = (gt_all==uk_index).squeeze().tolist()

        for i in gt_all.unique().tolist():
            mask = gt_all==i
            count = mask.sum().float()
            correct = (pred_label==gt_all) * mask
            correct = correct.sum().float()
            acc_dict[i] = ((correct/count).item(), count.item())
        accs.update(acc_dict)
        
        acc = np.mean(list(accs.avg.values()))
        self.print_acc(accs.avg)
        if uk_index not in accs.avg:
            self.model.train(True)
            self.neptune_metric('memo-val/Test Accuracy[center]', acc)    
            return acc, acc, 0.0, 0.0, 0.0
        bi_rec = metrics.recall_score(gt_binary, pred_binary, zero_division=0)
        bi_prec = metrics.precision_score(gt_binary, pred_binary, zero_division=0)
        self.neptune_metric('val/bi recall[center]', bi_rec)
        self.neptune_metric('val/bi prec[center]', bi_prec)

        k_acc = (acc * len(accs.avg) - accs.avg[uk_index])/(len(accs.avg)-1)
        uk_acc = accs.avg[uk_index]
        common_sum = 0.0
        common_cnt = 0.0
        for k, v in accs.sum.items():
            if k != uk_index:
                common_sum += v
                common_cnt += accs.count[k]
        common_acc = common_sum / common_cnt
        h_score = 2 * (common_acc * uk_acc) / (common_acc + uk_acc)
        self.neptune_metric('memo-val/H-score', h_score)
        self.model.train(True)
        self.neptune_metric('memo-val/Test Accuracy[center]', acc)
        self.neptune_metric('memo-val/UK classification accuracy[center]', accs.avg[uk_index])
        self.neptune_metric('memo-val/Known category accuracy[center]', k_acc)
        return acc, k_acc, h_score, bi_rec, bi_prec

    def open_validate(self, i_iter):
        self.model.train(False)
        knows = 0.0
        unknows = 0.0
        accs = GroupAverageMeter()
        t_centers = self.memory.memory

        length = len(self.test_loader.sampler)
        cls_pred_all = torch.zeros(length).cuda()
        memo_pred_all = torch.zeros(length).cuda()
        gt_all = torch.zeros(length).cuda()
        uk_index = self.config.num_classes

        cnt = 0
        for _, batch in tqdm(enumerate(self.test_loader)):
            acc_dict = {}
            img, label, name, _ = batch
            label = label.cuda()
            img = img.cuda()
            with torch.no_grad():
                _, neck, pred, pred2 = self.model(img)
            N = neck.shape[0]
            simi2cluster = self.cos_simi(F.normalize(neck, p=2, dim=-1), t_centers)
            clus_index = simi2cluster.argmax(dim=-1)
            cls_pred = pred2.argmax(-1)
            cls_pred_all[cnt:cnt+N] = cls_pred.squeeze()
            memo_pred_all[cnt:cnt+N] = clus_index.squeeze()
            gt_all[cnt:cnt+N] = label.squeeze()
            cnt+=N

        clus_mapping = self.cluster_mapping # mapping between source label and target cluster index 

        uk_null = torch.ones_like(memo_pred_all).float().cuda() * uk_index
        map_mask =  torch.zeros_like(memo_pred_all).float().cuda() 

        for k,v in self.cluster_mapping.items():
            if v in self.global_label_set:
                map_mask += torch.where(memo_pred_all==k, torch.Tensor([1.0]).cuda().float(), map_mask.float()) 


        pred_label = torch.where(map_mask>0, cls_pred_all, uk_null)

        gt_all = torch.where(gt_all>=self.config.num_classes, torch.Tensor([uk_index]).cuda(), gt_all.float())
        mask = pred_label!=uk_index
        pred_binary = (pred_label==uk_index).squeeze().tolist()
        gt_binary = (gt_all==uk_index).squeeze().tolist()

        for i in gt_all.unique().tolist():
            mask = gt_all==i
            count = mask.sum().float()
            correct = (pred_label==gt_all) * mask
            correct = correct.sum().float()
            acc_dict[i] = ((correct/count).item(), count.item())
        accs.update(acc_dict)
        
        acc = np.mean(list(accs.avg.values()))
        self.print_acc(accs.avg)
        if uk_index not in accs.avg:
            self.model.train(True)
            self.neptune_metric('memo-val/Test Accuracy[center]', acc)    
            return acc, acc, 0.0, 0.0, 0.0
        bi_rec = metrics.recall_score(gt_binary, pred_binary, zero_division=0)
        bi_prec = metrics.precision_score(gt_binary, pred_binary, zero_division=0)
        self.neptune_metric('val/bi recall[center]', bi_rec)
        self.neptune_metric('val/bi prec[center]', bi_prec)

        k_acc = (acc * len(accs.avg) - accs.avg[uk_index])/(len(accs.avg)-1)
        uk_acc = accs.avg[uk_index]
        common_sum = 0.0
        common_cnt = 0.0
        for k, v in accs.sum.items():
            if k != uk_index:
                common_sum += v
                common_cnt += accs.count[k]
        common_acc = common_sum / common_cnt
        h_score = 2 * (common_acc * uk_acc) / (common_acc + uk_acc)
        self.neptune_metric('memo-val/H-score', h_score)
        self.model.train(True)
        self.neptune_metric('memo-val/Test Accuracy[center]', acc)
        self.neptune_metric('memo-val/UK classification accuracy[center]', accs.avg[uk_index])
        self.neptune_metric('memo-val/Known category accuracy[center]', k_acc)
        return acc, k_acc, h_score, bi_rec, bi_prec

    def get_src_centers(self):

        self.model.eval()
        num_cls = self.config.cls_share + self.config.cls_src
        if self.config.model!='res50':
            s_center = torch.zeros((num_cls, 100)).float().cuda()
        else:
            s_center = torch.zeros((num_cls, 256)).float().cuda()
        if not self.config.bottleneck:
            s_center = torch.zeros((num_cls, 2048)).float().cuda()
        
        counter = torch.zeros((num_cls,1)).float().cuda()
        s_feats = []
        s_labels = []
        for _, batch in tqdm(enumerate(self.src_loader)):
            acc_dict = {}
            img, label, _, _ = batch
            label = label.cuda().squeeze()
            with torch.no_grad():
                _, neck, _, _  = self.model(img.cuda())
            neck = F.normalize(neck, p=2, dim=-1)
            N, C = neck.shape
            s_labels.extend(label.tolist())
            s_feats.extend(torch.chunk(neck, N, dim=0))
        s_feats = torch.stack(s_feats).squeeze()
        s_labels = torch.from_numpy(np.array(s_labels)).cuda()
        for i in s_labels.unique():
            i_msk = s_labels==i
            index = i_msk.squeeze().nonzero(as_tuple=False)
            i_feat = s_feats[index, :].mean(0)
            i_feat = F.normalize(i_feat, p=2, dim=1)
            s_center[i, :] = i_feat

        return s_center, s_feats, s_labels
    def sklearn_kmeans(self, feat, num_centers, init=None):
        if self.config.task in ['domainnet', 'visda']:
            return self.faiss_kmeans(feat, num_centers, init=init)
        if init is not None:
            kmeans = KMeans(n_clusters=num_centers, init=init, random_state=0).fit(feat.cpu().numpy())
        else:
            kmeans = KMeans(n_clusters=num_centers, random_state=0).fit(feat.cpu().numpy())
        center, t_codes = kmeans.cluster_centers_, kmeans.labels_
        score = sklearn.metrics.silhouette_score(feat.cpu().numpy(), t_codes)
        return torch.from_numpy(center).cuda(), torch.from_numpy(t_codes).cuda(), score

    def faiss_kmeans(self, feat, K, init=None, niter=500):
        import faiss
        feat = feat.cpu().numpy()
        d = feat.shape[1]
        kmeans = faiss.Kmeans(d, K, niter=niter, verbose=False,  spherical=True)
        kmeans.train(feat)
        center = kmeans.centroids
        D, I = kmeans.index.search(feat, 1)
        center = torch.from_numpy(center).cuda()
        I= torch.from_numpy(I).cuda()
        D= torch.from_numpy(D).cuda()
        center = F.normalize(center, p=2 , dim=-1)
        return center, I.squeeze(), D
    def save_npy(self, name, tensor, iter=None):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()
        if name[-4:]!='.npy':
            name = name+ '.npy'
        if self.config.transfer_all:
            name = '{}_{}_'.format(self.config.source, self.config.target)+name
        if iter is not None:
            name = '{}_'+name
            np.save(osp.join(self.config.snapshot, name.format(iter)), tensor)
        else:
            np.save(osp.join(self.config.snapshot, name), tensor)
    def save_pickle(self, name, dic, iter=None):
        if name[-4:]!='.pkl':
            name = name+'.pkl'
        if self.config.transfer_all:
            name = '{}_{}_'.format(self.config.source, self.config.target)+name
        if iter is None:
            path = osp.join(self.config.snapshot, name)
        else:
            name = '{}_'+name
            path = osp.join(self.config.snapshot, name.format(iter))
        print(path)
        files = open(path, 'wb')
        pickle.dump(dic, files)
        files.close()

