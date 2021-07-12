from .base_trainer import *
from model import *
from dataset import *
import sklearn
from utils.memory import Memory

def ExpWeight(step, gamma=3, max_iter=5000, reverse=False):
    step = max_iter-step
    ans = 1.0 * (np.exp(- gamma * step * 1.0 / max_iter))
    return float(ans)

class Trainer(BaseTrainer):

    def iter(self, i_iter):
        self.losses = edict({})

        self.optimizer.zero_grad()
        inverseDecaySheduler(self.optimizer, i_iter, self.config.lr, gamma=self.config.gamma, power=self.config.power, num_steps=self.config.num_steps)
        s_batch = next(self.s_loader)
        _, s_batch = s_batch
        s_img, s_label, _, _ = s_batch
        s_img = s_img.cuda()
        s_label = s_label.cuda()
        _, s_neck, s_prob, _ = self.model(s_img)
        s_loss = F.cross_entropy(s_prob, s_label.squeeze())
        loss = s_loss
        loss.backward()

        if self.warmup and i_iter+1 <= self.config.warmup_steps:
            self.optimizer.step()
            return 0
        del s_prob, s_neck, _, s_label

        
        # Cross-domain alignment on identified common samples with CDD

        batch = next(self.loader)
        _, batch = batch
        s_img, t_img, s_label, _, _ = batch
        n, k, c, h, w= s_img.shape
        s_img = s_img.cuda().view(-1, c, h, w)
        t_img = t_img.cuda().view(-1, c, h, w)
        s_label = s_label.cuda().view(-1)
        _, s_neck, s_prob, s_af_softmax = self.model(s_img)
        _, t_neck, _, t_af_softmax = self.model(t_img)
        counter = torch.ones(self.config.num_sample) * self.config.num_pclass
        counter = counter.long().tolist()
        cdd_loss = self.cdd.forward([s_neck], [t_neck], counter, counter)['cdd']
        s_loss = F.cross_entropy(s_prob, s_label)

        self.losses.source_loss = s_loss
        self.losses.cdd_loss = cdd_loss
        loss = self.config.lamb * cdd_loss + s_loss
        loss.backward()
        del _, s_neck, t_neck, s_prob, s_af_softmax, t_af_softmax 


        t_batch = next(self.t_loader)
        _, t_batch = t_batch
        t_img, t_label, _, _ = t_batch
        t_label = t_label.cuda()

        n, k, c, h, w  = t_img.shape
        t_img = t_img.view(-1, c, h, w).cuda()
        t_label = t_label.view(-1).cuda()

        b_pred, t_feat, _, t_prob = self.model(t_img.cuda())
        en_loss = self.memory.forward(t_feat, t_label, t=self.config.t, joint=False)

        en_loss = en_loss * ExpWeight(i_iter, gamma=self.config.gm)
        self.losses.entropy_loss = en_loss
        en_loss.backward()

        self.optimizer.step()

    def optimize(self):
        for i_iter in tqdm(range(self.config.stop_steps)):
            self.model = self.model.train()
            self.losses = edict({})
            losses = self.iter(i_iter)
            if i_iter % self.config.print_freq ==0:
                self.print_loss(i_iter)
            if not self.warmup or  i_iter+1>=self.config.warmup_steps:
                if (i_iter+1) % self.config.stage_size ==0:
                    self.class_set = self.re_clustering(i_iter)
                if (i_iter+1) % self.config.val_freq ==0:
                    self.validate(i_iter, self.class_set)

    def train(self):

        self.model = init_model(self.config)

        self.unknown = []
        self.model = self.model.train()
        self.center_history = []
        self.optimizer = optim.SGD(self.model.optim_parameters(self.config.lr),lr=self.config.lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay,nesterov=True)

        self.feat_dim = 256
        self.warmup = self.config.warmup

        if not self.warmup:
            t_centers, plabel_dict, class_set = self.cluster_matching(0)
            self.memory = Memory(self.config.num_centers, feat_dim=self.feat_dim)
            self.memory.init(t_centers)
            self.class_set = class_set
            self.cdd = CDD(1, (5,5), (2,2), num_classes=len(class_set))
            
            self.loader = init_class_dataset(self.config, plabel_dict=plabel_dict, src_class_set=class_set, tgt_class_set=class_set, length=self.config.stage_size)
            self.s_loader, self.t_loader = init_pair_dataset(self.config, length=self.config.stage_size, plabel_dict=self.cluster_label, binary_label=None)
            self.t_loader = init_target_dataset(self.config, length=self.config.stage_size, plabel_dict=self.cluster_label, tgt_class_set=[i for i in range(self.config.num_centers)])
        else:
            self.s_loader, self.t_loader = init_pair_dataset(self.config, length=self.config.warmup_steps, plabel_dict=None, binary_label=None)

        self.optimize()

        # save the final results to a txt file 
        self.save_txt()

    def re_clustering(self, i_iter):
        t_centers, plabel_dict, class_set = self.cluster_matching(i_iter)
        self.memory.init(t_centers)
        self.memory.init_source(self.init_src_private_centers)
        self.cdd = CDD(1, (5,5), (2,2), num_classes=len(class_set))
        # del self.loader, self.s_loader, self.t_loader
        if self.warmup and i_iter<self.config.warmup_steps:
            del self.s_loader, self.t_loader
        else:
            del self.loader, self.s_loader, self.t_loader
        self.loader = init_class_dataset(self.config, plabel_dict=plabel_dict, src_class_set=class_set, tgt_class_set=class_set, length=self.config.stage_size)
        self.s_loader, self.t_loader = init_pair_dataset(self.config, length=self.config.stage_size, plabel_dict=self.cluster_label, binary_label=None)
        self.t_loader =  init_target_dataset(self.config, length=self.config.stage_size, plabel_dict=self.cluster_label, tgt_class_set=[i for i in range(self.config.num_centers)])
        return class_set

    def consensus_score(self, t_feats, t_codes, t_centers, s_feats, s_labels, s_centers, step):
        # Calculate the consensus score of cross-domain matching 
        s_centers = F.normalize(s_centers, p=2, dim=-1)
        t_centers = F.normalize(t_centers, p=2, dim=-1)
        simis = torch.matmul(s_centers, t_centers.transpose(0, 1))
        s_index = simis.argmax(dim=1)
        t_index=  simis.argmax(dim=0)
        map_s2t =  [(i, s_index[i].item()) for i in range(len(s_index))]
        map_t2s =  [(t_index[i].item(), i) for i in range(len(t_index))]
        inter = [a for a in map_s2t if a in map_t2s]

        p_score = 0.0
        filtered_inter = []
        t_score = 0.0  
        s_score = 0.0 
        scores = []
        score_dict = {}
        score_vector = torch.zeros(s_centers.shape[0]).float().cuda()
        for i,j in inter:
            si_index = (s_labels == i).squeeze().nonzero(as_tuple=False)
            tj_index = (t_codes  == j).squeeze().nonzero(as_tuple=False)
            si_feat = s_feats[si_index, :]
            tj_feat = t_feats[tj_index, :]

            s2TC =torch.matmul(si_feat, t_centers.transpose(0, 1))
            s2TC = s2TC.argmax(dim=-1)
            p_i2j = (s2TC==j).sum().float()/len(s2TC)
            t2SC = torch.matmul(tj_feat, s_centers.transpose(0, 1))
            t2SC = t2SC.argmax(dim=-1)
            p_j2i = (t2SC==i).sum().float()/len(t2SC)

            cu_score =  (p_j2i + p_i2j)/2 
            score_dict[(i,j)] = (p_j2i,  p_i2j)
            filtered_inter.append((i,j))
            t_score += p_j2i.item()
            s_score += p_i2j.item()
            p_score += cu_score.item()
            scores.append(cu_score.item())
            score_vector[i] += cu_score.item()

        score = p_score/len(filtered_inter)
        t_score = t_score/len(filtered_inter)
        s_score = s_score/len(filtered_inter)
        min_score = np.min(scores)
        return score, score_vector, filtered_inter, scores, score_dict 

    def get_tgt_centers(self, step, s_centers, s_feats, s_labels):
        # Perform target clustering and then matching it with source clusters 
        self.model.eval()
        t_feats, t_gts, t_preds = self.gather_feats()
        init_step = 0 
        if self.config.warmup: 
            init_step = self.config.warmup_steps-1

        if step==init_step:
            init_center = self.config.num_classes 
        else:
            init_center = self.config.interval 
        max_center  = self.config.num_classes * self.config.max_search 
        
        gt_vector = torch.stack(list(t_gts.values()))
        id_, cnt = gt_vector.unique(return_counts=True)
        freq_dict= {}
        for i, cnt_ in zip(id_.tolist(), cnt.tolist()):
            freq_dict[i] = cnt_

        interval = int(self.config.interval)

        best_score = 0.0
        final_n_center = None
        final_t_codes = None
        final_t_centers = None
        score_dict = {}
        inter_memo = {}
        search = True
        if self.config.search_stop and self.fix_k(self.center_history, 2):
            search = False 
            self.k_converge=True
        n_center = init_center 
        score_his = []
        t_codes_dic = {}
        t_centers_dic = {}
        sub_scores = {}
        score_dicts = {}
        if search:
            while search and n_center <= max_center:
                t_centers, t_codes, sh_score = self.sklearn_kmeans(t_feats, n_center)
                mean_score, score_vector, inter, scores, sub_dict = self.consensus_score(t_feats, t_codes, t_centers, s_feats, s_labels, s_centers, step)
                inter_memo[n_center] = inter
                sub_scores[n_center] = scores
                score = mean_score 
                score_dict[n_center] = scores
                t_codes_dic[n_center] = t_codes
                t_centers_dic[n_center] = t_centers
                score_dicts[n_center] = sub_dict 
                if score > best_score:
                    final_n_center = n_center
                    best_score = score
               
                score_his.append(score)
                if self.config.drop_stop and self.detect_continuous_drop(score_his, n=self.config.drop, con=self.config.drop_con):
                    final_n_center = n_center - interval * (self.config.drop-1)
                    search=False
                n_center += interval 
                   
            inter = inter_memo[final_n_center]
            n_center = final_n_center
            t_centers = t_centers_dic[final_n_center]
            t_codes = t_codes_dic[final_n_center]
            final_sub_score = score_dict[final_n_center]
            final_score_dict = score_dicts[final_n_center]
            print('Num Centers: ', n_center)
        else:
            t_centers, t_codes, _  = self.sklearn_kmeans(t_feats, self.config.num_centers, init=self.memory.memory.cpu().numpy())
            st_score, t_score, inter, scores, sub_dict = self.consensus_score(t_feats, t_codes, t_centers, s_feats, s_labels, s_centers, step)
            n_center = self.config.num_centers
            final_sub_score = scores 
            final_score_dict = sub_dict 

        gt_vector = torch.stack(list(t_gts.values()))
        self.center_history.append(n_center)
        self.config.num_centers=n_center
        self.memory = Memory(self.config.num_centers, feat_dim=self.feat_dim)
        self.neptune_metric('cluster/num_centers', self.config.num_centers)

        names = list(t_gts.keys())
        id_dict = {}
        for i in t_codes.unique().tolist():
            msk = (t_codes==i).squeeze()
            i_index = msk.nonzero(as_tuple=False)
            id_dict[i] = [names[a] for a in i_index]
        t_centers = F.normalize(t_centers, p=2, dim=-1)    
        return t_feats, t_codes, t_centers, id_dict, t_gts, freq_dict, inter, final_sub_score 

    def target_filtering(self, t_feats, t_codes, t_gts, s_feats, s_codes, s_centers, t_centers, cycle_pair):
        index2name = list(t_gts.keys())

        filtered_cluster_label = {}
        filtered_pair = []
        n_src = s_centers.shape[0]
        n_tgt = t_centers.shape[0]
        freq = {}
        for s_index, t_index in cycle_pair:
            t_mask = t_codes == t_index
            if t_mask.sum()<=self.config.num_pclass:
                continue
            i_index = t_mask.squeeze().nonzero(as_tuple=False)
            i_names = [index2name[i[0]] for i in i_index.tolist()]
            
            filtered_pair.append((s_index, t_index))
            for n in i_names:
                filtered_cluster_label[n] = s_index
            freq[s_index] = len(i_names)
        return filtered_cluster_label, filtered_pair, freq

    def clus_acc(self, t_codes, t_gt, mapping, gt_freq, sub_score):
        # Print the status of matching 
        print(mapping, len(sub_score))
        for i, (src, tgt) in enumerate(mapping):
            mask = t_codes==tgt
            i_gt = torch.masked_select(t_gt, mask)
            i_acc = ((i_gt==src).sum().float())/len(i_gt)
            if src in gt_freq:
                gt_cnt = gt_freq[src]
            else:
                gt_cnt = 1.0
            recall = i_acc * len(i_gt) / gt_cnt
            print('{:0>2d}th Cluster ACC:{:.2f} Correct/Total/GT {:0>2d}/{:0>2d}/{:0>2d} Precision:{:.3f} Recall:{:.3f} Score:{:.2f}'.format(src, i_acc.item(), (i_gt==src).sum().item(), len(i_gt), int(gt_cnt), i_acc, recall, sub_score[i]))

    def cluster_matching(self, step):
        # Clustering matching 
        self.model.eval()
        s_centers, s_feats, s_labels = self.get_src_centers()

        t_feats, t_codes, t_centers, id_dict, gt_dict, gt_freq, inter, sub_scores   = self.get_tgt_centers(step, s_centers, s_feats, s_labels)
        gt_vector = torch.stack(list(gt_dict.values()))
        s_centers = F.normalize(s_centers, p=2, dim=-1)
        t_centers = F.normalize(t_centers, p=2, dim=-1)

        filtered_cluster_label, filtered_pair, freq = self.target_filtering(t_feats, t_codes, gt_dict, s_feats, s_labels, s_centers, t_centers, inter)
        self.clus_acc(t_codes.squeeze(), gt_vector.squeeze(), inter, gt_freq, sub_scores)

        correct = 0.0
        for name, plabel in filtered_cluster_label.items():
            if gt_dict[name] ==plabel:
                correct +=1

        label_set = [i[0] for i in filtered_pair]
        plabel_acc = correct/len(filtered_cluster_label)
        self.neptune_metric('cluster/Plabel Acc', plabel_acc)
        self.cluster_mapping = {i[1]:i[0] for i in filtered_pair}
        self.common_cluster_set = [i[1] for i in filtered_pair]
        self.private_label_set = [i for i in range(self.config.num_classes) if i not in label_set]
        self.private_mapping = {self.private_label_set[i]:i for i in range(len(self.private_label_set))}
        self.init_src_private_centers = s_centers[self.private_label_set, :]
        
        self.global_label_set = label_set 


        self.cluster_label = {}
        for k, names in id_dict.items():
            for n in names:
                self.cluster_label[n] = k

        return t_centers, filtered_cluster_label, label_set

    def fix_k(self, scores, n=3):
        # Stopping critetion: stop searching if K holds a certain value for n times.
        if len(scores) < n:
            return False
        scores = scores[-n:]
        flag = 0.0
        for i in scores:
            if i == scores[-n]:
                flag+=1
        if flag == n:
            return True
        else:
            return False

    def detect_continuous_drop(self, scores, n=3, con=False):
        # Stopping Criterion: stop searching in a round if the score drops continuously for n times.
        if len(scores) < n:
            return False
        scores = scores[-n:]
        flag = 0.0 
        if con:
            for i in range(1, n):
                if scores[-i] <= scores[-(i+1)]:
                    flag+=1
        else:
            flag = 0.0
            for i in scores:
                if i <= scores[-n]:
                    flag+=1
        if flag >= n-1:
            return True
        else:
            return False
