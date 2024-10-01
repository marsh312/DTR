from __future__ import print_function
import sys
sys.path.append('.')
import numpy as np
import torch
from .Network_Model import Network
from .Cluster_Faiss import Cluster_Faiss
import math
from .Tree_Model import Tree
from .tree_learning import Tree_Learner
from .negative_sampling import top_down_sample,uniform_sampling_multiclassifcation,softmax_sampling,all_negative_sampling
import copy
from .calculate_weights import calulate_calibrated_weight_in_full_softmax, calculate_max_weight
import time

# from lib.sasrec import SASRec
# from .sasrec import SASRec

global calculate_weight_time
global train_time
calculate_weight_time = 0
train_time = 0


class TrainModel:
    def __init__(self,
                 ids,
                 codes,
                 all_training_instance=None,
                 item_user_pair_dict=None,
                 embed_dim=24,
                 feature_groups=[20,20,10,10,2,2,2,1,1,1],
                 optimizer=lambda params: torch.optim.Adam(params, lr=1e-3, amsgrad=True),
                 parall=3,
                 temperature=1.0,
                 N=40,
                 sampling_method='uniform_multiclass',
                 tree_learner_mode='jtm',
                 item_node_share_embedding=True,
                 gamma=1.0e-1,
                 device='cuda',
                 calibrate_weight_mode='full', #'sampled', 'sasrec', 'none', 'max'
                 steps_calculate_weight=0,
                 max_calculate_calibrated_weight_num=300000,
                 sample_num_for_calibrated_weight=None,
                 max_calculate_num_for_tree_learner=30000,
                 sasrec_model=None,
                 scheduler_mode='exponential',
                 dropout_p=0.0,
                 calibrated_mode='normalized'):

        if calibrate_weight_mode == 'sasrec':
            assert sasrec_model is not None, 'sasrec_model_path should not be None when calibrate_weight_mode is sasrec'
            self.sasrec_model = sasrec_model
            self.sasrec_model.eval()
        elif calibrate_weight_mode == 'sampled':
            assert sample_num_for_calibrated_weight is not None, 'sample_num_for_calibrated_weight should not be None when calibrate_weight_mode is sampled'

        self.calibrated_weight_mode = calibrate_weight_mode
        self.calibrated_mode = calibrated_mode
        self.steps_calculate_weight = steps_calculate_weight
        self.max_calculate_calibrated_weight_num = max_calculate_calibrated_weight_num
        self.sample_num_for_calibrated_weight = sample_num_for_calibrated_weight
        self.scheduler_mode = scheduler_mode
        self.dropout_p = dropout_p

        self.device=device
        self.embed_dim=embed_dim
        self.feature_groups=feature_groups
        self.opti=optimizer
        self.parall=parall
        self.item_num=len(ids)
        self.feature_num=sum(feature_groups)
        self.temperature=temperature
        self.sample_method=sampling_method
        self.item_node_share_embedding=item_node_share_embedding
        self.N=N
        self.tree_learner_mode=tree_learner_mode
        self.gamma=gamma

        self.tree=Tree(ids,codes, device=self.device)
        self.tree_list=[self.tree]
        # self.sample_num=sum([min([self.N+1,self.tree.layer_node_num_list[layer]])\
        #                                        for layer in range(1,self.tree.max_layer_id+1)])
        # model part
        self.network_model=Network(embed_dim=embed_dim,feature_groups=feature_groups,item_num=self.item_num,
                                node_num=self.tree.node_num,
                                drop_out=True,
                                dropout_p=dropout_p,
                                item_node_share_embedding= self.item_node_share_embedding).to(self.device)
        self.model_list=[self.network_model]
        # optimizer
        self.optimizer = optimizer(self.network_model.parameters())
        if self.tree_learner_mode=='jtm':
            self.tree_learner=Tree_Learner(all_training_instance,item_user_pair_dict,max_calculate_num=max_calculate_num_for_tree_learner)
        elif self.tree_learner_mode=='tdm':
            self.cluster=Cluster_Faiss(parall=parall)
        else:
            assert False,'invalid tree learner mode : {}'.format(self.tree_learner_mode)
        self.batch_num=0

    def update_learning_rate(self, t, learning_rate_base=1e-3, warmup_steps=5000, alpha=2.0,
                         decay_rate=1./3, learning_rate_min=1e-6, mode='exponential', poly_power=3.0):
        if mode == 'exponential':
            lr = learning_rate_base * np.minimum(
                (t + 1.0) / warmup_steps,
                np.exp(decay_rate * ((warmup_steps - t - 1.0) / warmup_steps)),
            )
        elif mode == 'linear':
            lr = learning_rate_base * np.minimum(
                (t + 1.0) / warmup_steps,
                1.0 - (t / warmup_steps)
            )
        elif mode == 'cosine':
            lr = learning_rate_base * np.minimum(
                (t + 1.0) / warmup_steps,
                0.5 * (1 + np.cos(np.pi * t / (alpha * warmup_steps)))
            )
        elif mode == 'polynomial':
            lr = learning_rate_base * np.minimum(
                (t + 1.0) / warmup_steps,
                (1 - t / warmup_steps) ** poly_power
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}. Supported modes are: 'exponential', 'linear', 'cosine', 'polynomial'")

        lr = np.maximum(lr, learning_rate_min)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def update_network_model(self,batch_x,batch_y, lr_base=1e-3, min_lr=1e-6, warmup_steps=5000):#batch_x is the rows of training instances, batch_y is the corresponding labels.
        self.batch_num+=1
        train_start_time = time.time()
        all_leaf_codes=torch.LongTensor([self.tree.item_id_leaf_code[label.item()] for label in batch_y]).to(self.device).view(-1,1)
        loss=self.generate_training_instances(batch_x,all_leaf_codes)#column major
        loss.backward()#compute the gradient
        self.optimizer.step()# update the parameters
        self.optimizer.zero_grad()# clean the gradient
        self.update_learning_rate(self.batch_num, learning_rate_base=lr_base, learning_rate_min=min_lr, mode=self.scheduler_mode, warmup_steps=warmup_steps)
        global train_time
        train_time += time.time() - train_start_time
        return loss

    def train_specified_layer(self, batch_x, batch_y, layer_id, layer_N, min_lr=1e-6):
        start_label = sum([self.tree.layer_node_num_list[layer] for layer in range(layer_id)])
        layer_node_num = self.tree.layer_node_num_list[layer_id]
        sample_num = min([layer_N + 1, layer_node_num])

        self.batch_num += 1
        Bs, d = batch_x.shape
        batch_x = batch_x.to(self.device)

        samples_labels = torch.full((Bs, sample_num),-1,device=self.device, dtype=torch.int64)
        log_q_matrix=torch.full(samples_labels.shape,0.0,device=self.device,dtype=torch.float32)
        """positive nodes"""
        samples_labels[:, 0:1]=self.tree.item_id_node_ancestor_id[batch_y.to(self.device), layer_id].view(-1, 1)
        """"negative samples"""
        if layer_node_num <= layer_N+1:
            all_labels=torch.arange(start_label, start_label+layer_node_num, device=self.device).expand(Bs, layer_node_num)
            samples_labels[:, 1:] = all_labels[all_labels != samples_labels[:, 0:1] ].view(Bs, layer_node_num - 1)        
        else:
            negative_labels = torch.randint(start_label, start_label+layer_node_num, device=self.device, size=(Bs, layer_N))
            effective_negative_labels_index = negative_labels != samples_labels[:, 1:]
            samples_labels[:, 1:][effective_negative_labels_index] = negative_labels[effective_negative_labels_index]
            log_q_matrix[:, 1:][effective_negative_labels_index] = torch.log(
                effective_negative_labels_index.sum(-1).view(Bs, 1)
                / (layer_node_num - 1.0)
            ).expand(Bs, layer_N)[effective_negative_labels_index]

        effective_index = samples_labels>=0 #[Bs, sample_num]
        training_labels = samples_labels[effective_index].view(-1, 1) #[N_effective_lable, 1]

        o_pi=torch.full(samples_labels.shape,-1e9,device=self.device,dtype=torch.float32)

        if not self.item_node_share_embedding:
            raise Exception('not implemented')
        user_index = torch.arange(Bs).view(-1,1).expand(samples_labels.shape).to(self.device)[effective_index] #[N_effective_lable, 1]
        new_batch_user = batch_x[user_index] #[N_effective_lable, d]
        effective_item_index = new_batch_user>=0
        effective_items = new_batch_user[effective_item_index]
        new_batch_user[effective_item_index] = self.tree.item_id_node_ancestor_id[effective_items, layer_id]
        o_pi[effective_index] = self.network_model.preference(new_batch_user, training_labels)[:, 0] - log_q_matrix[effective_index]
        layer_loss = (torch.logsumexp(o_pi.view(-1,sample_num),dim=1)-o_pi.view(-1,sample_num)[:,0])
        layer_loss = layer_loss.mean()
        layer_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.update_learning_rate(self.batch_num, learning_rate_min=min_lr, mode=self.scheduler_mode)

        return layer_loss

    def use_all_negative(self,batch_x, all_leaf_codes):  # all_codes is a tensor with 1 column
        m,d=batch_x.shape
        # all_negative_sampling(batch_size,tree,all_leaf_codes,device)
        sample_labels=all_negative_sampling(m,self.tree,all_leaf_codes,self.device)
        start_col,loss=0,0.0
        batch_user=batch_x.to(self.device)
        effective_item_index= batch_user >= 0 #[Bs, d]
        effective_items=batch_user[effective_item_index]
        if self.item_node_share_embedding:
            # real_item_ids = torch.full((m, d), -1, dtype=torch.int64)
            for p_layer in range(self.tree.max_layer_id):
                layer_node_num=self.tree.layer_node_num_list[p_layer+1]
                all_labels=sample_labels[:,start_col:start_col+layer_node_num]

                batch_user[effective_item_index]=self.tree.item_id_node_ancestor_id[effective_items,p_layer+1]

                o_pi= self.network_model.preference(batch_user.repeat(1,layer_node_num).view(-1,d),\
                                                                  all_labels.reshape(-1,1))[:, 0].view(m,layer_node_num)
                loss += (torch.logsumexp(o_pi, dim=1) - o_pi[:, 0]).mean(-1)
                start_col += layer_node_num
        else:
            for p_layer in range(self.tree.max_layer_id):
                layer_node_num = self.tree.layer_node_num_list[p_layer + 1]
                all_labels=sample_labels[:,start_col:start_col+layer_node_num]
                o_pi = self.network_model.preference(batch_x.to(self.device).repeat(1, layer_node_num).view(-1, d), \
                                                 all_labels.view(-1, 1))[:, 0].view(m, layer_node_num)
                loss += (torch.logsumexp(o_pi, dim=1) - o_pi[:, 0]).mean(-1)
                start_col += layer_node_num
        return loss

    """penalty is the bias between exp{parent} and sumexp{children}, root node isn't in the penalty"""
    def generate_top_down_sample(self, batch_user, all_leaf_codes):  # all_leaf_codes is a tensor with 1 column
        m, d = batch_user.shape
        # def top_down_sample(batch_user,tree,network_model,all_leaf_codes,N,device):
        if self.gamma>0.0:
            samples, sibling_labels, effective_index, log_q_matrix,start_layer =\
                top_down_sample(batch_user, self.tree,self.network_model,all_leaf_codes,self.N,self.device,gamma=self.gamma)
            effective_sibling_index = (sibling_labels >= 0) & effective_index  # start from layer 2
            exp_o_sibling = torch.full(sibling_labels.shape, 0.0, device=self.device, dtype=torch.float32)
        else:
            samples, effective_index, log_q_matrix, start_layer = \
                top_down_sample(batch_user, self.tree, self.network_model, all_leaf_codes, self.N, self.device,
                                gamma=self.gamma)

        o_pi = torch.full(samples.shape, -1.0e9, device=self.device, dtype=torch.float32)
        ####test#########
        # assert (samples[effective_index]<0).sum().item()==0
        o_samples=torch.full(samples.shape,-1.0e9, device=self.device, dtype=torch.float32)

        if self.item_node_share_embedding:
            new_batch_user=\
                batch_user[torch.arange(m,device=self.device).view(-1, 1).expand(samples.shape)[effective_index]]
            label_layer=torch.arange(1,self.tree.max_layer_id+1,device=self.device).view(-1,1)\
                                            .repeat(1,self.N+1).view(-1).expand(samples.shape)[effective_index].view(-1,1)
            effective_item_index = new_batch_user>=0
            new_batch_user[effective_item_index]=\
                self.tree.item_id_node_ancestor_id[new_batch_user[effective_item_index],\
                                                        label_layer.expand(new_batch_user.shape)[effective_item_index]]

            o_samples[effective_index] = \
                self.network_model.preference(new_batch_user, samples[effective_index].view(-1, 1))[:, 0]
            o_pi[effective_index] = o_samples[effective_index] - log_q_matrix[effective_index]

            if self.gamma>0.0:
                samples[effective_index]=torch.arange(new_batch_user.shape[0],device=self.device)
                exp_o_sibling[effective_sibling_index]=torch.exp(
                        self.network_model.preference(new_batch_user[samples[effective_sibling_index]],\
                                              sibling_labels[effective_sibling_index].view(-1, 1))[:,0])
        else:
            pass

        all_loss = (torch.logsumexp(o_pi.view(-1,self.N+1),dim=1)-o_pi.view(-1,self.N+1)[:,0]) #[Bs*max_layer_id, ]

        assert self.calibrated_weight_mode in ['sampled', 'sasrec', 'full', 'none', 'max']
        if self.calibrated_weight_mode == 'none' or self.batch_num < self.steps_calculate_weight:
            all_loss =  all_loss.mean()
        else:
            calculate_weight_start = time.time()
            if self.calibrated_weight_mode == 'sampled':
                weight = self.calculate_sampled_calibrated_weight(batch_user, all_leaf_codes)
            elif self.calibrated_weight_mode == 'sasrec':
                weight = self.calculate_calibrated_weight_use_sasrec(batch_user, all_leaf_codes)
            elif self.calibrated_weight_mode == 'full':
                weight = self.calculate_calibrated_weight(batch_user, all_leaf_codes)
            elif self.calibrated_weight_mode == 'max':
                weight = calculate_max_weight(batch_user, self.tree, self.network_model, all_leaf_codes, self.device)
            global calculate_weight_time
            calculate_weight_time += time.time() - calculate_weight_start
            all_loss = (all_loss * weight).mean()

        if self.gamma>0.0:
            start_col=(self.N+1)*(start_layer-1)
            exp_o_samples = torch.exp(o_samples)
            all_penalty = torch.abs((2.0*exp_o_samples[:,start_col:-(self.N+1)] -\
                            (exp_o_samples[:,start_col+self.N+1:]+exp_o_sibling[:,start_col+self.N+1:]))\
                                        [effective_index[:,start_col:-(self.N+1)]]).mean()
            return all_loss+self.gamma*all_penalty
        else:
            return all_loss

    def generate_training_instances(self,batch_x, all_leaf_codes):  # all_codes is a array
        m,d=batch_x.shape
        batch_user=batch_x.to(self.device)
        if self.sample_method=='softmax':
            samples,log_q_matrix=softmax_sampling(batch_x,self.tree,self.network_model,all_leaf_codes,self.N,self.device,
                                                   temperature=self.temperature,
                                                   itme_node_share_embedding=self.item_node_share_embedding)
        elif self.sample_method=='uniform_multiclass':
            # uniform_sampling_multiclassifcation(batch_size,tree,all_leaf_codes,N,device):
            samples,log_q_matrix=uniform_sampling_multiclassifcation(m,self.tree,all_leaf_codes,self.N,self.device)
        elif self.sample_method=='top_down':
            return self.generate_top_down_sample(batch_user,all_leaf_codes)
        elif self.sample_method=='all_negative_sampling':
            return self.use_all_negative(batch_x, all_leaf_codes)
        else:
            assert False,'{} sampling is an invalid choice'.format(self.sample_method)

        # sample_num=samples.shape[-1]
        effective_index=samples>=0
        training_labels = samples[effective_index].view(-1, 1)

        o_pi=torch.full(samples.shape,-1e9,device=self.device,dtype=torch.float32) #[Bs, (N+1)*max_layer_id]

        if self.item_node_share_embedding:
            user_index = torch.arange(m).view(-1,1).expand(samples.shape).to(self.device)[effective_index]
            new_user_batch=batch_user[user_index]

            label_layer=torch.arange(1,self.tree.max_layer_id+1,device=self.device).view(-1,1)\
                                            .repeat(1,self.N+1).view(-1).expand(m,-1)[effective_index].view(-1,1)
            effective_item_index=new_user_batch>=0

            new_user_batch[effective_item_index]=\
                self.tree.item_id_node_ancestor_id[new_user_batch[effective_item_index],\
                                                   label_layer.expand(-1,d)[effective_item_index]]
            o_pi[effective_index]=\
                self.network_model.preference(new_user_batch,training_labels)[:,0]-log_q_matrix[effective_index]
        else:

            user_index = torch.arange(m).view(-1, 1).expand(samples.shape)[effective_index]

            o_pi[effective_index] = self.network_model.preference(batch_x.to(self.device)[user_index],
                                                              training_labels)[:, 0] - log_q_matrix[effective_index]

        # net_loss=(torch.logsumexp(o_pi.view(-1,self.N+1),dim=1)-o_pi.view(-1,self.N+1)[:,0]).mean(-1)
        net_loss = (torch.logsumexp(o_pi.view(-1,self.N+1),dim=1)-o_pi.view(-1,self.N+1)[:,0]) #[Bs*max_layer_id, ]

        assert self.calibrated_weight_mode in ['sampled', 'sasrec', 'full', 'none', 'max']
        if self.calibrated_weight_mode == 'none' or self.batch_num < self.steps_calculate_weight:
            net_loss =  net_loss.mean()
        else:
            calculate_weight_start = time.time()
            if self.calibrated_weight_mode == 'sampled':
                weight = self.calculate_sampled_calibrated_weight(batch_user, all_leaf_codes)
            elif self.calibrated_weight_mode == 'sasrec':
                weight = self.calculate_calibrated_weight_use_sasrec(batch_user, all_leaf_codes)
            elif self.calibrated_weight_mode == 'full':
                weight = self.calculate_calibrated_weight(batch_user, all_leaf_codes)
            elif self.calibrated_weight_mode == 'max':
                weight = calculate_max_weight(batch_user, self.tree, self.network_model, all_leaf_codes, self.device)
            global calculate_weight_time
            calculate_weight_time += time.time() - calculate_weight_start
            net_loss = (net_loss * weight).mean()
        return net_loss

    def update_tree(self,d=7,discriminator=None):
        if self.tree_learner_mode=='jtm':
            pi_new=self.tree_learner.tree_learning(d,self.tree,self.network_model)
            ids,codes=[],[]
            for item_id, leaf_code in pi_new.items():
                ids.append(item_id)
                codes.append(leaf_code)
            self.tree = Tree(ids, codes)
            self.network_model = Network(embed_dim=self.embed_dim, feature_groups=self.feature_groups,
                                         item_num=self.item_num,
                                         node_num=self.tree.node_num,
                                         drop_out=True,
                                         dropout_p=self.dropout_p,
                                         item_node_share_embedding=self.item_node_share_embedding).to(self.device)
            self.network_model.load_state_dict(self.model_list[-1].state_dict())
            self.optimizer = self.opti(self.network_model.parameters())
        elif self.tree_learner_mode=='tdm':
            if self.item_node_share_embedding:
                item_id_leaf_codes=[self.tree.item_id_leaf_code[item_id] for item_id in range(self.item_num)]
                leaf_node_id=[self.tree.node_code_node_id[code] for code in item_id_leaf_codes]
                cluster_data=self.network_model.node_embedding.weight.data[leaf_node_id].cpu().numpy()
            else:
                cluster_data = self.network_model.item_embedding.weight.data.cpu().numpy()[0:self.item_num, :]
            item_num,embed_dim=cluster_data.shape
            assert item_num==self.item_num and embed_dim==self.embed_dim
            print('cluster data shape {},{}'.format(*cluster_data.shape))
            ids, codes = self.cluster.train(ids=np.arange(self.item_num), data=cluster_data)  # return ids, codes
            self.tree = Tree(ids, codes)
            self.network_model = Network(embed_dim=self.embed_dim, feature_groups=self.feature_groups,
                                         item_num=self.item_num,
                                         node_num=self.tree.node_num,
                                         drop_out=True,
                                         dropout_p=self.dropout_p,
                                         item_node_share_embedding=self.item_node_share_embedding).to(self.device)
            # self.network_model.load_state_dict(self.model_list[-1].state_dict())
            self.optimizer = self.opti(self.network_model.parameters())
        else:
            assert False,'invalid tree learner mode : {}'.format(self.tree_learner_mode)

        self.tree_list.append(self.tree)
        self.model_list.append(self.network_model)
        # del self.network_model
        if self.device !='cpu':
            torch.cuda.empty_cache()
        self.batch_num=0
        # self.sample_num=sum([min([self.N+1,self.tree.layer_node_num_list[layer]])\
        #                                        for layer in range(1,self.tree.max_layer_id+1)])

    def predict_share_embedding(self, user_x, N, topk, discriminator=None, tree=None):
        if discriminator is None:
            discriminator = self.network_model
        if tree is None:
            tree = self.tree
        candidate_codes = {0}
        effective_index = user_x >= 0
        effective_items = user_x[effective_index]
        result_set_A = set()  # result_set_A contains leaf node codes
        while len(candidate_codes) > 0:
            leaf_nodes = {code for code in candidate_codes if int(math.log(code + 1, 2)) == tree.max_layer_id}
            result_set_A |= leaf_nodes  # result_set_A contains leaf node codes
            candidate_codes -= leaf_nodes
            if len(candidate_codes) <= 0: break
            test_labels = torch.LongTensor([tree.node_code_node_id[code] for code in candidate_codes]).view(-1, 1)
            label_layer_ids = torch.LongTensor([int(math.log(code + 1, 2)) for code in candidate_codes]).view(-1, 1)
            test_codes = torch.LongTensor([code for code in candidate_codes])

            user_x_mat = torch.full((len(candidate_codes), user_x.shape[-1]), -1, dtype=torch.int64, device=self.device)
            user_x_mat[effective_index.expand(user_x_mat.shape)] = \
                tree.item_id_node_ancestor_id[effective_items.repeat(len(candidate_codes)),
                                              label_layer_ids.expand(user_x_mat.shape)[
                                                  effective_index.expand(user_x_mat.shape)]]
            with torch.no_grad():
                log_probs = discriminator.preference(user_x_mat, test_labels.to(self.device))[:, 0]
            _, index = log_probs.sort(dim=-1)
            selected_codes = test_codes.gather(index=index[-N:].cpu(), dim=-1)
            candidate_codes.clear()
            for code in selected_codes:
                c = int(code.item())
                if 2 * c + 1 in tree.node_code_node_id:
                    candidate_codes.add(2 * c + 1)
                if 2 * c + 2 in tree.node_code_node_id:
                    candidate_codes.add(2 * c + 2)
        result_codes = torch.tensor([code for code in result_set_A]).view(-1)
        result_labels = torch.LongTensor([tree.node_code_node_id[code] for code in result_set_A])
        user_x_mat = torch.full(user_x.shape, -1, dtype=torch.int64, device=self.device)
        user_x_mat[effective_index] = tree.item_id_node_ancestor_id[effective_items, tree.max_layer_id]
        with torch.no_grad():
            log_probs = discriminator.preference(user_x_mat.expand(len(result_set_A), user_x.shape[-1]),
                                                 result_labels.to(self.device).view(-1, 1))[:, 0]
        sorted_values, index = log_probs.sort(dim=-1)
        # return [tree.leaf_code_item_id[code] for code in\
        #        result_codes.gather(index=index.cpu()[-topk:],dim=-1).numpy()]
        return [(code, tree.leaf_code_item_id[code]) for v, code in zip(sorted_values.cpu().numpy(),
                                                                        result_codes.gather(index=index.cpu(),
                                                                                            dim=-1).numpy())]

    def predict_share_embedding_parallel(self,user_ids,N,topk,discriminator=None,tree=None):
        ##user_ids is [bs,1]
        batch_size=user_ids.shape[0]
        user_x=user_ids.to(self.device)

        effective_user_index=user_x>=0
        effective_items=user_x[effective_user_index] 
        if discriminator is None:
            discriminator=self.network_model
        if tree is None:
            tree=self.tree

        selected_index=torch.arange(batch_size,device=self.device).view(-1,1).expand(batch_size,2*N)

        parent_code=torch.zeros((batch_size,N),device=self.device,dtype=torch.int64)
        candidate_codes = torch.zeros((batch_size,2*N),dtype=torch.int64,device=self.device)
        preference_mat=torch.full_like(candidate_codes,-1.0e9,device=self.device,dtype=torch.float32)

        layer_id=0
        assert tree.layer_node_num_list[0]==1 and len(tree.layer_node_num_list)==tree.max_layer_id+1
        while layer_id<tree.max_layer_id:

            p_code_num=min([tree.layer_node_num_list[layer_id],N])

            double_p_code=2*parent_code[:,0:p_code_num]
            candidate_codes[:,0:2*p_code_num:2]=double_p_code+1
            candidate_codes[:,1:2*p_code_num:2]=double_p_code+2

            all_labels =tree.node_code_node_id_array[candidate_codes[:,:2*p_code_num]].to(self.device)
            effective_index=all_labels>=0

            # test_codes = torch.LongTensor([code for code in candidate_codes])
            user_x[effective_user_index]=tree.item_id_node_ancestor_id[effective_items,layer_id+1]

            preference_mat[:,:]=-1.0e9
            with torch.no_grad():
                preference_mat[:,:2*p_code_num][effective_index] =\
                    torch.exp(discriminator.preference(user_x[selected_index[:,:2*p_code_num][effective_index]],\
                                                                            all_labels[effective_index].view(-1,1))[:, 0])

            index = preference_mat[:,:2*p_code_num].argsort(dim=-1) # ascending order
            selected_p=min([N,tree.layer_node_num_list[layer_id+1]])
            parent_code = candidate_codes[:,:2*p_code_num].gather(index=index[:,-selected_p:], dim=-1)
            layer_id+=1
        result_code_matrix=candidate_codes[:,:2*p_code_num].gather(index=index[:,-topk:], dim=-1).cpu().numpy()
        result_items=np.zeros((batch_size,topk),dtype=np.int32)
        for i,r_code in enumerate(result_code_matrix):
            result_items[i:i+1,:]=np.array([tree.leaf_code_item_id[code] for code in r_code])
        return result_items

    def predict(self, user_x, N, topk, out_put_each_tree_result=False, forest=True):
        if not forest:
            if self.item_node_share_embedding:
                # full_result= self.predict_share_embedding(user_x, N, topk)
                # return [item_id for (_,item_id) in full_result[-topk:]]
                return self.predict_share_embedding_parallel(user_x, N, topk)
            else:
                assert False, 'no predict way!'
        # result = []
        each_tree_result = []
        all_result_items = set()
        for tree_id in range(len(self.tree_list)):
            discriminator = self.model_list[tree_id]

            discriminator.eval()
            re = self.predict_share_embedding(user_x, N, topk, \
                                              discriminator=discriminator, tree=self.tree_list[tree_id])
            discriminator.train()

            # result.extend(re)
            each_tree_re = [item_id for (_, item_id) in re[-topk:]]
            each_tree_result.append(each_tree_re)

            # all_result_codes.update([code for (code, _) in re])
            all_result_items.update([item_id for (_, item_id) in re])
            # network_model.train()
        discriminator = self.model_list[-1]
        tree=self.tree_list[-1]
        discriminator.eval()
        result_codes = torch.LongTensor([tree.item_id_leaf_code[item_id] for item_id in all_result_items])
        with torch.no_grad():
            effective_index = user_x >= 0
            effective_items = user_x[effective_index]
            user_x_mat = torch.full(user_x.shape, -1, dtype=torch.int64, device=self.device)
            user_x_mat[effective_index] = tree.item_id_node_ancestor_id[effective_items, tree.max_layer_id]
            result_labels = torch.LongTensor([tree.node_code_node_id[code.item()] for code in result_codes])
            log_probs = discriminator.preference(user_x_mat.expand(len(result_codes), user_x.shape[-1]),
                                                 result_labels.to(self.device).view(-1, 1))[:, 0]
        _, index = log_probs.sort(dim=-1)
        selected_codes = result_codes.gather(index=index[-topk:].cpu(), dim=-1).tolist()
        selected_items = [tree.leaf_code_item_id[code] for code in selected_codes]
        discriminator.train()
        if out_put_each_tree_result:
            return selected_items, each_tree_result
        else:
            return selected_items

    def calculate_calibrated_weight(self, batch_user, target_leaf_codes):
        m, d = batch_user.shape
        with torch.no_grad():
            weight = None
            leaf_num = self.tree.layer_node_num_list[-1]
            calculate_weight_batch_size = int(self.max_calculate_calibrated_weight_num / leaf_num)
            calculate_weight_steps = math.ceil(m / calculate_weight_batch_size)
            calculate_weight_steps = max(calculate_weight_steps, 1)
            for i in range(calculate_weight_steps):
                start = i * calculate_weight_batch_size
                end = min((i + 1) * calculate_weight_batch_size, m)
                new_batch_user = batch_user[start:end]

                if not self.item_node_share_embedding:
                    raise Exception('item_node_share_embedding should be True')
                new_batch_user = new_batch_user.to(self.device)
                effective_item_index = new_batch_user >= 0
                effective_items = new_batch_user[effective_item_index]
                new_batch_user[effective_item_index] = self.tree.item_id_node_ancestor_id[effective_items, self.tree.max_layer_id]

                start_label = sum([self.tree.layer_node_num_list[layer] for layer in range(self.tree.max_layer_id)])
                leaf_labels = torch.arange(start_label, start_label + leaf_num, device=self.device).repeat(end - start, 1)

                effective_leaf_scores = self.network_model.preference(new_batch_user.repeat(1, leaf_num).view(-1, d), \
                                                                        leaf_labels.reshape(-1, 1))[:, 0].view(end - start, leaf_num)

                cur_weight = calulate_calibrated_weight_in_full_softmax(
                    effective_leaf_scores,
                    target_leaf_codes[start:end],
                    self.tree.max_layer_id,
                    self.tree.leaf_index_complete_leaf_index_array,
                    self.device,
                    self.calibrated_mode
                )
                assert cur_weight.shape[0] == (end - start)* self.tree.max_layer_id
                assert len(cur_weight.shape) == 1
                if weight is None:
                    weight = cur_weight
                else:
                    weight = torch.cat((weight, cur_weight), dim=0)
                torch.cuda.empty_cache()
        return weight

    def calculate_sampled_calibrated_weight(self, batch_user, target_leaf_codes):
        Bs, d = batch_user.shape
        with torch.no_grad():
            max_weight = torch.ones((Bs, self.tree.max_layer_id), device=self.device)
            node_codes = target_leaf_codes
            user_index = torch.arange(Bs, device=self.device).view(-1,1).repeat(1, 2)
            child_preference = torch.full((Bs, 2), -torch.inf, device=self.device)
            for layer in range(self.tree.max_layer_id, 1, -1):
                is_left = node_codes % 2 == 1
                sibling_codes = torch.where(is_left, node_codes + 1, node_codes - 1)
                codes = torch.cat((node_codes.view(-1,1), sibling_codes.view(-1,1)), dim=-1) #[Bs, 2]
                all_labels = self.tree.node_code_node_id_array[codes]
                effective_index = all_labels >= 0
                effective_labels = all_labels[effective_index]

                cur_user_index = user_index[effective_index].view(-1)
                new_batch_user = batch_user[cur_user_index]
                effective_item_index = new_batch_user >= 0
                new_batch_user[effective_item_index] = \
                    self.tree.item_id_node_ancestor_id[new_batch_user[effective_item_index], layer]

                child_preference[:] = -torch.inf
                child_preference[effective_index] = self.network_model.preference(new_batch_user, effective_labels.view(-1,1))[:, 0]
                softmax_child_preference = torch.nn.functional.softmax(child_preference, dim=-1)
                max_weight[:, layer-2] = (softmax_child_preference[:,0] >= softmax_child_preference[:,1]).float() 
                node_codes = torch.div(node_codes-1, 2, rounding_mode='floor')
            max_weight = max_weight.view(-1)

            approximate_expectation = None
            leaf_num = self.tree.layer_node_num_list[-1]
            calculate_expectation_batch_size = int(self.max_calculate_calibrated_weight_num / self.sample_num_for_calibrated_weight)
            calculate_expectation_steps = math.ceil(Bs / calculate_expectation_batch_size)
            calculate_expectation_steps = max(calculate_expectation_steps, 1)
            for i in range(calculate_expectation_steps):
                start = i * calculate_expectation_batch_size
                end = min((i + 1) * calculate_expectation_batch_size, Bs)
                new_batch_user = batch_user[start:end]
                new_leaf_codes = target_leaf_codes[start:end]

                if not self.item_node_share_embedding:
                    raise Exception('item_node_share_embedding should be True')
                new_batch_user = new_batch_user.to(self.device)
                effective_item_index = new_batch_user >= 0
                effective_items = new_batch_user[effective_item_index]
                new_batch_user[effective_item_index] = self.tree.item_id_node_ancestor_id[effective_items, self.tree.max_layer_id]

                target_leaf_labels= self.tree.node_code_node_id_array[new_leaf_codes].to(self.device) #[Bs, 1]
                start_label = sum([self.tree.layer_node_num_list[layer] for layer in range(self.tree.max_layer_id)])
                negative_labels = torch.randint(start_label, start_label+leaf_num, device=self.device, size=(Bs, self.sample_num_for_calibrated_weight)) #[Bs, N]
                negative_labels[negative_labels == target_leaf_labels.repeat(1, self.sample_num_for_calibrated_weight)] = -1
                new_labels = torch.cat((target_leaf_labels.repeat(1, 1), negative_labels), dim=1) #[Bs, N+1]
                effective_leaf_scores = torch.full(new_labels.shape, -torch.inf, device=self.device, dtype=torch.float32)

                effective_index = new_labels >= 0
                effective_labels = new_labels[effective_index].view(-1, 1)

                effective_leaf_scores[effective_index] = self.network_model.preference(
                    new_batch_user.repeat(1, self.sample_num_for_calibrated_weight + 1).view(-1, d)[effective_index.view(-1)],
                    effective_labels,
                )[:, 0]

                cur_approximate_expectation = torch.softmax(effective_leaf_scores, dim=1)[:, 0]
                assert cur_approximate_expectation.shape[0] == (end - start)
                assert len(cur_approximate_expectation.shape) == 1
                if approximate_expectation is None:
                    approximate_expectation = cur_approximate_expectation
                else:
                    approximate_expectation = torch.cat((approximate_expectation, cur_approximate_expectation), dim=0)
                torch.cuda.empty_cache()
            approximate_expectation = approximate_expectation.view(-1,1).repeat(1, self.tree.max_layer_id-1)
            approximate_expectation = torch.cat((approximate_expectation, torch.ones((Bs, 1), device=self.device)), dim=1).view(-1)
            sampled_weight = max_weight / approximate_expectation
        return sampled_weight

    def calculate_calibrated_weight_use_sasrec(self, batch_user, target_leaf_codes):
        with torch.no_grad():
            logits = self.sasrec_model(batch_user)
        weight = calulate_calibrated_weight_in_full_softmax(
                    logits,
                    target_leaf_codes,
                    self.tree.max_layer_id,
                    # self.tree.leaf_index_complete_leaf_index_array,
                    self.tree.item_id_complete_leaf_index_array,
                    self.device,
                    self.calibrated_mode
                )
        torch.cuda.empty_cache()
        return weight
