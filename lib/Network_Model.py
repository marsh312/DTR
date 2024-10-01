from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearLayer(nn.Module):
    def __init__(self,input_dim,out_dim,active_op='prelu',use_batch_norm=False,drop_out=False,drop_out_prob=0.2):
        super(LinearLayer,self).__init__()
        self.register_buffer('use_batch_norm',torch.tensor(use_batch_norm,dtype=torch.bool))
        self.register_buffer('drop_out',torch.tensor(drop_out,dtype=torch.bool))
        self.register_buffer('drop_out_prob',torch.tensor(drop_out_prob,dtype=torch.float32))
        self.fc=nn.Linear(input_dim,out_dim)
        if self.use_batch_norm:
            self.bn=nn.BatchNorm1d(out_dim)
        if self.drop_out:
            self.drop_out=nn.Dropout(p=self.drop_out_prob)
        if active_op=='prelu':
            self.active_op=torch.nn.PReLU()
        else:
            self.active_op=torch.nn.Sigmoid()
        #initialize the layer
        init_mean = 0.0
        init_stddev = 1.   # 0.001
        init_value = (init_stddev * np.random.randn(out_dim, input_dim).astype(
            np.float32) + init_mean) / np.sqrt(input_dim)
        self.fc.weight.data=torch.from_numpy(init_value)
        self.fc.bias.data=torch.zeros(out_dim)+0.1

        #initiallize the batch norm parameters
        if self.use_batch_norm:
            self.bn_gamma=1.0
            self.bn_beta=0.0
            self.bn.weight.data=torch.ones(out_dim)*self.bn_gamma
            self.bn.bias.data=torch.ones(out_dim)*self.bn_beta

    def forward(self,input_data):
        hidden=self.fc(input_data)
        if self.use_batch_norm:
            hidden=self.bn(hidden)
        if self.drop_out:
            hidden=self.drop_out(hidden)
        return self.active_op(hidden)

class ActiveUnit(nn.Module):
    def __init__(self,emb_dim, active_layer_node_num=[36,1],
                 active_op='prelu',use_batch_norm=False,):
        super(ActiveUnit,self).__init__()

        self.layers=nn.ModuleList()
        for i,num in enumerate(active_layer_node_num):
            if i==0:
                self.layers.append(LinearLayer(3*emb_dim,num,active_op=active_op,use_batch_norm=use_batch_norm))
            else:
                self.layers.append(LinearLayer(active_layer_node_num[i-1], num, active_op=active_op, use_batch_norm=use_batch_norm))


    def forward(self,batch_item_embedding,batch_node_embedding):
        hidden = torch.cat((batch_item_embedding, batch_item_embedding * batch_node_embedding, batch_node_embedding),-1)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden

class Network(nn.Module):
    def __init__(self,embed_dim=24,
                feature_groups=[20,20,10,10,2,2,2,1,1,1],
                layer_nodes=[128,64,24,1],
                item_num=10,
                node_num=10,
                active_op='prelu',
                drop_out=False,
                dropout_p=0.0,
                item_node_share_embedding=True):
        super(Network,self).__init__()
        self.linearparts= nn.ModuleList()
        self.register_buffer('window_num', torch.tensor(len(feature_groups),dtype=torch.int64))
        self.register_buffer('feature_num',torch.tensor(sum(feature_groups),dtype=torch.int64))
        self.register_buffer('embed_dim',torch.tensor(embed_dim,dtype=torch.int64))
        self.register_buffer('item_node_share_embedding',torch.tensor(item_node_share_embedding,dtype=torch.bool))

        start_dim=(len(feature_groups)+1)*embed_dim
        for i in range(len(layer_nodes)):
            if i==0 and drop_out:
                self.linearparts.append(LinearLayer(start_dim,layer_nodes[i],active_op=active_op,\
                                                    use_batch_norm=True,drop_out=drop_out, drop_out_prob=dropout_p))
            else:
                self.linearparts.append(LinearLayer(start_dim, layer_nodes[i], active_op=active_op,use_batch_norm=True))
            start_dim=layer_nodes[i]

        self.active_unit = ActiveUnit(embed_dim)

        self.node_embedding = nn.Embedding(node_num, embed_dim)

        if not item_node_share_embedding:
            self.item_embedding=nn.Embedding(item_num, embed_dim)
        #self.sigmoid=nn.Sigmoid()

        # window matrix for each window's weight sum
        window_matrix = torch.zeros(len(feature_groups),sum(feature_groups))
        start_index = 0
        for i, feature in enumerate(feature_groups):
            window_matrix[i, start_index:start_index + feature] = 1.0
            start_index += feature
        self.register_buffer('window_matrix',window_matrix)
    def LinearPart(self,input):
        h=input
        for layer in self.linearparts:
            h=layer(h)
        return h#self.sigmoid(h)#h#F.softmax(h,dim=-1)


    def get_active_weight_item_embedding(self,item_index,node_index):
        if self.item_node_share_embedding:
            batch_item_embedding=self.node_embedding(item_index)
        else:
            batch_item_embedding=self.item_embedding(item_index)

        return batch_item_embedding*self.active_unit(batch_item_embedding, self.node_embedding(node_index))

    def get_log_prob(self,labels,effective_item_index,weight_sum):
        pre_linear_part_inputs = torch.zeros((effective_item_index.numel(), self.embed_dim),device=weight_sum.device)

        pre_linear_part_inputs[effective_item_index.view(-1)]=weight_sum

        weight_batch_sum=torch.matmul(self.window_matrix,pre_linear_part_inputs.
                                      view(len(labels),self.feature_num,self.embed_dim)).view(-1,self.window_num*self.embed_dim)
        linear_part_input=torch.cat((weight_batch_sum, self.node_embedding(labels.view(-1))), -1)
        return self.LinearPart(linear_part_input)

    def preference(self,batch_item,batch_labels):#len(batch_item_index)==len(batch_labels). two dimension matrix
        pre_linear_part_inputs = torch.zeros((len(batch_item) * self.feature_num, self.embed_dim),
                                             device=batch_item.device)
        effective_item_index=batch_item>=0#negative index means the absence of user behaviour
        effective_items=batch_item[effective_item_index]
        effective_labels=batch_labels.expand(batch_item.shape)[effective_item_index]
        if self.item_node_share_embedding:
            batch_item_embedding=self.node_embedding(effective_items)
        else:
            batch_item_embedding=self.item_embedding(effective_items)

        pre_linear_part_inputs[effective_item_index.view(-1)]=batch_item_embedding*\
                                                              self.active_unit(batch_item_embedding,self.node_embedding(effective_labels))

        weight_batch_sum=torch.matmul(self.window_matrix,pre_linear_part_inputs.
                                      view(len(batch_labels),self.feature_num,self.embed_dim)).view(-1,self.window_num*self.embed_dim)
        linear_part_input=torch.cat((weight_batch_sum, self.node_embedding(batch_labels.view(-1))), -1) #[Bs, 10*emb_dim+emb_dim]
        return self.LinearPart(linear_part_input)







