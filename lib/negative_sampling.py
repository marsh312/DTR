import torch
import math
import numpy as np
import time
import torch.nn.functional as F

def all_negative_sampling(batch_size,tree,all_leaf_codes,device):#all_leaf_codes is tensor with 1 column
    #m, d = batch_x.shape
    m = batch_size
    sample_labels = torch.full((batch_size, sum(tree.layer_node_num_list[1:])), -1, device=device, dtype=torch.int64)
    codes = all_leaf_codes
    for p_layer in range(tree.max_layer_id - 1, -1, -1):
        start_col = sum(tree.layer_node_num_list[1:p_layer+1])
        """positive nodes"""
        sample_labels[:, start_col:start_col+1] = tree.node_code_node_id_array[codes]
        start_col+=1

        """negative nodes"""
        start_label = sum(tree.layer_node_num_list[0:p_layer + 1])
        end_label = sum(tree.layer_node_num_list[0:p_layer + 2])
        layer_node_num = tree.layer_node_num_list[p_layer + 1]
        all_labels = torch.arange(start_label, end_label, device=device).expand(m, layer_node_num)
        negative_index = all_labels != sample_labels[:, start_col - 1:start_col]  # non-positie node
        sample_labels[:,start_col:start_col+layer_node_num-1]=all_labels[negative_index].view(m, layer_node_num - 1)
        codes = (codes - 1) // 2
    #assert (codes==0).sum().item()==m,'not root node , start cvol {}'.format(start_col)
    return sample_labels


def softmax_sampling(batch_x,tree,network_model,all_leaf_codes,N,device,temperature=1.0,eps=1e-9,\
                     itme_node_share_embedding=True):
    m,d = batch_x.shape
    sample_labels = torch.full((m, tree.max_layer_id * (N + 1)), -1, device=device, dtype=torch.int64)
    log_q_matrix = torch.full(sample_labels.shape, 0, device=device, dtype=torch.float32)

    codes = torch.from_numpy(all_leaf_codes)
    for p_layer in range(tree.max_layer_id - 1, -1, -1):
        start_col=p_layer*(N+1)+1
        """positive nodes"""
        sample_labels[:, start_col-1] = tree.node_code_node_id_array[codes].to(device)
        #log_q_matrix[:,start_col-1]=0 default
        #codes = ((codes - 1) / 2).long()
        codes = (codes - 1) // 2

        """negative nodes"""
        start_label = sum(tree.layer_node_num_list[0:p_layer + 1])
        end_label = sum(tree.layer_node_num_list[0:p_layer + 2])
        layer_node_num=tree.layer_node_num_list[p_layer+1]
        all_labels= torch.arange(start_label, end_label, device=device).expand(m,layer_node_num)
        negative_index = all_labels != sample_labels[:,start_col - 1:start_col]#non-positie node
        negative_labels = all_labels[negative_index].view(m, layer_node_num-1)
        if layer_node_num <= N+1:
            sample_labels[:,start_col:start_col+layer_node_num-1]=negative_labels
            #log_q_matrix[:,:,start_col:+start_col+tree.layer_node_num_list[p_layer+1]-1]=0 default
        else:
            if itme_node_share_embedding:
                reall_item_ids=torch.full(batch_x.shape,-1,dtype=torch.int64)
                effective_index=batch_x>=0
                effective_items=batch_x[effective_index]
                reall_item_ids[effective_index]=tree.item_id_node_ancestor_id[effective_items, p_layer+1]
                with torch.no_grad():
                    out = network_model.preference(reall_item_ids.to(device).repeat(1,layer_node_num-1).view(-1,d),\
                                                     negative_labels.view(-1,1))[:,0].view(m,layer_node_num-1)
            else:
                with torch.no_grad():
                    out = network_model.preference(batch_x.to(device).repeat(1,layer_node_num-1).view(-1,d),\
                                                     negative_labels.view(-1,1))[:,0].view(m,layer_node_num-1)
            sample_probs=F.softmax(out,-1)
            selected_index=torch.multinomial(sample_probs,N,replacement=True)
            sample_labels[:,start_col:start_col+N]=negative_labels.gather(index=selected_index,dim=1)
            log_q_matrix[:,start_col:start_col+N]=torch.log(N*sample_probs.gather(index=selected_index,dim=1))
    return sample_labels,log_q_matrix


def uniform_sampling_multiclassifcation(batch_size,tree,all_leaf_codes,N,device):
    samples_labels = torch.full((batch_size, tree.max_layer_id*(N+1)),-1,device=device, dtype=torch.int64)
    log_q_matrix=torch.full(samples_labels.shape,0.0,device=device,dtype=torch.float32)
    codes=all_leaf_codes
    for layer in range(tree.max_layer_id-1,-1,-1):
        start_col=layer*(N+1) # [layer_1, ..., layer_1, ......, layer_max_layer_id, ..., layer_max_layer_id] in one row
        """positive nodes"""
        samples_labels[:,start_col:start_col+1]=tree.node_code_node_id_array[codes]
        #codes=((codes-1)/2).long()
        codes = (codes - 1) // 2

        """"negative samples"""
        layer_node_num=tree.layer_node_num_list[layer+1]
        start_label = sum(tree.layer_node_num_list[0:layer+1])
        end_label = sum(tree.layer_node_num_list[0:layer + 2])

        if layer_node_num <= N+1:
            all_labels=torch.arange(start_label, end_label, device=device).expand(batch_size, layer_node_num)
            samples_labels[:,start_col+1:start_col+layer_node_num]=\
                all_labels[all_labels != samples_labels[:, start_col:start_col+1] ].view(batch_size, layer_node_num - 1)
            #log_q_matrix[:,:,start_col:+start_col+tree.layer_node_num_list[p_layer+1
        else:
            negative_labels=torch.randint(start_label, end_label, device=device,size=(batch_size,N))
            effective_negative_labels_index=negative_labels != samples_labels[:, start_col:start_col+1]
            samples_labels[:,start_col+1:start_col+1+N][effective_negative_labels_index] = \
                                                        negative_labels[effective_negative_labels_index]
            log_q_matrix[:,start_col+1:start_col+1+N][effective_negative_labels_index]=torch.log(\
                            effective_negative_labels_index.sum(-1).view(batch_size,1)/(layer_node_num-1.0)).\
                                                                expand(batch_size,N)[effective_negative_labels_index]
    #log_q_matrix[:,:]=0
    return samples_labels,log_q_matrix

def top_down_sample(batch_user,tree,network_model,all_leaf_codes,N,device,eps=1.0e-9,gamma=0.0):
    #all_leaf_codes is a tensor with 1 column,batch_user has been put into device
    m, d = batch_user.shape
    double_N=2*N
    batch_user_id=torch.arange(m,device=device).view(-1,1)#.expand(m,double_N+2)

    sample_labels = torch.full((m, tree.max_layer_id*(N+1)), -1, device=device, dtype=torch.int64)
    if gamma>0.0:
        sibling_labels=torch.full(sample_labels.shape,-1,device=device,dtype=torch.int64)
    effective_index_matrix=torch.full(sample_labels.shape,False,device=device,dtype=torch.bool)
    log_q_matrix = torch.full(sample_labels.shape, 0.0, device=device, dtype=torch.float32)

    parent_code=torch.zeros((m,N),device=device,dtype=torch.int64)
    path_prob_matrix=torch.ones((m,N),device=device,dtype=torch.float32)
    child_matrix=torch.full((m,double_N),-1,device=device,dtype=torch.int64)
    child_preference=torch.full((m,double_N+2), -1.0e9,dtype=torch.float32, device=device)
    selected_indicator=torch.full((m,double_N),False,dtype=torch.bool,device=device)

    """positive nodes, don't contain root node"""
    codes = all_leaf_codes
    effective_index_matrix[:,::N+1]=True
    for layer in range(tree.max_layer_id, 0,-1):
        start_col=(layer-1)*(N+1)       # [layer_1, ..., layer_1, ......, layer_max_layer_id, ..., layer_max_layer_id] in one row
        sample_labels[:,start_col:start_col+1]=tree.node_code_node_id_array[codes]
        if gamma>0.0:
            sibling_labels[:,start_col:start_col+1]=tree.node_code_node_id_array[torch.where(codes%2==0,codes-1,codes+1)]
        codes = (codes - 1) // 2

    real_start_layer_id=1
    """"negative samples"""
    for layer in range(1,tree.max_layer_id+1):
        start_col,layer_node_num = (layer-1) * (N + 1) + 1,tree.layer_node_num_list[layer]
        if layer_node_num<=N+1:
            start_label = sum(tree.layer_node_num_list[0:layer])
            end_label =start_label+layer_node_num #sum(tree.layer_node_num_list[0:layer+1])
            all_labels=torch.arange(start_label, end_label, device=device).expand(m, layer_node_num)
            sample_labels[:,start_col:start_col+layer_node_num-1]=\
                all_labels[all_labels!=sample_labels[:,start_col-1:start_col]].view(m,layer_node_num-1)
            effective_index_matrix[:,start_col:start_col+layer_node_num-1]=True
            continue
        elif layer_node_num<=double_N+2:
            all_codes = torch.arange(2 ** layer - 1, 2 ** (layer + 1) - 1, device=device)
            all_labels = tree.node_code_node_id_array[all_codes]
            effective_index = all_labels >= 0
            all_labels = all_labels[effective_index].view(1,layer_node_num).repeat(m,1)

            #user_index=torch.arange(m,device=device).view(-1,1).repeat(1,layer_node_num).view(-1)
            user_index=batch_user_id.repeat(1,layer_node_num).view(-1)
            new_batch_users=batch_user[user_index]
            effective_item_index=new_batch_users>=0
            new_batch_users[effective_item_index]=\
                tree.item_id_node_ancestor_id[new_batch_users[effective_item_index],layer]

            child_preference[:,:]=-1.0e9
            with torch.no_grad():
                child_preference[:,-layer_node_num:]=\
                    network_model.preference(new_batch_users,all_labels.view(-1,1)).view(m,layer_node_num)
            sample_prob=F.softmax(child_preference[:,-layer_node_num:],dim=-1)
            sampled_index = torch.multinomial(sample_prob, N, replacement=True)
            parent_code[:,:] = all_codes[effective_index].expand(m,layer_node_num).gather(index=sampled_index, dim=-1)
            if gamma>0.0:
                selected_sibling_labels = \
                    tree.node_code_node_id_array[torch.where(parent_code % 2 == 0, parent_code - 1, parent_code + 1)]
            path_prob_matrix[:, :] = sample_prob.gather(index=sampled_index, dim=-1)
            selected_labels = all_labels.gather(index=sampled_index, dim=-1)
            real_start_layer_id = layer
        else:
            double_p_code = 2 * parent_code
            child_matrix[:, 0::2] = double_p_code + 1  # left child, dobole_p_code ,max size is N+1
            child_matrix[:, 1::2] = double_p_code + 2  # right child
            all_labels = tree.node_code_node_id_array[child_matrix]
            effective_index = all_labels >= 0
            effective_labels = all_labels[effective_index]
            child_preference[:,:] = -1.0e9

            new_batch_users = batch_user[batch_user_id.expand(m,double_N)[effective_index]]
            effective_item_index=new_batch_users>=0
            new_batch_users[effective_item_index] =\
                tree.item_id_node_ancestor_id[new_batch_users[effective_item_index], layer]
            with torch.no_grad():
                child_preference[:,-double_N:][effective_index]=\
                    network_model.preference(new_batch_users,effective_labels.view(-1,1))[:,0]

            child_prob=F.softmax(child_preference.view(-1,2),-1).view(child_preference.shape)[:,-double_N:]

            selected_indicator[:,0::2]=torch.rand((m,N),device=device)<child_prob[:,0::2]
            selected_indicator[:, 1::2]=~selected_indicator[:,0::2]

            parent_code[:,:]=child_matrix[selected_indicator].view(m,N)
            path_prob_matrix *= child_prob[selected_indicator].view(m,N)
            selected_labels=all_labels[selected_indicator].view(m,N)#tree.node_code_node_id_array[parent_code].to(device)
            if gamma>0.0:
                selected_sibling_labels=all_labels[~selected_indicator].view(m,N)

        negative_index=selected_labels!=sample_labels[:,start_col-1:start_col]
        effective_index_matrix[:,start_col:start_col+N]=negative_index
        sample_labels[:,start_col:start_col+N]=selected_labels
        #log_q_matrix[:, start_col:start_col + N] = torch.log(negative_index.sum(-1).view(-1, 1)*path_prob_matrix+eps)
        log_q_matrix[:, start_col:start_col + N] = torch.log(negative_index.sum(-1)).view(-1,1)+torch.log(path_prob_matrix+eps)
        if gamma>0.0:
            sibling_labels[:,start_col:start_col+N]=selected_sibling_labels
    if gamma>0.0:
        return sample_labels,sibling_labels,effective_index_matrix,log_q_matrix,real_start_layer_id
    else:
        return sample_labels,effective_index_matrix,log_q_matrix,real_start_layer_id

def verify_sample_process(tree,all_leaf_codes,sample_labels,N):
    print('samples labels shape [{},{}]'.format(*sample_labels.shape))
    for i,code in enumerate(all_leaf_codes):
        all_labels=sample_labels[i]
        for p_layer in range(tree.max_layer_id-1,-1,-1):
            start_ind=p_layer*(N+1)
            positive_label=all_labels[start_ind]
            assert positive_label==tree.node_code_node_id[code],'wrong positive label'
            negative_labels=all_labels[start_ind+1:start_ind+1+N]
            start_label = sum(tree.layer_node_num_list[0:p_layer + 1])
            end_label = sum(tree.layer_node_num_list[0:p_layer + 2])
            for negative_label in negative_labels:
                if negative_label>=0:
                    assert negative_label>=start_label and negative_label<end_label,\
                        'wrong negative label,start_labe {},end_label {},current_label {}'.format(start_label,end_label,negative_label)
            code=(code-1)//2
        assert code==0,'wrong code'



def stochastic_beamsearch(batch_x,tree,network_model,all_leaf_codes,N,device,temperature=1.0,eps=1e-9):
    m,d=batch_x.shape
    extra_negative_node_nums = [min([tree.layer_node_num_list[layer], N]) for layer in range(1, tree.max_layer_id+1)]
    #negative_node_num=sum(extra_negative_node_nums)

    samples_labels = torch.full((m, tree.max_layer_id*(N+1)),-1,device=device, dtype=torch.int64)
    log_q_matrix=torch.full(samples_labels.shape,0,device=device,dtype=torch.float32)

    """positive nodes"""
    codes=torch.from_numpy(all_leaf_codes)
    for layer in range(tree.max_layer_id-1,-1,-1):
        samples_labels[:,layer*(N+1)]=tree.node_code_node_id_array[codes].to(device)
        #log_q_matrix[:,layer*(N+1)]=0
        codes=((codes-1)/2).long()

    S_pi=torch.tensor([1,2],dtype=torch.int64,device=device).expand(m,-1)
    G_S_pi=torch.full((m,2*N), -1e9,dtype=torch.float32, device=device)
    phi_S_pi=torch.full(G_S_pi.shape, -1e9,dtype=torch.float32, device=device)
    phi_S=torch.full(G_S_pi.shape,0,device=device,dtype=torch.float32)
    G_S=torch.full(G_S_pi.shape,0,device=device,dtype=torch.float32)
    out_matrix=torch.full(G_S_pi.shape,0,device=device,dtype=torch.float32)

    user_index=torch.arange(m).view(-1, 1)
    all_labels = tree.node_code_node_id_array[S_pi].to(device)

    for layer,sample_num in zip(range(1,tree.max_layer_id+1),extra_negative_node_nums):
        start_col=(layer-1)*(N+1)+1
        col_num = S_pi.shape[1]
        effective_index = all_labels >= 0
        with torch.no_grad():##return out
            out=network_model.preference(batch_x[user_index.expand(m,col_num)[effective_index]].to(device),\
                                                                        all_labels[effective_index].view(-1, 1))
        phi_S_pi[:,-col_num:][effective_index]= F.log_softmax(out,-1)[:,0]+ phi_S[:,-col_num:][effective_index]
        out_matrix[:,-col_num:][effective_index]=out[:,0]
        G_S_pi[:,-col_num:]= (phi_S_pi[:,-col_num:]-temperature*torch.log(-torch.log(torch.\
                                                                        rand((m,col_num),device=device)+eps)+eps))
        G_S_pi[:,-col_num:][~effective_index]=-1e9
        Z=torch.where(G_S_pi[:,-col_num:-int(col_num/2)]<G_S_pi[:,-int(col_num/2):]
                                    ,G_S_pi[:,-int(col_num/2):],G_S_pi[:,-col_num:-int(col_num/2)]).repeat(1,2)
        #hat_G_S_pi=-torch.log(torch.exp(-G_S[:,-col_num:])-torch.exp(-Z)+torch.exp(-G_S_pi[:,-col_num:]))
        v=G_S[:,-col_num:]-G_S_pi[:,-col_num:]+torch.log(1.0-torch.exp(G_S_pi[:,-col_num:]-Z))
        hat_G_S_pi=G_S[:,-col_num:]-torch.where(v>0,v,torch.zeros(v.shape,device=device))\
                   -torch.log(1.0+torch.exp(-v.abs()))
        hat_G_S_pi[~effective_index]=-1e9

        sorted_index =hat_G_S_pi.argsort(dim=-1)[:,-sample_num:]
        selected_codes = S_pi.gather(index=sorted_index, dim=1)
        selected_labels=all_labels.gather(index=sorted_index, dim=1)
        effective_negative_index=selected_labels!=samples_labels[:,start_col-1:start_col]

        samples_labels[:,start_col:start_col+sample_num][effective_negative_index]=\
                                                                selected_labels[effective_negative_index]
        #log_q_matrix[:,start_col:start_col+sample_num]=\
        #                    torch.log(1.0*effective_negative_index.sum(-1).view(-1,1))+\
        #                                    out_matrix[:,-col_num:].gather(index=sorted_index, dim=1)

        if layer < tree.max_layer_id:
            S_pi=torch.cat((2 * selected_codes + 1,2 * selected_codes + 2),dim=-1)
            all_labels = tree.node_code_node_id_array[S_pi].to(device)
            phi_S[:,-2*sample_num:]=phi_S_pi[:,-col_num:].gather(index=sorted_index, dim=1).repeat(1,2)
            G_S[:,-2*sample_num:]=hat_G_S_pi.gather(index=sorted_index, dim=1).repeat(1,2)
    return samples_labels,log_q_matrix






