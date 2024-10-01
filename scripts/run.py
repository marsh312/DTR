import sys
sys.path.append('..')
sys.path.append('.')
sys.path.append('../lib')
import torch
import numpy as np
import lib
import argparse
import os
import wandb
from pprint import pprint
import gc
import numpy as np
import time

# parametres
parser = argparse.ArgumentParser()

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--tree_num', type=int, default=12)
parser.add_argument('--sampling_method', type=str, default='top_down', choices=['top_down', 'uniform_multiclass', 'all_negative_sampling', 'softmax'])
parser.add_argument('--training_batch_size', type=int, default=100)
parser.add_argument('--validation_batch_size', type=int, default=50)
parser.add_argument('--max_calculate_calibrated_weight_num', type=int, default=250000)
parser.add_argument('--max_calculate_num_for_tree_learner', type=int, default=100000)
parser.add_argument('--sample_num_for_calibrated_weight', type=int, default=1000)
parser.add_argument('--layer_bs', type=int, default=128)
parser.add_argument('--calibrate_weight_mode', type=str, default='sasrec', choices=['sasrec', 'sampled', 'none', 'full'])
parser.add_argument('--train_leaf_layer_epoch', type=int, default=0)
parser.add_argument('--wd', type=float, default=1e-2)
parser.add_argument('--batch_gap_to_rebuild_tree', type=int, default=80000)
parser.add_argument('--sample_num', type=int, default=70)
parser.add_argument('--emb_dim', type=int, default=24)
parser.add_argument('--scheduler_mode', type=str, default='exponential', choices=['polynomial', 'exponential', 'cosine'])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--min_lr', type=float, default=1e-6)
parser.add_argument('--dropout_p', type=float, default=0.1)
parser.add_argument('--update_gap', type=int, default=7)
parser.add_argument('--warmup_steps', type=int, default=5000)
parser.add_argument('--calibrated_mode', type=str, default='unnormalized', choices=['normalized', 'unnormalized', 'tailored'])
parser.add_argument('--project_name', type=str, required=True)
parser.add_argument('--dataset', type=str, help='dataset', choices=['mind',  'movie', 'tmall', 'amazon'], required=True)
parser.add_argument('--run_name', type=str, required=True)
parser.add_argument('--run_time', type=int, default=1)

args = parser.parse_args()


def presision(result_list,gt_list,top_k):
    count=0.0
    for r,g in zip(result_list,gt_list):
       count+=len(set(r).intersection(set(g)))
    return count/(top_k*len(result_list))

def recall(result_list,gt_list):
    t=0.0
    for r,g in zip(result_list,gt_list):
        t+=1.0*len(set(r).intersection(set(g)))/len(g)
    return t/len(result_list)

def f_measure(result_list,gt_list,top_k,eps=1.0e-9):
    f=0.0
    for r,g in zip(result_list,gt_list):
        recc=1.0*len(set(r).intersection(set(g)))/len(g)
        pres=1.0*len(set(r).intersection(set(g)))/top_k
        if recc+pres<eps:
            continue
        f+=(2*recc*pres)/(recc+pres)
    return f/len(result_list)

def novelty(result_list,s_u,top_k):
    count=0.0
    for r,g in zip(result_list,s_u):
        count+=len(set(r)-set(g))
    return count/(top_k*len(result_list))

def hit_ratio(result_list,gt_list):
    intersetct_set=[len(set(r)&set(g)) for r,g in zip(result_list,gt_list)]
    return 1.0*sum(intersetct_set)/sum([len(gts) for gts in gt_list])

def NDCG(result_list,gt_list):
    t=0.0
    for re,gt in zip(result_list,gt_list):
        setgt=set(gt)
        indicator=np.asfarray([1 if r in setgt else 0 for r in re])
        sorted_indicator = np.ones(min(len(setgt), len(re)))
        if 1 in indicator:
            t+=np.sum(indicator / np.log2(1.0*np.arange(2,len(indicator)+ 2)))/\
               np.sum(sorted_indicator/np.log2(1.0*np.arange(2,len(sorted_indicator)+ 2)))
    return t/len(gt_list)

def MAP(result_list,gt_list,topk):
    t=0.0
    for re,gt in zip(result_list,gt_list):
        setgt=set(gt)
        indicator=np.asfarray([1 if r in setgt else 0 for r in re])
        t+=np.mean([indicator[:i].sum(-1)/i for i in range(1,topk+1)],axis=-1)
    return t/len(gt_list)


dir_path = './final_multi_trees_log_{}'.format(args.dataset)
os.makedirs(dir_path, exist_ok=True)
def train():
    wandb.init(project=args.project_name, name=args.run_name, dir=dir_path, config=vars(args))
    # repalce the attribute in the args by the attribute in the config
    print('Config:')
    pprint(wandb.config)
    print('\n')

    print('Arguments passed:')
    pprint(vars(args))
    print('\n')

    device = 'cuda:{}'.format(args.gpu_id)
    has_processed_data=True
    topk=20

    train_sample_seg_cnt=10#the training data is located in the train_sample_seg_cnt datafiles
    parall=4
    seq_len=70 # se_len-1 is the number of behaviours in all the windows
    min_seq_len=15
    test_user_num=6000# the number of user in test file
    tree_learner_mode='jtm'
    gamma=0.0

    item_node_share_embedding=True
    raw_data_file='../../data/{}/{}.txt'.format(args.dataset,args.dataset)
    data_file_prefix='../../data/{}/processed_dataset/'.format(args.dataset)

    if not has_processed_data:
        if os.path.exists(data_file_prefix):
            pass
        else:
            os.makedirs(data_file_prefix)
    train_instances_file = data_file_prefix + "train_instances"
    test_instances_file = data_file_prefix + "test_instances"
    validation_instances_file = data_file_prefix + "validation_instances"
    kv_file = data_file_prefix + "kv.txt"  # save the key vavlue (i.e. item_id:leaf_code)
    result_prefix = "../../data/{}/DeFoRec_{}_Calibrated_{}_final_newest/".format(
        args.dataset, args.sampling_method, args.calibrate_weight_mode
    ) + "result_of_N_{}_share_embedding_{}_wd_{}_dropout_{}_train_leaf_{}_min_lr_{}_batch_gap_{}_{}_run_time_{}_{}/".format(
        args.sample_num,
        item_node_share_embedding,
        args.wd,
        args.dropout_p,
        args.train_leaf_layer_epoch,
        args.min_lr,
        args.batch_gap_to_rebuild_tree,
        args.calibrated_mode,
        args.run_time,
        time.strftime("%Y-%m-%d-%H-%M-%S")
    )

    featrue_groups=[20,20,10,10,2,2,2,1,1,1]
    assert sum(featrue_groups)==seq_len-1

    eps=0.000001
    if device!='cpu':
        torch.cuda.set_device(device)
    print(result_prefix)

    assert has_processed_data, 'please process data first'
    if not has_processed_data: 
        from lib.Generate_Data_and_Tree import _read,_gen_train_sample,_gen_test_sample,_init_tree,_gen_discriminator_samples
        behavior_dict, train_sample, test_sample,validation_sample = _read(raw_data_file,test_user_num)  # 20 is the test users
        stat = _gen_train_sample(train_sample, train_instances_file,test_sample=test_sample,validation_sample=validation_sample,
                                                        train_sample_seg_cnt=train_sample_seg_cnt,
                                                        parall=parall, seq_len=seq_len, min_seq_len=min_seq_len)
        _gen_test_sample(test_sample, test_instances_file, seq_len=seq_len,min_seq_len=min_seq_len)
        _gen_test_sample(validation_sample, validation_instances_file, seq_len=seq_len,min_seq_len=min_seq_len)
        ids, codes = _init_tree(train_sample, test_sample,validation_sample, stat, kv_file=kv_file)
        del behavior_dict
        del train_sample
        del test_sample
        del stat
        gc.collect()
    else:
        ids=[]
        codes=[]
        assert kv_file is not None
        with open(kv_file) as f:
            while True:
                line=f.readline()
                if line:
                    id_code=line.split('::')
                    ids.append(int(id_code[0]))
                    codes.append(int(id_code[1]))
                else:
                    break
        ids=np.array(ids,dtype=np.int32)
        codes=np.array(codes,dtype=np.int32)
    print('min item id is {}, max item id is {}'.format(ids.min(),ids.max()))
    print('min leaf node code is {}, max leaf node code is {}'.format(codes.min(), codes.max()))

    ids_list,codes_list=[],[]
    for _ in range(args.tree_num):
        ids_list.append(ids)
        codes_list.append(codes)
    item_num=len(ids_list[0])
    print('item number is {}'.format(item_num))

    from lib.generate_training_batches import Train_instance
    train_instances=Train_instance(parall=parall)
    training_batch_generator=train_instances.training_batches(train_instances_file,train_sample_seg_cnt,batchsize=args.training_batch_size)
    validation_batch_generator=train_instances.validation_batches(validation_instances_file,batchsize=args.validation_batch_size)
    test_instances=train_instances.read_test_instances_file(test_instances_file)
    training_instance_index_pair=train_instances.get_item_instance_pair_index(train_instances_file,train_sample_seg_cnt)

    sasrec_model = None
    load_sasrec_mode='weight' #'model'
    if args.calibrate_weight_mode == 'sasrec':
        if load_sasrec_mode == 'model':
            from sasrec import SASRec
            sasrec_model_path = "../sasrec/model/{}_model.pt".format(args.dataset)
            sasrec_model = torch.load(sasrec_model_path)
            sasrec_model.device = device
            sasrec_model = sasrec_model.to(device)
            print(sasrec_model)
        elif load_sasrec_mode == 'weight':
            import json
            from lib.sasrec import SASRec
            sasrec_weight_path = "../sasrec/weights/{}_weight.pt".format(args.dataset)
            sasrec_config_path = "../sasrec/weights/{}_config.json".format(args.dataset)
            sasrec_config = json.load(open(sasrec_config_path))
            sasrec_model = SASRec(n_item=sasrec_config["item_num"], max_len=seq_len-1, device=device, args=argparse.Namespace(**sasrec_config))
            sasrec_model.load_state_dict(torch.load(sasrec_weight_path))
            sasrec_model = sasrec_model.to(device)
            print(sasrec_model)

    from lib.trainer import TrainModel
    lib.trainer.train_time = 0
    lib.trainer.calculate_weight_time = 0
    optimizer = lambda params: torch.optim.Adam(params, lr=args.lr, amsgrad=True, weight_decay=args.wd)
    train_model=TrainModel(ids,codes,
                        embed_dim=args.emb_dim,
                        feature_groups=featrue_groups,
                        all_training_instance=train_instances.training_data,
                        item_user_pair_dict=training_instance_index_pair,
                        parall=parall, 
                        optimizer=optimizer,
                        N=args.sample_num,
                        sampling_method=args.sampling_method,
                        tree_learner_mode=tree_learner_mode,
                        item_node_share_embedding=item_node_share_embedding,
                        device=device,
                        gamma=gamma,
                        steps_calculate_weight=0,
                        calibrate_weight_mode=args.calibrate_weight_mode,
                        max_calculate_calibrated_weight_num=args.max_calculate_calibrated_weight_num,
                        max_calculate_num_for_tree_learner=args.max_calculate_num_for_tree_learner,
                        sample_num_for_calibrated_weight=args.sample_num_for_calibrated_weight,
                        sasrec_model=sasrec_model,
                        scheduler_mode=args.scheduler_mode,
                        dropout_p=args.dropout_p,
                        calibrated_mode=args.calibrated_mode
                        )

    if os.path.exists(result_prefix):
        pass
    else:
        os.makedirs(result_prefix)
    print(train_model.network_model)
    print('tree number is {}'.format(args.tree_num))

    trainset = torch.utils.data.TensorDataset(train_instances.training_data, train_instances.training_labels)
    for epoch in range(args.train_leaf_layer_epoch):
        print('training leaf layer, {}-th epoch ...'.format(epoch))
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.layer_bs, shuffle=True)
        for batch_x, batch_y in train_dataloader:
            loss = train_model.train_specified_layer(batch_x, batch_y, train_model.tree.max_layer_id, 5*args.sample_num, args.min_lr)
            wandb.log({'Train_Leaf_Layer/loss': loss.item(),
                'Train_Leaf_Layer/lr': train_model.optimizer.param_groups[0]['lr']})  
        train_model.batch_num = 0
    if device!='cpu':
        torch.cuda.empty_cache()

    time_start_updating_network = time.time()
    for (batch_x,batch_y) in training_batch_generator:
        loss=train_model.update_network_model(batch_x, batch_y, lr_base=args.lr, min_lr=args.min_lr, warmup_steps=args.warmup_steps)
        wandb.log({'Train/loss': loss.item(),
                'Train/lr': train_model.optimizer.param_groups[0]['lr']})  

        if train_model.batch_num% 5000 ==0: #5000
            train_model.network_model.eval()
            bs=args.validation_batch_size
            bs_count=(len(test_instances)-1)//bs+1
            all_result=np.zeros((len(test_instances),topk),dtype=np.int32)

            for i in range(bs_count):
                bs_user=test_instances[i*bs:(i+1)*bs]
                all_result[i*bs:(i+1)*bs]=train_model.predict(bs_user,150,topk,forest=False)
            resutl_history=all_result.tolist()
            test_precision = presision(resutl_history,train_instances.test_labels,topk)
            test_recall = recall(resutl_history,train_instances.test_labels)
            test_f_measure = f_measure(resutl_history,train_instances.test_labels,topk)
            wandb.log(
                {
                    "Test/precision": test_precision,
                    "Test/recall": test_recall,
                    "Test/f_measure": test_f_measure,
                }
            )
            train_model.network_model.train()

        if train_model.batch_num% 100 == 0: #100
            train_model.network_model.eval()
            validation_batch,validation_index=validation_batch_generator.__next__()
            gt_history=[train_instances.validation_labels[i.item()] for i in validation_index]
            resutl_history=train_model.predict(validation_batch,150,topk,forest=False).tolist()
            dev_precision = presision(resutl_history,gt_history,topk)
            dev_recall = recall(resutl_history,gt_history)
            dev_f_measure = f_measure(resutl_history,gt_history,topk)
            wandb.log(
                {
                    "Validation/precision": dev_precision,
                    "Validation/recall": dev_recall,
                    "Validation/f_measure": dev_f_measure,
                }
            )    
            train_model.network_model.train()

        if train_model.batch_num>=args.batch_gap_to_rebuild_tree and len(train_model.tree_list)<=args.tree_num:#batch_gap_to_rebuild_tree 
            model_path=result_prefix+"{}_tree_{}_network_model.pth".format(tree_learner_mode,len(train_model.tree_list))
            if os.path.exists(model_path):
                os.remove(model_path)
            torch.save(train_model.network_model,model_path)
            tree_path=result_prefix+"{}_tree_{}_tree.txt".format(tree_learner_mode,len(train_model.tree_list))
            with open(tree_path,'w') as f:
                for item_id,code in train_model.tree.item_id_leaf_code.items():
                    line=str(item_id)+"::"+str(code)+'\n'
                    f.write(line)

            print('\t ********* training {}-th network cost {}s'.format(len(train_model.tree_list),time.time()-time_start_updating_network), flush=True)

            resutl_history=[]
            test_topk=60
            test_beam_size=150
            train_model.network_model.eval()
            test_bs=50
            bs_count=(len(test_instances)-1)//test_bs+1
            all_result=np.zeros((len(test_instances),test_topk),dtype=np.int32)

            st=time.time()
            for i in range(bs_count):
                bs_user=test_instances[i*bs:(i+1)*bs]
                all_result[i*bs:(i+1)*bs]=train_model.predict(bs_user,test_beam_size,test_topk,forest=False)
            print('testing cost time {:.4f}s'.format(time.time()-st))

            train_model.network_model.train()

            print('all result shape:', all_result.shape)

            all_result=all_result[:,::-1]

            columns = []
            datas = [[]]
            for k in [20, 40, 60]:
                pre=presision(all_result[:,:k].tolist(),train_instances.test_labels,k)
                rec=recall(all_result[:,:k].tolist(),train_instances.test_labels)
                f_mea=f_measure(all_result[:,:k].tolist(),train_instances.test_labels,k)
                print('top-k={}:   {:.4f},{:.4f},{:.4f}'.format(k, pre, rec, f_mea))
                columns.extend(["precision@{}".format(k), "recall@{}".format(k), "f_measure@{}".format(k)])
                datas[0].extend([pre, rec, f_mea])
            table = wandb.Table(columns=columns, data=datas)
            wandb.log({"Tree_{}/metrics/".format(len(train_model.tree_list), k): table})

            if len(train_model.tree_list)<args.tree_num:
                time_start_updating_tree = time.time()
                train_model.update_tree(d=args.update_gap)
                time_end_updating_tree = time.time()
                print('\t ********* updating {}-th tree cost {}s'.format(len(train_model.tree_list),time_end_updating_tree-time_start_updating_tree), flush=True)
                for epoch in range(args.train_leaf_layer_epoch):
                    print('training leaf layer, {}-th epoch ...'.format(epoch))
                    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.layer_bs, shuffle=True)
                    for batch_x, batch_y in train_dataloader:
                        loss = train_model.train_specified_layer(batch_x, batch_y, train_model.tree.max_layer_id, 5*args.sample_num, args.min_lr)
                        wandb.log({'Train_Leaf_Layer/loss': loss.item(),
                            'Train_Leaf_Layer/lr': train_model.optimizer.param_groups[0]['lr']})  
                    train_model.batch_num = 0
                if device!='cpu':
                    torch.cuda.empty_cache()
                time_start_updating_network = time.time()

        if train_model.batch_num>=args.batch_gap_to_rebuild_tree and len(train_model.tree_list)>=args.tree_num:
            print('OVER Training!!!')
            break

    print('*'*20)
    print('total train network time: {} s'.format(lib.trainer.train_time))
    print('total calculate weight time: {} s'.format(lib.trainer.calculate_weight_time))
    print('percentage : {}%'.format(lib.trainer.calculate_weight_time/lib.trainer.train_time*100))

train()