from __future__ import print_function
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from tqdm import tqdm

class Train_instance:
    def __init__(self,parall=4):
        self.parall=parall
        self.training_data=None
        self.training_labels=None

    def read_one_instrance_file(self,traing_instaces_path,sec_count_id):
        file_path=traing_instaces_path+'_{}'.format(sec_count_id)
        historys,labels=[],[]
        with open(file_path) as f:
            for line in f:
                user,history,label=line.split('|')
                historys.append([int(st) for st in history.split(',')])
                labels.append(int(label))
                #line = f.readline()
        one_file_data=torch.LongTensor(historys)
        #one_file_data[one_file_data<0]=item_num
        return one_file_data,torch.LongTensor(labels)

    def read_all_instances_files(self,traing_instaces_path,seg_couts):
        his_maxtix=None
        labels=None
        for i in range(seg_couts):
            part_his,part_labels=self.read_one_instrance_file(traing_instaces_path,i)
            if his_maxtix is not None:
                his_maxtix=torch.cat((his_maxtix,part_his),0)
                labels=torch.cat((labels,part_labels),0)
            else:
                his_maxtix=part_his
                labels=part_labels
        assert len(his_maxtix)==len(labels)
        return his_maxtix,labels


    def read_test_instances_file(self,test_instance_path):
        historys,labels=[],[]
        with open(test_instance_path) as f:
            line=f.readline()
            while line:
                user,history,label=line.split('|')
                historys.append([int(st) for st in history.split(',')])
                labels.append([int(st) for st in label.split(',')])
                line = f.readline()
        test_data=torch.LongTensor(historys)
        self.test_labels=labels
        return test_data

    def read_validation_instacne_file(self,validation_instance_file):
        historys,labels=[],[]
        with open(validation_instance_file) as f:
            line=f.readline()
            while line:
                user,history,label=line.split('|')
                historys.append([int(st) for st in history.split(',')])
                labels.append([int(st) for st in label.split(',')])
                line = f.readline()
        self.validation_labels=labels
        validation_data=torch.LongTensor(historys)
        return validation_data



    def training_batches(self,traing_instaces_path,seg_couts,batchsize=300):
        if (self.training_data is None) or (self.training_labels is None):
            history_matrix,positive_labels=self.read_all_instances_files(traing_instaces_path,seg_couts)
            self.training_data=history_matrix
            self.training_labels=positive_labels
        tensor_train_instances=TensorDataset(self.training_data,self.training_labels)
        train_loader=DataLoader(dataset=tensor_train_instances,batch_size=batchsize,shuffle=True,num_workers=4)
        while True:
            yield from train_loader

    '''
    def test_batches(self,validation_instance_file,batchsize=100):
        test_instances_matrix=self.read_test_instacne_file(validation_instance_file)
        mindex=torch.tensor(np.arange(len(test_instances_matrix)))
        tensor_test_instances=TensorDataset(test_instances_matrix,mindex)
        test_loader=DataLoader(dataset=tensor_test_instances,batch_size=batchsize,shuffle=True,num_workers=4)
        while True:
            yield from test_loader
    '''
    def validation_batches(self,validation_instances_path,batchsize=100):
        validation_instances_matrix=self.read_validation_instacne_file(validation_instances_path)
        mindex=torch.tensor(np.arange(len(validation_instances_matrix)))
        tensor_validation_instances=TensorDataset(validation_instances_matrix,mindex)
        validation_loader=DataLoader(dataset=tensor_validation_instances,batch_size=batchsize,shuffle=True,num_workers=4)
        while True:
            yield from validation_loader

    def get_item_instance_pair_index(self,traing_instaces_path,seg_couts, with_processed_matrixs=True):
        item_instance_dict=dict()
        if (self.training_data is None) or (self.training_labels is None):
            if with_processed_matrixs:
                training_matrix_path = os.path.join(os.path.dirname(traing_instaces_path), 'training_matrixs.pt')
                if os.path.exists(training_matrix_path):
                    history_matrix, positive_labels = torch.load(training_matrix_path)
                else:
                    history_matrix, positive_labels = self.read_all_instances_files(traing_instaces_path, seg_couts)
                    torch.save((history_matrix, positive_labels), training_matrix_path)
            else:
                history_matrix, positive_labels = self.read_all_instances_files(traing_instaces_path, seg_couts)
            self.training_data = history_matrix
            self.training_labels = positive_labels
            assert len(history_matrix)==len(positive_labels)
        for index_count, label in tqdm(
            enumerate(self.training_labels),
            desc="get item user pair dict",
            total=len(self.training_labels),
        ):
            if label.item() not in item_instance_dict:
                item_instance_dict[label.item()]=[]
            item_instance_dict[label.item()].append(index_count)
        print('item num is {}'.format(len(item_instance_dict)))

        return item_instance_dict


    def read_discriminator_batches(self,discriminator_instance_path,seg_couts):
        his_list,label_list=[],[]
        for sec_count_id in range(seg_couts):
            file_path=discriminator_instance_path+'_{}'.format(sec_count_id)
            historys= []
            with open(file_path) as f:
                for line in f:
                    user, history, label = line.split('|')
                    historys.append([int(st) for st in history.split(',')])
                    label_list.append(set([int(st) for st in label.split(',')]))
                    # line = f.readline()
            one_file_data = torch.LongTensor(historys)
            #one_file_data[one_file_data < 0] = item_num

            his_list.append(one_file_data)
        history_matirx=torch.cat(his_list,dim=0)
        print(len(history_matirx),len(label_list))
        assert len(history_matirx)==len(label_list)
        return history_matirx,label_list

    def discriminator_batches(self,discriminator_instance_path,seg_couts,batchsize=300):
        history_matirx, label_list=self.read_discriminator_batches(discriminator_instance_path,seg_couts)
        self.disriminator_label_list=label_list
        tensor_train_instances=TensorDataset(history_matirx,torch.arange(len(history_matirx),dtype=torch.int64))
        train_loader=DataLoader(dataset=tensor_train_instances,batch_size=batchsize,shuffle=True,num_workers=4)
        while True:
            yield from train_loader





if __name__=="__main__":
    train_instances=Train_instance()
    batch_generator=train_instances.training_batches('./data/mock/train_instances',10)
    test_generator=train_instances.test_batches('./data/mock/test_instances',batchsize=5)
    hh,ss=test_generator.__next__()
    print(hh)
    print(ss)
    #for h in enuhh):
    #    print(hh[i:i+1,:])








