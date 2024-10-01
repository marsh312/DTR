from __future__ import print_function


import time
import collections
#import argparse

#import multiprocessing as mp
import numpy as np
#import sklearn
#from sklearn.cluster import KMeans
import faiss
from joblib import Parallel,delayed
import queue as Que
from .tree_builder import TreeBuilder


class Cluster_Faiss:
    def __init__(self,filename=None,ofilename=None,id_offset=None,parall=16,
                 kv_file=None,stat_file=None,prev_result=None):
        self.filename = filename
        self.ofilename = ofilename
        self.ids = None
        self.data = None
        self.parall = parall
        self.queue = None
        self.timeout = 5
        self.id_offset = id_offset
        self.codes = None
        self.kv_file = kv_file
        self.stat_file = stat_file
        self.stat = None
        self.prev_result = prev_result
        #os.environ['OMP_NUM_THREADS']='10'

    def _read(self):
        t1 = time.time()
        ids = list()#item id
        data = list()
        with open(self.filename) as f:
            for line in f:
                arr = line.split(',')
                if not arr:
                    break
                ids.append(int(arr[0]))
                vector = list()
                for i in range(1, len(arr)):
                    vector.append(float(arr[i]))
                data.append(vector)
        self.ids = np.array(ids)
        self.data = np.array(data)
        t2 = time.time()

        if self.stat_file:
            self.stat = dict()
            with open(self.stat_file, "rb") as f:
                for line in f:
                    arr = line.split(",")
                    if len(arr) != 2:
                        break
                    self.stat[int(arr[0])] = float(arr[1])

        print("Read data done, {} records read, elapsed: {}".format(
            len(ids), t2 - t1))


    def train(self,ids=None,data=None):
        if data is not None:
            self.data = data
            self.ids = ids
        else:
            self._read()
        queue=Que.Queue()
        mini_batch_queue=Que.Queue()
        print('get into cluster training process')
        while True:
            try:
                queue.put((0, np.array(range(len(self.ids)))),timeout=self.timeout)
                break
            except:
                #print('put item into queue error!!')
                pass
        assert(queue.qsize()==1)
        print('start to cluster')
        while queue.qsize()>0:
            pcode,index=queue.get()
            if len(index)<=1024:
                #self._minbatch(pcode, index, code)
                while True:
                    try:
                        mini_batch_queue.put((pcode,index),timeout=self.timeout)
                        break
                    except:
                        pass
            else:
                tstart = time.time()
                left_index, right_index = self._cluster(index)
                print("Train iteration done, pcode:{}, "
                          "data size: {}, elapsed time: {}"
                          .format(pcode, len(index), time.time() - tstart))
                self.timeout = int(
                    0.4 * self.timeout + 0.6 * (time.time() - tstart))
                if self.timeout < 5:
                    self.timeout = 5

                if len(left_index) > 1:
                    while True:
                        try:
                            queue.put((2 * pcode + 1, left_index),
                                      timeout=self.timeout)  # cluster is from root to leaf node
                            break
                        except:
                            pass
                if len(right_index) > 1:
                    while True:
                        try:
                            queue.put((2 * pcode + 2, right_index), timeout=self.timeout)
                            break
                        except:
                            pass
        print('start to process mini-batch parallel.....................................')
        tstart=time.time()
        qcodes,indice=[],[]
        while mini_batch_queue.qsize()>0:
            pcode,index=mini_batch_queue.get()
            qcodes.append(pcode)
            indice.append(index)
        make_job = delayed(self._minbatch)
        re = Parallel(n_jobs=self.parall)(make_job(pcode,index) for pcode,index in zip(qcodes, indice))
        id_code_list=[]
        for r in re:
            id_code_list.extend(r)
        ids = np.array([id for (id, _) in id_code_list])
        codes=np.array([code for (_,code) in id_code_list])
        print('cluster all the nodes, cost {} s'.format(time.time()-tstart))
        #code=np.sort(np.array(code,dtype=np.int32))
        print('cluster all the nodes done,start to rebalance the tree')
        assert (codes<=0).sum()<=0
        assert queue.qsize()==0
        assert len(ids)==len(data)
        builder = TreeBuilder()
        return builder.build(ids, codes, stat=self.stat, kv_file=self.kv_file)


    def _minbatch(self, pcode, index, code=None):
            #pocde is paretn code,index is the assinged items' id, code is np.zeros(len(ids)), used tor recoder whetert the item processed as a leaf
            dq = collections.deque()
            dq.append((pcode, index))
            batch_size = len(index)
            id_code_list=[]
            tstart = time.time()
            while dq:
                pcode, index = dq.popleft()# pop the tuple which is added into the deque early

                if len(index) == 2:
                    #code[index[0]] = 2 * pcode + 1#left child
                    #code[index[1]] = 2 * pcode + 2#right child
                    id_code_list.append((index[0], 2 * pcode + 1))#(in,code) pair
                    id_code_list.append((index[1], 2 * pcode + 2))
                    continue
                left_index, right_index = self._cluster(index)# divide the index into two nodes
                if len(left_index) > 1:
                    dq.append((2 * pcode + 1, left_index))
                elif len(left_index) == 1:
                    #code[left_index] = 2 * pcode + 1
                    id_code_list.append((left_index[0], 2 * pcode + 1))

                if len(right_index) > 1:
                    dq.append((2 * pcode + 2, right_index))
                elif len(right_index) == 1:
                    #code[right_index] = 2 * pcode + 2
                    id_code_list.append((right_index[0], 2 * pcode + 2))
            print("Minbatch, batch size: {}, elapsed: {}".format(batch_size, time.time() - tstart))
            return id_code_list


    def _cluster(self, index):
        data = self.data[index]
        kmeans=faiss.Kmeans(data.shape[1],2,niter=1024,verbose=True)
        kmeans.train(data)
        km_distances,labels=kmeans.index.search(data,1)
        l_i = np.where(labels.reshape(-1) == 0)[0]# l_i is the index of the first cluster of data
        r_i = np.where(labels.reshape(-1) == 1)[0]# r_i is the index of the second cluster of data
        left_index = index[l_i]
        right_index = index[r_i]
        if len(right_index) - len(left_index) > 1:
            distances =km_distances[r_i] #kmeans.transform(data[r_i])
            left_index, right_index = self._rebalance(
                left_index, right_index, distances[:, 0])
        elif len(left_index) - len(right_index) > 1:
            distances = km_distances[l_i]#kmeans.transform(data[l_i])
            left_index, right_index = self._rebalance(
                right_index, left_index, distances[:, 0])
        return left_index, right_index
    def _rebalance(self, lindex, rindex, distances):
        sorted_index = rindex[np.argsort(distances)]
        idx = np.concatenate((lindex, sorted_index))
        mid = int(len(idx) / 2)
        return idx[mid:], idx[:mid]


if __name__ == "__main__":
    '''
    mycluster=Cluster(parall=16)
    #data=np.loadtxt('./data/test_item_embedding.txt',delimiter=',',dtype=np.float32)[:,1:]
    data=np.random.rand(7000,24)
    print(data.shape)
    ids=np.array(list(range(7000)))
    mycluster.train(ids,data)
    '''





