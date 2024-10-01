

from __future__ import print_function


import numpy as np

#



class TreeBuilder:
    def __init__(self):


        pass

    #ids = np.array([item.item_id for item in items])
    #codes = np.array([item.code for item in items])
    #data = np.array([[] for i in range(len(ids))])， item embedding,i.e. the item embedding corresponding to leaf nodes
    #builder = tree_builder.TreeBuilder(self.tree_pb_file)
    def build(self, ids, codes, stat=None, kv_file=None):
        print('bulder {},{}'.format(len(ids),len(codes)))
        #code[1]=5 means thath item 1 is corresponding to leaf node 5
        #ids is the item id and codes is the corrsponding leaf node id
        #data is item embeddings
        # process id offset

        #id_offset is the maxinmu item id add 1
        # sort by codes
        argindex = np.argsort(codes)#
        codes = codes[argindex]
        ids = ids[argindex]
        #data = data[argindex]

        # Trick, make all leaf nodes to be in same level
        min_code = 0
        max_code = codes[-1]
        while max_code > 0:
            min_code = min_code * 2 + 1
            max_code = int((max_code - 1) / 2)

        for i in range(len(codes)):
            while codes[i] < min_code:
                codes[i] = codes[i] * 2 + 1


        #pstat = None# record the click frequency of all nodes, i.e. the sum of the corresponding leaf nodes
        if stat:
            pstat = dict()
            for id, code in zip(ids, codes):# item id and the corresponding leaf node id
                ancs = self._ancessors(code)# obatain all the ancestors of current code
                for anc in ancs:
                    if id in stat:
                      if anc not in pstat:
                        pstat[anc] = 0.0
                      pstat[anc] += stat[id]
        if kv_file is not None:
            with open(kv_file,'w') as f:
                for id,code in zip(ids,codes):
                    line=str(id)+"::"+str(code)+'\n'
                    f.write(line)

        return ids,codes

    def _ancessors(self, code):
        ancs = []
        while code > 0:
            code = int((code - 1) / 2)
            ancs.append(code)
        return ancs
