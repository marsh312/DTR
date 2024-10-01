from __future__ import print_function
import math
import numpy as np
import queue
import torch


class TreeNode(object):
    def __init__(self, id=0, code=0, isleaf=False):
        self.id = id#local id
        self.code = code
        self.isleaf = isleaf
        self.corresponding_item_id = None

    def __le__(self, other):
        if self.id == other.id:
            return self.code <= other.code
        return self.id <= other.id

    def __lt__(self, other):
        if self.id == other.id:
            return self.code < other.code
        return self.id < other.id


class Tree(object):
    def __init__(self,item_ids,leaf_codes,tree_id=0,device='cuda'):
        print('construct tree, leaf node num is {}'.format(len(item_ids)))
        self.tree_id = tree_id
        self.max_layer_id = 0
        self.node_num=0
        self.layer_node_num_list = []#node number at each layer
        self.item_id_leaf_code = None
        self.leaf_code_item_id = None
        self.node_code_node_id = dict()
        self.node_code_node_id_array=None
        self.item_id_node_ancestor_id=None
        self.node_id_layer_id=None
        self.node_sampled_indicator=None
        self.maximum_assigned_item_num=dict()#code->maximum assigned item num

        #initialize the pi
        self.item_id_leaf_code = {int(id): int(code) for id, code in zip(item_ids, leaf_codes)}
        self.leaf_code_item_id = {int(code): int(id) for id, code in zip(item_ids, leaf_codes)}



        node_list = []
        node_code_set = set()
        for code in leaf_codes:
            cur_code = int(code)
            while cur_code >= 0:
                if cur_code not in node_code_set:
                    node_code_set.add(cur_code)
                    node_list.append(TreeNode(code=cur_code))
                    cur_code = int((cur_code - 1) / 2)
                else:
                    break
        node_list = sorted(node_list)
        self.node_id_layer_id=torch.zeros(len(node_list),dtype=torch.int64,device=device)
        cur_layer, cur_layer_node_num = 0, 0
        self.node_code_node_id_array=torch.full((node_list[-1].code+2,),-1,dtype=torch.int64,device=device)
        self.node_sampled_indicator=torch.full((node_list[-1].code+2,),0,dtype=torch.bool)
        for i, node in enumerate(node_list):
            node.id = i
            self.node_code_node_id[node.code]=i
            self.node_code_node_id_array[node.code]=i
            self.node_id_layer_id[i]=int(math.log(node.code+ 1, 2))

            if self.leaf_code_item_id.get(node.code) is not None:
                node.corresponding_item_id = self.leaf_code_item_id.get(node.code)
                node.isleaf = True
            if int(math.log(node.code + 1.0, 2)) == cur_layer:
                cur_layer_node_num += 1
            else:
                self.layer_node_num_list.append(cur_layer_node_num)
                cur_layer_node_num = 1
                cur_layer += 1
        self.layer_node_num_list.append(cur_layer_node_num)
        self.max_layer_id = int(math.log(leaf_codes[0] + 1, 2))
        assert self.max_layer_id == len(self.layer_node_num_list) - 1
        assert sum(self.layer_node_num_list) == len(node_list)
        assert len( self.node_code_node_id)==len(node_list)

        self.node_num = len(node_list)
        self.item_id_node_ancestor_id=torch.zeros((len(item_ids),self.max_layer_id+1),dtype=torch.int64,device=device)
        self.generate_item_id_ancestor_node_id()

        codes=list(self.node_code_node_id.keys())
        codes.sort(reverse=True)#sort descent
        for code in codes:
            self.maximum_assigned_item_num[code] = 0
            if int(math.log(code + 1, 2))==self.max_layer_id:#leaf node
                self.maximum_assigned_item_num[code]=+1
            else:
                if 2*code+1 in self.node_code_node_id:
                    self.maximum_assigned_item_num[code] += self.maximum_assigned_item_num[2*code+1]
                if 2*code+2 in self.node_code_node_id:
                    self.maximum_assigned_item_num[code] += self.maximum_assigned_item_num[2*code+2]
        assert self.maximum_assigned_item_num[0]==len(self.item_id_leaf_code)
        print('Tree {},node number is {}, tree height is {}'.format(tree_id, len(node_list), self.max_layer_id))

        # leaf_index_complete_leaf_index
        leaf_num = self.layer_node_num_list[-1]
        complete_leaf_num = 2 ** self.max_layer_id
        node_exclude_leaf_num = self.node_num - leaf_num
        assert complete_leaf_num == node_exclude_leaf_num + 1
        self.leaf_index_complete_leaf_index_array = torch.full((leaf_num,), -1, dtype=torch.int64, device=device)
        for _, leaf_code in self.item_id_leaf_code.items():
            leaf_idx = self.node_code_node_id[leaf_code] - node_exclude_leaf_num # the index of leaf node in the leaf node list
            self.leaf_index_complete_leaf_index_array[leaf_idx] = leaf_code - node_exclude_leaf_num
        
        # item_id_complete_leaf_index
        print('calculate item_id_complete_leaf_index_array ... ...')
        self.item_id_complete_leaf_index_array = torch.full((len(item_ids),), -1, dtype=torch.int64, device=device)
        for item_id, leaf_code in self.item_id_leaf_code.items():
            self.item_id_complete_leaf_index_array[item_id] = leaf_code - node_exclude_leaf_num
        pass
    
    def generate_item_id_ancestor_node_id(self):
        #print('hh {},{},{}'.format(len(self.item_id_leaf_code.items()),*self.item_id_node_ancestor_id.shape))
        for item_id,leaf_code in self.item_id_leaf_code.items():
            layer=self.max_layer_id
            code=leaf_code
            while layer>=0:
                self.item_id_node_ancestor_id[item_id,layer]=self.node_code_node_id[code]
                code=int((code-1)/2)
                layer-=1
            assert code==0


    def get_ancestor(self, code, level):
        code_max = 2 ** (level + 1) - 1
        while code >= code_max:
            code = int((code - 1) / 2)
        return code


    def get_nodes_given_level(self, level):
        code_min = 2 ** level - 1
        code_max = 2 ** (level + 1) - 1
        res = []
        for code in range(code_min,code_max):
            if code in self.node_code_node_id:
                res.append(code)
        '''
        for code in self.node_code_node_id:
            if code >= code_min and code < code_max:
                res.append(code)
        '''
        return res

    def get_children_given_ancestor_and_level(self, ancestor, level):
        code_min = 2 ** level - 1
        code_max = 2 ** (level + 1) - 1

        parent_queue=queue.Queue()
        parent_queue.put(ancestor)
        res=[]
        while parent_queue.qsize()>0:
            parent_code=parent_queue.get()
            if parent_code>=code_min and parent_code<code_max:
                if parent_code in self.node_code_node_id:
                    res.append(parent_code)
            else:
                parent_queue.put(2*parent_code+1)
                parent_queue.put(2*parent_code+2)
        return res



    '''
    #error
    def get_children_given_ancestor_and_level(self, ancestor, level):
        code_min = 2 ** level - 1
        code_max = 2 ** (level + 1) - 1
        parent = [ancestor]
        res = []
        while True:
            children = []
            for p in parent:
                children.extend([2 * p + 1, 2 * p + 2])
            if code_min <= children[0] < code_max:
                break
            parent = children

        output = []
        for i in children:
        #for i in res: #this is the original codes
            if i in self.nodes:
                output.append(i)
        return output
    '''
    def get_parent_path(self, child, ancestor):
        res = []
        while child > ancestor:
            res.append(child)
            child = int((child - 1) / 2)
        return res

    def _ancessors(self, code):
        ancs = []
        while code > 0:
            code = int((code - 1) / 2)
            ancs.append(code)
        return ancs
    def get_off_spring_number(self,code):
        num=0
        que=queue.Queue()
        que.put(code)
        while que.qsize()>0:
            p_code=que.get()
            if 2 * p_code + 1 in self.node_code_node_id:
                num += 1
                que.put(2 * p_code + 1)
            if 2 * p_code + 2 in self.node_code_node_id:
                num += 1
                que.put(2 * p_code + 2)
        return num



'''
class Tree_Group(object):
    def __init__(self):
        pass

    @staticmethod
    def do(item_ids_list,leaf_node_codes_list):
        tree_list = []
        tree_num=len(item_ids_list)
        item_num=len(item_ids_list[0])
        total_node_num=0
        for tree_id,item_ids,leaf_node_codes in zip(range(tree_num),item_ids_list,leaf_node_codes_list):
            tree_list.append(Tree(item_ids,leaf_node_codes,tree_id=tree_id))
            total_node_num+=tree_list[-1].node_num
        global_leaf_node_ids=np.arange(total_node_num-item_num*tree_num,total_node_num-item_num*(tree_num-1))
        start_id=0
        for tree in tree_list:
            tree.node_local_id_global_id[:tree.node_num-item_num]=np.arange(start_id,start_id+tree.node_num-item_num)
            tree.node_local_id_global_id[-item_num:]=global_leaf_node_ids[:]
            start_id+=(tree.node_num-item_num)
        assert start_id==total_node_num-item_num*tree_num
        return tree_list

'''



if __name__ == '__main__':
    c1 = TreeNode(0, 11)
    c2 = TreeNode(0, 33)
    c3 = TreeNode(0, 55)
    c4 = TreeNode(0, 9)
    c5 = TreeNode(0, 100)
    c6 = TreeNode(3, 100)
    c7 = TreeNode(3, 66)

    ls = [c1, c2, c3, c4, c5, c6, c7]
    for c in ls:
        print(c.id, c.code)
    print('\n \n')
    lc = sorted(ls)
    for c in lc:
        print(c.id, c.code)

    import numpy as np
    item_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    leaf_codes = [1, 3, 5, 7, 9, 11, 13, 15, 17]