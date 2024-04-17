# -*- encoding: utf-8 -*-
'''
Filename         :Interact_dataset.py
Description      :
Time             :2023/10/16 21:08:58
Author           :Rigel Ma
Version          :1.0
'''

from utils.Libs import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
import pandas as pd
from pprint import pprint
import copy
import random
import scipy.sparse as sp


class Interact_dataset(Dataset):
    def __init__(self, **kwargs):
        super(Interact_dataset, self).__init__()

        # trans "" in kwargs to None
        for k,v in kwargs.items():
            if v == '':
                kwargs[k] = None
        
        logger.info(kwargs)

        # parameters
        self.dataset_name = kwargs.get('dataset')
        self.columns = kwargs.get('columns')
        self.map_columns = kwargs.get('map_columns')
        self.core_columns= kwargs.get('core_columns')
        self.core_settings= kwargs.get('core_settings')
        self.split_ratio = kwargs.get('split_ratio')
        self.order = kwargs.get('order')
        self.groupby = kwargs.get('groupby')
        self.batch_size = kwargs.get('batch_size')

        # load data
        separator = kwargs.get('separator')
        dataset_path = './dataset/{}/ratings.dat'.format(self.dataset_name)
        self.interactions = pd.read_csv(dataset_path, sep=separator, header=None, names=self.columns, engine='python')
        self.map_interactions = copy.deepcopy(self.interactions)


        # first is uid, and second is iid
        self.num_users = len(pd.unique(self.interactions.loc[:,self.columns[0]]))
        self.num_items = len(pd.unique(self.interactions.loc[:,self.columns[1]]))

        self.statistic()


    def statistic(self):
        string = f'===== {self.dataset_name} ======\n'
        string += f'num_users: {self.num_users}, num_items: {self.num_items}, sparsity: {len(self.map_interactions) / (self.num_users * self.num_items)}'
        logger.info(string)


    def size(self):
        return (self.num_users+self.num_items, self.num_items+self.num_users)
    


    def process(self):
        logger.info('===== Process =====')
        self.filter_core()
        self.id_map()
        self.split_dataset(self.split_ratio, self.order, self.groupby)
        self.create_user_item_dict()  # craete the user-item dict in adavance

    def create_user_item_dict(self):
        training_interactions_df = self.splited_interactions[0].loc[:,self.map_columns]

        # user-item index
        self.user_item_dict = defaultdict(list)
        for user, x in training_interactions_df.groupby(self.map_columns[0]):
            self.user_item_dict[user] = x.iloc[:,1].tolist()

    def sample_all_interactions(self):
        training_interactions_df = self.splited_interactions[0].loc[:,self.map_columns]
        training_interactions_df = training_interactions_df.sample(frac=1)
        training_interactions = training_interactions_df.to_numpy()
        interaction_num = len(training_interactions)
        batch_idx = 0

        while batch_idx < interaction_num:
            st = batch_idx
            ed = min(batch_idx + self.batch_size, interaction_num)
            users = training_interactions[st:ed,0]
            pos_items = training_interactions[st:ed,1]
            neg_items = []
            for user in users:
                neg_item = random.randint(0, self.num_items-1)
                while neg_item in self.user_item_dict[user]:
                    neg_item = random.randint(0, self.num_items-1)
                neg_items.append(neg_item)

            batch_idx += self.batch_size

            yield users, pos_items, neg_items

    def sample_users(self, n_batch):
        batch_idx = 0
        sample_users = np.random.randint(0, self.num_users, n_batch * self.batch_size)
        for batch_idx in range(n_batch):
            batch_users = sample_users[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]
            pos_items = []
            neg_items = []
            for user in batch_users:
                pos_items.append(random.choice(self.user_item_dict[user]))
                neg_item = random.randint(0, self.num_items-1)
                while neg_item in self.user_item_dict[user]:
                    neg_item = random.randint(0, self.num_items-1)
                neg_items.append(neg_item)
            
            yield batch_users, pos_items, neg_items

    
    def id_map(self, cols=None):
        if cols == None:
            cols = self.map_columns

        # map
        for col in cols:
            id_list = self.map_interactions.loc[:,col].tolist()
            unique_ids = np.unique(id_list)
            map_dict = defaultdict(int)
            for idx, id in enumerate(unique_ids):
                map_dict[id] = idx
            logger.info(f'strat mapping {col}, length:{len(unique_ids)}\nmapping finished!')
            map_id_list = [map_dict[v] for v in id_list]
            self.map_interactions.loc[:,col] = map_id_list


    def filter_core(self, cols=None, cores=None):
        if cols == None:
            cols = self.core_columns
        if cores == None:
            cores = self.core_settings

        # cols is a list, which contains the columm which need to be filtered.
        # cores mean the settings for all cols.
        while True:
            filter_flag = 0
            for idx, col in enumerate(cols):
                min_num =cores[idx]
                col_count = self.map_interactions.groupby(by=col).count()
                col_index = col_count.loc[col_count.iloc[:,0]<min_num].index
                filter_index = self.map_interactions[col].isin(col_index)

                self.map_interactions = self.map_interactions.loc[~filter_index]

                if filter_index.sum() != 0:
                    filter_flag = 1
            
            if filter_flag == 0:
                break
    
        self.num_users = len(pd.unique(self.map_interactions.loc[:,self.columns[0]]))
        self.num_items = len(pd.unique(self.map_interactions.loc[:,self.columns[1]]))        

        logger.info('filterring sucessfully! user:{}, item:{}, interactions:{}'.format(self.num_users,self.num_items,len(self.map_interactions)))


    def split_dataset(self, split_ratio=None, order=None, groupby=None):
        
        if split_ratio == None:
            split_ratio = self.split_ratio

        if order == None:
            order = self.order

        if groupby == None:
            groupby = self.groupby

        ratio = np.cumsum([v/sum(split_ratio) for v in split_ratio]).tolist()
        ratio = [0] + ratio

        splited_interactions = []

        if order != None:
            self.map_interactions.sort_values(by=order, inplace=True)

        if groupby != None:
            split_interactions = self.map_interactions.groupby(by=groupby)
        else:
            split_interactions = [(0, self.map_interactions)]

        
        # split
        for i in range(len(ratio)-1):
            idx_ = []
            for _, interaction_ in split_interactions:
                records_len = len(interaction_)
                idx_.extend(interaction_.iloc[int(ratio[i]*records_len):int(ratio[i+1]*records_len)].index.tolist())

            splited_interactions.append(self.map_interactions.loc[idx_])

        self.splited_interactions = splited_interactions

        for interaction_ in self.splited_interactions:
            print(len(interaction_))

        return self.splited_interactions
    
    def get_uid_iid(self, type='train'):
        types = ['train', 'val', 'test']
        if type not in types:
            raise ValueError("type must be 'train', 'val' or 'test'")
        interactions = self.splited_interactions[types.index(type)]
        uid_list = interactions.loc[:,self.map_columns[0]].to_numpy()
        iid_list = interactions.loc[:,self.map_columns[1]].to_numpy()

        return uid_list, iid_list
        


    # def get_sparse_graph(self, type='train'):
    #     '''
    #     type: ['train', 'val', 'test']
    #     '''

    #     if type not in TYPES:
    #         raise ValueError("type must be 'train', 'val' or 'test'")
        
    #     interactions = self.splited_interactions[TYPES.index(type)]
    #     uid_list = interactions.loc[:,self.map_columns[0]].tolist()
    #     iid_list = interactions.loc[:,self.map_columns[1]].tolist()
    #     uid_list = torch.tensor(uid_list)
    #     iid_list = torch.tensor(iid_list)
    #     row_list = torch.concat([uid_list, iid_list+self.num_users], dim=0)
    #     col_list = torch.concat([iid_list+self.num_users, uid_list], dim=0)

    #     values = torch.ones((len(row_list),1)).view(-1)
    #     sparse_graph = torch_sparse.SparseTensor(row=row_list, col=col_list, value=values, 
    #                                              sparse_sizes=(self.num_users+self.num_items, self.num_users+self.num_items))

    #     values = values.cuda()
    #     # Lap
    #     # A_indices = torch.tensor([row_list, col_list], dtype=torch.long).cuda()
    #     A_indices = torch.stack([row_list, col_list], dim=0).cuda()
    #     D_values = sparse_graph.sum(dim=1).pow(-0.5).cuda()
    #     D_indices = torch.tensor([list(range(self.num_users + self.num_items)), list(range(self.num_users + self.num_items))], dtype=torch.long).cuda()
        
    #     A_dim = self.num_users + self.num_items

    #     G_indices, G_values = torch_sparse.spspmm(D_indices, D_values, A_indices, values, A_dim, A_dim, A_dim)
    #     G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, D_indices, D_values, A_dim, A_dim, A_dim)

    #     sparse_graph = torch_sparse.SparseTensor(row=row_list.cuda(), col=col_list.cuda(), value=G_values, 
    #                                              sparse_sizes=(self.num_users+self.num_items, self.num_users+self.num_items))
        
    #     return sparse_graph
    