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
import os
import json
from data.download import download
import sys

SUPPORT_PRODUCT = {}

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
        self.dataset_file = kwargs.get('file')
        self.columns = kwargs.get('columns')
        self.map_columns = kwargs.get('map_columns')
        self.core_columns= kwargs.get('core_columns')
        self.core_settings= kwargs.get('core_settings')
        self.split_ratio = kwargs.get('split_ratio')
        self.order = kwargs.get('order')
        self.groupby = kwargs.get('groupby')
        self.batch_size = kwargs.get('batch_size')

        # check parameters


        # load data
        separator = kwargs.get('separator')
        dataset_root_path = './dataset/{}'.format(self.dataset_name)
        dataset_path = os.path.join(dataset_root_path, self.dataset_file)

        if not os.path.exists(dataset_root_path):
            download_flag = input(f'dataset {self.dataset_name} not found, try to download it?\n[y]es, [n]o:')
            if download_flag == 'y':
                if len(SUPPORT_PRODUCT) == 0:
                    self.get_support_product()
                
                if len(SUPPORT_PRODUCT) == 0:
                    print('sorry, there seems no dataset avaliable for download...>_<\nplease setup the dataset by self!')
                    sys.exit(0)
                
                print("Which of the following does this dataset belong to?")
                for idx, product in enumerate(SUPPORT_PRODUCT.keys()):
                    print(f"{idx}. {product}")
                
                try:
                    product_id = int(input("choose: "))
                except ValueError:
                    raise "invalid id!"

                url_root = dict(list(SUPPORT_PRODUCT.values())[product_id])
                download(url_root['url_root'], self.dataset_name)

            else:
                print("No dataset can be used.")
                sys.exit(0)
                    


        self.interactions = pd.read_csv(dataset_path, sep=separator, header=None, names=self.columns, engine='python')
        self.map_interactions = copy.deepcopy(self.interactions)


        # first is uid, and second is iid
        self.num_users = len(pd.unique(self.interactions.loc[:,self.columns[0]]))
        self.num_items = len(pd.unique(self.interactions.loc[:,self.columns[1]]))

        self.statistic()

        self.process()


    def check_params(self):
        require_params = [
            ""
        ]

    def statistic(self):
        string = f'===== {self.dataset_name} ======\n'
        string += f'num_users: {self.num_users}, num_items: {self.num_items}, sparsity: {len(self.map_interactions) / (self.num_users * self.num_items)}'
        logger.info(string)


    def size(self):
        return (self.num_users+self.num_items, self.num_items+self.num_users)
    

    def get_support_product(self):
        url_path = './data/url.json'
        global SUPPORT_PRODUCT
        SUPPORT_PRODUCT = json.loads(open(url_path).read())


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

        print(f"splited: [{[len(inter) for inter in self.splited_interactions]}]")

        return self.splited_interactions
    

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
    