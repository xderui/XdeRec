'''
Description: 
Author: Rigel Ma
Date: 2024-04-26 16:04:07
LastEditors: Rigel Ma
LastEditTime: 2024-04-27 00:13:35
FilePath: CF_dataset.py
'''

from data.interact_dataset import Interact_dataset
import random

class CF_dataset(Interact_dataset):
    def __init__(self, **kwars):
        super(CF_dataset, self).__init__(**kwars)



    def get_uid_iid(self, type='train'):
        types = ['train', 'val', 'test']
        if type not in types:
            raise ValueError("type must be 'train', 'val' or 'test'")
        interactions = self.splited_interactions[types.index(type)]
        uid_list = interactions.loc[:,self.map_columns[0]].to_numpy()
        iid_list = interactions.loc[:,self.map_columns[1]].to_numpy()

        return uid_list, iid_list
    
    
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