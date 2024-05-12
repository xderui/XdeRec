'''
Description: 
Author: Rigel Ma
Date: 2024-04-26 15:58:59
LastEditors: Rigel Ma
LastEditTime: 2024-05-12 15:48:31
FilePath: Sequential_dataset.py
'''


from data.interact_dataset import Interact_dataset
from utils.Libs import *
import copy
import yaml

class Sequential_dataset(Interact_dataset):
    def __init__(self, **kwargs):
        super(Sequential_dataset, self).__init__(**kwargs)

        self.user_col, self.item_col, self.rating_col, self.timestamp_col = self.columns
        self.max_seq_len = kwargs.get("max_seq_len")

        # check params
        self.check_params()

        self.splited_sequencce_data = []

        for interactions_ in self.splited_interactions:
            self.splited_sequencce_data.append(self.create_sequence_from_interaction(interactions_))

    def check_params(self):
        with open('./data/require_params.yaml', 'r', encoding='utf-8') as f:
            require_params = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        for require_key, require_desc in require_params[__name__.split('.')[-1]].items():
            if not getattr(self, require_key):
                raise KeyError(f"missing param '{require_key}' in dataset_config, and the meaning of it is '{require_desc}'")



    def create_sequence_from_interaction(self, interactions):
        interactions = interactions.sort_values(by=[self.user_col, self.timestamp_col], ascending=[True, True])
        last_user = -1
        last_time = -1
        new_interactions = []
        item_list = []
        timeinterval_list = []
        timelast_list = []
        timenow_list = []
        for i,x in interactions.iterrows():
            uid = x[self.user_col]
            line = x.tolist()
            if uid != last_user:
                item_list = []
                timeinterval_list = []
                timelast_list = []
                timenow_list = []

                last_user = uid
                last_time = x[self.timestamp_col]
            else:
                timenow_list = (np.array(timenow_list) + x[self.timestamp_col] - last_time).tolist()
                timenow_list.append(x[self.timestamp_col] - last_time)

                line.extend([item_list[-self.max_seq_len:], timeinterval_list[-self.max_seq_len:], timelast_list[-self.max_seq_len:]+[x[self.timestamp_col]-last_time], timenow_list[-self.max_seq_len:]])
                new_interactions.append(line)


            item_list.append(x[self.item_col])
            timeinterval_list.append(x[self.timestamp_col] - last_time)
            timelast_list.append(x[self.timestamp_col] - last_time)
            

            last_time = x[self.timestamp_col]

        new_interactions = pd.DataFrame(new_interactions, \
                                columns=['uid', 'iid', 'rating', 'timestamp', 'iid_list', 'timeinterval_list', 'timelast_list', 'timenow_list']
                            )

        return new_interactions

        
