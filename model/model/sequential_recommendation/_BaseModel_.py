import torch
import torch.nn as nn
from data.interact_dataset import Interact_dataset
from utils.Libs import *


# 序列推荐模型输入：
# (user, targetitem, target_cate, interact_seq, interact_cate_seq, )
'''


timeinterval_list: (time_list[i+1] - time_list[i]) / (3600*24) + [time_now - time_list[-1]]
timelast_list: [0,0] + (time_list[i+1] - time_list[i]) / (3600*24)
timenow_list: (time_now - time_list[i])
'''

class BaseModel(nn.Module):
    def __init__(self,
                 interactions: Interact_dataset,
                 param_dict: dict):
        super(BaseModel, self).__init__()

        self.interactions = interactions
        self.num_users, self.num_items = interactions.num_users, interactions.num_items
        
        self.init_param(param_dict)


    def init_param(self, param_dict):
        for k,v in param_dict.items():
            if isinstance(v, str):
                exec(f'self.{k}="{v}"')
            else:
                exec(f'self.{k}={v}')


        

