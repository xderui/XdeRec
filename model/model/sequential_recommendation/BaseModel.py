import torch
import torch.nn as nn

# 序列推荐模型输入：
# (user, targetitem, target_cate, interact_seq, interact_cate_seq, )
'''


timeinterval_list: (time_list[i+1] - time_list[i]) / (3600*24) + [time_now - time_list[-1]]
timelast_list: [0,0] + (time_list[i+1] - time_list[i]) / (3600*24)
timenow_list: (time_now - time_list[i])
'''

class BaseModel(nn.Module):
    def __init__(self, ):
        super(BaseModel, self).__init__()

        

        

