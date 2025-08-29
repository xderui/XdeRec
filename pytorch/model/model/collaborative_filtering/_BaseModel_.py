'''
Author: Rigel Ma
Date: 2023-11-24 15:46:58
LastEditors: Rigel Ma
LastEditTime: 2024-04-28 12:49:02
FilePath: BaseModel.py
Description: The class of other models will inherit on BaseModel
'''


from utils.Libs import *
from data.interact_dataset import Interact_dataset
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    def __init__(self, 
                 interactions: Interact_dataset,
                 param_dict: dict):
        super(BaseModel, self).__init__()
        self.interactions = interactions
        self.num_users, self.num_items = interactions.num_users, interactions.num_items

        # prepare for sparse graph
        self.uid_list, self.iid_list = interactions.get_uid_iid('train')

        # rows, cols in adj_mat
        self.rows = np.concatenate([self.uid_list, self.iid_list+self.num_users], axis=0).tolist()
        self.cols = np.concatenate([self.iid_list+self.num_users, self.uid_list], axis=0).tolist()


        #### information of adj mat ####

        # shape (dim, dim)
        self.A_dim = self.num_users + self.num_items
        self.A_shape = [self.A_dim] * 2
        self.A_indices = torch.tensor([self.rows, self.cols], dtype=torch.long).cuda()
        self.D_indices = torch.tensor([list(range(self.num_users + self.num_items)), list(range(self.num_users + self.num_items))], dtype=torch.long).cuda()

        #### information of laplacian mat ####

        # type
        self.rows = torch.LongTensor(self.rows).to(DEVICE())
        self.cols = torch.LongTensor(self.cols).to(DEVICE())

        # indices and values of laplacian mat
        self.G_dim = self.num_users + self.num_items
        self.G_shape = [self.G_dim] * 2
        self.G_indices, self.G_values = self.laplacian_adj()

        self.init_param(param_dict)
        

    def init_param(self, param_dict):
        for k,v in param_dict.items():
            if isinstance(v, str):
                exec(f'self.{k}="{v}"')
            else:
                exec(f'self.{k}={v}')
    

    def laplacian_adj(self):
        A_values = torch.ones((len(self.rows), 1)).view(-1).cuda()

        A_indices_sparse = torch_sparse.SparseTensor(row=self.rows, col=self.cols, value=A_values, sparse_sizes=self.A_shape).cuda()

        D_values = A_indices_sparse.sum(dim=-1).pow(-0.5)

        G_indices, G_values = torch_sparse.spspmm(self.D_indices, D_values, self.A_indices, A_values, self.A_shape[0], self.A_shape[1], self.A_shape[1])

        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices, D_values, self.A_shape[0], self.A_shape[1], self.A_shape[1])

        return G_indices, G_values

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def predict(self):
        pass
    
    def sample(self):
        pass