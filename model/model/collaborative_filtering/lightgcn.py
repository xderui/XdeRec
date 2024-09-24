"""
# -*- encoding: utf-8 -*-
Descripttion: Parameter settings of LightGCN
version: 1.0.0
Author: Rigel Ma
Date: 2023-11-21 17:07:46
LastEditors: Rigel Ma (rigelma01@gmail.com)
LastEditTime: 2023-11-22 17:14:55
---------------
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

Ref. @author: Jianbai Ye (gusye@mail.ustc.edu.cn)
---------------
"""

"""
模型部分需要声明的部分：
1. embedding, layer之类的内容 (参照LightGCN, 其中有加载预训练的部分，如何加载预训练模型写在模型定义还是主函数比较好，有待斟酌)

2. forward部分 (前向返回的应该是向量, 以便在Trainer中定义device类型)

3. 损失计算部分 (考虑到不同模型会使用特定的损失函数，故此部分也自己声明最好)
   该部分将使用forward的输出作为输入，然后得到loss

4. predict部分，用来预测topk结果
"""

from utils.Libs import *
from model.model.collaborative_filtering._BaseModel_ import BaseModel
from data.interact_dataset import Interact_dataset


class LightGCN(BaseModel):
    def __init__(self,
                 interactions: Interact_dataset,
                 param_dict: dict):
        super(LightGCN, self).__init__(interactions, param_dict)
        
        self.embedding_user = nn.Embedding(
            num_embeddings=self.interactions.num_users, 
            embedding_dim=self.latent_dim
        )
        
        self.embedding_item = nn.Embedding(
            num_embeddings=self.interactions.num_items,
            embedding_dim=self.latent_dim
        )

        self._init_weight()
    
    def _init_weight(self):
        nn.init.xavier_normal_(self.embedding_user.weight)
        nn.init.xavier_normal_(self.embedding_item.weight)


    def __dropout(self, keep_prob):
        
        def dropout(x):
            size = x.sizes()
            rows, cols, values = x.coo()
            indices = torch.stack([rows,cols],dim=0)  # (2,n)
            random_index = torch.rand(len(values)) + keep_prob
            random_index = random_index.int().bool()
            indices = indices[:,random_index]
            values = values[random_index]/keep_prob
            g = torch_sparse.SparseTensor(row=indices[0,:], col=indices[1,:], value=values, 
                                                 sparse_sizes=size)
        
            return g

        if self.A_split: 
            graph = []
            for g in self.graph:
                graph.append(dropout(g))
        else:
            graph = dropout(self.graph)
    
        return graph

    def computer(self):
        user_emb = self.embedding_user.weight
        item_emb = self.embedding_item.weight

        embs = torch.concat([user_emb, item_emb], dim=0)
        all_embs = [embs]

        for _ in range(self.n_layers):
            embs = torch_sparse.spmm(self.G_indices, self.G_values, self.A_dim, self.A_dim, embs)
            all_embs.append(embs)

        all_embs = torch.stack(all_embs, dim=1)
        all_embs = torch.mean(all_embs, dim=1, keepdim=False)

        self.user_emb, self.item_emb = torch.split(all_embs, [self.num_users, self.num_items])

        return self.user_emb, self.item_emb


    def forward(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        user_embs = all_users[users]
        pos_item_embs = all_items[pos_items]
        neg_item_embs = all_items[neg_items]

        pre_user_embs = self.embedding_user(users)
        pre_pos_item_embs = self.embedding_item(pos_items)
        pre_neg_item_embs = self.embedding_item(neg_items)
        
        loss = self.bpr_loss(user_embs, pos_item_embs, neg_item_embs)
        
        loss = loss + self.emb_reg * (pre_user_embs.norm(2).pow(2) + pre_pos_item_embs.norm(2).pow(2) \
                    + pre_neg_item_embs.norm(2).pow(2))

        return loss
        

    def predict(self, users):
        all_users, all_items = self.user_emb, self.item_emb
        user_embs = all_users[users]
        ratings = torch.matmul(user_embs, all_items.t())   
        return ratings

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    

    def bpr_loss(self, user_embs, pos_item_embs, neg_item_embs):  
        pos_scores = torch.sum(user_embs * pos_item_embs, dim=-1)
        neg_scores = torch.sum(user_embs * neg_item_embs, dim=-1)
        
        bpr_loss_ = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

        return bpr_loss_
    

        
