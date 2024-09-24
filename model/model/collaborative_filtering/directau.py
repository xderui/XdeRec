'''
Description: 
Author: Rigel Ma
Date: 2024-04-21 17:06:09
LastEditors: Rigel Ma
LastEditTime: 2024-04-28 12:52:11
FilePath: DirectAU.py
Paper: Towards representation alignment and uniformity in collaborative filtering, KDD 2022
'''

from utils.Libs import *
from model.model.collaborative_filtering._BaseModel_ import BaseModel
from data.interact_dataset import Interact_dataset
from utils.loss import reg_loss, alignment_loss, uniformity_loss


class DirectAU(BaseModel):
    def __init__(self,
                 interactions: Interact_dataset,
                 param_dict: dict):
        super(DirectAU, self).__init__(interactions, param_dict)
        
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

        # alignment
        loss = alignment_loss(user_embs, pos_item_embs)

        # uniformity
        loss = loss + self.uniform_reg * (uniformity_loss(user_embs) + uniformity_loss(pos_item_embs))

        # reg loss
        loss = loss + self.emb_reg * reg_loss(user_embs, pos_item_embs)


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

        
