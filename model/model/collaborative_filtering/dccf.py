'''
Description: 
Author: Rigel Ma
Date: 2024-04-21 17:06:09
LastEditors: Rigel Ma
LastEditTime: 2024-05-12 15:51:08
FilePath: DCCF.py
Paper: Disentangled contrastive collaborative filtering, SIGIR 2023
'''
from utils.Libs import *
from model.model.collaborative_filtering.BaseModel import BaseModel
from data.interact_dataset import Interact_dataset
from utils.loss import bpr_loss, reg_loss_pow


class DCCF(BaseModel):
    def __init__(self,
                 interactions: Interact_dataset,
                 param_dict: dict):
        super(DCCF, self).__init__(interactions, param_dict)
        
        self.embedding_user = nn.Embedding(
            num_embeddings=self.interactions.num_users, 
            embedding_dim=self.latent_dim
        )
        
        self.embedding_item = nn.Embedding(
            num_embeddings=self.interactions.num_items,
            embedding_dim=self.latent_dim
        )

        _user_intent = torch.empty(self.latent_dim, self.n_intents)
        nn.init.xavier_normal_(_user_intent)
        self.user_intent = torch.nn.Parameter(_user_intent, requires_grad=True)
        _item_intent = torch.empty(self.latent_dim, self.n_intents)
        nn.init.xavier_normal_(_item_intent)
        self.item_intent = torch.nn.Parameter(_item_intent, requires_grad=True)

        self._init_weight()
    
    def _init_weight(self):
        nn.init.xavier_normal_(self.embedding_user.weight)
        nn.init.xavier_normal_(self.embedding_item.weight)


    def _adaptive_mask(self, head_embeddings, tail_embeddings):

        head_embeddings = torch.nn.functional.normalize(head_embeddings)
        tail_embeddings = torch.nn.functional.normalize(tail_embeddings)
        edge_alpha = (torch.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2

        A_tensor = torch_sparse.SparseTensor(row=self.rows, col=self.cols, value=edge_alpha, sparse_sizes=self.A_shape).cuda()
        D_scores_inv = A_tensor.sum(dim=1).pow(-1).nan_to_num(0, 0, 0).view(-1)

        G_indices = torch.stack([self.rows, self.cols], dim=0)
        G_values = D_scores_inv[self.rows] * edge_alpha

        return G_indices, G_values
    

    def computer(self):
        user_emb = self.embedding_user.weight
        item_emb = self.embedding_item.weight

        embs = torch.concat([user_emb, item_emb], dim=0)
        all_embs = [embs]
        gnn_embs = []
        int_embs = []
        gaa_embs = []
        iaa_embs = []

        for i in range(self.n_layers):
            gnn_emb = torch_sparse.spmm(self.G_indices, self.G_values, self.A_dim, self.A_dim, all_embs[i])

            # Intent-aware Information Aggregation
            u_embeddings, i_embeddings = torch.split(all_embs[i], [self.num_users, self.num_items], 0)
            u_int_emb = torch.softmax(u_embeddings @ self.user_intent, dim=1) @ self.user_intent.T
            i_int_emb = torch.softmax(i_embeddings @ self.item_intent, dim=1) @ self.item_intent.T
            int_emb = torch.concat([u_int_emb, i_int_emb], dim=0)

            # Adaptive Augmentation
            gnn_head_emb = torch.index_select(gnn_emb, 0, self.rows)
            gnn_tail_emb = torch.index_select(gnn_emb, 0, self.cols)
            int_head_emb = torch.index_select(int_emb, 0, self.rows)
            int_tail_emb = torch.index_select(int_emb, 0, self.cols)
            G_graph_indices, G_graph_values = self._adaptive_mask(gnn_head_emb, gnn_tail_emb)
            G_inten_indices, G_inten_values = self._adaptive_mask(int_head_emb, int_tail_emb)

            gaa_emb = torch_sparse.spmm(G_graph_indices, G_graph_values, self.A_dim, self.A_dim, all_embs[i])
            iaa_emb = torch_sparse.spmm(G_inten_indices, G_inten_values, self.A_dim, self.A_dim, all_embs[i])

            gnn_embs.append(gnn_emb)
            int_embs.append(int_emb)
            gaa_embs.append(gaa_emb)
            iaa_embs.append(iaa_emb)

            all_embs.append(gnn_emb + int_emb + gaa_emb + iaa_emb + all_embs[i])


        all_embs = torch.stack(all_embs, dim=1)
        all_embs = torch.sum(all_embs, dim=1, keepdim=False)

        self.user_emb, self.item_emb = torch.split(all_embs, [self.num_users, self.num_items])

        return gnn_embs, int_embs, gaa_embs, iaa_embs
    
    def cal_cl_loss(self, users, items, gnn_emb, int_emb, gaa_emb, iaa_emb):
        users = torch.unique(users)
        items = torch.unique(items)

        cl_loss = 0.0

        def cal_loss(emb1, emb2):
            pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.temperature)
            neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.temperature), axis=1)
            loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
            loss /= pos_score.shape[0]
            return loss

        for i in range(len(gnn_emb)):
            u_gnn_embs, i_gnn_embs = torch.split(gnn_emb[i], [self.num_users, self.num_items], 0)
            u_int_embs, i_int_embs = torch.split(int_emb[i], [self.num_users, self.num_items], 0)
            u_gaa_embs, i_gaa_embs = torch.split(gaa_emb[i], [self.num_users, self.num_items], 0)
            u_iaa_embs, i_iaa_embs = torch.split(iaa_emb[i], [self.num_users, self.num_items], 0)

            u_gnn_embs = F.normalize(u_gnn_embs[users], dim=1)
            u_int_embs = F.normalize(u_int_embs[users], dim=1)
            u_gaa_embs = F.normalize(u_gaa_embs[users], dim=1)
            u_iaa_embs = F.normalize(u_iaa_embs[users], dim=1)

            i_gnn_embs = F.normalize(i_gnn_embs[items], dim=1)
            i_int_embs = F.normalize(i_int_embs[items], dim=1)
            i_gaa_embs = F.normalize(i_gaa_embs[items], dim=1)
            i_iaa_embs = F.normalize(i_iaa_embs[items], dim=1)

            cl_loss += cal_loss(u_gnn_embs, u_int_embs)
            cl_loss += cal_loss(u_gnn_embs, u_gaa_embs)
            cl_loss += cal_loss(u_gnn_embs, u_iaa_embs)

            cl_loss += cal_loss(i_gnn_embs, i_int_embs)
            cl_loss += cal_loss(i_gnn_embs, i_gaa_embs)
            cl_loss += cal_loss(i_gnn_embs, i_iaa_embs)

        return cl_loss


    def forward(self, users, pos_items, neg_items):
        gnn_embs, int_embs, gaa_embs, iaa_embs = self.computer()
        user_embs = self.user_emb[users]
        pos_item_embs = self.item_emb[pos_items]
        neg_item_embs = self.item_emb[neg_items]

        pre_user_embs = self.embedding_user(users)
        pre_pos_item_embs = self.embedding_item(pos_items)
        pre_neg_item_embs = self.embedding_item(neg_items)
        
        # bpr 
        loss = bpr_loss(user_embs, pos_item_embs, neg_item_embs)
        
        # reg
        loss = loss + self.emb_reg * (pre_user_embs.norm(2).pow(2) + pre_pos_item_embs.norm(2).pow(2) \
                    + pre_neg_item_embs.norm(2).pow(2))

        # intent
        loss = loss + self.cen_reg * (self.user_intent.norm(2).pow(2) + self.item_intent.norm(2).pow(2))

        # self-supervise learning
        loss = loss + self.ssl_reg * self.cal_cl_loss(users, pos_items, gnn_embs, int_embs, gaa_embs, iaa_embs)

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

