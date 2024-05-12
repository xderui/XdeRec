from utils.Libs import *
from model.model.BaseModel import BaseModel
from data.interact_dataset import Interact_dataset
import random
import scipy.sparse as sp

class SimGCL(BaseModel):
    def __init__(self,
                 interactions: Interact_dataset,
                 param_dict: dict):
        super(SimGCL, self).__init__(interactions, param_dict)
        
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



    def InfoNCE(self, view1, view2):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def cal_cl_loss(self, users, pos_items):
        user_view_1, item_view_1 = self.computer(self.new_G_indices_1, self.new_G_values_1)
        user_view_2, item_view_2 = self.computer(self.new_G_indices_2, self.new_G_values_2)
        user_cl_loss = self.InfoNCE(user_view_1[users], user_view_2[users])
        item_cl_loss = self.InfoNCE(item_view_1[pos_items], item_view_2[pos_items])
        return user_cl_loss + item_cl_loss


    def computer(self, perturbed=False):
        user_emb = self.embedding_user.weight
        item_emb = self.embedding_item.weight

        embs = torch.concat([user_emb, item_emb], dim=0)
        all_embs = [embs]

        for _ in range(self.n_layers):
            embs = torch_sparse.spmm(self.G_indices, self.G_values, self.A_dim, self.A_dim, embs)
            
            if perturbed:
                random_noise = torch.rand_like(embs).cuda()
                embs += torch.sign(embs) * F.normalize(random_noise, dim=-1) * self.eps

            all_embs.append(embs)

        all_embs = torch.stack(all_embs, dim=1)
        all_embs = torch.mean(all_embs, dim=1, keepdim=False)

        self.user_emb, self.item_emb = torch.split(all_embs, [self.num_users, self.num_items])

        return self.user_emb, self.item_emb


    def forward(self, users, pos_items, neg_items):

        # cl loss
        cl_loss = self.cl_reg * self.cal_cl_loss(users, pos_items) 


        all_users, all_items = self.computer()
        user_embs = all_users[users]
        pos_item_embs = all_items[pos_items]
        neg_item_embs = all_items[neg_items]
        
        loss = self.bpr_loss(user_embs, pos_item_embs, neg_item_embs)
        
        loss = loss + self.emb_reg * (user_embs.norm(2) + pos_item_embs.norm(2))
        
        loss = loss + cl_loss

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
    

        
