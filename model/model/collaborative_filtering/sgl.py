from utils.Libs import *
from model.model.collaborative_filtering._BaseModel_ import BaseModel
from data.interact_dataset import Interact_dataset
import random
import scipy.sparse as sp

class SGL(BaseModel):
    def __init__(self,
                 interactions: Interact_dataset,
                 param_dict: dict):
        super(SGL, self).__init__(interactions, param_dict)
        
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


    def data_augment(self, aug_type, drop_rate):
        if aug_type == 'node':
            drop_user_idx = random.sample(range(self.num_users), int(self.num_users * drop_rate))
            drop_item_idx = random.sample(range(self.num_items), int(self.num_items * drop_rate))
            indicator_user = np.ones(self.num_users, dtype=np.float32)
            indicator_item = np.ones(self.num_items, dtype=np.float32)
            indicator_user[drop_user_idx] = 0.
            indicator_item[drop_item_idx] = 0.
            diag_indicator_user = sp.diags(indicator_user)
            diag_indicator_item = sp.diags(indicator_item)
            R = sp.csr_matrix(
                (np.ones_like(self.uid_list, dtype=np.float32), (self.uid_list, self.iid_list)), 
                shape=(self.num_users, self.num_items))
            R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
            (user_np_keep, item_np_keep) = R_prime.nonzero()
            ratings_keep = R_prime.data
            tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep+self.num_users)), shape=(self.num_users+self.num_items, self.num_users+self.num_items))
        if aug_type == 'edge':
            record_num = len(self.list_h_list) // 2
            keep_idx = random.sample(range(record_num), int(record_num * (1 - drop_rate)))
            user_np = np.array(self.uid_list)[keep_idx]
            item_np = np.array(self.iid_list)[keep_idx]
            ratings = np.ones_like(user_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.num_users)), shape=(self.num_users+self.num_items, self.num_users+self.num_items))


        adj_mat = tmp_adj + tmp_adj.T
        tmp = adj_mat.tocoo()
        new_h_list = tmp.row
        new_t_list = tmp.col
        A_in_shape = tmp.shape
        A_indices = torch.tensor([new_h_list, new_t_list], dtype=torch.long).cuda()
        D_indices = torch.tensor([list(range(self.num_users + self.num_items)), list(range(self.num_users + self.num_items))], dtype=torch.long).cuda()

        new_h_list = torch.LongTensor(new_h_list).cuda()
        new_t_list = torch.LongTensor(new_t_list).cuda()

        A_values = torch.ones(size=(len(new_h_list), 1)).view(-1).cuda()

        A_tensor = torch_sparse.SparseTensor(row=new_h_list, col=new_t_list, value=A_values, sparse_sizes=A_in_shape).cuda()
        D_values = A_tensor.sum(dim=1).pow(-0.5)

        G_indices, G_values = torch_sparse.spspmm(D_indices, D_values, A_indices, A_values, A_in_shape[0], A_in_shape[1], A_in_shape[1])
        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, D_indices, D_values, A_in_shape[0], A_in_shape[1], A_in_shape[1])

        return G_indices, G_values

    def generate_cl_view(self):
        self.new_G_indices_1, self.new_G_values_1 = self.data_augment(aug_type=self.aug_type, drop_rate=0.2)
        self.new_G_indices_2, self.new_G_values_2 = self.data_augment(aug_type=self.aug_type, drop_rate=0.2)

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


    def computer(self, new_G_indices=None, new_G_values=None):
        user_emb = self.embedding_user.weight
        item_emb = self.embedding_item.weight

        embs = torch.concat([user_emb, item_emb], dim=0)
        all_embs = [embs]

        if new_G_indices != None:
            G_indices, G_values = new_G_indices, new_G_values
        else:
            G_indices, G_values = self.G_indices, self.G_values


        for _ in range(self.n_layers):
            embs = torch_sparse.spmm(G_indices, G_values, self.A_dim, self.A_dim, embs)
            all_embs.append(embs)

        all_embs = torch.stack(all_embs, dim=1)
        all_embs = torch.mean(all_embs, dim=1, keepdim=False)

        self.user_emb, self.item_emb = torch.split(all_embs, [self.num_users, self.num_items])

        return self.user_emb, self.item_emb

    def sample(self):
        self.generate_cl_view()


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
    

        
