# from model.sequential_recommendation.BaseModel import BaseModel
import torch.nn as nn
import torch
from torch.nn import functional as F
from model.model.sequential_recommendation._BaseModel_ import BaseModel
from data.interact_dataset import Interact_dataset
from model.model.sequential_recommendation._modules_ import MultiHeadAttention, FeedForward


class SASRec(BaseModel):
    def __init__(self,
                 interactions: Interact_dataset,
                 param_dict: dict):
        super(SASRec, self).__init__(interactions, param_dict)
        self.item_emb = nn.Embedding(self.num_items+1, self.latent_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(self.interactions.max_seq_len, self.pos_emb_dim)
        
        self.multi_attention = nn.ModuleList([MultiHeadAttention(self.num_heads, self.latent_dim, self.latent_dim, self.dropout_rate) for _ in range(self.num_blocks)] )
        self.feedfoward = nn.ModuleList([FeedForward(self.latent_dim)  for _ in range(self.num_blocks)])
        self.layernorm = nn.ModuleList([nn.LayerNorm(self.latent_dim) for _ in range(2 * self.num_blocks)])
        self.final_layernorm = nn.LayerNorm(self.latent_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, input_seq, pos, neg):
        seq_len = input_seq.shape[-1]
        mask = (input_seq != 0).unsqueeze(-1).float()
        
        # seq embedding
        print(input_seq.shape)
        seq_emb = self.item_emb(input_seq) * (self.latent_dim ** 0.5)

        # print(seq_emb.shape. input_seq.shape)
        # pos embedding
        pos_emb = self.pos_emb(input_seq)
        # add
        print(seq_emb.shape, pos_emb.shape)
        seq_emb = seq_emb + pos_emb
        # dropout
        seq_emb = self.dropout(seq_emb)  
        # mask
        seq_emb = seq_emb * mask
        
        # blocks
        print('start')
        for i in range(self.num_blocks):
            norm_seq_emb = self.layernorm[2*i](seq_emb)
            print(norm_seq_emb.shape, seq_emb.shape)
            seq_emb = self.multi_attention[i](norm_seq_emb, seq_emb)
            seq_emb = self.feedfoward[i](self.layernorm[2*i+1](seq_emb), dim=-1)
            seq_emb = seq_emb * mask

        print('start2')
        seq_emb = self.final_layernorm(seq_emb)

        pos_emb = self.item_emb(pos)
        neg_emb = self.item_emb(neg)

        pos_logits = (pos_emb * seq_emb).sum(dim=-1)
        neg_logits = (neg_emb * seq_emb).sum(dim=-1)

        istarget = (pos!=0).float()

        loss = -torch.sum(torch.log(torch.sigmoid(pos_logits) + 1e-18) * istarget + torch.log(1 - torch.sigmoid(neg_logits) + 1e-18) * istarget) / torch.sum(istarget)

        return loss





# if __name__ == "__main__":
#     model = MultiHeadAttention(8,32,32,0.2)
#     x = torch.randn(size=(1024,50,32))

#     x2 = torch.randn(size=(1024,50))
#     # model2 = SASRec()
    
#     print(model(x,x).shape)


