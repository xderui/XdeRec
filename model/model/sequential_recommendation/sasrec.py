from model.sequential_recommendation.BaseModel import BaseModel
import torch.nn as nn
import torch
from torch.nn import functional as F

class SASRec(BaseModel):
    def __init__(self):
        super(SASRec, self).__init__()



class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, input_dim, attention_dim, dropout_rate):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.dropout_rate = dropout_rate
        
        assert attention_dim % num_heads == 0

        self.depth = attention_dim // num_heads

        self.Q_linear = nn.Linear(input_dim, attention_dim, bias=False)
        self.K_linear = nn.Linear(input_dim, attention_dim, bias=False)
        self.V_linear = nn.Linear(input_dim, attention_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key):
        # linear projection
        q = self.Q_linear(query)   # [batch, n, dim]
        k = self.K_linear(key)
        v = self.V_linear(key)

        # multi-head
        q_ = q.reshape(-1, query.shape[1], self.depth)
        k_ = k.reshape(-1, key.shape[1], self.depth)
        v_ = v.reshape(-1, key.shape[1], self.depth)

        # multiplication
        output = torch.einsum("bij, bji->bii", q_, k_.transpose(1,2)) # [batch, q_n, k_n]
        # scale
        output = output / torch.sqrt(self.depth)

        # key mask
        key_mask = torch.sign(torch.abs(torch.sum(key, dim=-1))) # [batch, k_n]
        key_mask = torch.tile(key_mask, [self.num_heads, 1]) # [batch*num_heads, k_n]
        key_mask = torch.tile(key_mask.unsqueeze(1), [1, query.shape[1], 1])  # [batch*num_heads, q_n, k_n]

        paddings = torch.ones_like(output) * (-(2**32)+1)

        output = torch.where(key_mask == 0, paddings, output)

        # future blinding(causality)
        diag_vals = torch.ones_like(output[0,:,:]) # [q_n, k_n]
        tril = torch.tril(diag_vals).to_dense() # [q_n, k_n]

        masks = torch.tile(tril.unsqueeze(1), [output.shape[0], 1, 1]) # [batch*num_heads, q_n, k_n]

        padings = torch.ones_like(masks) * (-(2**32) + 1)
        output = torch.where(masks == 0, paddings, output)

        # activation
        output = F.softmax(output, dim=-1)
        query_mask = torch.sign(torch.abs(torch.sum(query, dim=-1))) # [batch, q_n]
        query_mask = torch.tile(query_mask, [self.num_heads, 1]) # [batch*num_heads, q_n]
        query_mask = torch.tile(query_mask.unsqueeze(1), [1, key.shape[1], 1])  # [batch*num_heads, k_n, q_n]

        output = query_mask * output

        # dropout
        output = self.dropout(output)

        # weighted sum
        output = torch.matmul(output, v_)

        # concat multi-head
        output = torch.concat(torch.split(output, self.num_heads, dim=0), dim=2) # [batch, q_n, dim]

        # residual connection
        output = output + query

        return output
