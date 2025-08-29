import torch
import torch.nn as nn
import torch.nn.functional as F

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

        print(q_.shape, k_.shape)

        # multiplication
        output = torch.einsum("bij, bjk->bik", q_, k_.transpose(1,2)) # [batch, q_n, k_n]
        # scale
        output = output / self.depth ** 0.5

        # key mask
        key_mask = torch.sign(torch.abs(torch.sum(key, dim=-1))) # [batch, k_n]
        key_mask = torch.tile(key_mask, [self.num_heads, 1]) # [batch*num_heads, k_n]
        key_mask = torch.tile(key_mask.unsqueeze(1), [1, query.shape[1], 1])  # [batch*num_heads, q_n, k_n]

        paddings = torch.ones_like(output) * (-(2**32)+1)

        print(key_mask.shape, paddings.shape, output.shape)

        output = torch.where(key_mask == 0, paddings, output)

        # future blinding(causality)
        diag_vals = torch.ones_like(output[0,:,:]) # [q_n, k_n]
        tril = torch.tril(diag_vals).to_dense() # [q_n, k_n]

        masks = torch.tile(tril.unsqueeze(0), [output.shape[0], 1, 1]) # [batch*num_heads, q_n, k_n]

        paddings = torch.ones_like(masks) * (-(2**32) + 1)
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
        output = torch.concat(torch.chunk(output, self.num_heads, dim=0), dim=-1) # [batch, q_n, dim]
        # chunk 按块分
        # split 按每一块的大小分

        # residual connection
        output = output + query

        # normalization
        output = F.normalize(output, dim=-1)

        return output

class FeedForward(nn.Module):
    def __init__(self, attn_dim=32, filter_dim=[32,32], dropout=0.2, is_training=True, reuse=None):
        super(FeedForward, self).__init__()

        # self.feed_layer = nn.Sequential(
        #     nn.Conv1d(attn_dim, filter_dim[0], kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv1d(filter_dim[0], , kernel_size=1)
        # )

        self.feed_layer = nn.Sequential()
        in_dim = attn_dim
        for idx, f_dim in enumerate(filter_dim[:-1]):
            out_dim = f_dim
            self.feed_layer.add_module('Conv_{}'.format(idx), nn.Conv1d(in_dim, out_dim, kernel_size=1))
            self.feed_layer.add_module('ACT_{}'.format(idx), nn.ReLU())
            in_dim = filter_dim[idx]

        self.add_module('Conv_{}'.format(len(filter_dim)-1), nn.Conv1d(in_dim, filter_dim[-1], kernel_size=1))

    
    def forward(self, inputs):
        return self.feed_layer(inputs)