import tensorflow as tf
from typing_extensions import Optional
import tensorflow.compat.v1 as tf1
import random
import numpy as np
from utils.tools import Tools
from model.embedding import Embedding


class DNN(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_units : list,
                 activations : Optional[list],
                 use_ln : bool = False,
                 dropout_rate : Optional[float] = None):
        super(DNN, self).__init__()
        
        self.hidden_units = hidden_units
        self.activations = activations
        self.activations_num = len(activations)
        self.use_ln = use_ln
        self.dropout_rate = dropout_rate
        
        self.layers = []
        for i,unit in enumerate(self.hidden_units):
            
            if self.use_ln:
                gamma_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.3)
                beta_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.0001)
                
                ln_layer = tf.keras.layers.LayerNormalization(
                    gamma_initializer=gamma_initializer, beta_initializer=beta_initializer
                )
            
                self.layers.append(ln_layer)
            
            self.layers.append(
                tf.keras.layers.Dense(
                    units=unit,
                    activation=activations[i] if i < self.activations_num else None,
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros"
                )
            )
    

    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        
        return x


class RelativeBiasGenerator(tf.keras.layers.Layer):
    def __init__(self,
                 n_dims : int = 32,
                 max_len : int = 1024):
        
        super(RelativeBiasGenerator, self).__init__()
        self.n_dims = n_dims
        self.max_len = max_len
        
        self.pos_bias_w = self.add_weight(
            name='pos_bias_weight',
            shape=(2*self.max_len-1, ),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            trainable=True
        )
    

    def generate_position_bias(self, seq_len, offset=0):
        assert offset + seq_len <= self.max_len, f"relative position only support from -{self.max_len-1} to {self.max_len}"
        N = self.max_len
        rel_pos_emb = self.pos_bias_w
        # rel_pos_emb = self.pos_bias_w[self.max_len - N :]
        # if N < self.max_len:
        #     rel_pos_emb = rel_pos_emb[: N - self.max_len]
        rel_pos_emb = tf.pad(rel_pos_emb, paddings=[[0, N]]) # [2*N-1+N]
        rel_pos_emb = tf.tile(rel_pos_emb, [N]) # [N*(3*N-1)]
        rel_pos_emb = tf.reshape(rel_pos_emb[:-N], (N, 3*N-2))
        rel_pos_emb = rel_pos_emb[offset:offset+seq_len, N-1 : -(N-1)]
        
        return rel_pos_emb


    def call(self, inputs, seq_len, offset=0, bias_type : Optional[str] = None):
        if bias_type == 'position':
            return self.generate_position_bias(seq_len, offset)
        else:
            return self.generate_position_bias(seq_len, offset)
        

class AttentionNet(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_units : list,
                 activations : list,
                 n_heads : int = 4,
                 n_dims : int = 32,
                 ):
        super(AttentionNet, self).__init__()
        
        self.hidden_units = hidden_units
        self.activations = activations
        self.n_heads = n_heads
        self.n_dims = n_dims
        
        self.q_proj = DNN(hidden_units=[self.n_dims], activations=[None])
        self.k_proj = DNN(hidden_units=[self.n_dims], activations=[None])
        self.v_proj = DNN(hidden_units=[self.n_dims], activations=[None])

        gamma_initializer = tf1.truncated_normal_initializer(mean=0.0, stddev=0.3)
        beta_initializer = tf1.truncated_normal_initializer(mean=0.0, stddev=0.0001)
        self.layer_norm = tf.keras.layers.LayerNormalization(
            gamma_initializer=gamma_initializer,
            beta_initializer=beta_initializer
        )
        
        self.ffn = DNN(hidden_units=self.hidden_units, activations=self.activations)
        

    def build(self, input_shapes):
        self.max_len = input_shapes[1]

        
    def split_heads(self, x):
        self.head_dims = self.n_dims // self.n_heads
        return tf.transpose(tf.reshape(x, (-1, self.max_len, self.n_heads, self.head_dims)), (0, 2, 1, 3))

    
    def call(self, x, attn_mask=None):
        # self attention
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.k_proj(x)
        
        # multi-head
        q, k, v = [self.split_heads(_) for _ in [q, k, v]]
        
        # attn scores
        attn_scores = tf.nn.softmax(tf.einsum(
            "bhnd,bhmd->bhnm", q, k
        ) / tf.math.sqrt(tf.constant(self.head_dims, dtype=tf.float32)), axis=-1)
        
        if attn_mask is not None:
            attn_scores *= attn_mask
        
        x = tf.einsum("bhij,bhjk->bhik", attn_scores, v)
        x = self.layer_norm(x + q)
        x = tf.reshape(tf.transpose(x, (0, 2, 1, 3)), (-1, self.max_len, self.n_dims)) # [b, max_len, dims]
        
        output = self.ffn(x) + x
        
        return output
        

class HSTUBlock(tf.keras.layers.Layer):
    def __init__(self,
                 n_heads : Optional[int] = None,
                 n_dims : int = 32,
                 hidden_units : list = [],
                 activations : list = [],
                 u_enable : bool = True,
                 u_nums : Optional[int] = 2):
        super(HSTUBlock, self).__init__()
        
        self.n_heads = n_heads
        self.n_dims = n_dims
        self.head_dims = n_dims // n_heads
        self.hidden_units = hidden_units
        self.activations = activations
        self.u_enable = u_enable
        self.u_nums = u_nums
        
        self.u_proj = None
        
        if self.u_enable:
            self.u_proj = tf.keras.layers.Dense(
                units=u_nums * n_dims,
                activation=None,
                kernel_initializer='glorot_uniform',
                bias_initializer="zeros"
            )
        
        # q,k,v
        self.q_proj = DNN(hidden_units=[self.n_dims], activations=self.activations)
        self.k_proj = DNN(hidden_units=[self.n_dims], activations=self.activations)
        self.v_proj = DNN(hidden_units=[self.n_dims], activations=self.activations)
        
        gamma_initializer = tf1.truncated_normal_initializer(mean=0.0, stddev=0.3)
        beta_initializer = tf1.truncated_normal_initializer(mean=0.0, stddev=0.0001)
        self.ln_layer = tf.keras.layers.LayerNormalization(
            gamma_initializer=gamma_initializer,
            beta_initializer=beta_initializer
        )
        
        self.ffn = tf.keras.layers.Dense(
            units=n_dims,
            activation='relu',
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros"
        )
        
        
    def build(self, input_shapes):
        # bias
        self.max_len = input_shapes[1]
        self.relativeBiasGenerator = RelativeBiasGenerator(self.n_dims, self.max_len)
        
        
    def split_heads(self, x):
        return tf.transpose(
            tf.reshape(x, (-1, x.shape[1], self.n_heads, self.head_dims)),
            (0, 2, 1, 3)
        )
    

    def cal_attention_output(self, inputs, q, k, v, u, valid_attn_mask=None):
        q_seq_len = q.shape[-2]
        k_seq_len = k.shape[-2]

        if valid_attn_mask is None:   # 默认是full bi-attention
            valid_attn_mask = tf.linalg.band_part(tf.ones((q_seq_len, k_seq_len)), -1, 0)
            valid_attn_mask = tf.reshape(valid_attn_mask, (1, 1, *valid_attn_mask.shape))


        # position embeddings
        self.posBiasAttentionScores = self.relativeBiasGenerator(None, q_seq_len, offset=k_seq_len-q_seq_len)

        # relativeBias
        pos_attn_output = tf.einsum(
            "nm,bmd->bnd", self.posBiasAttentionScores, v
        )

        # multi-head attention
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        attn_scores = tf.einsum("bhnd,bhmd->bhnm", q, k) / tf.math.sqrt(float(self.head_dims))
        # attn_scores *= tf.where(tf.equal(valid_attn_mask, True), attn_scores, -np.inf)    # tf-macos np.inf有bug
        attn_scores *= tf.where(tf.equal(valid_attn_mask, tf.constant(1.0)), attn_scores, 0)  # silu

        self_attn_output = tf.einsum(
            "bhnm,bhmd->bhnd",
            tf.nn.silu(attn_scores), v
        )

        self_attn_output = tf.reshape(
            tf.transpose(self_attn_output, (0, 2, 1, 3)),
            (-1, q_seq_len, self.n_dims)
        )
        
        attn_output_list = [self_attn_output, pos_attn_output]
        attn_output = tf.concat(attn_output_list, axis=-1)  # [b, n, 2d]
        attn_output = self.ln_layer(attn_output)  # [b, max_len, n_dims * 2]
        
        output = self.ffn(u * attn_output) + inputs

        return output
    

    def call(self, inputs, attn_mask=None, kv_caches=False):
        x = inputs
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        kv_caches_dict = {}
        if kv_caches:
            kv_caches_dict = {"k": k, "v": v}
        
        u = tf.constant(1)
        if self.u_enable:
            u = self.u_proj(x)

        output = self.cal_attention_output(inputs, q, k, v, u, attn_mask)
                
        return output, kv_caches_dict
    
        
class HSTU(tf.keras.Model):
    def __init__(self,
                 n_layers : int = 3,
                 n_heads : Optional[int] = None,
                 n_dims : int = 32,
                 hidden_units : list = [],
                 activations : list = [],
                 u_enable : bool = True,
                 u_nums : Optional[int] = 2,
                 save_hidden_states : bool = False,
                 ):
        assert n_dims % n_heads == 0, "n_dims must be divided by n_heads"
            
        super(HSTU, self).__init__()
        
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_dims = n_dims
        self.hidden_units = hidden_units
        self.activations = activations
        self.u_enable = u_enable
        self.u_nums = u_nums
        self.save_hidden_states = save_hidden_states
        
        self.hstu_block = []
        for i in range(n_layers):
            self.hstu_block.append(
                HSTUBlock(
                    self.n_heads,
                    self.n_dims,
                    self.hidden_units,
                    self.activations,
                    self.u_enable,
                    u_nums=2
                )
            )
        
        self.hidden_states = None
        
    
    def build(self, input_shapes):
        self.max_len = input_shapes[1]
        
    
    def get_hidden_states(self):
        assert self.save_hidden_states, "Please set `save_hidden_states=True` in HSTU."
        assert self.hidden_states, "Please forward at least once."
        return self.hidden_states
        
            
    def call(self, inputs, kv_caches=False, training=True):
        # inputs: (batch, max_len, dims)
        x = inputs
        if self.save_hidden_states:
            self.hidden_states = []
        if kv_caches:
            self.kv_caches_list = []
        
        for i, hstu_layer in enumerate(self.hstu_block):
            x, kv_caches_dict = hstu_layer(x, attn_mask=None, kv_caches=kv_caches)
            if self.save_hidden_states:
                self.hidden_states.append(x)
            if kv_caches:
                self.kv_caches_list.append(kv_caches_dict)
        
        if self.save_hidden_states and training:
            self.hidden_states = tf.stack(self.hidden_states, axis=2)

        if kv_caches:
            return x, self.kv_caches_list

        return x
    

    @tf.function(autograph=False)
    def predict(self, inputs):
        return self(inputs, training=False)
    

    @tf.function(autograph=False)
    def infer(self, inputs, infer_steps, search_embeddings):
        '''
            inputs: (b, N, dims), N <= max_len
        '''
        results_token_id = []

        output, kv_caches_list = self(inputs, kv_caches=True, training=False)
        last_token_output = tf.expand_dims(output[:, -1, :], axis=1)

        for infer_step in range(infer_steps):
            token_id, token_embeddings = Tools.retrive_topk_embeddings(
                search_embeddings,
                last_token_output,
                topk=1,
                is_norm=False
            )
            results_token_id.append(token_id)
            last_token_output = token_embeddings
            if infer_step == infer_steps - 1:
                break

            for layer_id, hstu_layer in enumerate(self.hstu_block):
                cur_u = hstu_layer.u_proj(last_token_output)
                cur_q = hstu_layer.q_proj(last_token_output)
                cur_k = hstu_layer.k_proj(last_token_output)
                cur_v = hstu_layer.v_proj(last_token_output)
                prev_k = kv_caches_list[layer_id]["k"][:, -self.max_len+1:]
                prev_v = kv_caches_list[layer_id]["v"][:, -self.max_len+1:]
                cur_k = tf.concat([prev_k, cur_k], axis=1)
                cur_v = tf.concat([prev_v, cur_v], axis=1)
                kv_caches_list[layer_id]["k"] = cur_k
                kv_caches_list[layer_id]["v"] = cur_v
                layer_output = hstu_layer.cal_attention_output(last_token_output, cur_q, cur_k, cur_v, cur_u)
                last_token_output = tf.expand_dims(layer_output[:, -1, :], axis=1)
        
        return results_token_id
            

    def cal_loss(self, 
                model_output, 
                target_sequence_id, 
                target_sequence_len,
                all_embeddings,
                neg_num : int = 10):
        '''
            model_output: [b, seq_len, dim]
            target_sequence_id: [b, seq_len]
            target_sequence_len: [b]
            all_embeddings: [N, dim] - 用于负采样
        '''

        model_output_flat = tf.reshape(model_output, (-1, tf.shape(model_output)[-1])) # [b*seq_len, dim]
        target_sequence_flat = tf.reshape(target_sequence_id, (-1, 1))  # [b*seq_len, 1]
        
        rank_index = tf.range(model_output.shape[1])
        batch_seq_len = tf.reshape(target_sequence_len, (-1 ,1))   # [b, 1]
        model_output_flat_mask = tf.reshape(rank_index < batch_seq_len-1, (-1))
        target_flat_mask = tf.reshape((rank_index >= 1) & (rank_index < batch_seq_len), (-1))
        
        valid_model_output = tf.boolean_mask(model_output_flat, model_output_flat_mask)
        valid_target_embeddings = tf.boolean_mask(model_output_flat, target_flat_mask)
        valid_pos_num = valid_model_output.shape[0]

        # 对每一个正样本进行负采样
        neg_index = random.sample(range(all_embeddings.shape[0]), valid_pos_num * neg_num)
        neg_sample_embeddings = tf.gather(all_embeddings, neg_index)
        neg_sample_embeddings = tf.reshape(neg_sample_embeddings, (-1, neg_num, all_embeddings.shape[-1]))

        valid_model_output = tf.expand_dims(valid_model_output, axis=1)
        valid_target_embeddings = tf.expand_dims(valid_target_embeddings, axis=1)
        pos_scores = tf.reduce_sum(tf.multiply(valid_model_output, valid_target_embeddings), axis=-1)
        neg_scores = tf.reduce_sum(tf.multiply(valid_model_output, neg_sample_embeddings), axis=-1)
        
        return -1 * tf.reduce_mean(tf.nn.softplus(pos_scores - neg_scores))
    

def test(case_id):
    if case_id == 0:
        attn_network = AttentionNet([32], [None], n_dims=32)
        position_generator = RelativeBiasGenerator(max_len=50)
        
        inputs = tf.random.uniform((32,20,32))
        attn_output = attn_network(inputs)
        print(attn_output.shape)
        print(position_generator(attn_output, seq_len=1, offset=49))
        print(position_generator.pos_bias_w[:50])

    if case_id == 1:
        hstuModel = HSTU(n_heads=4, n_dims=32, hidden_units=[32], activations=[None])
        test_embeddings = Embedding(100, 32)
        inputs = tf.random.uniform((32,20,32))

        print(hstuModel.infer(inputs, 50, test_embeddings.weights[0]))

if __name__ == "__main__":
    test(case_id=1)
    

    

    

