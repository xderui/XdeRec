import tensorflow as tf
from typing_extensions import Optional


class Embedding(tf.keras.Model):
    def __init__(self,
                 embed_num : int,
                 embed_dim : int):
        super(Embedding, self).__init__()

        self.embed_num = embed_num
        self.embed_dim = embed_dim
        
        self.embeddings = self.add_weight(
            shape=(embed_num, embed_dim),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
            trainable=True,
        )


    def compute_distances(self, inputs):
        '''
            @ brief: 计算inputs和所有embedding之间的距离
            @ param: 
                inputs: [n, embed_dim]
        '''

        embedding_t = tf.transpose(self.embeddings)
        inputs_pow = tf.reduce_sum(tf.pow(inputs, 2), axis=-1, keepdims=True)     # [b, 1]
        embedding_t_pow = tf.reduce_sum(tf.pow(embedding_t, 2), axis=0, keepdims=True)    # [1, b]
        distance_matrix = inputs_pow + embedding_t_pow - 2 * inputs @ embedding_t

        return distance_matrix
    

    def find_nearest_embedding(self, inputs):
        '''
            @ brief: 找到每个向量最近的向量
            @ param:
                inputs: [n, embed_dim]
        '''

        distance_matrix = self.compute_distances(inputs)
        nearest_indices = tf.argmin(distance_matrix, axis=-1)

        return nearest_indices, tf.gather(params=self.embeddings, indices=nearest_indices)
    
        
    def call(self, indices):
        return tf.gather(self.embeddings, indices)
    

class VQEmbedding(Embedding):
    '''
        @ brief: 和一般的嵌入表相比，VQ嵌入表需要迭代更新
    '''
    def __init__(self,
                 embed_num : int,
                 embed_dim : int,
                 ema_decay : float = 0.9,
                 warm_codebook : bool = True):
        super(VQEmbedding, self).__init__(embed_num, embed_dim)

        self.ema_decay = ema_decay
        self.warm_codebook = warm_codebook
        self.cluster_size_ = tf.zeros(shape=(self.embed_num))
        self.cluster_vectors_ = tf.identity(self.embeddings)

    
    def tile_vectors_with_noise(self, inputs, tile_size):
        n_inputs, embed_dim = inputs.shape[0]
        repeat_num = tf.math.ceil(tile_size / n_inputs)
        
        tile_vectors = tf.repeat(inputs, repeats=repeat_num, axis=0)
        return tile_vectors + tf.random.uniform(shape=tile_vectors.shape) * 0.01 / (embed_dim ** 0.5)
    
    
    def update_buffers(self, vectors, indexes):
        '''
            vectors: [n, embed_dim], 被分配的向量组
            indexes: [n]，每个向量选择的嵌入索引
        '''

        n_vectors = vectors.shape[0]
        scatter_rows = indexes
        scatter_cols = tf.range(n_vectors, dtype=tf.int64)
        scatter_indexes = tf.stack([scatter_rows, scatter_cols], axis=-1)   # [n, 2]
        cluster_one_hot_mask = tf.tensor_scatter_nd_update( 
            tensor = tf.zeros((self.embed_num, n_vectors)),
            indices = scatter_indexes,
            updates = tf.ones((n_vectors,))
        )   # [embed_num, n]

        clustered_size = tf.reduce_sum(cluster_one_hot_mask, axis=-1)  # [embed_num]
        clustered_vectors = cluster_one_hot_mask @ vectors  # [embed_num, embed_dim]

        self.cluster_size_ = tf.add(self.ema_decay * self.cluster_size_, (1 - self.ema_decay) * clustered_size)
        self.cluster_vectors_ = tf.add(self.ema_decay * self.cluster_vectors_, (1 - self.ema_decay) * clustered_vectors)

        if self.warm_codebook:
            if n_vectors < self.embed_num:  # 保证能分配到嵌入表中
                vectors = self.tile_vectors_with_noise(vectors, self.embed_num)

            vectors_shuffle = tf.random.shuffle(vectors)    # 随机打乱，然后取embed_num赋值给未被更新到的code
            
            is_updated = tf.expand_dims(tf.cast(self.cluster_size_ > 0, tf.float32), axis=-1)
            self.cluster_vectors_ = tf.multiply(is_updated, self.cluster_vectors_) + tf.multiply(1 - is_updated, vectors_shuffle[: self.embed_num])
            is_updated = tf.squeeze(is_updated)
            self.cluster_size_ = is_updated * self.cluster_size_ + (1 - is_updated) * tf.ones_like(self.cluster_size_)

    
    def update_embeddings(self):
        self.embeddings = self.cluster_vectors_ / tf.expand_dims(self.cluster_size_, axis=-1)


    def call(self, inputs, training=True):
        code, quant = self.find_nearest_embedding(inputs)
        if training:
            self.update_buffers(inputs, code)
            self.update_embeddings()
        
        return code, quant
        
    
if __name__ == "__main__":
    embedding_layer = Embedding(embed_num=128, embed_dim=64)
    print(embedding_layer(tf.constant([0,1,8,7,2])))
    