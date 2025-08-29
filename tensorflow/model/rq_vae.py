
import tensorflow as tf
from embedding import VQEmbedding
import random

class RQVAE(tf.keras.Model):
    def __init__(self,
                 codebook_num = 4,
                 codebook_size = 128,
                 codebook_dim = 32,
                 k_means : bool = False):
        super(RQVAE, self).__init__()
        
        self.codebook_num = codebook_num
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.codebooks = [VQEmbedding(codebook_size, codebook_dim) for _ in range(codebook_num)]

        self.quantize_list = None
        self.code_list = None


    def k_means(self, inputs, k):
        '''
            inputs: 向量集合
            k: 簇中心数量
        '''

        points_size = inputs.shape[0]
        cluster_points = inputs[random.sample(list(range(points_size)), k)]
        cluster_points = tf.transpose(cluster_points, perm=(0, 2, 1))
        
        
        distances = tf.reduce_sum(tf.pow(inputs, 2), axis=-1) - \
                    tf.matmul(inputs, cluster_points) + \
                    tf.reduce_sum(tf.pow())
        
        ## TODO: k-means聚类

    def quantize(self, inputs):
        residual_embedding = inputs
        code_list = []
        quantize_list = []
        quantize_embedding = tf.ones_like(inputs)
        for codebook in self.codebooks:
            code, quant = codebook(residual_embedding)   # code(index), quant(embedding)
            quantize_embedding = tf.add(quantize_embedding, quant)
            residual_embedding = tf.subtract(residual_embedding, quant)
            code_list.append(code)
            quantize_list.append(tf.identity(quantize_embedding))
                
        return quantize_list, code_list
        

    def call(self, inputs, return_loss=False):
        self.quantize_list, self.code_list = self.quantize(inputs)
        
        loss = 0.0
        if return_loss:
            loss = self.cal_quantization_loss_detach(inputs, self.quantize_list)

        final_quantize_embedding = inputs + tf.stop_gradient(self.quantize_list[-1] - inputs)

        return final_quantize_embedding, loss   # 量化之后的embedding
    

    def cal_quantization_loss_detach(self, inputs, quantize_list):
        partial_loss_list = []
        for quantize_embedding in quantize_list:
            partial_loss = tf.reduce_mean(tf.pow(tf.stop_gradient(inputs - quantize_embedding), 2))
            partial_loss_list.append(partial_loss)

        return tf.reduce_mean(partial_loss_list)
    

if __name__ == "__main__":
    rq_vae_model = RQVAE()
    inputs = tf.random.uniform(shape=(512, 32))

    # test
    for i in range(10):
        f, loss = rq_vae_model(inputs, return_loss=True)
        print(loss)
    
    print(rq_vae_model.code_list)