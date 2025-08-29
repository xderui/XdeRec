import tensorflow as tf

class Tools:
    def __init__(self):
        pass

    
    @staticmethod
    def retrive_topk_embeddings(search_embeddings, target_embedding, topk=10, is_norm=False):
        if not is_norm:
            search_embeddings = tf.nn.l2_normalize(search_embeddings, axis=-1)
            target_embedding = tf.nn.l2_normalize(target_embedding, axis=-1)

        topk_result = tf.math.top_k(
            tf.reduce_sum(target_embedding * search_embeddings, axis=-1),
            k=topk
        )
        
        # print(topk_result.indices)
        indices = topk_result.indices
        topk_embeddings = tf.gather(
            search_embeddings,
            indices=tf.reshape(indices, [-1])
        )

        topk_embeddings = tf.reshape(topk_embeddings, (*indices.shape, -1))
        
        return indices, topk_embeddings