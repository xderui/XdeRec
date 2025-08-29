import tensorflow as tf


def InfoNCE_HSTU(src, pos, neg, temperature=1.0):
    """ InfoNCE

    Args:
        src (tf.Tensor): (batch, seq_len, dims)
        pos (tf.Tensor): (batch, seq_len, dims)
        neg (tf.Tensor): (batch, seq_len, neg_num, dims)
    """
    
    src = tf.nn.l2_normalize(src, axis=-1)
    pos = tf.nn.l2_normalize(pos, axis=-1)
    neg = tf.nn.l2_normalize(neg, axis=-1)
    
    pos_scores = tf.exp(tf.reduce_sum(src * pos, axis=-1) / temperature) # (batch, seq_len)
    neg_scores = tf.reduce_sum(tf.exp(tf.reduce_sum(tf.expand_dims(src, axis=-2) * neg, axis=-1) / temperature), axis=-1)
    
    return -1 * tf.math.log(pos_scores / neg_scores)


def InfoNce_HSTU_inbatch(src, pos, temperature=1.0):
    
    src = tf.nn.l2_normalize(src, axis=-1)
    pos = tf.nn.l2_normalize(pos, axis=-1)
    
    pos_scores = tf.exp(tf.reduce_sum(src * pos, axis=-1))
    neg_scores = tf.reduce_sum(tf.exp(tf.einsum("nd,md->nm", src, pos)), axis=-1) # [b, n]

    return -1 * tf.reduce_mean(tf.math.log(pos_scores / neg_scores))



def cross_entropy_(logits, labels):
    '''
        logits: [batch_size, num_classes], 未经softmax处理
        labels: [batch_size]
    '''
    
    labels = tf.cast(labels, tf.int32)
    logits = tf.gather(tf.nn.softmax(logits, axis=-1), indices=labels, batch_dims=1)
    return -tf.reduce_mean(tf.math.log(logits))
    

def cross_entropy(logits, labels):
    return tf.losses.sparse_softmax_cross_entropy(labels, logits)


def sampled_cross_entropy(last_hidden_states, class_matrix_weight, class_matrix_bias, labels, num_classes):
    '''
        last_hidden_states: (b, seq_len, dims)
        labels: (b, seq_len,)
    '''
    
    
    return tf.reduce_mean(tf.nn.sampled_softmax_loss(
        weights=tf.transpose(class_matrix_weight, (1,0)),
        biases=class_matrix_bias,
        labels=labels,
        inputs=last_hidden_states,
        num_classes=num_classes,
        num_sampled=1024
    ))


def dpo_loss(pred_logits, chosen_logits, rejected_logits):
    '''
        pred_logits, chosen_logits, rejected_logits: [b, sampled_seq_len, dim]
    '''

    pairwise_loss = tf.nn.softplus(tf.reduce_sum(pred_logits * chosen_logits, axis=-1)) + \
                    tf.nn.softplus(-1 * tf.reduce_sum(pred_logits * rejected_logits, axis=-1))
    # 等价于以下logsigmoid的形式
    # pos_scores = tf.reduce_sum(pred_logits * chosen_logits, axis=-1)
    # neg_scores = tf.reduce_sum(pred_logits * rejected_logits, axis=-1)
    # pairwise_loss = -tf.math.log_sigmoid(pos_scores) - tf.math.log_sigmoid(-1 * neg_scores)
    return pairwise_loss


def grpo_loss(pred_logits, ref_logits, ref_reward, ref_embedding):
    '''
        pred_logits: (b, seq_len, dim)
        ref_logits: (b, group_size, seq_len, dim)
        ref_reward: (b, group_size, seq_len)
        ref_index: (b, group_size, seq_len, dim)    每一个位置实际检索选取的embedding
    '''

    reward_mean = tf.reduce_mean(ref_reward, axis=-2, keepdims=True)   # [b, 1, seq_len]
    reward_std = tf.math.reduce_std(ref_reward, axis=-2, keepdims=True)    # [b, 1, seq_len]
    advantages = tf.divide(tf.subtract(ref_reward, reward_mean), reward_std)    # [b, g, seq_len]

    # 由于next token是根据输出的隐藏层状态进行ANN检索的，所以需要这里使用token相似度作为prob
    pred_logits = tf.expand_dims(pred_logits, axis=1)
    pred_prob = tf.reduce_sum(pred_logits * ref_embedding, axis=-1)
    ref_prob = tf.reduce_sum(ref_logits * ref_embedding, axis=-1)

    new_old_ratio = tf.divide(pred_prob, ref_prob)
    part1_loss = tf.reduce_mean(new_old_ratio * advantages)     # important-sampling advantages
    part2_loss = tf.reduce_mean((new_old_ratio - 1) - tf.math.log(new_old_ratio))   # kl

    return part1_loss + part2_loss