import tensorflow as tf
import importlib
import sys
sys.path.append('./')


## TODO: 奖励模型封装
class RewardModel(tf.keras.Model):
    def __init__(self,):
        super(RewardModel, self).__init__()


class SequentialRewardModel(RewardModel):
    def __init__(self,
                 module : str,
                 backbone : str,
                 model_config : dict,
                 is_critic : bool = False):
        super(SequentialRewardModel, self).__init__()

        self.is_critic = is_critic
        sequential_model_class = getattr(importlib.import_module(f"model.{module}"), backbone)
        self.sequential_model = sequential_model_class(**model_config)
    

    def build(self, inputs_shape):
        self.seq_len = inputs_shape[1]
        self.reward_head = tf.keras.layers.Dense(units=self.seq_len if self.is_critic else 1, 
                                                 activation='sigmoid', 
                                                 use_bias=True)


    def call(self, inputs):
        '''
            inputs: [b, seq_len, dim]
        '''
        seq_out = self.sequential_model(inputs)
        reward = self.reward_head(seq_out)

        return reward
    

    def generate_reward(self, labels, gamma=0.9):
        '''
            labels: [n, ]
            gamma: 衰减因子
        '''
        n_labels = len(labels)
        reward = tf.zeros((n_labels, self.seq_len if self.is_critic else 1))
        
        reward_size = reward.shape[1]
        reward = tf.tensor_scatter_nd_update(
            tensor=reward,
            indices=tf.stack([tf.range(n_labels), [reward_size-1]*n_labels], axis=-1),
            updates=labels
        )   # [n, reward_size]
        
        for i in reversed(range(reward_size-1)):
            reward = tf.tensor_scatter_nd_update(
                tensor=reward,
                indices=tf.stack([tf.range(n_labels), [i] * reward_size], axis=-1),
                updates=reward[:, i+1] * gamma
            )
        
        return reward
    
    
    def cal_reward_loss(self, pred_reward, labels):
        '''
            pred_reward: [n, reward_size]
            labels: [n, ]
        '''

        true_reward = self.generate_reward(labels, gamma=0.9)
        return tf.losses.MSE(true_reward, pred_reward)
    


if __name__ == "__main__":
    model_config = {
        "hidden_units":[32],
        "activations" : [None],
        "n_heads":4,
        "n_dims":32
    }
    sequential_reward_model = SequentialRewardModel("hstu_noaction", 'AttentionNet', model_config, True)
    inputs = tf.random.uniform((8, 4, 32))
    print(sequential_reward_model(inputs))
    print(sequential_reward_model.generate_reward([1,1,0,0], gamma=0.9))
    