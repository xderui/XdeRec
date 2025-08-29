import tensorflow as tf
from data import *
from model import *
import importlib
import datetime
import yaml
from evaluator import Evaluator
import os


class Trainer:
    def __init__(self,
                 *args,
                 **kwargs
                 ):
        print("using Trainer.")

    def __new__(cls, *args, **kwargs):
        if cls is Trainer:
            config_path = args[0] if args else ""
            assert config_path != "", "config_path is empty"
        
            config_dict = yaml.load(
                open(config_path, 'r'), Loader=yaml.FullLoader
            )

            trainer_type = config_dict.get("type", "default")
            trainer_type = config_dict['type'] if 'type' in config_dict else "default"
            trainer_class = {
                "sequential": SequentialTrainer,
                "default": Trainer
            }.get(trainer_type, Trainer)

            print(trainer_class)
            instance = super().__new__(trainer_class)
            print(config_dict)
            # instance.__init__(
            #     data_config=config_dict.get("data_config", {}),
            #     model_config=config_dict.get("model_config", {}),
            #     training_config=config_dict.get("training_config", {})
            # )
            instance.__init__(**config_dict['trainer_config'])

            return instance

        return super().__new__(cls)
        

class SequentialTrainer:
    def __init__(self,
                 data_config : dict,
                 model_config : dict,
                 training_config : dict,
                 **kwargs
                 ):
        self.data_config = data_config

        self.model_name = model_config.get("model_name", "")
        self.model_param = model_config.get("model_param", {})

        self.epoches = training_config.get('epoches', 1)
        self.log_dir = training_config.get('log_dir', "./logs")
        self.checkpoint_dir = training_config.get("checkpoint_dir", "")
        self.optimizer_name = training_config.get('optimizer', "Adam")
        self.optimizer_param = training_config.get("optimizer_param", {})

        # 载入数据
        self.seq_data = Dataset(**self.data_config)
        self.train_data_generator = self.seq_data.build_dataGenerator(self.data_config.get("batch_size", 32), mode='test')
        self.test_data_generator = self.seq_data.build_dataGenerator(self.data_config.get("batch_size", 32), mode='test')
        
        # 定义嵌入层
        self.user_num = self.seq_data.count_unique_key()
        self.ad_num = self.seq_data.count_unique_values()
        
        self.user_embeddings = Embedding(embed_num=self.user_num,
                                       embed_dim=self.model_param.get("n_dim", 32))
        self.ad_embeddings = Embedding(embed_num=self.ad_num,
                                       embed_dim=self.model_param.get("n_dim", 32))
        
        # 定义优化器
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        # self.optimizer = getattr(importlib.import_module("tensorflow.keras.optimizers"), 
        #                          self.optimizer_name)(learning_rate=self.optimizer_param.get('learning_rate', 1e-3))
        
        # 定义模型
        self.model = getattr(importlib.import_module("model"), 
                             self.model_name)(**self.model_param)
        
        # log
        self.tfboard_writer = self.get_tfboard_writer()
        

    def get_tfboard_writer(self):
        log_dir = os.path.join(
            self.log_dir, 
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        summary_writer = tf.summary.create_file_writer(log_dir)
        return summary_writer
    

    def resume_from_checkpoints(self, checkpoint_dir : Optional[str] = ""):
        if checkpoint_dir != "":
            self.model.load_weights(checkpoint_dir)
        else:
            self.model.load_weights(self.checkpoint_dir)


    def register_evaluator(self, evaluator: Evaluator):
        self.evaluator = evaluator


    def train(self):
        for epoch in range(self.epoches):
            index = 0
            for batch_x, batch_target_seq, batch_seq_len in self.train_data_generator:
                with tf.GradientTape() as tape:
                    user_embeddins = self.user_embeddings(batch_x)
                    inputEmbeddings = self.ad_embeddings(batch_target_seq)
                    model_output = self.model(inputEmbeddings, training=True) # [b, seq_len, dim]
                    
                    loss = self.model.cal_loss(
                        model_output, 
                        batch_target_seq,
                        batch_seq_len,
                        self.ad_embeddings.weights[0],
                        10
                    )
                    
                    trainable_vars = (
                        self.model.trainable_weights + 
                        list(self.ad_embeddings.trainable_variables)
                        # list(self.user_embeddings.trainable_variables)
                    )
                                
                    grads = tape.gradient(loss, trainable_vars)
                    self.optimizer.apply_gradients(zip(grads, trainable_vars))

                    with self.tfboard_writer.as_default():
                        tf.summary.scalar('training_loss', loss, step=self.optimizer.iterations)
                    
                    # Print loss every 100 steps
                    if index % 2 == 0:
                        print(f"Epoch {epoch+1}, Batch {index}, Loss: {loss:.4f}")
                    index += 1