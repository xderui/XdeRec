import tensorflow as tf
from embedding import VQEmbedding
import random
    

class OneRec(tf.keras.layers.Layer):
    def __init__(self):
        super(OneRec, self).__init__()
