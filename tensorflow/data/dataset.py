import tensorflow as tf
import random
import math

class Dataset:
    def __init__(self,
                 data_path : str,
                 data_sep : str = '\t',
                 data_is_seq : bool = True,
                 seq_max_len : int = 1024):
            
        self.data_path = data_path
        self.data_sep = data_sep
        self.data_is_seq = data_is_seq
        self.seq_max_len = seq_max_len
        
        self.data_dict = self.read_data(self.data_path, self.data_sep, self.data_is_seq)

        self.candidate_ads = self.get_ads_set(self.data_dict)
        self.user_set = set(self.data_dict.keys())
        
        self.user_seq_dict = self.data_dict

        self.user_map = self.remap(self.user_set)
        self.ad_map = self.remap(self.candidate_ads)
        
        self.user_num = len(self.user_map) + 1
        self.ad_num = len(self.ad_map) + 1
    
    
    def read_data(self, path, sep, isSeq = False):
        dict_ = {}
        with open(path, 'r') as f:
            for line in f:
                key, value = line.strip().split(sep)
                if isSeq:
                    dict_[key] = self.process_sequence(value, " ")
                else:
                    dict_[key] = value
        
        return dict_
    
    
    def remap(self, ids_set):
        dict_ = {x:i+1 for i,x in enumerate(ids_set)}
        dict_[-1] = 0  # 留出空的id
        return dict_   
    
    
    def get_ads_set(self, seq_dict):
        all_ads = []
        for ads in seq_dict.values():
            all_ads.extend(ads)
            
        return set(all_ads)
            
    def process_sequence(self, seqStr, sep):
        return seqStr.split(sep)
    
    def count_unique_key(self):
        return len(self.data_dict)
    
    def count_unique_values(self):
        return len(self.ad_map)
        
    
    def count_max_len(self):
        return max([len(values) for key, values in self.data_dict.items()])


    def truncate_or_padding_sequence(self, seq, mode='train', test_rate=0.2, needMap=False):
        seq_len = len(seq)
        if needMap:
            seq = [self.ad_map[_] for _ in seq]

        if not isinstance(seq, tf.Tensor):
            seq = tf.constant(seq, dtype=tf.int32)
            
        test_seq_len = math.ceil(seq_len * test_rate)
        if mode == 'train':
            seq = seq[:-test_seq_len]
            seq_len = seq_len - test_seq_len
        else:
            seq = seq[-test_seq_len:]
            seq_len = test_seq_len
        
        if seq_len <= self.seq_max_len:
            seq = tf.pad(seq, [[0, self.seq_max_len - seq_len]])
        else:
            seq_len = self.seq_max_len
            seq = seq[-self.seq_max_len:]
            
        # if len(seq) != self.seq_max_len:
        #     print(seq, origin_seq)
                
        return seq, seq_len
    
    def negative_sampling(self, target_sequence, neg_num):
        negatives = []
        for pos_ad in target_sequence:
            if pos_ad == 0:
                negatives.append([0] * neg_num)
            else:
                negatives_ = random.sample(self.ad_set, neg_num)
                negatives.append(negatives_)

        return tf.constant(negatives) # [seq_len, neg_num]
            
    
    def build_dataGenerator(self, batch_size = 64, mode='train', test_rate=0.2):
        '''
            先暂时实现序列回归生成
        '''
        # x_data = [self.user_map[_] for _ in list(user_seq_dict.keys())]
        x_data = []
        target_data = []
        target_data_len = []
        negative_data = []
        self.ad_set = set(self.ad_map.values())
        for uid, ad_seq in self.user_seq_dict.items():
            target_sequence, seq_len = self.truncate_or_padding_sequence(ad_seq, mode=mode, test_rate=test_rate, needMap=True)
            if seq_len <= 1:
                continue
            x_data.append(self.user_map[uid])
            target_data.append(target_sequence)
            target_data_len.append(seq_len)
            # negative_data.append(self.negative_sampling(target_sequence, 5))
            
        dataset = tf.data.Dataset.from_tensor_slices((x_data, target_data, target_data_len)).shuffle(1000).batch(batch_size).prefetch(1)
        
        return dataset


if __name__ == "__main__":
    data_path = '../data/sequence_data'
    seqData = Dataset(data_path=[data_path], data_sep=['\t'], data_is_seq=[True])

    print(seqData.count_unique_key())
    print(seqData.count_max_len())
    
    for batch_x, batch_y, batch_len in seqData.build_dataGenerator():
        print(tf.shape(batch_x), tf.shape(batch_y), tf.shape(batch_len),)
        break
    
    
        