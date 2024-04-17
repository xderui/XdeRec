'''
Description: 
Author: Rigel Ma
Date: 2023-12-08 20:05:40
LastEditors: Rigel Ma
LastEditTime: 2024-04-17 16:54:37
FilePath: Evaluator.py
'''


from utils.Libs import *
from model.model.BaseModel import BaseModel
import math
from pprint import pprint
import time
from collections import defaultdict

class Evaluator(nn.Module):
    def __init__(self,
                 model_:BaseModel, 
                 eval_config: dict):
        super(Evaluator, self).__init__()

        self.model_ = model_
        self.eval_config = eval_config
        self.train_interactions = self.model_.interactions.splited_interactions[0]
        self.eval_interactions = self.model_.interactions.splited_interactions[1]
        self.test_interactions = self.model_.interactions.splited_interactions[2]

        self.pre_process()
        
        
    def pre_process(self):
        self.users = pd.unique(self.train_interactions[self.model_.interactions.map_columns[0]])
        user_num = self.model_.interactions.num_users

        self.train_set = self.construct_dict(self.train_interactions)
        self.eval_set = self.construct_dict(self.eval_interactions)
        self.test_set = self.construct_dict(self.test_interactions)

    def construct_dict(self, df):
        dict_ = defaultdict(list)
        for user, x in df.groupby(self.model_.interactions.map_columns[0]):
            dict_[user] = x.iloc[:,1].tolist()

        return dict_
    

    def evaluate(self, type='val', rec_topk=None):
        self.model_.eval()
        if rec_topk == None:
            users = np.sort(self.users)
            ratings = self.model_.predict(users)
            
            # filter the items in train_set
            for idx, _ in enumerate(users):
                ratings[idx, self.train_set[self.users[idx]]] = -np.inf

            rec_topk = torch.argsort(ratings, dim=-1, descending=True)[:,:self.eval_config['topk']].cpu().detach().numpy()

        # rec_list -> 0,1 list
        pred_ = []
        for i, pred_topk in enumerate(rec_topk):
            gt = self.test_set[self.users[i]]
            correct_items = np.intersect1d(pred_topk, gt)
            pred = list(map(lambda x: x in correct_items, pred_topk))
            pred_.append(pred)

        pred_ = np.array(pred_, dtype='float')


        results = dict()
        for metric in self.eval_config['metrics']:
            eval_func = getattr(self, metric)
            result = eval_func(pred_)
            results[metric] = result

        return results

    def recall(self, pred_):
        assert len(pred_) == len(self.test_set)

        return np.mean(pred_.sum(axis=-1) / np.array([len(self.test_set[user]) for idx, user in enumerate(self.users)]), axis=-1)
        


    def ndcg(self, pred_):
        assert len(pred_) == len(self.test_set)
        k = pred_.shape[-1]
        test_mat = np.zeros_like(pred_)
        for i, items in enumerate(self.test_set.values()):
            len_ = k if k <= len(items) else len(items)
            test_mat[i, :len_] = 1

        idcg = np.sum(test_mat / np.log2(np.arange(2,k+2)), axis=1)
        dcg = np.sum(pred_ / np.log2(np.arange(2,k+2)), axis=1)

        idcg[idcg == 0] = 1
        ndcg = dcg / idcg
        ndcg[np.isnan(ndcg)] = 0

        return np.mean(ndcg)
        
                    

