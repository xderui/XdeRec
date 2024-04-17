'''
Description: 
Author: Rigel Ma
Date: 2023-12-01 18:45:49
LastEditors: Rigel Ma
LastEditTime: 2024-04-17 16:54:32
FilePath: Trainer.py
'''

from utils.Libs import *
from model.model.BaseModel import BaseModel
from model.Evaluator import Evaluator
from tqdm import tqdm
import time


class Trainer(nn.Module):
    def __init__(self,
                 train_config: dict,
                 eval_config: dict,
                 model_: BaseModel):
        super(Trainer, self).__init__()

        self.init_param(train_config)
        self.model_ = model_.to(DEVICE())
        self.evaluator = Evaluator(self.model_, eval_config)

        # self.model_ = model_.cuda()
        
    def init_param(self, param_dict):
        for k,v in param_dict.items():
            try:
                assert v != "" and v != None
                exec(f'self.{k}={v}')
            except:
                pass
            
    def train(self):

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        self.model_.train()

        for epoch_ in range(self.epoch):
            if hasattr(self, 'n_batch'):
                n_batch = self.n_batch
                loop = tqdm(self.model_.interactions.sample_users(n_batch), total=n_batch)
                loop.set_description("Sampling data")
            else:
                n_batch = len(self.model_.interactions.splited_interactions[0]) // self.batch_size + 1
                loop = self.model_.interactions.sample_all_interactions()
            train_loop = tqdm(enumerate(loop), total=n_batch)
            s_time = time.time()
            avg_loss = 0
            for batch_idx, batch in train_loop:
                batch = [torch.tensor(v).cuda() for v in batch]
                users, pos_items, neg_items = batch
                loss = self.model_(users, pos_items, neg_items)

                avg_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loop.set_description(f'Epoch [{epoch_}/{self.epoch}]')
                train_loop.set_postfix(loss = loss.item())
            results = self.evaluator.evaluate('val')
            print(f'Epoch [{epoch_}/{self.epoch}][{round(time.time() - s_time, 2)} + s]: avg_loss:{round(avg_loss.item()/n_batch, 6)}, ' + ', '.join([f'{k}:{round(v, 4)}' for k,v in results.items()]))

        # training finished
        self.evaluator.evaluate('test')
        

            
        


