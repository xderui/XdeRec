'''
Description: 
Author: Rigel Ma
Date: 2023-12-01 18:45:49
LastEditors: Rigel Ma
LastEditTime: 2024-05-12 15:20:52
FilePath: Trainer.py
'''

from utils.Libs import *
from model.Evaluator import Evaluator
from tqdm import tqdm
import time
from abc import ABC, abstractmethod

class AbstractTrainer(nn.Module):
    def __init__(self, model_type):
        super(AbstractTrainer, self).__init__()
    
    @abstractmethod
    def init_param(self, param_dict):
        pass

    @abstractmethod
    def train(self):
        pass

            

class Trainer(nn.Module):
    def __init__(self,
                 train_config: dict,
                 eval_config: dict,
                 model_):
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
        if hasattr(self, 'early_stop'):
            early_stop = self.early_stop
            acc_epoch = 0
        else:
            early_stop = None 

        best_status = None

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
            self.model_.sample()
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

            if not best_status:
                best_status = [epoch_, results]
            else:
                best_num = 0
                keys_len = len(results) // 2
                best_results = best_status[1]
                for k in results.keys():
                    if results[k] > best_results[k]:
                        best_num += 1
                if best_num >= keys_len:
                    best_status = [epoch_, results]
                
                if early_stop:
                    if best_num >= keys_len:
                        acc_epoch = 0
                    else:
                        acc_epoch += 1
            
            if early_stop and acc_epoch >= early_stop:
                break

        print('best val: epoch {}, '.format(best_status[0]) + ', '.join([f'{k}:{round(v, 4)}' for k,v in best_status[1].items()]))

        

            
class SeqTrainer(nn.Module):
    def __init__(self,
                 train_config: dict,
                 eval_config: dict,
                 model_):
        super(SeqTrainer, self).__init__()

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
        if hasattr(self, 'early_stop'):
            early_stop = self.early_stop
            acc_epoch = 0
        else:
            early_stop = None 

        best_status = None

        for epoch_ in range(self.epoch):
            if hasattr(self, 'n_batch'):
                n_batch = self.n_batch
                loop = tqdm(self.model_.interactions.sample_interactions(n_batch), total=n_batch)
                loop.set_description("Sampling data")
            else:
                n_batch = len(self.model_.interactions.splited_interactions[0]) // self.batch_size + 1
                loop = self.model_.interactions.sample_all_interactions(self.model_.need_attrs)
            train_loop = tqdm(enumerate(loop), total=n_batch)
            s_time = time.time()
            avg_loss = 0
            # self.model_.sample()
            for batch_idx, batch in train_loop:
                # print(len(batch), batch)
                batch = [torch.tensor(v).cuda() for v in batch]
                # users, pos_items, neg_items = batch
                users, pos_items, sequences, neg_items = batch
                loss = self.model_(sequences, pos_items, neg_items)

                avg_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loop.set_description(f'Epoch [{epoch_}/{self.epoch}]')
                train_loop.set_postfix(loss = loss.item())
            results = self.evaluator.evaluate('val')
            print(f'Epoch [{epoch_}/{self.epoch}][{round(time.time() - s_time, 2)} + s]: avg_loss:{round(avg_loss.item()/n_batch, 6)}, ' + ', '.join([f'{k}:{round(v, 4)}' for k,v in results.items()]))        

            if not best_status:
                best_status = [epoch_, results]
            else:
                best_num = 0
                keys_len = len(results) // 2
                best_results = best_status[1]
                for k in results.keys():
                    if results[k] > best_results[k]:
                        best_num += 1
                if best_num >= keys_len:
                    best_status = [epoch_, results]
                
                if early_stop:
                    if best_num >= keys_len:
                        acc_epoch = 0
                    else:
                        acc_epoch += 1
            
            if early_stop and acc_epoch >= early_stop:
                break

        print('best val: epoch {}, '.format(best_status[0]) + ', '.join([f'{k}:{round(v, 4)}' for k,v in best_status[1].items()]))

        
        
