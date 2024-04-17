'''
Description: 
Author: Rigel Ma
Date: 2023-10-16 20:49:33
LastEditors: Rigel Ma
LastEditTime: 2024-04-17 16:54:11
FilePath: main.py
'''

import importlib
import json
from utils.parser import main_parser
from data.Interact_dataset import Interact_dataset
from model.Trainer import Trainer
import os
from utils.Libs import *
from utils.set_color_log import init_logger


if __name__ == "__main__":
    main_parser = main_parser()
    args = main_parser.parse_args()

    logger = init_logger()
    
    # load config
    config = json.loads(open(args.config).read()) # dict
    train_config = config['train_config']
    update_DEVICE(torch.device(train_config['device']))

    # data #
    dataset_config = config['dataset_config']
    dataset_config['batch_size'] = train_config['batch_size'] # 生成数据需要batch_size
    interactions = Interact_dataset(**dataset_config)
    interactions.process()
    # train_data, val_data, test_data = interactions.split_dataset()

    # model #
    model_name = train_config['model']
    MODEL_Module = importlib.import_module('model.model.{}'.format(model_name))
    MODEL = getattr(MODEL_Module, model_name)

    # init model
    # parameters setting is in ./model/config/{model_name}.json
    # model_config_path = config['model_config']['path']
    model_config_path = f'./model/config/{model_name}.json'
    model_config = json.loads(open(model_config_path,'r').read())

    model_ = MODEL(interactions, model_config)

    # train #
    eval_config = config['eval_config']
    trainer = Trainer(train_config, eval_config, model_)
    trainer.train()