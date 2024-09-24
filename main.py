'''
Description: 
Author: Rigel Ma
Date: 2023-10-16 20:49:33
LastEditors: Rigel Ma
LastEditTime: 2024-05-12 20:27:58
FilePath: main.py
'''

import importlib
import json
from utils.parser import main_parser
from data.collaborative_filtering_dataset import CF_dataset
from data.sequential_recommendation_dataset import Sequential_dataset
from model.Trainer import Trainer
from model.Evaluator import Evaluator
import os
import imp
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

    # data & model

    dataset_config = config['dataset_config']
    dataset_config['batch_size'] = train_config['batch_size']
    
    
    # first find model module, ensuring which type of dataset should be loaded
    # model_name = train_config['model'].lower()
    model_name = train_config['model']
    MODEL_MODULE = None
    for model_type in MODEL_TYPES:
        model_module_path = '.'.join(['model', 'model', model_type, model_name])
        if importlib.util.find_spec(model_module_path, model_name):
            print('find!', model_type, f'model.model.{model_type}.{model_name}')
            MODEL_MODULE = importlib.import_module(f'model.model.{model_type}.{model_name}')
            model_type_ = model_type
            print('ok find')
            break
    if not MODEL_MODULE:
        raise ValueError(f"model {train_config['model']} is not exsits")
    
    MODEL_CLASS = getattr(MODEL_MODULE, train_config['model'])
    
    # init dataset
    DATASET_MODULE = importlib.import_module(f"data.{model_type_}_dataset", __name__)
    DATASET_CLASS = getattr(DATASET_MODULE, MODEL_TYPE2DATASET_TYPE[model_type_])
    
    interactions = DATASET_CLASS(**dataset_config)


    # init model
    # parameters setting is in ./model/config/{model_name}.json
    # model_config_path = config['model_config']['path']
    model_config_path = f'./model/config/{model_type_}/{model_name}.json'
    model_config = json.loads(open(model_config_path,'r').read())

    model_ = MODEL_CLASS(interactions, model_config)

    # train #
    eval_config = config['eval_config']
    TRAINER_MODULE = importlib.import_module(f"model.Trainer", __name__)
    TRAINER_CLASS = getattr(TRAINER_MODULE, MODEL_TYPE2TRAINER_TYPE[model_type_])
    trainer = TRAINER_CLASS(train_config, eval_config, model_)
    trainer.train()

    # test #
    test_config = config['test_config']
    evaluator = Evaluator(model_, test_config)
    test_results = evaluator.evaluate('test')
    print('final test: ' + ', '.join([f'{k}:{round(v, 4)}' for k,v in test_results.items()]))
    