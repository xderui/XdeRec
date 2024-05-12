'''
Description: Some general library
Author: Rigel Ma
Date: 2023-11-21 17:10:46
LastEditors: Rigel Ma
LastEditTime: 2024-05-12 20:24:13
FilePath: Libs.py
'''


import torch
from torch import nn
from torch.nn import functional as F
import torch_sparse
import pandas as pd
import numpy as np
from utils.set_color_log import init_logger

DATA_TYPES = ['train', 'val', 'test']
MODEL_TYPES = ['collaborative_filtering', 'sequential_recommendation']
MODEL_TYPE2DATASET_TYPE = {
    "collaborative_filtering" : "CF_dataset",
    "sequential_recommendation" : "Sequential_dataset"
}

logger = init_logger()


DEVICE_ = 'cuda:0'

def update_DEVICE(device):
    global DEVICE_
    DEVICE_ = device

def DEVICE():
    return DEVICE_