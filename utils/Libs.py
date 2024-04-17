'''
Description: Some general library
Author: Rigel Ma
Date: 2023-11-21 17:10:46
LastEditors: Rigel Ma
LastEditTime: 2024-04-17 16:54:27
FilePath: Libs.py
'''


import torch
from torch import nn
from torch.nn import functional as F
import torch_sparse
import pandas as pd
import numpy as np
from utils.set_color_log import init_logger

TYPES = ['train', 'val', 'test']

logger = init_logger()
DEVICE = 'cuda:0'
def update_DEVICE(device):
    global DEVICE
    DEVICE = device
def DEVICE():
    global DEVICE
    return DEVICE