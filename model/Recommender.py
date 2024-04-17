'''
Description: 
Author: Rigel Ma
Date: 2023-11-14 16:54:28
LastEditors: Rigel Ma
LastEditTime: 2024-04-17 16:53:59
FilePath: Recommender.py
'''
# -*- encoding: utf-8 -*-

import torch
from torch import nn


class Recommender(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    