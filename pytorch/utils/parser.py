# -*- encoding: utf-8 -*-
'''
Filename         :parser.py
Description      :
Time             :2023/11/14 17:17:04
Author           :Rigel Ma
Version          :1.0
'''

import argparse

def main_parser():
    parser = argparse.ArgumentParser(description='Main configuration')
    parser.add_argument('--config', type=str, required=True)

    return parser

    