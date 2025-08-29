'''
Description: 
Author: Rigel Ma
Date: 2023-12-20 15:48:06
LastEditors: Rigel Ma
LastEditTime: 2024-04-17 16:54:16
FilePath: set_color_log.py
'''
import logging
import colorlog

def init_logger():
    fmt_string = "%(log_color)s%(message)s"
    log_colors = {
        'DEBUG': 'white',
        'INFO': 'bold_blue',
        'WARNING': 'bold_yellow',
        'ERROR': 'bold_red',
        'CRITICAL': 'bold_purple'
    }

    fmt = colorlog.ColoredFormatter(fmt=fmt_string, log_colors=log_colors)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt=fmt)
    logger = logging.getLogger("user")
    if len(logger.handlers) == 0:
        logger.setLevel(logging.INFO)
        logger.addHandler(stream_handler)

    return logger