import torch
import numpy as np
import os
import json
import pandas as pd
import datetime

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

'''
description: 日志工具
created by zpp 2021/07/24
'''
class Logger(object):
    def __init__(self, args):
        self.output_path = args
        # path initial
        self.create_time = datetime.datetime.now().strftime('%m%d-%H%M')
        self.output_path = os.path.join(self.output_path, self.create_time)
        self.log_path = os.path.join(self.output_path, 'log')
        self.data_path = os.path.join(self.output_path, 'data')
        self.fig_path = os.path.join(self.output_path, 'fig')
        # summary writer
        '''
        使用：tensorboard --logdir=yourlogpath
        command example: tensorboard --logdir=./output/0724-2221/log/
        then CTRL+C click the url in the terminal
        '''
        self.summary_writer = SummaryWriter(self.log_path)
        self.mem_dict = {}

    def mkdir(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

    def log_info(self, data_name, data, time_stamp):
        self.summary_writer.add_scalar(data_name, data, time_stamp)
    
    def save_to_csv(self, filename, save_path = None):
        '''
        csv 要求数组长度一致，mem_dict中的数组长度不一定一致，故分开保存
        '''
        new_dict = {}
        new_dict[filename] = self.mem_dict[filename]
        output_df = pd.DataFrame(new_dict)
        if save_path:
            self.mkdir(save_path)    
            output_df.to_csv(os.path.join(save_path, '{}.csv'.format(filename)), float_format='%.4f', index=False)
        else:
            self.mkdir(self.data_path)
            output_df.to_csv(os.path.join(self.data_path, '{}.csv'.format(filename)), float_format='%.4f', index=False)

    def save_to_json(self, filename, save_path = None):
        if save_path:
            self.mkdir(save_path)
            with open(os.path.join(save_path, '{}.json'.format(filename)),'w') as f:
                json.dump(self.mem_dict, f) 
        else:
            self.mkdir(self.data_path)
            with open(os.path.join(self.data_path, '{}.json'.format(filename)),'w') as f:
                json.dump(self.mem_dict, f)

    def save_data(self, data_name, data, auto_save = True):
        self.mem_dict[data_name] = self.mem_dict.get(data_name, [])
        self.mem_dict[data_name].append(data)
        if auto_save:
            self.save_to_csv(data_name)
            self.save_to_json('data')

    def plot(self):
        # wait to write/add/supply/update
        self.mkdir(self.fig_path)
        for key, value in self.mem_dict.items():
            fig = plt.figure()
            x = [i for i in range(len(value))]
            plt.plot(x, value)
            plt.xlabel('x')
            plt.ylabel(key)
            plt.savefig(os.path.join(self.fig_path, '{}.png'.format(key)))



'''
test logger
'''
if __name__ == "__main__":
    logger = Logger(args = ('output'))
    for i in range(100):
        data = 2 * i
        logger.log_info(data_name = 'index', data = data, time_stamp = i)
        logger.save_data(data_name = 'index', data = i)
        logger.log_info(data_name = 'test', data = 4*i, time_stamp = i)
        logger.save_data(data_name = 'test', data = 4*i)
    logger.plot()
    # logger.save_to_csv(filename = 'test')     # auto save
    # logger.save_to_json(filename = 'test')
