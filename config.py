#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import argparse
import torch
import os
import random
import json
import matplotlib
import numpy as np


class Config(object):
    def __init__(self):
        # get init config
        args = self.__get_config()
        for key in args.__dict__:
            setattr(self, key, args.__dict__[key])

        # select device
        self.device = None
        if self.cuda >= 0 and torch.cuda.is_available():
            self.device = torch.device('cuda:{}'.format(self.cuda))
        else:
            self.device = torch.device('cpu')

        # determine the model name and model dir
        if self.model_name is None:
            self.model_name = '{}+{}'.format(self.encoder,
                                             self.selector).upper()
        self.model_dir = os.path.join(self.output_dir, self.model_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # data cache dir
        if self.cache_dir is None:
            self.cache_dir = './data_cache'
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # backup data
        self.__config_backup(args)

        # set the random seed
        self.__set_seed(self.seed)

        # set the environment
        self.__init_environment()

    def __init_environment(self):
        matplotlib.use('Agg')

    def __set_seed(self, seed=1234):
        os.environ['PYTHONHASHSEED'] = '{}'.format(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # set seed for cpu
        torch.cuda.manual_seed(seed)  # set seed for current gpu
        torch.cuda.manual_seed_all(seed)  # set seed for all gpu

    def __get_config(self):
        parser = argparse.ArgumentParser()
        parser.description = 'config for models'

        # data directory
        parser.add_argument('--data_dir', type=str,
                            default='./processed',
                            help='dir to load data')
        parser.add_argument('--output_dir', type=str,
                            default='./output',
                            help='dir to save models')
        parser.add_argument('--cache_dir', type=str,
                            default=None,
                            help='dir to save data cache')

        # word/char embedding
        parser.add_argument('--embedding_path', type=str,
                            default='./processed/word2vec_50d.txt',
                            help='path to load pre-trained word/char embedding')
        parser.add_argument('--embedding_dim', type=int,
                            default=50,
                            help='dimension of embedding')
        parser.add_argument('--min_freq', type=float,
                            default=0,
                            help='minimum token frequency when constructing vocabulary list')

        # train settings
        parser.add_argument('--encoder', type=str,
                            default='cnn', choices=['cnn', 'pcnn'],
                            help='encoder for sentence')
        parser.add_argument('--selector', type=str,
                            default='one', choices=['one', 'att', 'avg'],
                            help='selector for model')
        parser.add_argument('--model_name', type=str,
                            default=None,
                            help='model name')
        parser.add_argument('--mode', type=int,
                            default=0,
                            choices=[0, 1],
                            help='running mode: 0: train; 1: test')
        parser.add_argument('--seed', type=int,
                            default=2020,
                            help='random seed')
        parser.add_argument('--cuda', type=int,
                            default=0,
                            help='num of gpu device, if -1, select cpu')
        parser.add_argument('--epoch', type=int,
                            default=5,
                            help='max epoches during training')

        # hyper parameters
        parser.add_argument('--batch_size', type=int,
                            default=160,
                            help='batch size')
        parser.add_argument('--dropout', type=float,
                            default=0.5,
                            help='the possiblity of dropout')
        parser.add_argument('--lr', type=float,
                            default=0.001,
                            help='learning rate')
        parser.add_argument('--max_len', type=int,
                            default='120',
                            help='max length of sentence')
        parser.add_argument('--pos_dis', type=int,
                            default=45,
                            help='max distance of position embedding')
        parser.add_argument('--pos_dim', type=int,
                            default=5,
                            help='dimension of position embedding')
        parser.add_argument('--filter_num', type=int,
                            default=230,
                            help='the number of filters in convolution')
        parser.add_argument('--window', type=int,
                            default=3,
                            help='the size of window in convolution')

        args = parser.parse_args()
        return args

    def __config_backup(self, args):
        config_backup_path = os.path.join(self.model_dir, 'config.json')
        with open(config_backup_path, 'w', encoding='utf-8') as fw:
            json.dump(vars(args), fw, ensure_ascii=False)

    def print_config(self):
        for key in self.__dict__:
            print(key, end=' = ')
            print(self.__dict__[key])


if __name__ == '__main__':
    config = Config()
    config.print_config()
