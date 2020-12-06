#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import os
import json
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from tqdm import tqdm

from config import Config
from utils import EmbeddingLoader, RelationLoader, NYTDataLoader
from model import DS_Model
from evaluate import Eval
from plot import Canvas


class Runner(object):
    def __init__(self, emb, class_num, loader, config):
        self.class_num = class_num
        self.loader = loader
        self.config = config

        self.model = DS_Model(emb, class_num, config)
        self.model = self.model.to(config.device)
        self.eval_tool = Eval(class_num, config)
        self.plot_tool = Canvas(config)

    def train(self):
        train_loader, test_loader = self.loader
        optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        print(self.model)
        print('traning model parameters:')
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print('%s :  %s' % (name, str(param.data.shape)))
        print('--------------------------------------')
        print('start to train the model ...')
        max_auc = -float('inf')
        for epoch in range(1, 1+self.config.epoch):
            train_loss = 0.0
            data_iterator = tqdm(train_loader, desc='Train')
            for step, (data, label, scope) in enumerate(data_iterator):
                self.model.train()
                data = data.to(self.config.device)
                label = label.to(self.config.device)

                optimizer.zero_grad()
                loss, _ = self.model(data, scope, label)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
            train_loss = train_loss / len(train_loader)

            auc, test_loss, precision, recall = self.eval_tool.evaluate(
                self.model, test_loader
            )
            print('[%03d] train_loss: %.3f | test_loss: %.3f | auc on test: %.3f'
                  % (epoch, train_loss, test_loss, auc), end=' ')
            if auc > max_auc:
                max_auc = auc
                torch.save(self.model.state_dict(), os.path.join(
                    self.config.model_dir, 'model.pkl'))
                print('>>> save models!')
            else:
                print()

    def test(self):
        print('-------------------------------------')
        print('start test ...')

        _, test_loader = self.loader
        self.model.load_state_dict(torch.load(
            os.path.join(config.model_dir, 'model.pkl')))
        auc, test_loss, precision, recall = self.eval_tool.evaluate(
            self.model, test_loader
        )
        print('test_loss: %.3f | auc on test: %.3f' % (test_loss, auc))

        target_file = os.path.join(self.config.model_dir, 'pr.txt')
        with open(target_file, 'w', encoding='utf-8') as fw:
            for i in range(len(precision)):
                fw.write('%.6f \t %.6f \n' % (precision[i], recall[i]))
        self.plot_tool.plot(precision, recall, auc)


if __name__ == '__main__':
    config = Config()
    print('--------------------------------------')
    print('some config:')
    config.print_config()

    print('--------------------------------------')
    print('start to load data ...')
    token2id, emb = EmbeddingLoader(config).load_embedding()
    rel2id, id2rel, class_num = RelationLoader(config).get_relation()
    loader = NYTDataLoader(rel2id, token2id, config)

    train_loader, test_loader = None, None
    if config.mode == 0:  # train mode
        train_loader = loader.get_train()
        test_loader = loader.get_test()
    elif config.mode == 1:
        test_loader = loader.get_test()

    loader = [train_loader, test_loader]
    print('finish!')

    runner = Runner(emb, class_num, loader, config)
    if config.mode == 0:  # train mode
        runner.train()
        runner.test()
    elif config.mode == 1:
        runner.test()
    else:
        raise ValueError('invalid train mode!')
