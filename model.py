#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import CNN_Encoder
from selector import ONE, ATT, AVG


class DS_Model(nn.Module):
    def __init__(self, word_vec, class_num, config):
        super().__init__()
        self.word_vec = word_vec
        self.class_num = class_num

        # hyper parameters and others
        self.max_len = config.max_len
        self.embedding_dim = config.embedding_dim
        self.pos_dim = config.pos_dim
        self.pos_dis = config.pos_dis

        # net structures and operations
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,
        )
        self.pos1_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.pos2_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )

        if config.encoder in ['cnn', 'pcnn']:
            self.encoder = CNN_Encoder(config)

        if config.selector == 'one':
            self.selector = ONE(self.class_num, config)
        elif config.selector == 'att':
            self.selector = ATT(self.class_num, config)
        elif config.selector == 'avg':
            self.selector = AVG(self.class_num, config)

        self.criterion = nn.NLLLoss()

    def input(self, tokens, pos1, pos2):
        word_embedding = self.word_embedding(tokens)
        pos1_embedding = self.pos1_embedding(pos1)
        pos2_embedding = self.pos2_embedding(pos2)
        emb = torch.cat(
            tensors=[word_embedding, pos1_embedding, pos2_embedding], dim=-1)
        return emb

    def forward(self, data, scope, label):
        tokens = data[:, 0, :].view(-1, self.max_len)
        pos1 = data[:, 1, :].view(-1, self.max_len)
        pos2 = data[:, 2, :].view(-1, self.max_len)
        mask = data[:, 3, :].view(-1, self.max_len)
        emb = self.input(tokens, pos1, pos2)
        reps = self.encoder(emb, mask)
        probs = self.selector(reps, scope, label)
        loss = self.criterion(torch.log(probs), label)
        return loss, probs
