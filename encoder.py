#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # hyper parameters and others
        self.max_len = config.max_len
        self.embedding_dim = config.embedding_dim
        self.pos_dim = config.pos_dim
        self.pos_dis = config.pos_dis
        self.filter_num = config.filter_num
        self.window = config.window

        # net structures and operations
        self.dim = self.embedding_dim + 2 * self.pos_dim
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.filter_num,
            kernel_size=(self.window, self.dim),
            stride=(1, 1),
            bias=True,
            padding=(1, 0),  # same padding
            padding_mode='zeros'
        )
        self.maxpool = nn.MaxPool2d((self.max_len, 1))
        self.tanh = nn.Tanh()

        # mask operation for pcnn
        if self.config.encoder == 'pcnn':
            self.mask_embedding = nn.Embedding(4, 3)
            masks = torch.tensor(
                [[0, 0, 0],
                 [100, 0, 0],
                 [0, 100, 0],
                 [0, 0, 100]]
            )
            self.mask_embedding.weight.data.copy_(masks)
            self.mask_embedding.weight.requires_grad = False

    def conv_layer(self, emb, mask):
        emb = emb.unsqueeze(dim=1)  # B*1*L*D
        conv = self.conv(emb)  # B*C*L*1

        # mask, remove the effect of 'PAD'
        conv = conv.squeeze(dim=-1)  # B*C*L
        mask = mask.unsqueeze(dim=1)  # B*1*L
        mask = mask.expand(-1, self.filter_num, -1)  # B*C*L
        conv = conv.masked_fill_(mask.eq(0), float('-inf'))  # B*C*L
        conv = conv.unsqueeze(dim=-1)  # B*C*L*1
        return conv

    def piece_maxpool(self, x, mask):
        x = x.permute(0, 2, 1, 3)  # B*L*C*1
        mask_embed = self.mask_embedding(mask)  # B*L*3
        mask_embed = mask_embed.unsqueeze(dim=-2)  # B*L*1*3
        x = x + mask_embed  # B*L*C*3
        x = torch.max(x, dim=1)[0] - 100  # B*1*C*3
        x = x.view(x.shape[0], -1)  # B*(C*3)
        return x

    def single_maxpool(self, x):
        x = self.maxpool(x)  # B*C*1*1
        x = x.view(-1, self.filter_num)  # B*C
        return x

    def forward(self, emb, mask):
        conv = self.conv_layer(emb, mask)

        if self.config.encoder == 'pcnn':
            pool = self.piece_maxpool(conv, mask)
        else:
            pool = self.single_maxpool(conv)

        reps = self.tanh(pool)
        return reps
