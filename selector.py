#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import torch
import torch.nn as nn
import torch.nn.functional as F


class ONE(nn.Module):
    def __init__(self, class_num, config):
        super().__init__()
        self.class_num = class_num

        if config.encoder == 'cnn':
            self.piece = 1
        elif config.encoder == 'pcnn':
            self.piece = 3

        # hyper parameters and others
        self.filter_num = config.filter_num
        self.dropout_value = config.dropout

        # net structures and operations
        self.dropout = nn.Dropout(self.dropout_value)
        self.dense = nn.Linear(
            in_features=self.filter_num*self.piece,
            out_features=self.class_num,
            bias=True
        )

    def forward(self, reps, scope, label):
        reps = self.dropout(reps)
        batch_size = len(scope) - 1
        probs = []
        label = label.cpu().detach().tolist()
        for i in range(batch_size):
            sen_reps = reps[scope[i]:scope[i+1], :].\
                view(-1, self.filter_num*self.piece)  # n*C
            logits = self.dense(sen_reps)  # n*N
            prob = F.softmax(logits, dim=-1)  # n*N
            # prob = torch.clamp(prob, min=1.0e-10, max=1.0)
            if self.training:
                label_prob = prob[:, label[i]].view(-1)
                _, max_idx = torch.max(label_prob, dim=0)
                probs.append(prob[max_idx].view(1, self.class_num))
            else:
                # zeng2015
                row_prob, row_idx = torch.max(prob, dim=-1)
                if row_idx.sum() > 0:
                    mask = row_idx.view(-1, 1).expand(-1, self.class_num)
                    prob = prob.masked_fill_(mask.eq(0), float('-inf'))
                    row_prob, _ = torch.max(prob[:, 1:], dim=-1)
                    _, row_idx = torch.max(row_prob, dim=0)
                else:
                    _, row_idx = torch.max(row_prob, dim=-1)
                probs.append(prob[row_idx].view(1, self.class_num))

                # lin2016
                # probs.append(prob.max(dim=0)[0].view(1, self.class_num))
        probs = torch.cat(probs, dim=0)
        return probs


class AVG(nn.Module):
    def __init__(self, class_num, config):
        super().__init__()
        self.class_num = class_num

        if config.encoder == 'cnn':
            self.piece = 1
        elif config.encoder == 'pcnn':
            self.piece = 3

        # hyper parameters and others
        self.filter_num = config.filter_num
        self.dropout_value = config.dropout

        # net structures and operations
        self.dropout = nn.Dropout(self.dropout_value)
        self.dense = nn.Linear(
            in_features=self.filter_num*self.piece,
            out_features=self.class_num,
            bias=True
        )

    def forward(self, reps, scope, label):
        # reps = self.dropout(reps)
        batch_size = len(scope) - 1
        bag_reps = []
        for i in range(batch_size):
            sen_reps = reps[scope[i]:scope[i+1], :].\
                view(-1, self.filter_num*self.piece)  # n*C
            bag_rep = sen_reps.mean(dim=0).view(1, -1)
            bag_reps.append(bag_rep)

        bag_reps = torch.cat(bag_reps, dim=0)
        bag_reps = self.dropout(bag_reps)
        logits = self.dense(bag_reps)
        probs = F.softmax(logits, dim=-1)
        return probs


class ATT(nn.Module):
    def __init__(self, class_num, config):
        super().__init__()
        self.class_num = class_num

        if config.encoder == 'cnn':
            self.piece = 1
        elif config.encoder == 'pcnn':
            self.piece = 3

        # hyper parameters and others
        self.filter_num = config.filter_num
        self.dropout_value = config.dropout

        # net structures and operations
        self.dropout = nn.Dropout(self.dropout_value)
        self.dense = nn.Linear(
            in_features=self.filter_num*self.piece,
            out_features=self.class_num,
            bias=True
        )
        self.attention = nn.Parameter(
            torch.rand(size=(1, self.filter_num*self.piece)).view(-1)
        )

    def forward(self, reps, scope, label):
        reps = self.dropout(reps)
        batch_size = len(scope) - 1
        probs = []
        label = label.cpu().detach().tolist()
        for i in range(batch_size):
            sen_reps = reps[scope[i]:scope[i+1], :].\
                view(-1, self.filter_num*self.piece)  # n*C
            att_sen_reps = torch.mul(sen_reps, self.attention)  # n*C
            rel_embedding = self.dense.weight.t()  # C*N
            att_score = torch.mm(att_sen_reps, rel_embedding)  # n*N
            att_weight = F.softmax(att_score, dim=0).t()  # N*n

            bag_reps = torch.mm(att_weight, sen_reps)  # N*C
            logits = self.dense(bag_reps)  # N*N (premise*predicted)
            prob = F.softmax(logits, dim=-1)  # N*N
            # prob = torch.clamp(prob, min=1.0e-10, max=1.0)
            if self.training:
                probs.append(prob[label[i]].view(1, self.class_num))
            else:
                predcited_prob = prob.diag().view(-1, self.class_num)
                probs.append(predcited_prob)
        probs = torch.cat(probs, dim=0)
        return probs
