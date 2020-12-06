#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class EmbeddingLoader(object):
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.embedding_path = config.embedding_path
        self.embedding_dim = config.embedding_dim
        self.min_freq = config.min_freq
        self.cache_dir = config.cache_dir

    def __build_vocab(self):
        vocab = {}
        file_type = ['train', 'test']
        for filetype in file_type:
            label_file = '{}.json'.format(filetype)
            print('building vocaburary from %s' % label_file)
            with open(os.path.join(self.data_dir, label_file), 'r', encoding='utf-8') as fr:
                for line in fr:
                    sentence = json.loads(line.strip())['sentence'].split()
                    for token in sentence:
                        vocab[token] = vocab.get(token, 0) + 1
        vocab = set(
            [token for token in vocab if vocab[token] >= self.min_freq])
        return vocab

    def __load_embedding(self):
        vocab = self.__build_vocab()
        token2id = {}
        token2id['PAD'] = len(token2id)
        token2id['UNK'] = len(token2id)
        special_num = len(token2id)
        token_emb = []
        with open(self.embedding_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip().split()
                if len(line) != self.embedding_dim + 1:
                    continue
                if line[0] not in vocab:
                    continue
                token2id[line[0]] = len(token2id)
                token_emb.append(np.asarray(line[1:], dtype=np.float32))
        token_emb = np.stack(token_emb).reshape(-1, self.embedding_dim)
        emb_mean, emb_std = token_emb.mean(), token_emb.std()
        special_emb = np.random.normal(emb_mean, emb_std,
                                       size=(special_num, self.embedding_dim))
        token_emb = np.concatenate((special_emb, token_emb), axis=0)

        token_emb = token_emb.astype(
            np.float32).reshape(-1, self.embedding_dim)
        token_emb = torch.from_numpy(token_emb)
        return token2id, token_emb

    def load_embedding(self):
        data_cache = os.path.join(self.cache_dir, 'embedding.pkl')
        if not os.path.exists(data_cache):
            token2id, token_emb = self.__load_embedding()
            torch.save([token2id, token_emb], data_cache)
        else:
            token2id, token_emb = torch.load(data_cache)
        print('embedding scale: {}*{}d'.format(len(token2id), self.embedding_dim))
        print('finish loading embeddng!')
        return token2id, token_emb


class RelationLoader(object):
    def __init__(self, config):
        self.data_dir = config.data_dir

    def __load_relation(self):
        relation_file = os.path.join(self.data_dir, 'relation.json')
        with open(relation_file, 'r', encoding='utf-8') as fr:
            rel2id = json.load(fr)
        id2rel = {num: rel for rel, num in rel2id.items()}
        return rel2id, id2rel, len(rel2id)

    def get_relation(self):
        return self.__load_relation()


class NYTCorpus(object):
    def __init__(self, rel2id, token2id, config):
        self.rel2id = rel2id
        self.token2id = token2id
        self.data_dir = config.data_dir
        self.cache_dir = config.cache_dir
        self.max_len = config.max_len
        self.pos_dis = config.pos_dis
        self.class_num = len(rel2id)

    def __get_pos_index(self, x):
        if x < -self.pos_dis:
            return 0
        if x >= -self.pos_dis and x <= self.pos_dis:
            return x + self.pos_dis + 1
        if x > self.pos_dis:
            return 2 * self.pos_dis + 2

    def __symbolize_sentence(self, e1, e2, sentence):
        sentence = sentence.split()
        e1_pos = -1
        e2_pos = -1
        mask = []
        mask_flag = 1
        for i in range(len(sentence)):
            if e1_pos == -1 and sentence[i] == e1:
                e1_pos = i
                mask_flag += 1
            if e2_pos == -1 and sentence[i] == e2:
                e2_pos = i
                mask_flag += 1
            mask.append(mask_flag)
        words = []
        pos1 = []
        pos2 = []
        length = min(self.max_len, len(sentence))
        mask = mask[:length]
        for i in range(length):
            words.append(self.token2id.get(sentence[i], self.token2id['UNK']))
            pos1.append(self.__get_pos_index(i - e1_pos))
            pos2.append(self.__get_pos_index(i - e2_pos))
        if length < self.max_len:
            for i in range(length, self.max_len):
                mask.append(0)  # 'PAD' mask is zero
                words.append(self.token2id['PAD'])
                pos1.append(self.__get_pos_index(i - e1_pos))
                pos2.append(self.__get_pos_index(i - e2_pos))
        unit = np.asarray([words, pos1, pos2, mask], dtype=np.int64)
        unit = np.reshape(unit, newshape=(1, 4, self.max_len))
        return unit

    def __load_data(self, filetype):
        data_cache = os.path.join(self.cache_dir, '{}.pkl'.format(filetype))
        if os.path.exists(data_cache):
            bag_data, bag_label = torch.load(data_cache)
        else:
            label_file = '{}.json'.format(filetype)
            label_file_path = os.path.join(self.data_dir, label_file)

            data = {}
            labels = {}
            with open(label_file_path, 'r', encoding='utf-8') as fr:
                for line in fr:
                    line = json.loads(line.strip())
                    e1 = line['head']
                    e2 = line['tail']
                    rel = line['relation']
                    sentence = line['sentence']
                    label_idx = self.rel2id[rel]
                    sen_reps = self.__symbolize_sentence(e1, e2, sentence)

                    # in order to keep the test bag have only one label
                    k = e1 + '\t' + e2 + '\t' + str(label_idx)
                    if k not in data.keys():
                        data[k] = [sen_reps]
                        labels[k] = label_idx
                    else:
                        data[k].append(sen_reps)

            bag_data = []
            bag_label = []
            for k in data.keys():
                unit_data = np.asarray(data[k], dtype=np.int64)
                unit_data = np.reshape(
                    unit_data, newshape=(-1, 4, self.max_len))
                bag_data.append(unit_data)
                bag_label.append(labels[k])
            data_labels = [bag_data, bag_label]
            torch.save(data_labels, data_cache)
        return bag_data, bag_label

    def load_corpus(self, filetype):
        """
        filetype:
            train: 加载训练数据
            dev  : 加载验证数据
            test : 加载测试数据
        """
        if filetype in ['train', 'test']:
            return self.__load_data(filetype)
        else:
            raise ValueError('mode error!')


class NYTDataset(Dataset):
    def __init__(self, data, labels):
        self.dataset = data
        self.label = labels

    def __getitem__(self, index):
        data = self.dataset[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.label)


class NYTDataLoader(object):
    def __init__(self, rel2id, token2id, config):
        self.event2id = rel2id
        self.token2id = token2id
        self.config = config
        self.corpus = NYTCorpus(rel2id, token2id, config)

    def __collate_fn(self, batch):
        data, label = zip(*batch)  # unzip the batch data
        data = list(data)
        label = list(label)
        scope = []
        total = 0
        for i in range(len(label)):
            scope.append(total)
            total += data[i].shape[0]
        scope.append(total)
        data = torch.from_numpy(np.concatenate(data, axis=0))
        label = torch.from_numpy(np.asarray(label, dtype=np.int64))
        return data, label, scope

    def __get_data(self, filetype, shuffle=False):
        data, labels = self.corpus.load_corpus(filetype)
        dataset = NYTDataset(data, labels)
        collate_fn = self.__collate_fn
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=2,
            collate_fn=collate_fn
        )
        return loader

    def get_train(self):
        ret = self.__get_data(filetype='train', shuffle=True)
        print('finish loading train!')
        return ret

    def get_test(self):
        ret = self.__get_data(filetype='test', shuffle=False)
        print('finish loading test!')
        return ret


if __name__ == '__main__':
    from config import Config
    config = Config()
    token2id, emb = EmbeddingLoader(config).load_embedding()

    rel2id, id2rel, class_num = RelationLoader(config).get_relation()
    print('class_num', class_num)
    print(rel2id)

    loader = NYTDataLoader(rel2id, token2id, config)
    test_loader = loader.get_test()

    for step, (data, label, scope) in enumerate(test_loader):
        print(type(data), data.shape)
        print(type(label), label.shape)
        print(type(scope), len(scope))
        break
    train_loader = loader.get_train()
