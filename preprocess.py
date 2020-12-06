#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import json
import os

from gensim.models.keyedvectors import KeyedVectors


def preprocess_relation(path_src, path_des):
    fw = open(path_des, 'w', encoding='utf-8')
    with open(path_src, 'r', encoding='utf-8') as fr:
        rel2id = {}
        for line in fr:
            line = line.strip().split()
            rel2id[line[0]] = int(line[1])
        json.dump(rel2id, fw, ensure_ascii=False)
    fw.close()
    return rel2id


def preprocess_data(path_src, path_des, rel2id):
    fw = open(path_des, 'w', encoding='utf-8')
    with open(path_src, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip().split('\t')
            assert len(line) == 7
            e1 = line[2]
            e2 = line[3]
            rel = line[4]
            sentence = line[5]
            if rel not in rel2id.keys():
                rel = 'NA'
            instance = dict(
                head=e1,
                tail=e2,
                relation=rel,
                sentence=sentence,
            )
            json.dump(instance, fw, ensure_ascii=False)
            fw.write('\n')
    fw.close()


def preprocess_embedding(path_src, path_des):
    model = KeyedVectors.load_word2vec_format(path_src, binary=True)
    model.save_word2vec_format(path_des, binary=False)


if __name__ == '__main__':
    src_dir = './data'
    des_dir = './processed'
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)
    rel2id = preprocess_relation(
        os.path.join(src_dir, 'RE/relation2id.txt'),
        os.path.join(des_dir, 'relation.json'),
    )
    preprocess_data(
        os.path.join(src_dir, 'RE/train.txt'),
        os.path.join(des_dir, 'train.json'),
        rel2id
    )
    preprocess_data(
        os.path.join(src_dir, 'RE/test.txt'),
        os.path.join(des_dir, 'test.json'),
        rel2id
    )
    preprocess_embedding(
        os.path.join(src_dir, 'vec.bin'),
        os.path.join(des_dir, 'word2vec_50d.txt'),
    )
