#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import matplotlib.pyplot as plt
import os


class Canvas(object):
    def __init__(self, config):
        self.model_dir = config.model_dir
        self.model_name = config.model_name

    def plot(self, precision, recall, auc=0.0):
        pic_title = 'Precision-Recall | AUC = {:.3f}'.format(auc)
        pic_label = '{}'.format(self.model_name)
        plt.figure()
        plt.plot(recall, precision, lw=2, label=pic_label)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.3, 1.0])
        plt.xlim([0.0, 0.4])
        plt.title(pic_title)
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig(os.path.join(self.model_dir, 'pr.png'), format='png')
        plt.close()
