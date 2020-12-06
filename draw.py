#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6


import matplotlib
matplotlib.use('Agg')


def get_pr(path_src):
    precision = []
    recall = []
    with open(path_src, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip().split()
            precision.append(float(line[0]))
            recall.append(float(line[1]))
    return precision, recall


if __name__ == '__main__':

    cnn_one = './output/CNN+ONE/pr.txt'
    cnn_att = './output/CNN+ATT/pr.txt'
    cnn_avg = './output/CNN+AVG/pr.txt'
    pcnn_one = './output/PCNN+ONE/pr.txt'
    pcnn_att = './output/PCNN+ATT/pr.txt'
    pcnn_avg = './output/PCNN+AVG/pr.txt'

    cnn_one_p, cnn_one_r = get_pr(cnn_one)
    cnn_att_p, cnn_att_r = get_pr(cnn_att)
    cnn_avg_p, cnn_avg_r = get_pr(cnn_avg)
    pcnn_one_p, pcnn_one_r = get_pr(pcnn_one)
    pcnn_att_p, pcnn_att_r = get_pr(pcnn_att)
    pcnn_avg_p, pcnn_avg_r = get_pr(pcnn_avg)

    import matplotlib.pyplot as plt

    pic_title = 'Precision-Recall (CNN)'
    plt.figure()
    plt.plot(cnn_one_r, cnn_one_p, lw=1, label='CNN+ONE')
    plt.plot(cnn_att_r, cnn_att_p, lw=1, label='CNN+ATT')
    plt.plot(cnn_avg_r, cnn_avg_p, lw=1, label='CNN+AVG')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.3, 1.0])
    plt.xlim([0.0, 0.4])
    plt.title(pic_title)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('./pr_cnn.png', format='png')
    plt.close()

    pic_title = 'Precision-Recall (PCNN)'
    plt.figure()
    plt.plot(pcnn_one_r, pcnn_one_p, lw=1, label='PCNN+ONE')
    plt.plot(pcnn_att_r, pcnn_att_p, lw=1, label='PCNN+ATT')
    plt.plot(pcnn_avg_r, pcnn_avg_p, lw=1, label='PCNN+AVG')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.3, 1.0])
    plt.xlim([0.0, 0.4])
    plt.title(pic_title)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('./pr_pcnn.png', format='png')
    plt.close()
