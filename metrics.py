from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np


def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) / len(ind)
    metrics['MR'] = np.median(ind) + 1
    return metrics

def compute_epic_metrics(x, labels):
    N = len(labels)
    order = np.argsort(-x, axis=0)
    mask = labels.reshape((1,-1)) == labels[order]
    mask = mask.T
    i = np.where(mask)[0]
    j = np.where(mask)[1]
    switch_j = j[np.where(i[1:] != i[:-1])[0]+1]
    switch_j = np.concatenate([[j[0]], switch_j])
    rank = np.ones(N) * N
    for col in range(len(switch_j)):
        wrong_rank_higher = labels[order][:,col][:switch_j[col]]
        rank[col] = len(np.unique(wrong_rank_higher))
    rank = rank.astype(int)
    metrics = {}
    metrics['R1'] = float(np.sum(rank == 0)) / len(rank)
    metrics['R5'] = float(np.sum(rank < 5)) / len(rank)
    metrics['R10'] = float(np.sum(rank < 10)) / len(rank)
    metrics['MR'] = np.median(rank) + 1 
    
    return metrics


def print_computed_metrics(metrics):
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    mr = metrics['MR']
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'.format(r1, r5, r10, mr))
