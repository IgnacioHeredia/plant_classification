#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Results of the paper

Author: Ignacio Heredia
Date: December 2016

Description:
This script contains all the commands to reproduce the results and figures
of our paper on plant classification.
"""

import numpy as np
import json
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import os
import sys
homedir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(homedir, 'scripts'))

# %% Load prediction dict and synsets

pred_dict = json.load(open("test_predictions/resnet_predictions_inaturalist.json", "rb"))
metadata = np.genfromtxt(os.path.join(homedir, 'data', 'data_splits', 'synsets.txt'), dtype='str', delimiter='/n')
metadata_binomial = np.genfromtxt(os.path.join(homedir, 'data', 'data_splits', 'synsets_binomial.txt'), dtype='str', delimiter='/n')

# %% Overall accuracy

top1, top5 = 0, 0
for i, l in enumerate(pred_dict['true_lab']):
    top1 += 1. * (l == pred_dict['pred_lab'][i][0])
    top5 += 1. * (l in pred_dict['pred_lab'][i])
print 'Top 1 Accuracy: {}'.format(top1 / len(pred_dict['true_lab']))
print 'Top 5 Accuracy: {}'.format(top5 / len(pred_dict['true_lab']))

# %% Prediction accuracy vs cutoff


def top_acc(pred_dict, cutoff=0):
    """
    Computes top1 and top5 accuracy only for predictions for which the first
    predicted label probability is above a cutoff.
    """
    args = np.argwhere(np.array(pred_dict['pred_prob'])[:, 0] > cutoff).flatten()
    pred_lab, true_lab = np.array(pred_dict['pred_lab'])[args], np.array(pred_dict['true_lab'])[args]
    top1, top5, counts = 0, 0, 0
    for i in range(len(pred_lab)):
        top1 += 1. * (true_lab[i] == pred_lab[i, 0])
        top5 += 1. * (true_lab[i] in pred_lab[i])
        counts += 1.
    return top1, top5, counts


def acc_vs_cutoff(pred_dict):
    acc_dict = {'top1': [], 'top5': [], 'counts': [], 'cutoff': []}
    for i in np.linspace(0, 1, 100):
        t1, t5, c = top_acc(pred_dict, cutoff=i)
        if c == 0:
            break
        acc_dict['cutoff'].append(i)
        acc_dict['top1'].append(t1/c)
        acc_dict['top5'].append(t5/c)
        acc_dict['counts'].append(c)
    return (np.array(acc_dict['cutoff']), np.array(acc_dict['top1']),
            np.array(acc_dict['top5']), np.array(acc_dict['counts']))

x, top1, top5, counts = acc_vs_cutoff(pred_dict)
fig, [ax1, ax2] = plt.subplots(1, 2)
ax1.plot(x, top5, c='b', label='Top5 accuracy')
ax1.plot(x, top1, c='g', label='Top1 accuracy')
ax1.set_xlabel('Confidence cutoff'), ax1.legend()
ax2.plot(x, counts/counts[0], c='b', label='Counts (%)')
ax2.set_xlabel('Confidence cutoff'), ax2.legend()

# %% Matrix Mixing

cla = len(metadata)

## Select only top1 predictions
#pred_lab = np.array(pred_dict['pred_lab'])[:, 0]
#pred_prob = np.array(pred_dict['pred_prob'])[:, 0]
#true_lab = np.array(pred_dict['true_lab'])

# Select top5 predictions
pred_lab = np.array(pred_dict['pred_lab']).flatten()
pred_prob = np.array(pred_dict['pred_prob']).flatten()
true_lab = np.repeat(pred_dict['true_lab'], 5)

# Remove true labels with few occurrences to make the matrix more dense
lab_distr = {'lab': [], 'occurences': []}
for i in set(true_lab):
    lab_distr['lab'].append(i)
    occ = np.sum(true_lab == i)
    lab_distr['occurences'].append(occ)
new_true_lab = np.array(lab_distr['lab'])[np.array(lab_distr['occurences']) > 100]  # we cutooff=100 we dscard 50% (1.5K labels) of the less common labels
args = []
for i, j in enumerate(true_lab):
    if j in new_true_lab:
        args.append(i)
pred_lab, true_lab = pred_lab[args], true_lab[args]

# Fill matrix
M = np.zeros((cla, cla))
for i, j in zip(pred_lab, true_lab):
    M[i, j] += 1. * pred_prob[i]  # weight counts with the confidence of the prediction
M = M / np.sum(M, axis=0)  # normalize so that every true label has sum=1


def plot_matrix(M, labels=None):
    fig, ax = plt.subplots(1)
    M = np.ma.masked_where(M == 0, M)
    im = ax.imshow(M, interpolation='none', vmin=0.0001, cmap='gray_r', norm=LogNorm(vmin=0.0001, vmax=1.))
    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_xlabel('True label', fontsize=14)
    ax.set_ylabel('Predicted label', fontsize=14)
    fig.colorbar(im, fraction=0.046, pad=0.04)


def plot_multi_matrix(M, labels):
    fig, [ax1, ax2] = plt.subplots(1, 2)
    M = np.ma.masked_where(M == 0, M)
    
    ax1.imshow(M, interpolation='none', vmin=0.0001, cmap='gray_r', norm=LogNorm(vmin=0.0001, vmax=1.))
    ax1.set_xticks(range(len(labels)))
    ax1.set_yticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=90)
    ax1.set_yticklabels(labels)
    
    ax2.imshow(M, interpolation='none', vmin=0.0001, cmap='gray_r', norm=LogNorm(vmin=0.0001, vmax=1.))
    ax2.yaxis.tick_right()
    ax2.yaxis.set_ticks_position('both')
    ax2.set_xticks(range(len(labels)))
    ax2.set_yticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=90)
    ax2.set_yticklabels(labels)
    
    ax1.set_position([0.2, 0.5, 0.35, 0.35])
    ax2.set_position([0.41, 0.5, 0.35, 0.35])

#plot_matrix(M, metadata)

# Creating the family tree
with open(os.path.join('plant_utils', 'family_tree.json')) as json_file:
    fam = json.load(json_file)

# Map between old ordering and new ordering
ord_spe, ord_fam = [], []  # ordered list of species and families
for i in sorted(fam.keys()):
    ord_spe += sorted(fam[i])
    ord_fam += [i.encode('ascii', 'ignore')] * len(fam[i])

map_array = np.array([list(metadata).index(i) for i in ord_spe])
assert (ord_spe == metadata[map_array]).all()

ord_spe = list(metadata_binomial[map_array])  # transform to binomial name
#print 'Old order: {}'.format(metadata)
#print 'New order: {}'.format(zip(ord_spe, ord_fam))

def labels_mapping(old_labels, mapping_array):
    """
    Map the old labels to new labels following the order defined in mapping array.
    """
    new_labels = np.zeros_like(old_labels)
    for i, j in enumerate(mapping_array):
        new_labels[old_labels == j] = i
    return new_labels

pred_lab_mapped = labels_mapping(pred_lab, map_array)
assert (metadata[pred_lab] == metadata[map_array][pred_lab_mapped]).all()

# Reorder matrix
M_mapped = M[map_array, :][:, map_array]

# Chop off labels non present in out test set to make the matrix more dense
true_lab_meta = metadata_binomial[list(set(true_lab))]
args = []
for i, spe in enumerate(ord_spe):
    if spe in true_lab_meta:
        args.append(i)
ord_spe, ord_fam = list(np.array(ord_spe)[args]), list(np.array(ord_fam)[args])
M_mapped = M_mapped[:, args][args, :]

labels = [ord_spe[i] + ' | ' + ord_fam[i] for i in range(len(ord_spe))]

##Full matrix plot
#plot_matrix(M_mapped)
##plt.savefig('full_matrix.png', dpi=600)

##Zoom plots
#plot_multi_matrix(M_mapped, labels)
##plt.savefig('mixing_zoom.png', dpi=600)

M = M_mapped

fig, [ax1, ax2] = plt.subplots(1, 2)
M = np.ma.masked_where(M == 0, M)
im1 = ax1.imshow(M, interpolation='none', vmin=0.0001, cmap='gray_r', norm=LogNorm(vmin=0.0001, vmax=1.))
ax1.set_xticks(range(len(labels))), ax1.set_yticks(range(len(labels)))
ax1.set_xticklabels(labels, rotation=90), ax1.set_yticklabels(labels)
im2 = ax2.imshow(M, interpolation='none', vmin=0.0001, cmap='gray_r', norm=LogNorm(vmin=0.0001, vmax=1.))
ax2.yaxis.tick_right()
ax2.yaxis.set_ticks_position('both')
ax2.set_xticks(range(len(labels))), ax2.set_yticks(range(len(labels)))
ax2.set_xticklabels(labels, rotation=90), ax2.set_yticklabels(labels)

ax1.set_xlabel('True label', fontsize=14)
ax1.set_ylabel('Predicted label', fontsize=14, rotation=270, labelpad=20)
ax1.xaxis.set_label_position('top')
ax1.yaxis.set_label_position('right')

ax2.set_xlabel('True label', fontsize=14)
ax2.set_ylabel('Predicted label', fontsize=14)
ax2.xaxis.set_label_position('top')

cbaxes = fig.add_axes([0.49, 0.5, 0.01, 0.35]) 
cb = plt.colorbar(im1, cax = cbaxes)  

ax1.set_position([0.2, 0.5, 0.35, 0.35])
ax2.set_position([0.45, 0.5, 0.35, 0.35])

#plt.savefig('mixing_zoom.png', dpi=600)

# %% Multimage prediction (original)

pred_dict = json.load(open("test_predictions/resnet_predictions_inaturalist_multiimage.json", "rb"))
pred_dict1 = json.load(open("test_predictions/resnet_predictions_inaturalist.json", "rb"))
pred_dict['im_per_obs'] += [1] * len(pred_dict1['true_lab'])
pred_dict['true_lab'] += pred_dict1['true_lab']
pred_dict['pred_lab'] += pred_dict1['pred_lab']
pred_dict['pred_prob'] += pred_dict1['pred_prob']


def obs_dict(obs_num, pred_dict):
    """
    Returns a dict with the observations that have a number obs_num of images.
    """
    args = (np.array(pred_dict['im_per_obs']) == obs_num)
    new_true_lab = np.array(pred_dict['true_lab'])[args]
    new_pred_lab = np.array(pred_dict['pred_lab'])[args]
    new_pred_prob = np.array(pred_dict['pred_prob'])[args]
    return {'true_lab': list(new_true_lab),
            'pred_lab': list(new_pred_lab),
            'pred_prob': list(new_pred_prob)}

ims_per_obs = [4, 3, 2, 1]
var_list = []
for i in ims_per_obs:
    tmp_dict = obs_dict(i, pred_dict)
    var_tmp = acc_vs_cutoff(tmp_dict) #[x, top1, top5, counts]
    var_list.append(var_tmp)

# Reduced one-column image

fig, [ax1, ax2] = plt.subplots(1, 2)
#colors = ['g', 'm', 'b', 'k'] #colour
colors = ['#c62f0f', 'g', 'm', 'k'] #colour2
#colors = ['0.8', '0.5', '0.3', '0.0'] #greyscale
for i, ims in enumerate(ims_per_obs):
    x, top1, top5, counts = var_list[i]
    ax1.plot(x, top1, c=colors[i], ls='-', linewidth=3.0)
    ax1.plot(x, top5, c=colors[i], ls='--', linewidth=3.0)
    ax2.plot(x, 1. - counts/counts[0], c=colors[i], ls='-', linewidth=3.0, label='{} images'.format(ims))

fs=18
ax1.set_xlabel('Confidence cutoff', fontsize=fs)
ax1.set_ylabel('Accuracy', fontsize=fs)
ax2.set_xlabel('Confidence cutoff', fontsize=fs)
ax2.set_ylabel('Discarded observations', fontsize=fs)

leg = ax2.legend(loc='lower right', fontsize=16)
for legobj in leg.legendHandles:
    legobj.set_linewidth(3.0)

for ax in [ax1, ax2]:
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16) 


#plt.savefig('multi_observations.png', dpi=600)