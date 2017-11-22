#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Furthers results

Author: Ignacio Heredia
Date: May 2017

Description:
Couple figures not included in the original paper.
"""
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os.path as op


homedir = op.abspath(op.join(__file__, op.pardir, op.pardir, op.pardir, op.pardir))


def detailed_acc(outfile_name):
    
    with open(os.path.join('test_predictions', outfile_name + '.json'), 'rb') as f:
            pred_dict = json.load(f)
    
    true_lab = np.array(pred_dict['true_lab']) 
    pred_lab = np.array(pred_dict['pred_lab'])
    
    top1 = np.mean(true_lab == pred_lab[:, 0])
    top5 = np.mean([true_lab[i] in j for i,j in enumerate(pred_lab)])
    print 'Dataset: {}'.format(outfile_name)
    print 'Top1 accuracy: {}'.format(top1)
    print 'Top5 accuracy: {}'.format(top5)
    print ''
    
    set_list = np.array(list(set(true_lab)))
    top1_list, top5_list = [], []
    for i in set_list:
        args = (true_lab == i)
        tmp_pred = pred_lab[args]
        tmp_top1 = np.mean(i == tmp_pred[:, 0])
        tmp_top5 = np.mean([i in j for j in tmp_pred])
        top1_list.append(tmp_top1)
        top5_list.append(tmp_top5)
    top1_list, top5_list = np.array(top1_list), np.array(top5_list)
    
    # Order as a function of Top1
    args = np.argsort(top1_list)[::-1]
    set_list, top1_list, top5_list = set_list[args], top1_list[args], top5_list[args]
    
    # Order Top1==0 species by Top5
    args = (top1_list == 0)
    args2 = np.argsort(top5_list[args])[::-1]
    set_list[args] = set_list[args][args2]
    top1_list[args] = top1_list[args][args2]
    top5_list[args] = top5_list[args][args2]
        
    return set_list, top1_list, top5_list

#%%
#==============================================================================
# Detailed accuracies per species
#==============================================================================
#
##outfile_names = ['plantnettool_predictions_google',
##                 'plantnettool_predictions_ptflora',
##                 'plantnettool_predictions_inaturalist', 
##                 'resnet_predictions_google',
##                 'resnet_predictions_ptflora',
##                 'resnet_predictions_inaturalist']
##fig, axes = plt.subplots(2,3)
##axes = axes.flatten()

outfile_names = ['resnet_predictions_google',
                 'resnet_predictions_ptflora',
                 'resnet_predictions_inaturalist']
fig, axes = plt.subplots(1,3)
axes = axes.flatten()

for i, ax in enumerate(axes):
    outfile_name = outfile_names[i]
    set_list, top1_list, top5_list = detailed_acc(outfile_name)
       
    ind = np.arange(len(set_list))    # the x locations for the groups

    filtered = savgol_filter(top5_list, 25, 3)
    ax.fill_between(ind, top5_list, color='#0074D9', label='Top5')
    ax.fill_between(ind, top1_list, color='#FFDC00', label='Top1') 
    
    ax.set_xlabel('Species')
    ax.set_ylabel('Accuracy')
    ax.legend(frameon=False)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, len(set_list)])
#    ax.tick_params(axis='x', top='off') 
#    ax.set_title(outfile_name)

#%%
#==============================================================================
# Accuracy per specie vs images in database
#==============================================================================

#outfile_names = ['plantnettool_predictions_google',
#                 'plantnettool_predictions_ptflora',
#                 'plantnettool_predictions_inaturalist', 
#                 'resnet_predictions_google',
#                 'resnet_predictions_ptflora',
#                 'resnet_predictions_inaturalist']
#fig, axes = plt.subplots(2,3)
#axes = axes.flatten()

outfile_names = ['resnet_predictions_google',
                 'resnet_predictions_ptflora',
                 'resnet_predictions_inaturalist']
fig, axes = plt.subplots(1,3)
axes = axes.flatten()


metadata_im_per_specie = np.genfromtxt(os.path.join(homedir, 'data', 'data_splits', 'synsets_binomial_with_im_number.txt'), dtype='str', delimiter='/n')
im_per_specie = [i.split('[')[1].split(' ')[0] for i in metadata_im_per_specie]
im_per_specie = np.array(im_per_specie, dtype='int')

for i, ax in enumerate(axes):
    set_list, top1_list, top5_list = detailed_acc(outfile_names[i])

    ax.scatter(im_per_specie[set_list], top1_list, s=5, color='#0074D9')    
    ax.set_xlabel('Images in database')
    ax.set_ylabel('Accuracy')
    ax.set_xlim([1,1000])
    ax.set_ylim([0, 1])
    ax.set_xscale('log')
#    ax.axis('equal')
