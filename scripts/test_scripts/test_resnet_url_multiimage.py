# -*- coding: utf-8 -*-
"""
Runfile for testing purposes

Author: Ignacio Heredia
Date: December 2016

Description:
This file contains the commands for testing a convolutional net for image
classification on some dataset file consisting in url and labels.
The difference with test_resrnet_url is that this implements the MULTI image prediction.
This script does parallezation fllowing the scheme:
* Fetch and augment N observations with n images in parallel (n depends on the
observation) using all available threads.
    --> output = megabatch(N, n, 10, 3, 224, 224)
* Processing sequentially the prediction part passing minibatches of (n, 10, 3, 224, 224)
* ... the same with the following N images ...
"""

import numpy as np
import os
import sys
import threading
import json
import time
homedir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(homedir, 'scripts'))
from data_utils import data_augmentation
from model_utils import load_model

# Select dataset
dataset = 'inaturalist'
print("Loading url data...")
metadata = np.genfromtxt(os.path.join(homedir, 'data', 'data_splits', 'synsets.txt'), dtype='str', delimiter='/n')
metadata_binomial = np.genfromtxt(os.path.join(homedir, 'data', 'data_splits', 'synsets_binomial.txt'), dtype='str', delimiter='/n')

if dataset == 'inaturalist':
    outfile_name = 'resnet_predictions_inaturalist_multiimage.json'
    im_list = np.genfromtxt(os.path.join('test_datasets', 'test_inaturalist_multiimage.txt'), dtype='str', delimiter=' ; ')
    X, y = im_list[:, 0], np.array([list(metadata_binomial).index(label) for label in im_list[:, 1]]).astype(np.int32)

# Loading the model
modelname = 'resnet50_6182classes_100epochs'
test_func = load_model(os.path.join(homedir, 'scripts', 'training_weights', modelname + '.npz'), output_dim=6182)

# Launching prediction
pred_dict = {'pred_lab': [], 'pred_prob': [], 'true_lab': [], 'im_per_obs': []}
acc_dict = {'top1': 0, 'top5': 0, 'counts': 0}
batch_size = 100  # batch shape (batchsize, 10, 224, 224, 3)
max_threads = 20  # parallel threads


def batch_fetching(image_list, image_labels, output):

    for i, im in enumerate(image_list):
        print 'Image number: {}'.format(i)
        batch_tmp = []
        for subim in im.split(' '):
            try:
                subbatch_tmp = data_augmentation([subim]*10, mode='test', filemode='url')
                batch_tmp.append(subbatch_tmp)
            except Exception:
                print 'Error at Image {}'.format(i)
                continue
        if len(batch_tmp) > 1:  # enough image queries have been successful as to have a multiimage prediction
            output['batch'].append(batch_tmp)
            output['labels'].append(image_labels[i])
            output['im_per_obs'].append(len(batch_tmp))


def parallel_batch_fetching(image_list, image_labels, thread_num=5, tmax=600):

    output = {'batch': [], 'labels': [], 'im_per_obs': []}
    print 'Fetching internet images and processing the batch in parallel ...'
    t_list = []
    for args in np.array_split(np.arange(len(image_list)), thread_num):
        X_tmp, y_tmp = image_list[args], image_labels[args]
        t = threading.Thread(target=batch_fetching, args=(X_tmp, y_tmp, output))
        t_list.append(t)

    for t in t_list:
        t.daemon = True
        t.start()

    t0 = time.time()
    for t in t_list:
        dt = time.time() - t0
        t.join(max(tmax-dt, 0.))

    return output['batch'], output['labels'], output['im_per_obs']


def run_parallel(image_list, image_labels, pred_dict, acc_dict, thread_num, tmax):
    """
    Parameters
    ----------
    image_list : array of strs
        List of urls to query
    image_labels : array of ints
        Label number of the images
    pred_dict : dict
        Storage of the prediction label and probabilities.
    acc_dict : dict
        Storage of intermediate accuarcy results
    thread_num : int
        Number of threads to use
    tmax : float
        Maximal time allowed to the program to finish the task. We set this so
        that the whole script isn't blocked if a thread gets stucked for some reason.
        Should be ~ batchsize / (thread_num * internet_speed).
    batchsize : int
        Number of images to process before doing a backup
    """

    for args in np.array_split(np.arange(len(image_list)), np.ceil(1.*len(image_list)/batch_size)):
        X_tmp, y_tmp = image_list[args], image_labels[args]
        batch, labels, im_per_obs = parallel_batch_fetching(X_tmp, y_tmp, thread_num, tmax)
        print 'Sequentially predicting label ...'
        pred_lab, pred_prob = [], []
        for i, minibatch in enumerate(batch):  # shape (n,10,3,224,224)
            print 'Observation number: {}'.format(i+acc_dict['counts'])
            pred_tmp = []
            for subbatch in minibatch:  # shape (10,3,224,224)
                predsub_raw = test_func(subbatch)  # probabilities for all labels for all 10 crops
                predsub_tmp = np.sum(predsub_raw, axis=0) / len(subbatch)  # mean probabilities across crops
                pred_tmp.append(predsub_tmp)
            pred_tmp = np.sum(pred_tmp, axis=0) / len(pred_tmp)  # mean probabilities across the n images
            args = pred_tmp.argsort()[-5:][::-1]  # top5 predicted labels
            pred_lab.append(args)
            pred_prob.append(pred_tmp[args])
        # Storing predicted labels, predicted probabilities, true labels and number of images per observation
        pred_dict['pred_lab'] += pred_lab
        pred_dict['pred_prob'] += pred_prob
        pred_dict['true_lab'] += labels
        pred_dict['im_per_obs'] += im_per_obs
        # Updating the accuracy results with the current's batch predictions
        pred_lab = np.array(pred_lab)
        for i in range(len(pred_lab)):
            acc_dict['top1'] += 1. * (labels[i] == pred_lab[i, 0])
            acc_dict['top5'] += 1. * (labels[i] in pred_lab[i])
            acc_dict['counts'] += 1.
        print acc_dict
        # Save tmp results
        pred_dict_tmp = pred_dict
        pred_dict_tmp['true_lab'] = np.array(pred_dict_tmp['true_lab']).tolist()
        pred_dict_tmp['pred_lab'] = np.array(pred_dict_tmp['pred_lab']).tolist()
        pred_dict_tmp['pred_prob'] = np.array(pred_dict_tmp['pred_prob']).tolist()
        with open(os.path.join('test_predictions', 'tmp_' + outfile_name), 'wb') as f:
            json.dump(pred_dict_tmp, f)


run_parallel(X, y, pred_dict, acc_dict, thread_num=max_threads, tmax=600.)

print 'Top1 accuracy: {}'.format(1.*acc_dict['top1']/acc_dict['counts'])
print 'Top5 accuracy: {}'.format(1.*acc_dict['top5']/acc_dict['counts'])

pred_dict['true_lab'] = np.array(pred_dict['true_lab']).tolist()
pred_dict['pred_lab'] = np.array(pred_dict['pred_lab']).tolist()
pred_dict['pred_prob'] = np.array(pred_dict['pred_prob']).tolist()

with open(os.path.join('test_predictions', outfile_name), 'wb') as f:
        json.dump(pred_dict, f)
