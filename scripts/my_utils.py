# -*- coding: utf-8 -*-
"""
Miscellanous functions used to train and evaluate in image recognition.

Author: Ignacio Heredia
Date: November 2016
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import resnet50_class as resnet
from data_utils import data_augmentation, standard_tencrop_batch


def test_predictions(test_func, im_list, aug_params=None, crop_mode='random'):
    """
    Function for testing single images with random ten crop.

    Parameters
    ----------
    test_func : theano function
        Function to make predictions.
    im_list : list
        List of image filepaths or urls.
    aug_params : dict, None, optional
        Parameters for data augmentation.
    crop_mode : {'random','standard'}
        Modality of croppping. Random usually works better.

    Returns
    -------
    Arrays with top 5 predicted labels numbers and their corresponding probabilities.

    """
    if aug_params is None:
        aug_params = {}
    aug_params.pop('mode', None)
    pred_lab, pred_prob = [], []
    for i, im in enumerate(im_list):
        print 'Image number: {}'.format(i)
        try:
            if crop_mode == 'random':
                batch = data_augmentation([im]*10, mode='test', **aug_params)
            elif crop_mode == 'standard':
                batch = standard_tencrop_batch(im, **aug_params)
        except Exception:
            print 'Error at Image {}'.format(i)
            pred_lab.append([0]*5)
            pred_prob.append([0]*5)
            continue
        pred_raw = test_func(batch)  # probabilities for all labels for all 10 crops
        pred_tmp = np.sum(pred_raw, axis=0) / 10.  # mean probabilities across crops
        args = pred_tmp.argsort()[-5:][::-1]  # top5 predicted labels
        pred_lab.append(args)
        pred_prob.append(pred_tmp[args])
    return np.array(pred_lab), np.array(pred_prob)


def single_prediction(test_func, im_list, aug_params=None, crop_mode='random'):
    """
    Function for identying a SINGLE plant with one or more images.
    It combines the predictions for all the images to output the best possible 
    labels overall.

    Parameters
    ----------
    test_func : theano function
        Function to make predictions
    im_list : list
        List of image filepaths or urls.
    aug_params : dict, None, optional
        Parameters for data augmentation.
    crop_mode : {'random','standard'}
        Modality of croppping. Random usually works better.

    Returns
    -------
    Arrays with top 5 predicted labels numbers and their corresponding probabilities.

    """
    if aug_params is None:
        aug_params = {}
    aug_params.pop('mode', None)
    pred = []
    for i, im in enumerate(im_list):
        print 'Image number: {}'.format(i)
        try:
            if crop_mode == 'random':
                batch = data_augmentation([im]*10, mode='test', **aug_params)
            if crop_mode == 'standard':
                batch = standard_tencrop_batch(im, **aug_params)
        except Exception:
            print 'Error at Image {}'.format(i)
            continue
        pred_raw = test_func(batch)  # probabilities for all labels for all 10 crops
        pred_tmp = np.sum(pred_raw, axis=0) / 10.  # mean probabilities across crops
        pred.append(pred_tmp)
    pred_prob = np.sum(pred, axis=0) / len(im_list)  # mean probabilities across images
    args = pred_prob.argsort()[-5:][::-1]  # top5 predicted labels
    pred_lab = args
    return np.array(pred_lab), np.array(pred_prob[args])


def train_model(X_train, y_train, X_val=None, y_val=None,
                net_params=None, aug_params=None):
    """
    Train model

    Parameters
    ----------
    X_train, y_train : numpy array
        Training data and targets
    X_val, y_val : numpy array, optional
        Validation data and targets
    net_params : dict, None, optional
        Net parameters (check the net_class).
    aug_params : dict, None, optional
        Data augmentation parameters (check the data_augmentation function).

    """
    if net_params is None:
        net_params = {}

    # Build the net and train
    net = resnet.prediction_net(**net_params)
    test_func = net.build_and_train(X_train, y_train, X_val, y_val,
                                    aug_params=aug_params)

    # Final predictions and accuracy
    train_pred, train_prob = test_predictions(test_func, X_train)
    train_acc = np.mean(train_pred[:, 0] == y_train)
    print 'Final training accuracy: {}%'.format(train_acc*100)

    if X_val is not None:
        val_pred, val_prob = test_predictions(test_func, X_val)
        val_acc = np.mean(val_pred[:, 0] == y_val)
        print 'Final validation accuracy: {}%'.format(val_acc*100)
