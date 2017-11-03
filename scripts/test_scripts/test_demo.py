# -*- coding: utf-8 -*-
"""
Runfile for demo purposes

Author: Ignacio Heredia
Date: December 2016

Description:
This file contains several commands for testing a convolutional net for image
classification.
"""

import numpy as np
import os
import sys
import json
homedir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(homedir, 'scripts'))
from my_utils import single_prediction, test_predictions
from model_utils import load_model
from plot_utils import augmentation_demo, training_plots, test_plot

metadata = np.genfromtxt(os.path.join(homedir, 'data', 'data_splits', 'synsets.txt'), dtype='str', delimiter='/n')
metadata_binomial = np.genfromtxt(os.path.join(homedir, 'data', 'data_splits', 'synsets_binomial.txt'), dtype='str', delimiter='/n')

modelname = 'resnet50_6182classes_100epochs'

# Load training info
info_file = os.path.join(homedir, 'scripts', 'training_info', modelname + '.json')
with open(info_file) as datafile:
    train_info = json.load(datafile)
mean_RGB = train_info['augmentation_params']['mean_RGB']
output_dim = train_info['training_params']['output_dim']

# Load net weights
test_func = load_model(os.path.join(homedir, 'scripts', 'training_weights', modelname + '.npz'), output_dim=output_dim)

# %% Demo data augmention

im_path = os.path.join(homedir, 'data', 'demo-images', 'image1.jpg')
augmentation_demo(im_path)

# %% Plot training information

#info_file = os.path.join(homedir, 'scripts', 'training_info', modelname + '.json')
training_plots(info_file)

# %% Predict list of local images

splits_dir = os.path.join(homedir, 'data', 'data_splits')
im_dir = '/media/ignacio/Datos/plant_net/images_ori'
data = np.genfromtxt(os.path.join(splits_dir, 'train.txt'), dtype='str', delimiter=' ')
#data = data[:100]
y_data = data[:, -1].astype(np.int32)
X_data = np.array([os.path.join(im_dir, i) for i in data[:, 0]])


def top5(val, l):
    return val in l[:5]
pred_lab, pred_prob = test_predictions(test_func, im_list=X_data, aug_params={'mean_RGB': mean_RGB})
print 'Top1 accuracy: {}%'.format(np.mean(pred_lab[:, 0] == y_data) * 100)
print 'Top5 accuracy: {}%'.format(np.mean(map(top5, y_data, pred_lab)) * 100)

# %% Predict single local image

im_dir = os.path.join(homedir, 'data', 'demo-images')
im_path = [os.path.join(im_dir, 'paper_demo.jpg')]
pred_lab, pred_prob = single_prediction(test_func, im_list=im_path, aug_params={'mean_RGB': mean_RGB})
test_plot(im_path[0], metadata_binomial[pred_lab], pred_prob, display=True)

# %% Predict single observation with multiple local images

im_dir = os.path.join(homedir, 'data', 'demo-images')
im_path = [os.path.join(im_dir, 'image1.jpg'),
           os.path.join(im_dir, 'image2.jpg')]
pred_lab, pred_prob = single_prediction(test_func, im_list=im_path, aug_params={'mean_RGB': mean_RGB})
test_plot(im_path, metadata_binomial[pred_lab], pred_prob, display=False)

# %% Predict single url image

url = ['https://s-media-cache-ak0.pinimg.com/736x/20/1c/e9/201ce94e7998a27257cdc2426ae7060c.jpg']
pred_lab, pred_prob = single_prediction(test_func, im_list=url, aug_params={'mean_RGB': mean_RGB, 'filemode':'url'})
test_plot(url, metadata_binomial[pred_lab], pred_prob, filemode='url', display=False)
