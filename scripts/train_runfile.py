# -*- coding: utf-8 -*-
"""
Runfile

Author: Ignacio Heredia
Date: September 2016

Description:
This file contains the commands for training a convolutional net for image
classification.
"""

from my_utils import model_ensembling
from data_utils import data_splits, meanRGB

tag = False
im_dir = '/media/ignacio/Datos/datasets/plant_net/images_ori'  # absolute path to file_dir
X_train, y_train, X_val, y_val, metadata = data_splits(im_dir, tag)

if tag:
    mean_RGB, std_RGB = meanRGB(X_train[0])
else:
    mean_RGB, std_RGB = meanRGB(X_train)

net_params = {'output_dim': len(metadata), 'batchsize': 32, 'num_epochs': 5}
aug_params = {'tag': tag, 'mean_RGB': mean_RGB}

model_ensembling(X_train, y_train, X_val, y_val, net_params=net_params, aug_params=aug_params)
