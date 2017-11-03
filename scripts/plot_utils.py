# -*- coding: utf-8 -*-
"""
Miscellanous functions used plot in image recognition.

Author: Ignacio Heredia
Date: November 2016
"""

import numpy as np
import matplotlib.pylab as plt
from data_utils import data_augmentation
from PIL import Image
from scipy.signal import savgol_filter
import requests
from io import BytesIO
import json


def training_plots(info_file, filtering=True):
    """
    Plot the training info.

    Parameters
    ----------
    info_file : str
        Path to json file that  contains a dict with the following keys:
        training_params, train_err, train_acc, val_err, val_acc.
    filtering : bool
        Apply filter to training values to smoothen the plot.

    """
    with open(info_file) as datafile:
        params = json.load(datafile)

    epochs = params['training_params']['num_epochs']
    fig, [ax1, ax2] = plt.subplots(1, 2)
    it_per_epoch = len(params['train_err']) / epochs

    def filter_func(l):
        return savgol_filter(l, 101, 3)

    # Training
    x_tr = 1. * np.arange(len(params['train_err'])) / it_per_epoch
    if filtering:
        ax1.plot(x_tr, filter_func(params['train_err']), label='Training (filtered)')
        ax2.plot(x_tr, filter_func(params['train_acc']), label='Training (filtered)')
    else:
        ax1.plot(x_tr, params['train_err'], label='Training')
        ax2.plot(x_tr, params['train_acc'], label='Training')

    # Validation
    val_err = np.split(np.array(params['val_err']), epochs)
    val_acc = np.split(np.array(params['val_acc']), epochs)
    val_err = [np.mean(l) for l in val_err]
    val_acc = [np.mean(l) for l in val_acc]
    x_val = np.arange(1, epochs+1)
    ax1.plot(x_val, val_err, label='Validation')
    ax2.plot(x_val, val_acc, label='Validation')

    ax2.set_ylim([0, 1])
    ax1.set_xlabel('Epochs'), ax1.set_title('Loss')
    ax2.set_xlabel('Epochs'), ax2.set_title('Accuracy')
    plt.legend()


def augmentation_demo(filename, it=20, mean_RGB=None):
    """
    Little demo to show how data augmentation is performed on a single image.

    Parameters
    ----------
    filename : str
        Path of the image
    it : int
        Number of examples of data augmentation

    """
    if mean_RGB is None:
        mean_RGB = np.array([107.59348955,  112.1047813,   80.9982362])
    else:
        mean_RGB = np.array(mean_RGB)  
    batch = data_augmentation([filename]*it, mean_RGB=mean_RGB)
    
    plt.ion()
    fig, [ax1, ax2] = plt.subplots(1, 2, num=1)
    ax1.set_title('Original image')
    ax2.set_title('Transformed image')
    image = Image.open(filename)
    ax1.imshow(np.asarray(image))
    
    mean_RGB = mean_RGB.astype(np.float32)
    for im in batch:
        im = im[::-1, :, :]
        im = np.transpose(im, (1, 2, 0))
        im = im + mean_RGB[None, None, :]
        ax2.imshow(im.astype(np.uint8))
        plt.waitforbuttonpress(1)


def test_plot(filename, pred_lab, pred_prob, true_lab=None, filemode=None, display=True):
    """
    Display image and predicted label in matplotlib plot.

    Parameters
    ----------
    filename : str
        Image path or url
    pred_lab : numpy array
        Top5 prediction labels
    pred_prob : numpy array
        Top 5 prediction probabilities
    True_lab : str, None, optional
        True label
    filemode : str, None
        Either None or 'url' to read internet images
    display : bool
        If True displays image + predicted labels in matplotlib plot.
        If False displays predicted labels in command line.

    """
    pred_tmp = ['{}.  {} | {:.0f} %'.format(str(i+1), p, pred_prob[i]*100) for i, p in enumerate(pred_lab)]
    text = r''
    if true_lab is not None:
        text += 'True label:\n\n     {}  \n\n'.format(true_lab)
    text += 'Predicted labels: \n\n    ' + '\n    '.join(pred_tmp)
    if display:
        if filemode == 'url':
            filename = BytesIO(requests.get(filename).content)
            im = Image.open(filename)
            im = im.convert('RGB')
        else:
            im = Image.open(filename)
        arr = np.asarray(im)
        fig = plt.figure(figsize=(20, 12))
        ax1 = fig.add_axes((.1, .1, .5, 0.9))
        ax1.imshow(arr)
        ax1.set_xticks([]), ax1.set_yticks([])
        ax1.set_xticklabels([]), ax1.set_yticklabels([])
        t = fig.text(.7, .5, text, fontsize=20)
        t.set_bbox(dict(color='white', alpha=0.5, edgecolor='black'))
    else:
        print text
