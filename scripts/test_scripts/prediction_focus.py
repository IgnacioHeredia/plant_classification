#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Prediction focus

Author: Ignacio Heredia
Date:August 2017

Implement sliding (black) square to find the most interesting parts of the 
image by occluding them and looking at the changes in the prediction confidence.
"""

import numpy as np
import os
import sys
import json
homedir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(homedir, 'scripts'))
from model_utils import load_model
from PIL import Image
import matplotlib.pylab as plt


# Load the image
img = '/home/ignacio/test_plants2/data/demo-images/demo1.jpg' #path to image
X, Y = 224, 224 #network's input size
batchsize = 64
ori = Image.open(img)
ori = ori.resize((X,Y))

# Load the model
modelname = 'resnet50_6182classes_100epochs'
metadata = np.genfromtxt(os.path.join(homedir, 'data', 'data_splits', 'synsets.txt'), dtype='str', delimiter='/n')
                       
# Load training info
info_file = os.path.join(homedir, 'scripts', 'training_info', modelname + '.json')
with open(info_file) as datafile:
    train_info = json.load(datafile)
output_dim = train_info['training_params']['output_dim']
mean_RGB = np.array(train_info['augmentation_params']['mean_RGB'])
if np.any(mean_RGB) == None:
    mean_RGB = np.array([107.59348955, 112.1047813, 80.9982362])

# Load net weights
test_func = load_model(os.path.join(homedir, 'scripts', 'training_weights', modelname + '.npz'), output_dim=output_dim)


def preprocess_batch(im_list):
    """
    Minimal preprocess of the image before feeding to network
    """
    im_list = np.array(im_list) - mean_RGB[None, None, None, :]  # mean centering
    im_list = im_list.transpose(0, 3, 1, 2)  # shape(N, 3, 224, 224)
    im_list = im_list[:, ::-1, :, :]  # switch from RGB to BGR   
    return im_list.astype(np.float32)


def monoscale(square_size=20, stride=4, square_RGBcolor=np.array([0,0,0]) ):
    """
    Returns a mask of probabilities that assses the relative importance of each pixel
    
    Parameters
    ----------
    square_size : positive int
        Size of the occluding square. Useful to see importance across scales.
    stride : positive int
        Step size when moving the square along the image. If different than 1 then 
        you kind of average for values in between. If equal to 1 then you do the 
        detailed prediction. The higher the value the faster the prediction is made.
    square_RGBcolor : numpy array. Shape (1,3)
        RGB color of the occluding square.
    """
    print 'Processing ocluded images...'
    
    # Main prediction (not ocludded)
    im_list = [np.array(ori)]
    im_list = preprocess_batch(im_list)
    pred_raw = test_func(im_list)
    arg_max = np.argmax(pred_raw)
    pred_ori = pred_raw[0, arg_max]
    print 'Predicted specie: {} ({}%)'.format(metadata[arg_max], int(100 * pred_ori))
    if pred_ori < 0.7:
        print 'WARNING: Low confidence prediction, possible meaningless results.'   
    
    # Predict occluded images
    black_square = np.ones((square_size,square_size,3)) * square_RGBcolor
    im_list, pred = [], np.array([]).reshape(0, output_dim)
    ind = 0
    
    x_range = np.arange(0, X-square_size, stride)
    y_range = np.arange(0, Y-square_size, stride)
    tot = x_range.shape[0] * y_range.shape[0]
    num_batch = np.ceil(1.*tot/batchsize)
    
    for y in y_range:
        for x in x_range:
            img_occ = np.array(ori)
            img_occ[y:y+square_size, x:x+square_size, :] = black_square
            im_list.append(img_occ)
            ind += 1
            if (ind % batchsize)==0 or ind==tot:
                print 'Batch {:.0f}/{:.0f}'.format(np.ceil(1.*ind/batchsize), num_batch)
                im_list = preprocess_batch(im_list)
                pred_raw = test_func(im_list)
                pred = np.vstack((pred, pred_raw))
                im_list = []
    
    # Construct prediction mask
    pred_prob = pred[:, arg_max] #select probabilities corresponding to the main prediction
    pred_mask = np.zeros(ori.size)
    count_mask = np.zeros(ori.size)
    i = 0
    for y in y_range:
        for x in x_range:
            pred_mask[y:y+square_size, x:x+square_size] += pred_prob[i] * np.ones((square_size,square_size))
            count_mask[y:y+square_size, x:x+square_size] += np.ones((square_size,square_size))
            i += 1
            
    # Replace the zeros (by edge padding) in count_mask to avoid division by zero
    diffx = X - (np.amax(x_range) + square_size)
    diffy = Y - (np.amax(y_range) + square_size)
    pred_mask = pred_mask[:-diffy, :-diffx]
    count_mask = count_mask[:-diffy, :-diffx]
    pred_mask = np.pad(pred_mask, ((0,diffy), (0, diffx)), mode='edge')
    count_mask = np.pad(count_mask, ((0,diffy), (0, diffx)), mode='edge')
    
    # Normalize pred_mask
    pred_mask /= count_mask #average
    pred_mask = 1 - pred_mask/pred_ori #positive value -> prediction went down
    pred_mask /= square_size #make it scale free, because bigger square size will tend to reduce more the prob irrespectively of the importance
    
    return pred_mask


def multiscale(scale_list=[20,50,100]):
    """
    Merges the mask created by monoscale() at different scales (square sizes)
    
    Parameters
    ----------
    scale_list : list of ints
        Size of the squares to make the maps that will be merged
    """
    pred = []   
    for scale in scale_list:
        print 'Computing at scale {} ...'.format(scale)
        pred_prob = monoscale(square_size=scale)
        pred.append(pred_prob)
        
    return np.mean(pred, axis=0)
    

def plots(pred_prob):
    """
    Positive/Negative values mean that when occluding that pixel the prediction went down/up.
    
    Parameters
    ----------
    pred_prob : np.array. shape(x,y)
        Prediction mask to be plotted
    """
    
    # Plot mask and original image
    fig, ax = plt.subplots(1,2, figsize=(12, 4))
    cf = ax[0].imshow(pred_prob, clim=[np.amin(pred_prob), np.amax(pred_prob)])
    fig.colorbar(cf, ax=ax[0], fraction=0.046, pad=0.04)
    ax[0].set_title('Importance mask')
    x,y = pred_prob.shape
    ori_patch = np.array(ori)[(X-x)/2:(X+x)/2, (Y-y)/2:(Y+y)/2] #central patch
    ax[1].imshow(ori_patch)
    ax[1].set_title('Original image (patch)')
    
    # Plot contour plots of the mask over the image
    plt.figure()
    plt.imshow(ori_patch)
    CS = plt.contour(np.arange(pred_prob.shape[0]), np.arange(pred_prob.shape[1]), pred_prob)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Contour plot')    
    
    # Plot RGBA image (soft mask)
    plt.figure()
    new_prob = pred_prob - np.amin(pred_prob)
    new_prob /= np.amax(new_prob)
    alpha_channel = np.round(new_prob*255)
    rgba_arr = np.dstack((ori_patch, alpha_channel)).astype(np.uint8)
    plt.imshow(rgba_arr)
    plt.title('RGBA image')
    

if __name__ == "__main__":
    pred = multiscale()
    plots(pred)
    